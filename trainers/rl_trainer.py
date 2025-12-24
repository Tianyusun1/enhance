# File: trainers/rl_trainer.py (V11.2: Model-Derived Heatmap & Bold Explorer - Full Version)

import torch
import torch.nn.functional as F
import torch.optim as optim
from .trainer import LayoutTrainer
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from trainers.loss import compute_kl_loss

# --- [V10.0 保留] 物理常识锚点 ---
CLASS_SIZE_PRIORS = {
    0: 0.30, 1: 0.30, 2: 0.35, 3: 0.30, 6: 0.20, 5: 0.15,
    7: 0.10, 4: 0.08, 10: 0.08, 8: 0.04, 9: 0.02
}

class RLTrainer(LayoutTrainer):
    """
    [V11.2 Final] "大胆探索" + "模型热力图" 完整版
    
    核心特性：
    1. [实时引导] 直接利用模型 forward_rl 产出的 pred_heatmaps 计算 Align 奖励，无需硬盘读取。
    2. [大胆试错] 非对称 Advantage 机制，奖励成功尝试，宽容失败探索。
    3. [物理常识] 强力锚点权重，纠正物体大小失衡。
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        # RL 超参数
        self.rl_lr = float(config['training'].get('rl_learning_rate', 1e-6))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        
        # 探索非对称系数
        self.success_scale = 2.0
        self.failure_scale = 0.1
        
        # 奖励权重配置
        reward_cfg = config['training'].get('reward_weights', {})
        self.w_iou = 2.0    
        self.w_rel = 3.0    
        self.w_physics = 4.0  # 物理锚点
        self.w_align = 5.0    # 语义对齐(实时热力图)
        self.w_hm_size = 1.0  # 语义大小微调
        self.w_disp = 3.0     # 空间分散
        self.w_overlap = -5.0 # 重叠惩罚
        self.w_bound = -2.0   # 边界惩罚

        self.last_reward_stats = {}
        self.reward_history = []
        self.plot_path_reward = os.path.join(self.output_dir, "rl_reward_trajectory.png")

        print(f"[RLTrainer V11.2] Model-Heatmap & Risk-Seeking Mode Initialized.")
        print(f" -> Heatmap: Real-time from Model Forward")
        print(f" -> Exploration: Asymmetric Advantage (Success x2.0, Fail x0.1)")

    def compute_reward(self, dynamic_layout, batch, attention_maps=None):
        """
        计算奖励：集成物理锚点、模型实时产出的热力图引力和空间斥力
        """
        B, T, _ = dynamic_layout.shape
        device = dynamic_layout.device
        
        pred_boxes = dynamic_layout[..., :4]
        loss_mask = batch['loss_mask']          
        target_boxes = batch['target_boxes'][..., :4] 
        kg_spatial_matrix = batch.get('kg_spatial_matrix')
        kg_class_ids = batch['kg_class_ids']    
        
        obj_rewards = torch.zeros(B, T, device=device)
        
        # 1. 物理锚点 (我是鸟，我就得小)
        r_physics = self._calculate_physics_size_reward(pred_boxes, kg_class_ids) * self.w_physics

        # 2. 基础真值一致性
        iou = self._calculate_iou(pred_boxes, target_boxes)
        r_iou = iou * loss_mask * self.w_iou

        r_rel = torch.zeros(B, T, device=device)
        if kg_spatial_matrix is not None:
            rel_scores = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids)
            r_rel = rel_scores * self.w_rel

        # 3. 语义引导 (直接利用传入的模型产出的实时热力图)
        r_align = torch.zeros(B, T, device=device)
        r_hm_size = torch.zeros(B, T, device=device)
        if attention_maps is not None:
            r_align = self._calculate_attention_alignment(attention_maps, pred_boxes) * self.w_align
            r_hm_size = self._calculate_heatmap_area_reward(attention_maps, pred_boxes) * self.w_hm_size

        # 4. 空间斥力与围栏
        r_disp = self._calculate_dispersion_reward(pred_boxes, loss_mask) * self.w_disp
        r_bound = self._calculate_boundary_penalty(pred_boxes) * self.w_bound
        overlap_penalty = self._calculate_overlap_penalty(pred_boxes)
        
        # 熔断机制
        veto_factor = (1.0 - overlap_penalty * 3.0).clamp(min=0.0)
        
        # 5. 汇总
        obj_rewards += (r_physics + r_align) 
        obj_rewards += (r_iou + r_hm_size + r_rel + r_disp) * veto_factor
        obj_rewards += r_bound 
        r_over = overlap_penalty * self.w_overlap 
        obj_rewards += r_over

        self.last_reward_stats = {
            'Phy': r_physics.mean().item(),
            'Align': r_align.mean().item(),
            'Disp': r_disp.mean().item(),
            'Over': r_over.mean().item(),
        }

        valid_count = loss_mask.sum(dim=1).clamp(min=1.0)
        return (obj_rewards * loss_mask).sum(dim=1) / valid_count

    def train_rl_epoch(self, epoch):
        self.model.train()
        total_reward = 0
        steps = 0
        
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"RL Epoch {epoch}")):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
            
            # 1. Baseline (Greedy)
            self.model.eval()
            with torch.no_grad():
                # [V11.2 关键] 接收三元组返回值
                baseline_out = self.model.forward_rl(
                    batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                    batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'),
                    sample=False
                )
                baseline_layout, _, baseline_heatmaps = baseline_out
                reward_baseline = self.compute_reward(baseline_layout, batch, attention_maps=baseline_heatmaps)
            
            # 2. Sample (Stochastic)
            self.model.train()
            sample_out = self.model.forward_rl(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'),
                sample=True
            )
            sample_layout, log_probs, sample_heatmaps = sample_out
            if log_probs.dim() > 1: log_probs = log_probs.sum(dim=1)
            
            reward_sample = self.compute_reward(sample_layout, batch, attention_maps=sample_heatmaps)
            
            # 3. 非对称优势计算 (Risk-Seeking)
            raw_advantage = reward_sample - reward_baseline 
            if raw_advantage.std() > 1e-6:
                norm_advantage = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-8)
            else:
                norm_advantage = raw_advantage

            pos_mask = (norm_advantage > 0).float()
            neg_mask = (norm_advantage <= 0).float()
            final_advantage = norm_advantage * (pos_mask * self.success_scale + neg_mask * self.failure_scale)
            
            rl_loss = -(log_probs * final_advantage).mean()
            
            # 4. Supervised Auxiliary Loss
            mu, logvar, dynamic_layout_sup, decoder_output, _ = self.model(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'),
                target_boxes=batch['target_boxes']
            )
            
            loss_tuple = self.model.get_loss(
                pred_cls=None, pred_bbox_ids=None, pred_boxes=dynamic_layout_sup, 
                pred_count=None, layout_seq=None, layout_mask=batch['loss_mask'], 
                num_boxes=batch['num_boxes'], target_coords_gt=batch['target_boxes'],
                kg_spatial_matrix=batch.get('kg_spatial_matrix'), kg_class_weights=batch.get('kg_class_weights'),
                kg_class_ids=batch['kg_class_ids'], decoder_output=decoder_output, gestalt_mask=batch.get('gestalt_mask') 
            )
            
            supervised_loss = loss_tuple[0]
            l_gestalt = loss_tuple[-1]
            kl_loss = compute_kl_loss(mu, logvar) if mu is not None else torch.tensor(0.0)

            total_combined_loss = rl_loss + 0.2 * (supervised_loss + kl_loss)
            
            self.optimizer.zero_grad()
            total_combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_reward += reward_sample.mean().item()
            steps += 1
            
            if (step + 1) % 10 == 0:
                stats = self.last_reward_stats
                print(f"[RL V11.2] Step {step+1} | R:{reward_sample.mean().item():.2f} | "
                      f"Phy:{stats.get('Phy',0):.2f} | Align:{stats.get('Align',0):.2f} | "
                      f"Disp:{stats.get('Disp',0):.2f}")

        avg_reward = total_reward / steps
        self.reward_history.append(avg_reward)
        self._plot_reward_history()
        return avg_reward

    # ================= 辅助函数 =================

    def _calculate_physics_size_reward(self, pred_boxes, kg_class_ids):
        if kg_class_ids is None: return torch.zeros(pred_boxes.shape[:2], device=pred_boxes.device)
        rewards = torch.zeros(pred_boxes.shape[:2], device=pred_boxes.device)
        pred_area = (pred_boxes[..., 2] * pred_boxes[..., 3]).clamp(min=1e-6)
        target_priors = torch.zeros_like(pred_area)
        for cid, prior_area in CLASS_SIZE_PRIORS.items():
            mask = (kg_class_ids == cid)
            if mask.any(): target_priors[mask] = prior_area
        target_priors[target_priors == 0] = 0.15
        log_diff = torch.abs(torch.log(pred_area) - torch.log(target_priors))
        return torch.exp(-0.5 * (log_diff ** 2))

    def _calculate_heatmap_area_reward(self, attn_maps, boxes):
        B, T, H, W = attn_maps.shape
        rewards = torch.zeros(B, T, device=boxes.device)
        pred_area = (boxes[..., 2] * boxes[..., 3]).clamp(min=1e-6)
        for b in range(B):
            for t in range(T):
                attn = attn_maps[b, t]
                threshold = attn.max() * 0.4
                active_pixels = (attn > threshold).float().sum()
                heatmap_coverage = (active_pixels / (H * W)).clamp(min=0.01, max=0.95)
                area_diff = torch.abs(torch.log(pred_area[b, t]) - torch.log(heatmap_coverage))
                rewards[b, t] = torch.exp(-area_diff)
        return rewards

    def _calculate_dispersion_reward(self, pred_boxes, mask):
        B, T, _ = pred_boxes.shape
        centers = pred_boxes[..., :2]
        disp_rewards = torch.zeros(B, T, device=pred_boxes.device)
        for b in range(B):
            valid_idx = torch.nonzero(mask[b]).squeeze(1)
            N = len(valid_idx)
            if N < 2: continue 
            valid_centers = centers[b, valid_idx]
            dist_mat = torch.cdist(valid_centers, valid_centers, p=2)
            dist_mat.masked_fill_(torch.eye(N, device=pred_boxes.device).bool(), 0.0)
            avg_dist = dist_mat.sum(dim=1) / (N - 1 + 1e-6)
            disp_rewards[b, valid_idx] = torch.clamp(avg_dist * 2.5, max=1.0)
        return disp_rewards

    def _calculate_boundary_penalty(self, pred_boxes):
        cx, cy = pred_boxes[..., 0], pred_boxes[..., 1]
        margin = 0.05
        pen = F.relu(margin - cx) + F.relu(cx - 0.95) + F.relu(margin - cy) + F.relu(cy - 0.95)
        return torch.clamp(pen * 5.0, max=2.0)
    
    def _calculate_iou(self, pred, target):
        px1, py1 = pred[..., 0]-pred[..., 2]/2, pred[..., 1]-pred[..., 3]/2
        px2, py2 = pred[..., 0]+pred[..., 2]/2, pred[..., 1]+pred[..., 3]/2
        tx1, ty1 = target[..., 0]-target[..., 2]/2, target[..., 1]-target[..., 3]/2
        tx2, ty2 = target[..., 0]+target[..., 2]/2, target[..., 1]+target[..., 3]/2
        ix1, iy1 = torch.max(px1, tx1), torch.max(py1, ty1)
        ix2, iy2 = torch.min(px2, tx2), torch.min(py2, ty2)
        inter = (ix2-ix1).clamp(min=0)*(iy2-iy1).clamp(min=0)
        union = pred[..., 2]*pred[..., 3] + target[..., 2]*target[..., 3] - inter
        return inter / (union + 1e-6)

    def _calculate_relation_reward(self, pred_boxes, kg_spatial_matrix, kg_class_ids):
        if kg_spatial_matrix is None or kg_class_ids is None: return torch.zeros(pred_boxes.shape[:2], device=pred_boxes.device)
        B, T, _ = pred_boxes.shape; device = pred_boxes.device
        cls_indices = (kg_class_ids - 2).clamp(min=0, max=8).long()
        row_idx = cls_indices.unsqueeze(2).expand(B, T, T); col_idx = cls_indices.unsqueeze(1).expand(B, T, T) 
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, T)
        rel_matrix_seq = kg_spatial_matrix[batch_idx, row_idx, col_idx] 
        cy = pred_boxes[..., 1]; cx = pred_boxes[..., 0]; cy_i, cy_j = cy.unsqueeze(2), cy.unsqueeze(1)
        diff_above, diff_below = cy_j - cy_i, cy_i - cy_j
        dist_sq = (cx.unsqueeze(2)-cx.unsqueeze(1))**2 + (cy.unsqueeze(2)-cy.unsqueeze(1))**2
        mask_above, mask_below, mask_inside = (rel_matrix_seq==1)|(rel_matrix_seq==5), rel_matrix_seq==2, rel_matrix_seq==3
        total_rewards = torch.zeros_like(dist_sq)
        total_rewards = torch.where(mask_above, torch.sigmoid(diff_above*5.0), total_rewards)
        total_rewards = torch.where(mask_below, torch.sigmoid(diff_below*5.0), total_rewards)
        total_rewards = torch.where(mask_inside, torch.exp(-dist_sq*10.0), total_rewards)
        valid_rel = mask_above | mask_below | mask_inside
        final = total_rewards.sum(dim=2) / valid_rel.float().sum(dim=2).clamp(min=1.0)
        return final * (valid_rel.sum(dim=2)>0).float()

    def _calculate_overlap_penalty(self, pred_boxes):
        B, T, _ = pred_boxes.shape
        x1, y1 = pred_boxes[..., 0]-pred_boxes[..., 2]/2, pred_boxes[..., 1]-pred_boxes[..., 3]/2
        x2, y2 = pred_boxes[..., 0]+pred_boxes[..., 2]/2, pred_boxes[..., 1]+pred_boxes[..., 3]/2
        area = pred_boxes[..., 2] * pred_boxes[..., 3]
        ix1, iy1 = torch.max(x1.unsqueeze(2), x1.unsqueeze(1)), torch.max(y1.unsqueeze(2), y1.unsqueeze(1))
        ix2, iy2 = torch.min(x2.unsqueeze(2), x2.unsqueeze(1)), torch.min(y2.unsqueeze(2), y2.unsqueeze(1))
        inter = (ix2-ix1).clamp(min=0)*(iy2-iy1).clamp(min=0)
        iou_mat = inter / (area.unsqueeze(2)+area.unsqueeze(1)-inter+1e-6)
        iou_mat.masked_fill_(torch.eye(T, device=pred_boxes.device).bool().unsqueeze(0).expand(B, T, T), 0.0)
        return F.relu(iou_mat.max(dim=2)[0] - 0.05)

    def _calculate_attention_alignment(self, attn_maps, boxes):
        B, T, H, W = attn_maps.shape
        alignment_scores = torch.zeros(B, T, device=boxes.device)
        for b in range(B):
            for t in range(T):
                box = boxes[b, t]
                x1, y1 = int((box[0]-box[2]/2)*W), int((box[1]-box[3]/2)*H)
                x2, y2 = int((box[0]+box[2]/2)*W), int((box[1]+box[3]/2)*H)
                attn_map = attn_maps[b, t]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if x2 > x1 and y2 > y1:
                    alignment_scores[b, t] = attn_map[y1:y2, x1:x2].sum() / (attn_map.sum() + 1e-6)
        return alignment_scores

    def _plot_reward_history(self):
        if not self.reward_history: return
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.reward_history)
            plt.title("RL Reward Trajectory (V11.2 Risk-Seeking)")
            plt.savefig(self.plot_path_reward); plt.close()
        except Exception: pass