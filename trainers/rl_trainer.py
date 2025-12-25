# File: trainers/rl_trainer.py (V11.6: Fix Dimension Bug & Gravity)

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
    [V11.6 Fixed] 修复 Tensor 维度崩溃 + 坚持 "顶部留白/物体沉底" 策略
    
    修正记录:
    1. [CRITICAL FIX] 修复 train_rl_epoch 中的维度错误: [128, 7] * [128] -> 崩溃。
       修正为先对 log_prob 求和得到 [128]，再与 Advantage 相乘。
    2. [Deep Gravity] 既然你说 "留白应该在上边"，这意味着 "物体应该在下边"。
       保持重心目标 Target Y = 0.75 (画布中下部)，权重 5.0，强力把物体往下拉。
    3. [Relation Fix] 包含完整的 _calculate_relation_reward 函数。
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        self.rl_lr = float(config['training'].get('rl_learning_rate', 1e-6))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        self.success_scale = 2.0
        self.failure_scale = 0.1
        
        # --- 权重策略 ---
        self.w_iou = 2.0    
        self.w_rel = 4.0      
        self.w_physics = 6.0    
        self.w_align = 4.0      
        self.w_hm_size = 2.5    
        
        # 斥力与惩罚
        self.w_disp = 7.0      
        self.w_overlap = -15.0 
        self.w_bound = -3.0
        
        # [关键策略] 重力沉底，留白在顶
        self.w_balance = 5.0   # 强力拉拽物体向下 (Target Y=0.75)
        self.w_center = 1.5    # 保持水平居中倾向

        self.last_reward_stats = {}
        self.reward_history = []
        self.plot_path_reward = os.path.join(self.output_dir, "rl_reward_trajectory.png")

        print(f"[RLTrainer V11.6] System Initialized.")
        print(f" -> Gravity Target Y=0.75 (Objects Bottom, Space Top) | Weight: {self.w_balance}")

    def compute_reward(self, dynamic_layout, batch, attention_maps=None):
        """
        计算奖励：集成物理、热力图、斥力以及重力平衡系统
        """
        B, T, _ = dynamic_layout.shape
        device = dynamic_layout.device
        
        pred_boxes = dynamic_layout[..., :4]
        loss_mask = batch['loss_mask']          
        target_boxes = batch['target_boxes'][..., :4] 
        kg_spatial_matrix = batch.get('kg_spatial_matrix')
        kg_class_ids = batch['kg_class_ids']    
        
        obj_rewards = torch.zeros(B, T, device=device)
        
        # 1. 物理大小
        r_physics = self._calculate_physics_size_reward(pred_boxes, kg_class_ids) * self.w_physics

        # 2. IoU
        r_iou = self._calculate_iou(pred_boxes, target_boxes) * loss_mask * self.w_iou

        # 3. 空间关系
        r_rel = torch.zeros(B, T, device=device)
        if kg_spatial_matrix is not None:
            r_rel = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids) * self.w_rel

        # 4. 热力图对齐
        r_align = torch.zeros(B, T, device=device)
        r_hm_size = torch.zeros(B, T, device=device)
        if attention_maps is not None:
            hi_res = F.interpolate(attention_maps, size=(256, 256), mode='bilinear', align_corners=False)
            potential_field = F.avg_pool2d(hi_res, kernel_size=5, stride=1, padding=2)
            r_align = self._calculate_attention_alignment(potential_field, pred_boxes) * self.w_align
            r_hm_size = self._calculate_heatmap_area_reward(potential_field, pred_boxes) * self.w_hm_size

        # 5. 分散与边界
        r_disp = self._calculate_dispersion_reward(pred_boxes, loss_mask) * self.w_disp
        r_bound = self._calculate_boundary_penalty(pred_boxes) * self.w_bound
        
        # 6. [重力系统]
        # 垂直重力：把物体拉到 Y=0.75 (底部)，从而在顶部留白
        r_balance = self._calculate_vertical_balance_reward(pred_boxes, loss_mask) * self.w_balance
        # 水平居中：防止贴边
        r_center = self._calculate_horizontal_centering_reward(pred_boxes) * self.w_center

        # 7. 重叠惩罚
        overlap_penalty = self._calculate_overlap_penalty(pred_boxes)
        r_over = overlap_penalty * self.w_overlap 
        veto_factor = (1.0 - overlap_penalty * 5.0).clamp(min=0.0)
        
        # --- 汇总 ---
        obj_rewards += (r_physics + r_align + r_balance + r_center) 
        obj_rewards += (r_iou + r_hm_size + r_rel + r_disp) * veto_factor
        obj_rewards += r_bound 
        obj_rewards += r_over 

        self.last_reward_stats = {
            'Phy': r_physics.mean().item(),
            'Align': r_align.mean().item(),
            'Disp': r_disp.mean().item(),
            'Bal': r_balance.mean().item(), 
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
            
            # Baseline
            self.model.eval()
            with torch.no_grad():
                b_out = self.model.forward_rl(batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                                             batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'), sample=False)
                reward_baseline = self.compute_reward(b_out[0], batch, attention_maps=b_out[2])
            
            # Sample
            self.model.train()
            s_out = self.model.forward_rl(batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                                         batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'), sample=True)
            reward_sample = self.compute_reward(s_out[0], batch, attention_maps=s_out[2])
            
            # Advantage
            raw_adv = reward_sample - reward_baseline
            std = raw_adv.std()
            norm_adv = (raw_adv - raw_adv.mean()) / (std + 1e-8) if std > 1e-6 else raw_adv
            final_adv = norm_adv * torch.where(norm_adv > 0, self.success_scale, self.failure_scale)
            
            # === [CRITICAL FIX] ===
            # s_out[1] shape: [Batch, Time] (e.g., 128, 7)
            # final_adv shape: [Batch] (e.g., 128)
            # 必须先将 log_prob 在时间维度求和，得到整句话的 log_prob，再乘 advantage
            
            # 1. Sum log_probs over sequence (dim=1) -> [Batch]
            seq_log_prob = s_out[1].sum(dim=1)
            
            # 2. Calculate Policy Gradient Loss -> [Batch] -> scalar
            rl_loss = -(seq_log_prob * final_adv).mean()
            # ======================
            
            # Auxiliary Loss
            if step % 5 == 0:
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
                total_combined_loss = rl_loss + 0.2 * (loss_tuple[0] + (compute_kl_loss(mu, logvar) if mu is not None else 0.0))
            else:
                total_combined_loss = rl_loss

            self.optimizer.zero_grad()
            total_combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_reward += reward_sample.mean().item()
            steps += 1
            if (step + 1) % 10 == 0:
                s = self.last_reward_stats
                print(f"[Step {step+1}] R:{reward_sample.mean().item():.2f} | Phy:{s.get('Phy',0):.2f} | Bal:{s.get('Bal',0):.2f} | Over:{s.get('Over',0):.2f}")

        # Plotting
        avg_reward = total_reward / steps
        self.reward_history.append(avg_reward)
        self._plot_reward_history()
        return avg_reward

    # ================= 辅助函数 =================
    
    def _calculate_relation_reward(self, pred_boxes, kg_spatial_matrix, kg_class_ids):
        """计算空间关系奖励 (Above/Below/Inside)"""
        B, T, _ = pred_boxes.shape
        device = pred_boxes.device
        rewards = torch.zeros(B, T, device=device)
        
        for b in range(B):
            valid_indices = torch.nonzero(kg_class_ids[b] >= 2).squeeze(1)
            if len(valid_indices) < 2: continue

            for i in valid_indices:
                for j in valid_indices:
                    if i == j: continue
                    cid_i = kg_class_ids[b, i].item(); cid_j = kg_class_ids[b, j].item()
                    idx_i, idx_j = int(cid_i) - 2, int(cid_j) - 2
                    if not (0 <= idx_i < 9 and 0 <= idx_j < 9): continue
                    
                    rel = kg_spatial_matrix[b, idx_i, idx_j].item()
                    if rel == 0: continue 
                    
                    box_a = pred_boxes[b, i]; box_b = pred_boxes[b, j]
                    current_reward = 0.0
                    
                    if rel in [1, 5]: # ABOVE: A.y < B.y
                        diff = box_a[1] - box_b[1]
                        if diff < 0: current_reward = 1.0
                        else: current_reward = torch.exp(-3.0 * diff)
                    elif rel == 2: # BELOW: A.y > B.y
                        diff = box_b[1] - box_a[1]
                        if diff < 0: current_reward = 1.0
                        else: current_reward = torch.exp(-3.0 * diff)
                    elif rel == 3: # INSIDE
                        a_x1, a_y1 = box_a[0]-box_a[2]/2, box_a[1]-box_a[3]/2
                        a_x2, a_y2 = box_a[0]+box_a[2]/2, box_a[1]+box_a[3]/2
                        b_x1, b_y1 = box_b[0]-box_b[2]/2, box_b[1]-box_b[3]/2
                        b_x2, b_y2 = box_b[0]+box_b[2]/2, box_b[1]+box_b[3]/2
                        violation = F.relu(b_x1 - a_x1) + F.relu(a_x2 - b_x2) + F.relu(b_y1 - a_y1) + F.relu(a_y2 - b_y2)
                        if violation < 1e-4: current_reward = 1.0
                        else: current_reward = torch.exp(-5.0 * violation)
                    
                    rewards[b, i] += current_reward
        return torch.clamp(rewards, max=1.0)

    def _calculate_vertical_balance_reward(self, pred_boxes, mask):
        """
        [Deep Gravity] 
        目标重心设为 0.75 (3/4处，靠近底部)。
        这样物体会沉底，留白自然就出现在顶部了。
        """
        cy = pred_boxes[..., 1]
        weighted_sum_cy = (cy * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        mean_cy = weighted_sum_cy / count
        # 惩罚系数 4.0，严厉打击任何飘在上面的布局
        return torch.exp(-4.0 * (mean_cy.unsqueeze(1) - 0.75) ** 2)

    def _calculate_horizontal_centering_reward(self, pred_boxes):
        cx = pred_boxes[..., 0]
        dist_to_center = torch.abs(cx - 0.5)
        return torch.exp(-2.0 * dist_to_center ** 2)

    def _calculate_physics_size_reward(self, pred_boxes, kg_class_ids):
        B, T = pred_boxes.shape[:2]
        pred_area = (pred_boxes[..., 2] * pred_boxes[..., 3]).clamp(min=1e-6)
        target_priors = torch.full((B, T), 0.15, device=pred_boxes.device)
        for cid, prior_area in CLASS_SIZE_PRIORS.items():
            target_priors = torch.where(kg_class_ids == cid, torch.tensor(prior_area, device=pred_boxes.device), target_priors)
        return torch.exp(-1.0 * (torch.abs(torch.log(pred_area) - torch.log(target_priors)) ** 2))

    def _calculate_heatmap_area_reward(self, attn_maps, boxes):
        B, T, H, W = attn_maps.shape
        max_vals = attn_maps.view(B, T, -1).max(dim=-1)[0].view(B, T, 1, 1)
        active_pixels = (attn_maps > (max_vals * 0.4)).float().sum(dim=[-1, -2])
        heatmap_coverage = (active_pixels / (H * W)).clamp(min=0.01, max=0.45)
        pred_area = (boxes[..., 2] * boxes[..., 3]).clamp(min=1e-6)
        return torch.exp(-1.5 * torch.abs(torch.log(pred_area) - torch.log(heatmap_coverage)))

    def _calculate_attention_alignment(self, attn_maps, boxes):
        B, T, H, W = attn_maps.shape
        device = boxes.device
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, H, device=device), torch.linspace(0, 1, W, device=device), indexing='ij')
        x1, y1 = (boxes[..., 0]-boxes[..., 2]/2).view(B,T,1,1), (boxes[..., 1]-boxes[..., 3]/2).view(B,T,1,1)
        x2, y2 = (boxes[..., 0]+boxes[..., 2]/2).view(B,T,1,1), (boxes[..., 1]+boxes[..., 3]/2).view(B,T,1,1)
        mask = (grid_x.view(1,1,H,W) >= x1) & (grid_x.view(1,1,H,W) <= x2) & (grid_y.view(1,1,H,W) >= y1) & (grid_y.view(1,1,H,W) <= y2)
        align = (attn_maps * mask.float()).sum(dim=[-1, -2]) / (attn_maps.sum(dim=[-1, -2]) + 1e-6)
        return torch.pow(align, 2)

    def _calculate_dispersion_reward(self, pred_boxes, mask):
        B, T, _ = pred_boxes.shape
        dist_mat = torch.cdist(pred_boxes[..., :2], pred_boxes[..., :2], p=2)
        m1, m2 = mask.unsqueeze(1).bool(), mask.unsqueeze(2).bool()
        v_mask = m1 & m2 & (~torch.eye(T, device=pred_boxes.device).bool().unsqueeze(0))
        sum_dist = (dist_mat * v_mask.float()).sum(dim=2)
        return torch.clamp((sum_dist / v_mask.float().sum(dim=2).clamp(min=1.0)) * 4.0, max=1.0)

    def _calculate_boundary_penalty(self, pred_boxes):
        cx, cy = pred_boxes[..., 0], pred_boxes[..., 1]
        # 放宽底部边界限制(0.98)，允许物体沉底
        pen = F.relu(0.05 - cx) + F.relu(cx - 0.95) + F.relu(0.05 - cy) + F.relu(cy - 0.98)
        return torch.clamp(pen * 5.0, max=2.0)
    
    def _calculate_iou(self, pred, target):
        px1, py1 = pred[..., 0]-pred[..., 2]/2, pred[..., 1]-pred[..., 3]/2
        px2, py2 = pred[..., 0]+pred[..., 2]/2, pred[..., 1]+pred[..., 3]/2
        tx1, ty1 = target[..., 0]-target[..., 2]/2, target[..., 1]-target[..., 3]/2
        tx2, ty2 = target[..., 0]+target[..., 2]/2, target[..., 1]+target[..., 3]/2
        ix = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0) * (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
        return ix / (pred[..., 2]*pred[..., 3] + target[..., 2]*target[..., 3] - ix + 1e-6)

    def _calculate_overlap_penalty(self, pred_boxes):
        B, T, _ = pred_boxes.shape
        x1, y1 = pred_boxes[..., 0]-pred_boxes[..., 2]/2, pred_boxes[..., 1]-pred_boxes[..., 3]/2
        x2, y2 = pred_boxes[..., 0]+pred_boxes[..., 2]/2, pred_boxes[..., 1]+pred_boxes[..., 3]/2
        area = pred_boxes[..., 2] * pred_boxes[..., 3]
        ix = (torch.min(x2.unsqueeze(2), x2.unsqueeze(1)) - torch.max(x1.unsqueeze(2), x1.unsqueeze(1))).clamp(min=0)
        iy = (torch.min(y2.unsqueeze(2), y2.unsqueeze(1)) - torch.max(y1.unsqueeze(2), y1.unsqueeze(1))).clamp(min=0)
        iou = (ix * iy) / (area.unsqueeze(2) + area.unsqueeze(1) - ix * iy + 1e-6)
        iou.masked_fill_(torch.eye(T, device=pred_boxes.device).bool().unsqueeze(0), 0.0)
        return F.relu(iou.max(dim=2)[0] - 0.05)

    def _plot_reward_history(self):
        if not self.reward_history: return
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.reward_history, marker='o', color='b', label='Avg Reward')
            plt.xlabel('Epochs')
            plt.ylabel('Reward Value')
            plt.title('RL Training Progress (Gravity Edition)')
            plt.grid(True)
            plt.legend()
            plt.savefig(self.plot_path_reward)
            plt.close()
        except Exception: pass