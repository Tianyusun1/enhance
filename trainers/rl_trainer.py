# File: trainers/rl_trainer.py (V11.3: Gravity & Balance Optimized)

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
    [V11.3 Final] "重力平衡" + "去角点效应" 修正版
    
    针对 400 轮后出现的 "布局偏上" 和 "死板站位(左上右下)" 进行物理修正：
    1. [新增] 全局重心奖励 (Global Balance)：强制画面重心下沉，解决下方空洞。
    2. [新增] 水平向心力 (Centering)：对抗过强的斥力，防止物体被锁死在画布四角。
    3. [微调] 稍微降低 Dispersion 权重 (10.0 -> 7.0)，给布局灵活性留出空间。
    4. [保留] 超分热力图、重叠熔断、自动绘图等所有核心功能。
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        self.rl_lr = float(config['training'].get('rl_learning_rate', 1e-6))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        self.success_scale = 2.0
        self.failure_scale = 0.1
        
        # --- 权重策略：引入重力，打破死板 ---
        self.w_iou = 2.0    
        self.w_rel = 4.0      
        self.w_physics = 6.0    
        self.w_align = 4.0      
        self.w_hm_size = 2.5    
        
        # [调整] 斥力从 10.0 降为 7.0，防止把物体推死在角落
        self.w_disp = 7.0      
        self.w_overlap = -15.0 
        self.w_bound = -3.0
        
        # [新增] 重心与平衡权重
        self.w_balance = 3.0   # 垂直重心：解决"偏上"
        self.w_center = 1.5    # 水平向心：解决"贴边"

        self.last_reward_stats = {}
        self.reward_history = []
        self.plot_path_reward = os.path.join(self.output_dir, "rl_reward_trajectory.png")

        print(f"[RLTrainer V11.3] Gravity System Initialized.")
        print(f" -> Balance(Vertical): {self.w_balance} | Center(Horizontal): {self.w_center}")

    def compute_reward(self, dynamic_layout, batch, attention_maps=None):
        """
        计算奖励：集成物理、热力图、斥力以及新的重力平衡系统
        """
        B, T, _ = dynamic_layout.shape
        device = dynamic_layout.device
        
        pred_boxes = dynamic_layout[..., :4]
        loss_mask = batch['loss_mask']          
        target_boxes = batch['target_boxes'][..., :4] 
        kg_spatial_matrix = batch.get('kg_spatial_matrix')
        kg_class_ids = batch['kg_class_ids']    
        
        obj_rewards = torch.zeros(B, T, device=device)
        
        # 1. 物理大小约束
        r_physics = self._calculate_physics_size_reward(pred_boxes, kg_class_ids) * self.w_physics

        # 2. 真值一致性
        r_iou = self._calculate_iou(pred_boxes, target_boxes) * loss_mask * self.w_iou

        # 3. 空间关系
        r_rel = torch.zeros(B, T, device=device)
        if kg_spatial_matrix is not None:
            r_rel = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids) * self.w_rel

        # 4. 超分热力图引导 (保留 256x256 逻辑)
        r_align = torch.zeros(B, T, device=device)
        r_hm_size = torch.zeros(B, T, device=device)
        if attention_maps is not None:
            hi_res = F.interpolate(attention_maps, size=(256, 256), mode='bilinear', align_corners=False)
            potential_field = F.avg_pool2d(hi_res, kernel_size=5, stride=1, padding=2)
            r_align = self._calculate_attention_alignment(potential_field, pred_boxes) * self.w_align
            r_hm_size = self._calculate_heatmap_area_reward(potential_field, pred_boxes) * self.w_hm_size

        # 5. 空间力场 (斥力 + 边界)
        r_disp = self._calculate_dispersion_reward(pred_boxes, loss_mask) * self.w_disp
        r_bound = self._calculate_boundary_penalty(pred_boxes) * self.w_bound
        
        # 6. [新增] 重力与平衡系统
        # (A) 垂直重心奖励：把布局拉下来
        r_balance = self._calculate_vertical_balance_reward(pred_boxes, loss_mask) * self.w_balance
        # (B) 水平向心奖励：把物体从角落拉出来
        r_center = self._calculate_horizontal_centering_reward(pred_boxes) * self.w_center

        # 7. 重叠惩罚与熔断
        overlap_penalty = self._calculate_overlap_penalty(pred_boxes)
        r_over = overlap_penalty * self.w_overlap 
        veto_factor = (1.0 - overlap_penalty * 5.0).clamp(min=0.0)
        
        # --- 最终汇总 ---
        # 物理、对齐、重力是基础分
        obj_rewards += (r_physics + r_align + r_balance + r_center) 
        
        # 关系、分散等受到熔断限制
        obj_rewards += (r_iou + r_hm_size + r_rel + r_disp) * veto_factor
        
        # 惩罚项无条件生效
        obj_rewards += r_bound 
        obj_rewards += r_over 

        self.last_reward_stats = {
            'Phy': r_physics.mean().item(),
            'Align': r_align.mean().item(),
            'Disp': r_disp.mean().item(),
            'Bal': r_balance.mean().item(), # 监控重心
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
            
            rl_loss = -(s_out[1] * final_adv).mean()
            
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
                # 打印 Balance 指标
                print(f"[Step {step+1}] R:{reward_sample.mean().item():.2f} | Phy:{s.get('Phy',0):.2f} | Bal:{s.get('Bal',0):.2f} | Over:{s.get('Over',0):.2f}")

        # Plotting
        avg_reward = total_reward / steps
        self.reward_history.append(avg_reward)
        self._plot_reward_history()
        return avg_reward

    # ================= 新增：重力与平衡函数 =================

    def _calculate_vertical_balance_reward(self, pred_boxes, mask):
        """
        [新增] 全局重心奖励：解决布局整体偏上的问题
        计算所有有效框的 Y 中心均值，鼓励其接近 0.6 (中下部)，惩罚 < 0.4 (太高)
        """
        cy = pred_boxes[..., 1]
        # 使用 mask 计算有效物体的平均 Y
        # 注意：需要防止除以 0，且只计算 mask=1 的物体
        weighted_sum_cy = (cy * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        mean_cy = weighted_sum_cy / count
        
        # 目标重心：0.6 (微微偏下)。如果重心太靠上(0.3)或太靠下(0.9)都会扣分
        return torch.exp(-5.0 * (mean_cy.unsqueeze(1) - 0.6) ** 2)

    def _calculate_horizontal_centering_reward(self, pred_boxes):
        """
        [新增] 水平向心力：解决物体死贴左右边缘的问题
        给一个微弱的力，让物体 X 轴倾向于靠近 0.5，打破角点最优解
        """
        cx = pred_boxes[..., 0]
        # 计算离中心的距离
        dist_to_center = torch.abs(cx - 0.5)
        # 距离越近奖励越高，但不用太强，主要是为了打破对称僵局
        return torch.exp(-2.0 * dist_to_center ** 2)

    # ================= 原有辅助函数 =================

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
        # 降权到 4.0，防止过度推离
        return torch.clamp((sum_dist / v_mask.float().sum(dim=2).clamp(min=1.0)) * 4.0, max=1.0)

    def _calculate_boundary_penalty(self, pred_boxes):
        cx, cy = pred_boxes[..., 0], pred_boxes[..., 1]
        pen = F.relu(0.05 - cx) + F.relu(cx - 0.95) + F.relu(0.05 - cy) + F.relu(cy - 0.95)
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