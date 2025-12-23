# File: trainers/rl_trainer.py (V8.0: Pure Data-Driven RL & Crash Fix)

import torch
import torch.nn.functional as F
import torch.optim as optim
from .trainer import LayoutTrainer
import time
import os
import matplotlib.pyplot as plt
import numpy as np

from trainers.loss import compute_kl_loss

class RLTrainer(LayoutTrainer):
    """
    [V8.0 Upgrade] 纯数据驱动 RL 训练器
    
    1. 移除了所有人为硬编码的“态势奖励” (Gestalt Reward)，完全信任模型从原图学到的物理规律。
    2. 修复了 get_loss 解包数量不匹配导致的崩溃问题 (适配 12 个返回值)。
    3. 专注于优化“空间布局合理性” (Overlap, Relation, Alignment)，而将“笔墨物理”留给监督头。
    """
    def __init__(self, model, train_loader, val_loader, config, tokenizer, example_poem, test_loader):
        super().__init__(model, train_loader, val_loader, config, tokenizer, example_poem, test_loader)
        
        # RL 超参数
        self.rl_lr = float(config['training'].get('rl_learning_rate', 5e-6))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.rl_lr)
        
        # 奖励权重配置
        reward_cfg = config['training'].get('reward_weights', {})
        self.w_iou = float(reward_cfg.get('iou', 2.0))              
        self.w_rel = float(reward_cfg.get('relation', 5.0)) 
        self.w_align = float(reward_cfg.get('alignment', 8.0)) 
        
        # 激进的重叠惩罚 (保留 V7.3 的防堆叠逻辑)
        cfg_overlap = float(reward_cfg.get('overlap', -1.0))
        if cfg_overlap > -5.0: 
            self.w_overlap = -15.0
        else:
            self.w_overlap = cfg_overlap

        self.last_reward_stats = {}
        self.reward_history = []
        self.plot_path_reward = os.path.join(self.output_dir, "rl_reward_trajectory.png")

        print(f"[RLTrainer V8.0] Data-Driven Mode. Hard-coded Gestalt Rules REMOVED.")

    def compute_reward(self, dynamic_layout, batch, attention_maps=None):
        """
        计算奖励。
        注意：V8.0 不再对 'gestalt' (后4维) 进行人工规则奖励，
        只关注几何合理性 (IoU, Relation, Overlap) 和 语义对齐 (Alignment)。
        """
        B, T, _ = dynamic_layout.shape
        device = dynamic_layout.device
        
        # 提取基础坐标 (前 4 维)
        pred_boxes = dynamic_layout[..., :4]
        # 态势参数 (后 4 维) - RL 阶段不再直接干预它，让它保持预训练的分布
        # gestalt_params = dynamic_layout[..., 4:] 
        
        loss_mask = batch['loss_mask']          
        target_boxes = batch['target_boxes'][..., :4] # 仅对比坐标部分
        kg_spatial_matrix = batch.get('kg_spatial_matrix')
        kg_class_ids = batch['kg_class_ids']    
        
        obj_rewards = torch.zeros(B, T, device=device)
        
        # --- 1. 计算各项原始奖励 ---
        
        # A. IoU 奖励 (保底，防止位置跑偏太远)
        iou = self._calculate_iou(pred_boxes, target_boxes)
        r_iou = iou * loss_mask * self.w_iou

        # B. 关系奖励 (Full Vectorized)
        r_rel = torch.zeros(B, T, device=device)
        if kg_spatial_matrix is not None:
            rel_scores = self._calculate_relation_reward(pred_boxes, kg_spatial_matrix, kg_class_ids)
            r_rel = rel_scores * self.w_rel

        # C. 语义对齐奖励 (Attention Alignment)
        r_align = torch.zeros(B, T, device=device)
        if attention_maps is not None:
            r_align = self._calculate_attention_alignment(attention_maps, pred_boxes)
            r_align = r_align * self.w_align

        # D. [REMOVED] 态势奖励 (Gestalt Reward)
        # V8.0: 删除人工规则，信任 VisualGestaltExtractor 的监督结果
        # r_gestalt = torch.zeros(B, T, device=device)

        # --- 2. 计算重叠惩罚 & 熔断系数 ---
        
        # 重叠惩罚 [B, T]
        overlap_penalty = self._calculate_overlap_penalty(pred_boxes)
        
        # Veto Factor (熔断系数)
        # 严重重叠时，强制降低其他正向奖励的权重，迫使模型优先解决重叠
        veto_factor = (1.0 - overlap_penalty * 2.5).clamp(min=0.0)
        
        # --- 3. 汇总 ---
        
        obj_rewards += r_iou 
        obj_rewards += r_rel * veto_factor
        obj_rewards += r_align * veto_factor
        # obj_rewards += r_gestalt * veto_factor # 已移除
        
        # 加上重叠惩罚 (负分)
        r_over = overlap_penalty * self.w_overlap 
        obj_rewards += r_over

        # 记录明细
        self.last_reward_stats = {
            'IoU': r_iou.mean().item(),
            'Rel': r_rel.mean().item(),
            'Align': r_align.mean().item(),
            'Over': r_over.mean().item(),
            'Veto': veto_factor.mean().item()
        }

        return obj_rewards.sum(dim=1) / (T + 1e-6)

    def _calculate_iou(self, pred, target):
        """计算预测框与GT的IoU"""
        # pred: [B, T, 4], target: [B, T, 4]
        pred_x1 = pred[..., 0] - pred[..., 2] / 2
        pred_y1 = pred[..., 1] - pred[..., 3] / 2
        pred_x2 = pred[..., 0] + pred[..., 2] / 2
        pred_y2 = pred[..., 1] + pred[..., 3] / 2
        
        tgt_x1 = target[..., 0] - target[..., 2] / 2
        tgt_y1 = target[..., 1] - target[..., 3] / 2
        tgt_x2 = target[..., 0] + target[..., 2] / 2
        tgt_y2 = target[..., 1] + target[..., 3] / 2
        
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        pred_area = pred[..., 2] * pred[..., 3]
        tgt_area = target[..., 2] * target[..., 3]
        union_area = pred_area + tgt_area - inter_area
        
        return inter_area / (union_area + 1e-6)

    def _calculate_relation_reward(self, pred_boxes, kg_spatial_matrix, kg_class_ids):
        """
        [完备版] 全向量化关系奖励计算。
        能够并行处理 Above(1), Below(2), Inside(3), OnTop(5) 等关系。
        """
        if kg_spatial_matrix is None or kg_class_ids is None:
            return torch.zeros(pred_boxes.shape[:2], device=pred_boxes.device)

        B, T, _ = pred_boxes.shape
        device = pred_boxes.device

        # 1. 构建当前序列的成对关系矩阵 [B, T, T]
        # kg_class_ids: [B, T] -> 映射到 indices (0-8)
        cls_indices = (kg_class_ids - 2).clamp(min=0, max=8).long()
        
        row_idx = cls_indices.unsqueeze(2).expand(B, T, T) # [B, T, T] (Source)
        col_idx = cls_indices.unsqueeze(1).expand(B, T, T) # [B, T, T] (Target)
        
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, T)
        rel_matrix_seq = kg_spatial_matrix[batch_idx, row_idx, col_idx] # [B, T, T]
        
        # 2. 准备几何参数用于广播计算
        cx = pred_boxes[..., 0]
        cy = pred_boxes[..., 1]
        w  = pred_boxes[..., 2]
        h  = pred_boxes[..., 3]
        
        # 扩展维度以进行 NxN 比较
        cy_i = cy.unsqueeze(2); cy_j = cy.unsqueeze(1)
        
        # 3. 计算关系满足度 (Scores)
        
        # --- Relation 1 & 5: ABOVE / ON_TOP (i 在 j 上方 -> i.y < j.y) ---
        diff_above = cy_j - cy_i 
        score_above = torch.sigmoid(diff_above * 5.0) 
        
        # --- Relation 2: BELOW (i 在 j 下方 -> i.y > j.y) ---
        diff_below = cy_i - cy_j
        score_below = torch.sigmoid(diff_below * 5.0)
        
        # --- Relation 3: INSIDE ---
        # 略微简化，只计算中心点距离是否够近
        # (严谨的 Inside 计算比较耗时，RL 中用距离代理通常够用)
        dist_sq = (cx.unsqueeze(2) - cx.unsqueeze(1))**2 + (cy.unsqueeze(2) - cy.unsqueeze(1))**2
        score_inside = torch.exp(-dist_sq * 10.0) # 距离越近分越高
        
        # 4. 根据关系矩阵聚合奖励
        mask_above = (rel_matrix_seq == 1) | (rel_matrix_seq == 5)
        mask_below = (rel_matrix_seq == 2)
        mask_inside = (rel_matrix_seq == 3)
        
        total_rewards_matrix = torch.zeros_like(score_above)
        total_rewards_matrix = torch.where(mask_above, score_above, total_rewards_matrix)
        total_rewards_matrix = torch.where(mask_below, score_below, total_rewards_matrix)
        total_rewards_matrix = torch.where(mask_inside, score_inside, total_rewards_matrix)
        
        valid_rel_mask = mask_above | mask_below | mask_inside
        
        obj_rel_sum = (total_rewards_matrix * valid_rel_mask.float()).sum(dim=2)
        obj_rel_count = valid_rel_mask.float().sum(dim=2).clamp(min=1.0)
        
        final_rel_reward = obj_rel_sum / obj_rel_count
        has_relation = (valid_rel_mask.sum(dim=2) > 0).float()
        
        return final_rel_reward * has_relation

    def _calculate_overlap_penalty(self, pred_boxes):
        """计算重叠惩罚 (Penalty)。向量化版本。"""
        B, T, _ = pred_boxes.shape
        
        # 展开计算 Pairwise IoU
        x1 = pred_boxes[..., 0] - pred_boxes[..., 2]/2
        y1 = pred_boxes[..., 1] - pred_boxes[..., 3]/2
        x2 = pred_boxes[..., 0] + pred_boxes[..., 2]/2
        y2 = pred_boxes[..., 1] + pred_boxes[..., 3]/2
        area = pred_boxes[..., 2] * pred_boxes[..., 3]
        
        # Broadcasting
        ix1 = torch.max(x1.unsqueeze(2), x1.unsqueeze(1))
        iy1 = torch.max(y1.unsqueeze(2), y1.unsqueeze(1))
        ix2 = torch.min(x2.unsqueeze(2), x2.unsqueeze(1))
        iy2 = torch.min(y2.unsqueeze(2), y2.unsqueeze(1))
        
        i_w = (ix2 - ix1).clamp(min=0)
        i_h = (iy2 - iy1).clamp(min=0)
        inter_area = i_w * i_h
        
        union_area = area.unsqueeze(2) + area.unsqueeze(1) - inter_area
        iou_mat = inter_area / (union_area + 1e-6)
        
        # 排除自身
        diag_mask = torch.eye(T, device=pred_boxes.device).bool().unsqueeze(0).expand(B, T, T)
        iou_mat.masked_fill_(diag_mask, 0.0)
        
        # 找到每个物体最大的重叠对象
        max_iou, _ = iou_mat.max(dim=2) 
        penalty = F.relu(max_iou - 0.05)
        
        return penalty

    def _calculate_attention_alignment(self, attn_maps, boxes):
        """计算 Cross-Attention Map 与 矩形框的重合度。"""
        B, T, H, W = attn_maps.shape
        alignment_scores = torch.zeros(B, T, device=boxes.device)
        
        for b in range(B):
            for t in range(T):
                box = boxes[b, t]
                x1, y1 = int((box[0]-box[2]/2)*W), int((box[1]-box[3]/2)*H)
                x2, y2 = int((box[0]+box[2]/2)*W), int((box[1]+box[3]/2)*H)
                
                attn_map = attn_maps[b, t]
                total_attn = attn_map.sum() + 1e-6
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                if x2 > x1 and y2 > y1:
                    inner_attn = attn_map[y1:y2, x1:x2].sum()
                    alignment_scores[b, t] = inner_attn / total_attn
        
        return alignment_scores

    def _plot_reward_history(self):
        if not self.reward_history: return
        plt.figure(figsize=(10, 5))
        plt.plot(self.reward_history)
        plt.title("RL Reward Trajectory (V8.0 Data-Driven)")
        plt.xlabel("Epoch")
        plt.ylabel("Average Reward")
        plt.grid(True)
        try:
            plt.savefig(self.plot_path_reward)
            plt.close()
        except Exception: pass

    def train_rl_epoch(self, epoch):
        self.model.train()
        total_reward = 0
        steps = 0
        
        for step, batch in enumerate(self.train_loader):
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
            
            # --- SCST 训练逻辑 ---
            self.model.eval()
            with torch.no_grad():
                # Baseline: Greedy Decode
                baseline_layout, _ = self.model.forward_rl(
                    batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                    batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'),
                    sample=False
                )
                reward_baseline = self.compute_reward(baseline_layout, batch)
            
            self.model.train()
            # Sample: Stochastic Decode
            sample_layout, log_probs = self.model.forward_rl(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'),
                sample=True
            )
            
            reward_sample = self.compute_reward(sample_layout, batch)
            
            advantage = reward_sample - reward_baseline
            rl_loss = -(log_probs.sum(dim=1) * advantage).mean()
            
            # 监督辅助 (Supervised Loss for Regularization)
            mu, logvar, dynamic_layout_sup, decoder_output = self.model(
                batch['input_ids'], batch['attention_mask'], batch['kg_class_ids'], 
                batch['padding_mask'], batch.get('kg_spatial_matrix'), batch.get('location_grids'),
                target_boxes=batch['target_boxes']
            )
            
            # [CRITICAL FIX V8.0] 必须适配 12 个返回值，否则会崩溃
            loss_tuple = self.model.get_loss(
                pred_cls=None, pred_bbox_ids=None, 
                pred_boxes=dynamic_layout_sup, 
                pred_count=None, layout_seq=None, 
                layout_mask=batch['loss_mask'], 
                num_boxes=batch['num_boxes'], 
                target_coords_gt=batch['target_boxes'],
                kg_spatial_matrix=batch.get('kg_spatial_matrix'),
                kg_class_weights=batch.get('kg_class_weights'),
                kg_class_ids=batch['kg_class_ids'],
                decoder_output=decoder_output,
                
                # [关键修复] 传入 gestalt_mask，确保 Loss 忽略无效的态势数据
                gestalt_mask=batch.get('gestalt_mask') 
            )
            
            # V8.0 Unpacking (12 values)
            (supervised_loss, l_rel, l_over, l_reg, l_iou, l_size, l_area, 
             l_align, l_bal, l_clus, l_cons, l_gestalt) = loss_tuple
            
            kl_loss = compute_kl_loss(mu, logvar) if mu is not None else torch.tensor(0.0)

            # 总损失 = RL Loss + (Supervised + Gestalt + Consistency) * small_weight
            # 这里的 supervised_loss 已经包含了 reg, relation, gestalt 等所有项
            total_combined_loss = rl_loss + (0.2 * supervised_loss + 0.1 * kl_loss)
            
            self.optimizer.zero_grad()
            total_combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_reward += reward_sample.mean().item()
            steps += 1
            
            if (step + 1) % 10 == 0:
                stats = self.last_reward_stats
                print(f"[RL V8.0] Step {step+1} | R:{reward_sample.mean().item():.2f} | "
                      f"Over:{stats.get('Over',0):.2f} | Veto:{stats.get('Veto',0):.2f} | "
                      f"GestLoss:{l_gestalt.item():.3f}") # 监控 Gestalt Loss

        avg_reward = total_reward / steps
        self.reward_history.append(avg_reward)
        self._plot_reward_history()
        return avg_reward