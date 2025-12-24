# models/poem2layout.py (V9.0: Heatmap Projection & Generation - Full Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
try:
    from torchvision.ops import complete_box_iou_loss
except ImportError:
    def complete_box_iou_loss(boxes1, boxes2, reduction='none'):
        return F.smooth_l1_loss(boxes1, boxes2, reduction=reduction)

from .decoder import LayoutDecoder

# ==========================================
# [V8.2] 定义类别“基准”面积先验 (Base Area Ratio)
# ==========================================
CLASS_AREA_PRIORS = {
    2: 0.30,   # Mountain (山)
    3: 0.30,   # Water (水)
    4: 0.06,   # People (人)
    5: 0.15,   # Tree (树)
    6: 0.15,   # Building (建筑)
    7: 0.10,   # Bridge (桥)
    8: 0.03,   # Flower (花)
    9: 0.015,  # Bird (鸟)
    10: 0.05   # Animal (动物)
}

# === 1. 图神经网络关系先验 (R-GAT) ===
class GraphRelationPriorNet(nn.Module):
    def __init__(self, num_relations, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        
        self.rel_embed_k = nn.Embedding(num_relations, hidden_dim)
        self.rel_embed_v = nn.Embedding(num_relations, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, node_features, spatial_matrix):
        B, T, D = node_features.shape
        H = self.num_heads
        d_k = self.head_dim

        q = self.q_proj(node_features).view(B, T, H, d_k)
        k = self.k_proj(node_features).view(B, T, H, d_k)
        v = self.v_proj(node_features).view(B, T, H, d_k)

        r_k = self.rel_embed_k(spatial_matrix).view(B, T, T, H, d_k)
        r_v = self.rel_embed_v(spatial_matrix).view(B, T, T, H, d_k)

        q = q.unsqueeze(2) 
        k_prime = k.unsqueeze(1) + r_k 
        
        scores = (q * k_prime).sum(dim=-1) / (d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=2)
        attn_weights = self.dropout(attn_weights)

        v_prime = v.unsqueeze(1) + r_v 
        agg = (attn_weights.unsqueeze(-1) * v_prime).sum(dim=2)
        
        agg = agg.view(B, T, D)
        output = self.out_proj(agg)
        output = output + node_features
        output = self.norm(output)
        output = self.activation(output)
        
        return output

# === 2. 布局变换编码器 ===
class LayoutTransformerEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_size=768, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 50, hidden_size))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, 
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, boxes, mask=None):
        B, T, _ = boxes.shape
        x = self.input_proj(boxes)
        
        if T <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :T, :]
        else:
            x = x + self.pos_embed[:, :self.pos_embed.size(1), :]
            
        x = self.transformer(x, src_key_padding_mask=mask)
        
        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1).float()
            sum_feat = (x * valid_mask).sum(dim=1)
            count = valid_mask.sum(dim=1).clamp(min=1e-6)
            global_feat = sum_feat / count
        else:
            global_feat = x.mean(dim=1)
            
        return global_feat

# === 主模型: Poem2LayoutGenerator ===
class Poem2LayoutGenerator(nn.Module):
    def __init__(self, bert_path: str, num_classes: int, hidden_size: int = 768, bb_size: int = 64, 
                 decoder_layers: int = 4, decoder_heads: int = 8, dropout: float = 0.1, 
                 reg_loss_weight: float = 1.0, iou_loss_weight: float = 1.0, area_loss_weight: float = 1.0,
                 relation_loss_weight: float = 5.0,  
                 overlap_loss_weight: float = 2.0,   
                 size_loss_weight: float = 2.0,
                 alignment_loss_weight: float = 0.5,
                 balance_loss_weight: float = 0.5,
                 clustering_loss_weight: float = 1.0, 
                 consistency_loss_weight: float = 1.0, 
                 gestalt_loss_weight: float = 2.0, 
                 latent_dim: int = 32,               
                 **kwargs): 
        super(Poem2LayoutGenerator, self).__init__()
        
        self.num_element_classes = num_classes 
        self.hidden_size = hidden_size
        self.bb_size = bb_size
        self.latent_dim = latent_dim
        
        # Loss weights
        self.reg_loss_weight = reg_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.area_loss_weight = area_loss_weight
        self.relation_loss_weight = relation_loss_weight
        self.overlap_loss_weight = overlap_loss_weight
        self.size_loss_weight = size_loss_weight
        self.alignment_loss_weight = alignment_loss_weight
        self.balance_loss_weight = balance_loss_weight
        self.clustering_loss_weight = clustering_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.gestalt_loss_weight = gestalt_loss_weight 
        
        self.cond_dropout = nn.Dropout(0.25)

        # 1. Text Encoder
        self.text_encoder = BertModel.from_pretrained(bert_path)
        
        # 2. Object Query Embedding
        self.obj_class_embedding = nn.Embedding(num_classes + 2, bb_size, padding_idx=0)
        
        # 3. Sequence Positional Embedding
        self.seq_pos_embedding = nn.Embedding(50, bb_size)

        # 4. Spatial Bias Embedding
        self.num_spatial_relations = 7
        self.spatial_bias_embedding = nn.Embedding(self.num_spatial_relations, decoder_heads)

        # 5. Position Encoders
        self.grid_encoder = nn.Sequential(
            nn.Linear(64, bb_size), 
            nn.ReLU(),
            nn.Linear(bb_size, bb_size),
            nn.Dropout(dropout)
        )
        
        self.gnn_prior = GraphRelationPriorNet(
            num_relations=self.num_spatial_relations,
            input_dim=bb_size, 
            hidden_dim=bb_size,
            num_heads=4,
            dropout=dropout
        )
        
        # 6. CVAE Components
        self.layout_encoder = LayoutTransformerEncoder(
            input_dim=4,
            hidden_size=hidden_size,
            num_layers=2,
            nhead=4,
            dropout=dropout
        )
        self.mu_head = nn.Linear(hidden_size, latent_dim)
        self.logvar_head = nn.Linear(hidden_size, latent_dim)
        self.z_proj = nn.Linear(latent_dim, bb_size)
        
        # 7. KG Projection
        self.kg_projection = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 8. Decoder
        self.layout_decoder = LayoutDecoder(
            hidden_size=hidden_size,
            bb_size=bb_size,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=dropout
        )

        # === 9. 双头预测系统 ===
        decoder_output_size = hidden_size + bb_size
        
        # A. 基础坐标回归头
        self.reg_head = nn.Sequential(
            nn.Linear(decoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4)
        )

        # B. 形态态势头
        self.gestalt_dir_head = nn.Sequential(
            nn.Linear(decoder_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3), 
            nn.Tanh() 
        )

        self.gestalt_flow_head = nn.Sequential(
            nn.Linear(decoder_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()
        )
        
        # 10. Consistency Projection
        self.consistency_proj = nn.Linear(bb_size, hidden_size)

        # [NEW V9.0] 热力图投影头 (Heatmap Projection Head)
        # 目的：从 Decoder 输出中恢复出 64x64 的空间热力图，用于 RL 对齐
        self.heatmap_small_dim = 16
        self.heatmap_head = nn.Sequential(
            nn.Linear(decoder_output_size, self.heatmap_small_dim * self.heatmap_small_dim),
            nn.ReLU(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def construct_spatial_bias(self, cls_ids, kg_spatial_matrix):
        if kg_spatial_matrix is None:
            return None
        B, S = cls_ids.shape
        map_ids = cls_ids - 2 
        gather_ids = map_ids.clamp(min=0, max=self.num_element_classes - 1)
        b_idx = torch.arange(B, device=cls_ids.device).view(B, 1, 1).expand(-1, S, S)
        row_idx = gather_ids.view(B, S, 1).expand(-1, -1, S)
        col_idx = gather_ids.view(B, 1, S).expand(-1, S, -1)
        rel_ids = kg_spatial_matrix[b_idx, row_idx, col_idx] 
        is_valid_obj = (map_ids >= 0)
        valid_pair_mask = is_valid_obj.unsqueeze(2) & is_valid_obj.unsqueeze(1)
        rel_ids = rel_ids.masked_fill(~valid_pair_mask, 0)
        spatial_bias = self.spatial_bias_embedding(rel_ids) 
        spatial_bias = spatial_bias.permute(0, 3, 1, 2).contiguous() 
        return spatial_bias

    # [NEW V9.0] 生成热力图的辅助函数
    def generate_heatmaps(self, decoder_output):
        B, T, _ = decoder_output.shape
        # 1. Project to small grid features: [B, T, 16*16]
        flat_maps = self.heatmap_head(decoder_output)
        
        # 2. Reshape: [B*T, 1, 16, 16]
        small_maps = flat_maps.view(B * T, 1, self.heatmap_small_dim, self.heatmap_small_dim)
        
        # 3. Upsample to 64x64: [B*T, 1, 64, 64]
        # 使用 bilinear 插值获得平滑的热力图
        large_maps = F.interpolate(small_maps, size=(64, 64), mode='bilinear', align_corners=False)
        
        # 4. Activate: [B, T, 64, 64]
        heatmaps = torch.sigmoid(large_maps).view(B, T, 64, 64)
        return heatmaps

    def forward(self, input_ids, attention_mask, kg_class_ids, padding_mask, 
                kg_spatial_matrix=None, location_grids=None, target_boxes=None):
        
        text_encoded = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state 

        content_embed = self.obj_class_embedding(kg_class_ids) 
        content_embed = self.cond_dropout(content_embed)
        
        if target_boxes is not None:
            target_geom = target_boxes[..., :4]
            global_layout_feat = self.layout_encoder(target_geom, mask=padding_mask) 
            mu = self.mu_head(global_layout_feat)
            logvar = self.logvar_head(global_layout_feat)
            z = self.reparameterize(mu, logvar)
        else:
            B = input_ids.shape[0]
            mu = None
            logvar = None
            z = torch.randn(B, self.latent_dim, device=input_ids.device)
            
        z_feat = self.z_proj(z).unsqueeze(1) 

        layout_content = content_embed + z_feat 
        pos_feat = torch.zeros_like(content_embed)
        
        if location_grids is not None:
            B, T, H, W = location_grids.shape
            grid_flat = location_grids.view(B, T, -1).to(content_embed.device)
            handcrafted_pos = self.grid_encoder(grid_flat) 
            pos_feat = pos_feat + handcrafted_pos

        if kg_spatial_matrix is not None:
            B, T = kg_class_ids.shape
            map_ids = kg_class_ids - 2 
            gather_ids = map_ids.clamp(min=0, max=self.num_element_classes - 1)
            b_idx = torch.arange(B, device=kg_class_ids.device).view(B, 1, 1).expand(-1, T, T)
            row_idx = gather_ids.view(B, T, 1).expand(-1, -1, T)
            col_idx = gather_ids.view(B, 1, T).expand(-1, T, -1)
            seq_rel_ids = kg_spatial_matrix[b_idx, row_idx, col_idx]
            
            learned_pos = self.gnn_prior(content_embed, seq_rel_ids)
            layout_content = layout_content + learned_pos 

        pos_feat = self.cond_dropout(pos_feat)
        B, T = kg_class_ids.shape
        seq_ids = torch.arange(T, device=kg_class_ids.device).unsqueeze(0).expand(B, -1)
        seq_embed = self.seq_pos_embedding(seq_ids) 
        layout_pos = pos_feat + seq_embed

        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if padding_mask is not None:
            trg_mask = torch.zeros_like(padding_mask, dtype=torch.float)
            trg_mask = trg_mask.masked_fill(padding_mask, float('-inf'))
            trg_mask = trg_mask.unsqueeze(1).unsqueeze(2)
        else:
            trg_mask = None

        spatial_bias = self.construct_spatial_bias(kg_class_ids, kg_spatial_matrix)

        decoder_output = self.layout_decoder(
            layout_content, layout_pos, text_encoded, 
            src_mask, trg_mask, spatial_bias=spatial_bias
        ) 

        # === Outputs ===
        pred_boxes = torch.sigmoid(self.reg_head(decoder_output))
        pred_dir = self.gestalt_dir_head(decoder_output)   
        pred_flow = self.gestalt_flow_head(decoder_output) 
        pred_gestalt = torch.cat([pred_dir, pred_flow], dim=-1)
        
        dynamic_layout = torch.cat([pred_boxes, pred_gestalt], dim=-1)
        
        # [NEW V9.0] 在 forward 中计算热力图并返回 (第5个参数)
        pred_heatmaps = self.generate_heatmaps(decoder_output) # [B, T, 64, 64]
        
        return mu, logvar, dynamic_layout, decoder_output, pred_heatmaps

    # [NEW V9.0] 适配 RL 训练，返回三元组
    def forward_rl(self, input_ids, attention_mask, kg_class_ids, padding_mask, 
                   kg_spatial_matrix=None, location_grids=None, target_boxes=None, 
                   sample=True):
        # 接收 5 个返回值
        _, _, dynamic_layout_mu, decoder_output, pred_heatmaps = self.forward(
            input_ids, attention_mask, kg_class_ids, padding_mask, 
            kg_spatial_matrix, location_grids, target_boxes=None
        )
        
        if not sample:
            # 返回: 布局, 概率(None), 热力图
            return dynamic_layout_mu, None, pred_heatmaps

        std = torch.ones_like(dynamic_layout_mu) * 0.1
        dist = torch.distributions.Normal(dynamic_layout_mu, std)
        
        action_layout = dist.sample()
        
        # RL 探索时的物理约束截断
        coords = torch.clamp(action_layout[..., :4], 0.0, 1.0)
        g_dir = torch.clamp(action_layout[..., 4:7], -1.0, 1.0)
        g_flow = torch.clamp(action_layout[..., 7:8], -1.0, 1.0) 
        
        final_action = torch.cat([coords, g_dir, g_flow], dim=-1)
        log_prob = dist.log_prob(final_action).sum(dim=-1)
        
        if padding_mask is not None:
             log_prob = log_prob.masked_fill(padding_mask, 0.0)

        # 返回: 布局, 概率, 热力图
        return final_action, log_prob, pred_heatmaps

    # [NEW V9.0] 辅助函数：根据 GT 生成高斯分布掩码，用于监督热力图
    def _generate_gaussian_masks(self, boxes, grid_size=64):
        B, T, _ = boxes.shape
        device = boxes.device
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, grid_size, device=device), 
            torch.linspace(0, 1, grid_size, device=device)
        )
        grid = torch.stack([x_grid, y_grid], dim=-1).view(1, 1, grid_size, grid_size, 2)
        cx = boxes[..., 0].view(B, T, 1, 1)
        cy = boxes[..., 1].view(B, T, 1, 1)
        w  = boxes[..., 2].view(B, T, 1, 1)
        h  = boxes[..., 3].view(B, T, 1, 1)
        sigma_x = (w / 4.0).clamp(min=0.01)
        sigma_y = (h / 4.0).clamp(min=0.01)
        dist_sq = ((grid[..., 0] - cx)**2) / (2*sigma_x**2) + ((grid[..., 1] - cy)**2) / (2*sigma_y**2)
        masks = torch.exp(-dist_sq)
        return masks

    def get_loss(self, pred_cls, pred_bbox_ids, pred_boxes, pred_count, layout_seq, layout_mask, num_boxes, 
                 target_coords_gt=None, kg_spatial_matrix=None, kg_class_weights=None, kg_class_ids=None, 
                 decoder_output=None, gestalt_mask=None): 
        loss_mask = layout_mask 
        pred_coords = pred_boxes[..., :4]
        pred_gestalt = pred_boxes[..., 4:]
        
        has_gestalt_gt = False
        if target_coords_gt is not None and target_coords_gt.shape[-1] >= 8:
            target_boxes = target_coords_gt[..., :4]
            target_gestalt = target_coords_gt[..., 4:]
            has_gestalt_gt = True
        else:
            target_boxes = target_coords_gt
            target_gestalt = None
        
        if loss_mask.dim() == 1:
             loss_mask = loss_mask.view(pred_coords.shape[0], -1)
        
        num_valid = loss_mask.sum().clamp(min=1)

        # 1. 基础几何损失
        loss_reg = F.smooth_l1_loss(pred_coords, target_boxes, reduction='none') 
        loss_reg = (loss_reg.mean(dim=-1) * loss_mask).sum() / num_valid
        
        loss_iou = self._compute_iou_loss(pred_coords, target_boxes, loss_mask)
        
        pred_w, pred_h = pred_coords[..., 2], pred_coords[..., 3]
        tgt_w, tgt_h = target_boxes[..., 2], target_boxes[..., 3]
        pred_area = pred_w * pred_h
        tgt_area = tgt_w * tgt_h
        loss_area = F.smooth_l1_loss(pred_area, tgt_area, reduction='none')
        loss_area = (loss_area * loss_mask).sum() / num_valid
        
        # 2. 高级布局损失
        loss_relation = self._compute_relation_loss(pred_coords, loss_mask, kg_spatial_matrix, kg_class_ids)
        loss_overlap = self._compute_overlap_loss(pred_coords, loss_mask, kg_spatial_matrix, kg_class_ids)
        loss_size_prior = self._compute_size_loss(pred_coords, loss_mask, num_boxes, kg_class_ids=kg_class_ids)
        loss_alignment = self._compute_alignment_loss(pred_coords, loss_mask)
        loss_balance = self._compute_balance_loss(pred_coords, loss_mask)
        loss_clustering = self._compute_clustering_loss(pred_coords, loss_mask, kg_class_ids)
        loss_consistency = self._compute_consistency_loss(decoder_output, loss_mask)

        # 3. 像素级弱监督态势损失
        loss_gestalt = torch.tensor(0.0, device=pred_boxes.device)
        if has_gestalt_gt:
            loss_g_vec = F.smooth_l1_loss(pred_gestalt, target_gestalt, reduction='none')
            loss_g_val = loss_g_vec.mean(dim=-1)
            if gestalt_mask is not None:
                combined_mask = loss_mask * gestalt_mask if gestalt_mask.shape == loss_mask.shape else loss_mask
            else:
                combined_mask = loss_mask
            num_gestalt_valid = combined_mask.sum().clamp(min=1)
            loss_gestalt = (loss_g_val * combined_mask).sum() / num_gestalt_valid

        # 4. [NEW V9.0] 热力图监督损失
        loss_heatmap = torch.tensor(0.0, device=pred_boxes.device)
        if decoder_output is not None:
            # 重新生成 Heatmap 用于计算 Loss
            pred_heatmaps = self.generate_heatmaps(decoder_output) 
            # 生成 GT Gaussian Mask
            gt_heatmaps = self._generate_gaussian_masks(target_boxes, grid_size=64)
            # 计算 MSE Loss
            hm_loss_map = F.mse_loss(pred_heatmaps, gt_heatmaps, reduction='none').mean(dim=[2, 3])
            loss_heatmap = (hm_loss_map * loss_mask).sum() / num_valid

        total_loss = self.reg_loss_weight * loss_reg + \
                      self.iou_loss_weight * loss_iou + \
                      self.area_loss_weight * loss_area + \
                      self.relation_loss_weight * loss_relation + \
                      self.overlap_loss_weight * loss_overlap + \
                      self.size_loss_weight * loss_size_prior + \
                      self.alignment_loss_weight * loss_alignment + \
                      self.balance_loss_weight * loss_balance + \
                      self.clustering_loss_weight * loss_clustering + \
                      self.consistency_loss_weight * loss_consistency + \
                      self.gestalt_loss_weight * loss_gestalt + \
                      5.0 * loss_heatmap # 强监督热力图
                      
        return total_loss, loss_relation, loss_overlap, \
               loss_reg, loss_iou, loss_size_prior, loss_area, \
               loss_alignment, loss_balance, loss_clustering, loss_consistency, loss_gestalt

    def _compute_consistency_loss(self, decoder_output, mask):
        if decoder_output is None:
            return torch.tensor(0.0, device=mask.device)
        text_stream_feat = decoder_output[..., :self.hidden_size] 
        layout_stream_feat = decoder_output[..., self.hidden_size:] 
        layout_projected = self.consistency_proj(layout_stream_feat) 
        text_norm = F.normalize(text_stream_feat, p=2, dim=-1)
        layout_norm = F.normalize(layout_projected, p=2, dim=-1)
        cosine_sim = (text_norm * layout_norm).sum(dim=-1) 
        loss = 1.0 - cosine_sim
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss

    def _compute_iou_loss(self, pred, target, mask):
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
        iou = inter_area / (union_area + 1e-6)
        loss = (1.0 - iou) * mask
        return loss.sum() / (mask.sum().clamp(min=1))

    def _compute_relation_loss(self, pred_boxes, mask, kg_spatial_matrix, kg_class_ids):
        if kg_spatial_matrix is None or kg_class_ids is None:
            return torch.tensor(0.0, device=pred_boxes.device)
        loss = torch.tensor(0.0, device=pred_boxes.device)
        B, S, _ = pred_boxes.shape
        count = 0
        for b in range(B):
            valid_indices = torch.nonzero(mask[b]).squeeze(1)
            if len(valid_indices) < 2: continue
            for i in valid_indices:
                for j in valid_indices:
                    if i == j: continue
                    cid_i = kg_class_ids[b, i].item()
                    cid_j = kg_class_ids[b, j].item()
                    idx_i = int(cid_i) - 2
                    idx_j = int(cid_j) - 2
                    if not (0 <= idx_i < 9 and 0 <= idx_j < 9): continue
                    rel = kg_spatial_matrix[b, idx_i, idx_j].item()
                    if rel == 0: continue
                    box_a = pred_boxes[b, i]
                    box_b = pred_boxes[b, j]
                    if rel in [1, 5]: # ABOVE / ON_TOP
                        dist = box_a[1] - box_b[1] + 0.05
                        if dist > 0:
                            loss += dist
                            count += 1
                    elif rel == 2: # BELOW
                        dist = box_b[1] - box_a[1] + 0.05
                        if dist > 0:
                            loss += dist
                            count += 1
                    elif rel == 3: # INSIDE
                        a_x1, a_y1 = box_a[0]-box_a[2]/2, box_a[1]-box_a[3]/2
                        a_x2, a_y2 = box_a[0]+box_a[2]/2, box_a[1]+box_a[3]/2
                        b_x1, b_y1 = box_b[0]-box_b[2]/2, box_b[1]-box_b[3]/2
                        b_x2, b_y2 = box_b[0]+box_b[2]/2, box_b[1]+box_b[3]/2
                        l_inside = F.relu(b_x1 - a_x1) + F.relu(a_x2 - b_x2) + \
                                   F.relu(b_y1 - a_y1) + F.relu(a_y2 - b_y2)
                        if l_inside > 0:
                            loss += l_inside
                            count += 1
        if count > 0: return loss / count
        return loss

    def _compute_overlap_loss(self, pred_boxes, mask, kg_spatial_matrix, kg_class_ids):
        loss = torch.tensor(0.0, device=pred_boxes.device)
        B, S, _ = pred_boxes.shape
        count = 0
        for b in range(B):
            valid_indices = torch.nonzero(mask[b]).squeeze(1)
            if len(valid_indices) < 2: continue
            boxes = pred_boxes[b, valid_indices]
            N = boxes.shape[0]
            x1 = boxes[:, 0] - boxes[:, 2]/2
            y1 = boxes[:, 1] - boxes[:, 3]/2
            x2 = boxes[:, 0] + boxes[:, 2]/2
            y2 = boxes[:, 1] + boxes[:, 3]/2
            areas = boxes[:, 2] * boxes[:, 3]
            inter_x1 = torch.max(x1.unsqueeze(1), x1.unsqueeze(0))
            inter_y1 = torch.max(y1.unsqueeze(1), y1.unsqueeze(0))
            inter_x2 = torch.min(x2.unsqueeze(1), x2.unsqueeze(0))
            inter_y2 = torch.min(y2.unsqueeze(1), y2.unsqueeze(0))
            inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
            union_area = areas.unsqueeze(1) + areas.unsqueeze(0) - inter_area
            iou_mat = inter_area / (union_area + 1e-6)
            ignore_overlap = torch.eye(N, device=pred_boxes.device).bool()
            
            if kg_spatial_matrix is not None and kg_class_ids is not None:
                for i_local, i_global in enumerate(valid_indices):
                    for j_local, j_global in enumerate(valid_indices):
                        if i_local == j_local: continue
                        cid_i = kg_class_ids[b, i_global].item()
                        cid_j = kg_class_ids[b, j_global].item()
                        idx_i = int(cid_i) - 2
                        idx_j = int(cid_j) - 2
                        if not (0 <= idx_i < 9 and 0 <= idx_j < 9): continue
                        rel = kg_spatial_matrix[b, idx_i, idx_j].item()
                        if rel in [3, 4]: 
                            ignore_overlap[i_local, j_local] = True
                            ignore_overlap[j_local, i_local] = True
                            
            triu_mask = torch.triu(torch.ones(N, N, device=pred_boxes.device), diagonal=1).bool()
            target_mask = triu_mask & (~ignore_overlap)
            if target_mask.sum() > 0:
                bad_iou = iou_mat[target_mask]
                loss += F.relu(bad_iou - 0.1).sum()
                count += target_mask.sum()
        if count > 0: return loss / count
        return loss

    def _compute_size_loss(self, pred_boxes, mask, num_boxes, kg_class_ids=None):
        pred_areas = pred_boxes[..., 2] * pred_boxes[..., 3] 
        if num_boxes is None:
            N_per_sample = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        else:
            N_per_sample = num_boxes.float().clamp(min=1).unsqueeze(1)
        base_expected_area = (0.5 / torch.sqrt(N_per_sample)).expand_as(pred_areas)
        target_areas = base_expected_area.clone()

        if kg_class_ids is not None:
            prior_lookup = torch.full((20,), -1.0, device=pred_boxes.device)
            for cid, area in CLASS_AREA_PRIORS.items():
                prior_lookup[cid] = area
            class_priors = prior_lookup[kg_class_ids]
            density_scale = torch.ones_like(N_per_sample)
            density_scale[N_per_sample == 1] = 4.0
            density_scale[N_per_sample == 2] = 2.5
            density_scale[(N_per_sample > 2) & (N_per_sample <= 5)] = 1.5
            scaled_priors = class_priors * density_scale
            scaled_priors = torch.clamp(scaled_priors, max=0.90)
            has_prior_mask = (class_priors > 0).float()
            target_areas = has_prior_mask * scaled_priors + (1.0 - has_prior_mask) * target_areas

        loss = F.smooth_l1_loss(pred_areas, target_areas, reduction='none')
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss

    def _compute_alignment_loss(self, pred_boxes, mask):
        B, N, _ = pred_boxes.shape
        loss = torch.tensor(0.0, device=pred_boxes.device)
        count = 0
        for b in range(B):
            valid_indices = torch.nonzero(mask[b]).squeeze(1)
            num_valid = len(valid_indices)
            if num_valid < 2: continue
            boxes = pred_boxes[b, valid_indices]
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            left = cx - w / 2
            right = cx + w / 2
            top = cy - h / 2
            bottom = cy + h / 2
            x_vals = torch.stack([left, cx, right], dim=1) 
            y_vals = torch.stack([top, cy, bottom], dim=1)
            
            def min_dist_loss(vals):
                v1 = vals.unsqueeze(1) 
                v2 = vals.unsqueeze(0) 
                diff = torch.abs(v1.unsqueeze(3) - v2.unsqueeze(2)).view(num_valid, num_valid, -1)
                eye_mask = torch.eye(num_valid, device=vals.device).bool().unsqueeze(-1)
                diff = diff.masked_fill(eye_mask, 100.0)
                min_dists, _ = diff.min(dim=2) 
                min_dists, _ = min_dists.min(dim=1) 
                return min_dists.mean()

            loss += min_dist_loss(x_vals) + min_dist_loss(y_vals)
            count += 1
        if count > 0: return loss / count
        return loss

    def _compute_balance_loss(self, pred_boxes, mask):
        loss = torch.tensor(0.0, device=pred_boxes.device)
        B, N, _ = pred_boxes.shape
        count = 0
        target_center = 0.5
        margin = 0.15 
        for b in range(B):
            valid_indices = torch.nonzero(mask[b]).squeeze(1)
            if len(valid_indices) == 0: continue
            boxes = pred_boxes[b, valid_indices]
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = w * h
            total_area = areas.sum().clamp(min=1e-6)
            center_x = (cx * areas).sum() / total_area
            center_y = (cy * areas).sum() / total_area
            dist_x = F.relu(torch.abs(center_x - target_center) - margin)
            dist_y = F.relu(torch.abs(center_y - target_center) - margin)
            loss += (dist_x + dist_y)
            count += 1
        if count > 0: return loss / count
        return loss

    def _compute_clustering_loss(self, pred_boxes, mask, kg_class_ids):
        if kg_class_ids is None: return torch.tensor(0.0, device=pred_boxes.device)
        loss = torch.tensor(0.0, device=pred_boxes.device)
        B, N, _ = pred_boxes.shape
        count = 0
        max_dist_threshold = 0.35 
        for b in range(B):
            classes = kg_class_ids[b]
            valid_mask = mask[b]
            unique_classes = torch.unique(classes)
            for cls_id in unique_classes:
                if cls_id <= 2: continue 
                indices = torch.nonzero((classes == cls_id) & (valid_mask > 0)).squeeze(1)
                if len(indices) < 2: continue 
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1 = indices[i]; idx2 = indices[j]
                        box1 = pred_boxes[b, idx1]; box2 = pred_boxes[b, idx2]
                        dist = torch.sqrt((box1[0] - box2[0])**2 + (box1[1] - box2[1])**2)
                        if dist > max_dist_threshold:
                            loss += (dist - max_dist_threshold)
                            count += 1
        if count > 0: return loss / count
        return loss