import torch
import torch.nn as nn

class PoemLayoutEmbedding(nn.Module):
    """
    将 5 个离散 Token ID (类别 ID, cx_id, cy_id, w_id, h_id) 转换为拼接的向量嵌入。
    """
    def __init__(self, 
                 total_vocab_size: int, 
                 num_bbox_bins: int, 
                 bb_size: int, 
                 cls_embed_dim: int,
                 bbox_embed_dim: int, 
                 dropout: float = 0.1):
        super(PoemLayoutEmbedding, self).__init__()
        
        # 验证嵌入维度是否与 bb_size 匹配
        expected_bb_size = cls_embed_dim + 4 * bbox_embed_dim
        if bb_size != expected_bb_size:
            raise ValueError(
                f"Config Error: bb_size ({bb_size}) must equal "
                f"cls_embed_dim ({cls_embed_dim}) + 4 * bbox_embed_dim ({4 * bbox_embed_dim}) = {expected_bb_size}"
            )
            
        self.bb_size = bb_size
        
        # 1. 类别 ID 嵌入 (包含 BOS/EOS/元素类别)
        # total_vocab_size = 元素数(9) + BOS(1) + EOS(1) = 11
        # padding_idx=0 仅用于确保兼容性
        self.cls_embed = nn.Embedding(total_vocab_size, cls_embed_dim, padding_idx=0)
        
        # 2. 4 个 BBox 坐标 ID 嵌入 (使用新的 num_bbox_bins 作为词汇表大小)
        # num_bbox_bins = 1000
        self.cx_embed = nn.Embedding(num_bbox_bins, bbox_embed_dim, padding_idx=0)
        self.cy_embed = nn.Embedding(num_bbox_bins, bbox_embed_dim, padding_idx=0)
        self.w_embed = nn.Embedding(num_bbox_bins, bbox_embed_dim, padding_idx=0)
        self.h_embed = nn.Embedding(num_bbox_bins, bbox_embed_dim, padding_idx=0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_ids: torch.Tensor, bbox_ids: torch.Tensor):
        """
        Args:
            cls_ids: [B, T] (类别 ID, LongTensor)
            bbox_ids: [B, T, 4] (cx_id, cy_id, w_id, h_id, LongTensor)
        Returns:
            embeddings: [B, T, bb_size]
        """
        
        # 1. 嵌入类别 ID
        cls_emb = self.cls_embed(cls_ids) # [B, T, cls_embed_dim]
        
        # 2. 嵌入 BBox ID
        cx_emb = self.cx_embed(bbox_ids[..., 0]) # [B, T, bbox_embed_dim]
        cy_emb = self.cy_embed(bbox_ids[..., 1])
        w_emb = self.w_embed(bbox_ids[..., 2])
        h_emb = self.h_embed(bbox_ids[..., 3])
        
        # 3. 拼接所有嵌入
        # Order: [cls, cx, cy, w, h]
        embeddings = torch.cat([cls_emb, cx_emb, cy_emb, w_emb, h_emb], dim=-1) # [B, T, bb_size]
        
        return self.dropout(embeddings)