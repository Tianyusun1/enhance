import math
import torch
import torch.nn as nn
from torch import Tensor

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert size % num_heads == 0
        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        batch_size = k.size(0)
        num_heads = self.num_heads
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2) # [B, H, L_k, Dh]
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2) # [B, H, L_v, Dh]
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2) # [B, H, L_q, Dh]
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3)) # [B, H, L_q, L_k]
        if mask is not None:
            # Handle different mask types (bool, int, float)
            # Original mask shape: [B, L_q, L_k] or [L_q, L_k] or [1, L_q, L_k] etc.
            # Need to broadcast to [B, H, L_q, L_k]
            if mask.dim() == 2: # Assume [L_q, L_k] like trg_mask
                # Expand to [1, 1, L_q, L_k] then broadcast
                mask_expanded = mask.unsqueeze(0).unsqueeze(0) # [1, 1, L_q, L_k]
            elif mask.dim() == 3: # Assume [B, L_q, L_k] like src_mask
                # Expand to [B, 1, L_q, L_k] then broadcast
                mask_expanded = mask.unsqueeze(1) # [B, 1, L_q, L_k]
            elif mask.dim() == 4: # Already [B, H, L_q, L_k] or similar
                mask_expanded = mask
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")

            if mask_expanded.dtype == torch.bool:
                scores = scores.masked_fill(~mask_expanded, float('-inf'))
            else:
                # Assume float mask with 0 and -inf
                scores = scores.masked_fill(mask_expanded == float('-inf'), float('-inf'))
        attention = self.softmax(scores)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v) # [B, H, L_q, Dh_v]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size) # [B, L_q, D]
        output = self.output_layer(context)
        return output

class ContMultiHeadedAttention(nn.Module):
    # 修改初始化，接受 Q, K, V 的不同维度
    def __init__(self, num_heads: int, size_q: int, size_k: int, size_v: int, dropout: float = 0.1):
        super(ContMultiHeadedAttention, self).__init__()
        assert size_k % num_heads == 0
        assert size_v % num_heads == 0
        self.head_size_k = head_size_k = size_k // num_heads
        self.head_size_v = head_size_v = size_v // num_heads
        self.model_size_k = size_k
        self.model_size_v = size_v
        self.num_heads = num_heads
        
        # Content Layers
        self.k_layer = nn.Linear(size_k, num_heads * head_size_k)
        self.v_layer = nn.Linear(size_v, num_heads * head_size_v)
        self.q_layer = nn.Linear(size_q, num_heads * head_size_k) # Dh_q assumed same as Dh_k for matmul
        
        # [INNOVATION] Position Layers for Disentangled Attention
        # 假设位置编码维度与内容一致，这里使用 size_k 和 size_q 初始化
        self.pos_k_layer = nn.Linear(size_k, num_heads * head_size_k)
        self.pos_q_layer = nn.Linear(size_q, num_heads * head_size_k)

        # Output layer maps back to the original V space
        self.output_layer = nn.Linear(num_heads * head_size_v, size_v)
        
        # Adaptive Spatial Gating Parameter
        # 为每个 Head 学习一个门控生成器。输入是 Query 特征 (head_size_k)，输出是 1 个标量 (Gate)
        self.gate_proj = nn.Linear(head_size_k, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    # [MODIFIED] Added spatial_bias, pos_k, pos_q parameters
    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None, 
                spatial_bias: Tensor = None, pos_k: Tensor = None, pos_q: Tensor = None):
        """
        Args:
            k, v, q: Content tensors [B, L, D]
            mask: Attention mask
            spatial_bias: KG-based spatial bias
            pos_k: Position tensor for Keys (optional, for disentangled attn) [B, L, D]
            pos_q: Position tensor for Queries (optional, for disentangled attn) [B, L, D]
        """
        batch_size = k.size(0) 
        num_heads = self.num_heads
        head_size_k = self.head_size_k
        head_size_v = self.head_size_v

        # 1. Project Content
        k_c = self.k_layer(k).view(batch_size, -1, num_heads, head_size_k).transpose(1, 2) # [B, H, L_k, Dh_k]
        v_c = self.v_layer(v).view(batch_size, -1, num_heads, head_size_v).transpose(1, 2) # [B, H, L_v, Dh_v]
        q_c = self.q_layer(q).view(batch_size, -1, num_heads, head_size_k).transpose(1, 2) # [B, H, L_q, Dh_k]

        # 2. Calculate Attention Scores
        scale = 1.0 / (head_size_k ** 0.5)
        
        # [INNOVATION] Disentangled Attention Logic
        if pos_k is not None and pos_q is not None:
            # Project Positions
            # Note: Assuming pos_k dim matches size_k and pos_q dim matches size_q
            k_p = self.pos_k_layer(pos_k).view(batch_size, -1, num_heads, head_size_k).transpose(1, 2)
            q_p = self.pos_q_layer(pos_q).view(batch_size, -1, num_heads, head_size_k).transpose(1, 2)

            # Term 1: Content-to-Content (Qc * Kc^T)
            score_cc = torch.matmul(q_c, k_c.transpose(2, 3))
            
            # Term 2: Content-to-Position (Qc * Kp^T)
            score_cp = torch.matmul(q_c, k_p.transpose(2, 3))
            
            # Term 3: Position-to-Content (Qp * Kc^T)
            score_pc = torch.matmul(q_p, k_c.transpose(2, 3))
            
            # Sum terms (Using DeBERTa style disentangled attention)
            scores = (score_cc + score_cp + score_pc) * scale
        else:
            # Standard Scaled Dot-Product Attention
            scores = torch.matmul(q_c, k_c.transpose(2, 3)) * scale
        
        # [INNOVATION] Adaptive Spatial Gating Logic
        if spatial_bias is not None:
            # 计算 Gate 值: [B, H, L_q, 1]
            # 使用 Content Query (q_c) 来决定空间先验的重要性
            gate = torch.sigmoid(self.gate_proj(q_c))
            
            # 软注入 Spatial Bias
            scores = scores + gate * spatial_bias

        if mask is not None:
            if mask.dim() == 2:
                mask_expanded = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask_expanded = mask.unsqueeze(1)
            elif mask.dim() == 4:
                mask_expanded = mask
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")

            if mask_expanded.dtype == torch.bool:
                scores = scores.masked_fill(~mask_expanded, float('-inf'))
            else:
                scores = scores.masked_fill(mask_expanded == float('-inf'), float('-inf'))
        
        attention = self.softmax(scores) 
        attention = self.dropout(attention)
        
        # Context: [B, H, L_q, Dh_v]
        context = torch.matmul(attention, v_c) 
        
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * head_size_v) 
            
        output = self.output_layer(context) # [B, L_q, size_v]
        return output

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, ff_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, size: int, ff_size: int, num_heads: int, dropout: float = 0.1, src_trg_att: bool = True):
        super(TransformerDecoderLayer, self).__init__()
        self.size = size
        self.src_trg_att = src_trg_att
        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        if src_trg_att:
            self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)
        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor, trg_mask: Tensor):
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x
        if self.src_trg_att:
            h1_norm = self.dec_layer_norm(h1)
            h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)
            o = self.feed_forward(self.dropout(h2) + h1)
        else:
            o = self.feed_forward(h1)
        return o

class PoemLayoutDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, bb_size: int, ff_size: int, num_heads: int, dropout: float = 0.1):
        super(PoemLayoutDecoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.bb_size = bb_size
        
        # Self-attention: Q, K, V all from layout space (bb_size)
        # Disentangled Attention will reuse size_k/q for pos_k/q
        self.layout_self_attn = ContMultiHeadedAttention(num_heads, bb_size, bb_size, bb_size, dropout=dropout)
        
        # Cross-attention: Q from layout (bb_size), K,V from text (hidden_size)
        self.layout_text_attn = ContMultiHeadedAttention(num_heads, bb_size, hidden_size, hidden_size, dropout=dropout)
        self.layout_text_proj = nn.Linear(hidden_size, bb_size)
        self.layout_ffn = PositionwiseFeedForward(bb_size, ff_size)

        # Cross-attention: Q from text (hidden_size), K,V from layout (bb_size)
        self.text_layout_attn = ContMultiHeadedAttention(num_heads, hidden_size, bb_size, bb_size, dropout=dropout)
        self.text_layout_proj = nn.Linear(bb_size, hidden_size)
        self.text_ffn = PositionwiseFeedForward(hidden_size, ff_size)

        self.layout_norm1 = nn.LayerNorm(bb_size, eps=1e-6)
        self.layout_norm2 = nn.LayerNorm(bb_size, eps=1e-6)
        self.text_norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.text_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # [MODIFIED] Added layout_pos argument for Disentangled Attention
    def forward(self, layout_content: Tensor, text_x: Tensor, text_memory: Tensor, 
                src_mask: Tensor, trg_mask: Tensor, spatial_bias: Tensor = None, 
                layout_pos: Tensor = None):
        """
        Args:
            layout_content: [B, T, bb_size] (Content embeddings only)
            text_x: [B, T, hidden_size]
            text_memory: [B, L_text, hidden_size]
            spatial_bias: [B, H, T, T] or None
            layout_pos: [B, T, bb_size] (Position embeddings only) - NEW
        """
        
        # 1. Layout self-attention (Disentangled)
        layout_norm1 = self.layout_norm1(layout_content)
        
        # Pass layout_pos to pos_k and pos_q to trigger disentangled logic
        layout_self_out = self.layout_self_attn(
            k=layout_norm1, v=layout_norm1, q=layout_norm1, 
            mask=trg_mask, 
            spatial_bias=spatial_bias,
            pos_k=layout_pos, # [INNOVATION]
            pos_q=layout_pos  # [INNOVATION]
        )
        
        layout_self_out = self.dropout(layout_self_out) + layout_content # Residual

        # 2. Layout attends to Text (Standard Cross Attention)
        # Note: Cross Attention typically doesn't use relative position of layout
        layout_norm2 = self.layout_norm2(layout_self_out)
        layout_text_out = self.layout_text_attn(text_memory, text_memory, layout_norm2, mask=src_mask)
        layout_text_out_proj = self.layout_text_proj(layout_text_out)
        layout_out_pre_ffn = self.dropout(layout_text_out_proj) + layout_self_out
        layout_out = self.layout_ffn(layout_out_pre_ffn) 

        # 3. Text attends to Layout (Standard Cross Attention)
        text_norm1 = self.text_norm1(text_x)
        text_layout_out = self.text_layout_attn(layout_out, layout_out, text_norm1, mask=None)
        text_layout_out_to_hidden = self.text_layout_proj(text_layout_out)
        text_out_pre_ffn = self.dropout(text_layout_out_to_hidden) + text_x
        text_out = self.text_ffn(text_out_pre_ffn) 

        return layout_out, text_out