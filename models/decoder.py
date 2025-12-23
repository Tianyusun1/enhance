# models/decoder.py (V6.0: Disentangled Attention Support)

import torch
import torch.nn as nn
from .transformer_layers import PoemLayoutDecoderLayer

class LayoutDecoder(nn.Module):
    def __init__(self, hidden_size: int, bb_size: int, num_layers: int, num_heads: int, ff_size: int = None, dropout: float = 0.1):
        super(LayoutDecoder, self).__init__()
        if ff_size is None:
            ff_size = hidden_size * 4
        self.bb_size = bb_size
        
        self.layers = nn.ModuleList([
            PoemLayoutDecoderLayer(
                hidden_size=hidden_size,
                bb_size=bb_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # The final layer norm input size is hidden_size + bb_size
        self.layer_norm = nn.LayerNorm(hidden_size + bb_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    # [MODIFIED] 接口更新：分别接收 layout_content 和 layout_pos
    def forward(self, layout_content: torch.Tensor, layout_pos: torch.Tensor, text_features: torch.Tensor, src_mask: torch.Tensor, trg_mask: torch.Tensor, spatial_bias: torch.Tensor = None):
        """
        Args:
            layout_content: [B, T, bb_size] - 纯内容嵌入 (Object Embedding + Z)
            layout_pos:     [B, T, bb_size] - 纯位置嵌入 (Learnable Pos + Location Signal)
            text_features:  [B, L_text, hidden_size]
            ...
        """
        # layout_content: [B, T, bb_size]
        # text_features: [B, L_text, hidden_size]
        B, T, bb_size_dim = layout_content.shape
        B_text, L_text, hidden_size_dim = text_features.shape

        # === 关键修复区域 (V5.2 Fix) ===
        # 使用 BERT 的 [CLS] token (Index 0) 初始化 text_x 流。
        
        # 1. 提取 [CLS] token: [B, L_text, H] -> [B, 1, H]
        cls_token = text_features[:, 0, :].unsqueeze(1)
        
        # 2. 扩展到序列长度 T: [B, 1, H] -> [B, T, H]
        text_x = cls_token.expand(-1, T, -1).clone()
        
        # [INNOVATION] 初始的 layout_x 即为 layout_content
        # 在解耦注意力中，内容会随层更新，但位置信息通常作为辅助信息透传（或共享）
        layout_x = layout_content 

        # Iterate through the stack of layers
        for layer in self.layers:
            # [MODIFIED] 将 layout_pos 单独传入每一层
            # 注意：这意味着 transformer_layers.py 中的 PoemLayoutDecoderLayer 也必须同步修改接口
            layout_x, text_x = layer(
                layout_content=layout_x, 
                layout_pos=layout_pos, 
                text_x=text_x, 
                text_memory=text_features, 
                src_mask=src_mask, 
                trg_mask=trg_mask, 
                spatial_bias=spatial_bias
            )

        # After all layers, concatenate the final streams
        # Order: [text_features_stream, layout_features_stream]
        # 注意：输出仅包含细化后的内容特征 layout_x，不包含 layout_pos
        output = torch.cat([text_x, layout_x], dim=-1) # [B, T, hidden_size + bb_size]
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output # [B, T, hidden_size + bb_size]