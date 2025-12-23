import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple

# =========================================================================
# NEW CLASS: BBox Tokenizer (离散化坐标工具)
# =========================================================================

class BBoxTokenizer:
    """
    负责将 [0, 1] 范围的浮点坐标离散化为整数 Token ID，以及逆过程。
    Args:
        num_bins: BBox 坐标的离散化 bin 数量 (例如 1000)。
    """
    def __init__(self, num_bins: int = 1000):
        # 确保 num_bins 至少为 2，以避免除零或越界
        self.num_bins = max(2, num_bins)
        
        # Bins 数量 N 对应 Token ID 范围 [0, N-1]。
        # 量化公式使用 N-1 作为除数，以确保 1.0 被映射到最大的 ID (N-1)。
        self.range_divisor = self.num_bins - 1

    def tokenize(self, value: float) -> int:
        """将 [0, 1] 浮点值量化为 [0, num_bins - 1] 范围内的整数 ID。"""
        # 1. 钳位到 [0, 1] 确保安全
        value = np.clip(value, 0.0, 1.0)
        # 2. 缩放到 [0, num_bins - 1] 范围，并向下取整
        bin_id = int(np.floor(value * self.range_divisor))
        return bin_id

    def detokenize(self, bin_id: int) -> float:
        """将整数 ID [0, num_bins - 1] 还原为 [0, 1] 范围内的浮点值。"""
        # 还原为 bin 的起始值（或 mid-point，这里选择起始值以简化）
        value = float(bin_id) / self.range_divisor
        # 钳位到 [0, 1] 确保边界
        return np.clip(value, 0.0, 1.0).item()

# =========================================================================
# 原始 I/O 函数 (兼容 V8.0 9维数据)
# =========================================================================

def layout_seq_to_yolo_txt(layout_seq: List[Tuple], output_path: str):
    """
    将布局序列保存为 YOLO 格式的 .txt 文件。
    
    [V8.0 Fix] 兼容 9 维数据 (cls, cx, cy, w, h, bx, by, rot, flow)。
    函数会自动截取前 5 维几何信息写入文件，避免因长度检查导致文件为空。

    Args:
        layout_seq: 布局序列，每个元素至少包含 (cls_id, cx, cy, w, h)。
        output_path: 输出 .txt 文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in layout_seq:
            # [Fix] 放宽长度检查，只要包含基础的 5 维坐标即可
            if len(item) < 5:
                continue
            
            # [Fix] 仅解包前 5 个元素用于标准 YOLO 格式写入
            # 忽略后 4 维态势参数 (如需保存态势，建议另存为 JSON 或扩展格式)
            cls_id, cx, cy, w, h = item[:5]
            
            # 写入文件
            f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def yolo_txt_to_layout_seq(txt_path: str) -> List[Tuple]:
    """
    从 YOLO 格式的 .txt 文件加载布局序列 [(cls_id, cx, cy, w, h), ...]。
    
    WARNING: 此函数返回的 cls_id 范围是 2-10 (即原始 ID)

    Args:
        txt_path: 输入 .txt 文件路径
    Returns:
        layout_seq: 布局序列 [(cls_id, cx, cy, w, h)]
    """
    layout_seq = []
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                # [Fix] 同样放宽读取检查，兼容可能包含额外列的文件
                if len(parts) < 5:
                    continue
                
                # 读取时，确保 cls_id 是整数，坐标是浮点数
                cls_id = int(float(parts[0]))
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                layout_seq.append((cls_id, cx, cy, w, h))
    except FileNotFoundError:
        print(f"Error: YOLO file not found at {txt_path}")
    except Exception as e:
        print(f"Error reading YOLO file {txt_path}: {e}")
        
    return layout_seq

def get_device():
    """Returns the torch device (cuda if available, else cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# WARNING: 移除冗余的 collate_fn_for_layout，请使用 dataset.py 中的 layout_collate_fn