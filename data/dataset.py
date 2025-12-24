# File: data/dataset.py (V8.5: Final Path Fix with Auto-Resolve)

import os
import torch
import pandas as pd
import yaml 
from pathlib import Path
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Optional
import numpy as np 
import random
import cv2  # OpenCV for visual feature extraction

# --- 导入知识图谱模型 ---
from models.kg import PoetryKnowledgeGraph
# --- 导入位置引导信号生成器 ---
from models.location import LocationSignalGenerator
# ---------------------------

# 类别定义
CLASS_NAMES = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}
VALID_CLASS_IDS = set(CLASS_NAMES.keys())

class VisualGestaltExtractor:
    """
    [V8.5 升级版] 视觉态势提取器
    利用水墨画的像素灰度（浓淡）直接计算物理势能。
    包含路径检查，防止因找不到图片导致训练崩溃。
    """
    def extract(self, image_path: str, box: List[float]) -> Tuple[List[float], float]:
        """
        输入: 全图路径, 归一化 Box [cx, cy, w, h]
        输出: 
            1. 态势参数 [bias_x, bias_y, rotation, flow]
            2. 有效性 (validity): 1.0 表示提取成功，0.0 表示失败
        """
        try:
            # 1. 安全性检查
            if not os.path.exists(image_path):
                # 路径不存在时返回无效，允许训练继续
                return [0.0, 0.0, 0.0, 0.0], 0.0
            
            # 读取灰度图
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return [0.0, 0.0, 0.0, 0.0], 0.0
                
            H, W = img.shape
            cx, cy, w, h = box
            
            # 2. 裁切物体 (Crop)
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)
            
            # 边界保护
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            # 区域过小则视为无效
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                return [0.0, 0.0, 0.0, 0.0], 0.0
                
            crop = img[y1:y2, x1:x2]
            
            # 3. 水墨预处理：反色 + 软性降噪
            # 纸白(255) -> 0, 墨黑(0) -> 255
            ink_map = 255.0 - crop.astype(float)
            
            # 底噪过滤
            ink_map[ink_map < 30] = 0 
            
            total_ink = np.sum(ink_map)
            if total_ink < 100: 
                return [0.0, 0.0, 0.0, 0.0], 0.0

            # === A. 计算 Bias (重心偏移) & Rotation (主轴) ===
            M = cv2.moments(ink_map.astype(np.float32), binaryImage=False)
            
            bias_x, bias_y = 0.0, 0.0
            rotation = 0.0
            
            if M["m00"] != 0:
                # 重心
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
                
                # 几何中心
                h_crop, w_crop = ink_map.shape
                geo_cX = w_crop / 2.0
                geo_cY = h_crop / 2.0
                
                # 归一化偏移 (-1.0 ~ 1.0)
                bias_x = (cX - geo_cX) / (geo_cX + 1e-6)
                bias_y = (cY - geo_cY) / (geo_cY + 1e-6)
                bias_x = np.clip(bias_x, -1.0, 1.0)
                bias_y = np.clip(bias_y, -1.0, 1.0)
                
                # 主轴角度
                mu20 = M["mu20"] / M["m00"]
                mu02 = M["mu02"] / M["m00"]
                mu11 = M["mu11"] / M["m00"]
                
                theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
                rotation = theta / (np.pi / 2) # 归一化到 -1 ~ 1
            
            # === B. 计算 Flow (洇散度/墨韵) ===
            h_crop, w_crop = ink_map.shape
            avg_density = total_ink / (w_crop * h_crop * 255.0)
            
            # 边缘强度 (梯度)
            sobelx = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            avg_grad = np.mean(grad_mag) / 255.0 
            
            raw_flow = avg_density / (avg_grad + 0.01)
            
            # [核心逻辑] 建立从物理湿润度到 [-1, 1] 的映射
            # Pivot (干湿分界线) = 0.6
            pivot = 0.6 
            
            if raw_flow > pivot:
                # 湿润区间 (Wet Mode): (0, 1]
                flow = (raw_flow - pivot) / (3.0 - pivot + 1e-6)
                flow = np.clip(flow, 0.01, 1.0)
            else:
                # 枯笔区间 (Dry Mode): [-1, 0)
                flow = (raw_flow - pivot) / pivot
                flow = np.clip(flow, -1.0, -0.01)
            
            return [float(bias_x), float(bias_y), float(rotation), float(flow)], 1.0
            
        except Exception as e:
            # 发生任何错误均视为提取失败
            return [0.0, 0.0, 0.0, 0.0], 0.0


class PoegraphLayoutDataset(Dataset):
    def __init__(
        self,
        xlsx_path: str,
        labels_dir: str,
        bert_model_path: str = "/home/610-sty/huggingface/bert-base-chinese",
        max_layout_length: int = 30, 
        max_text_length: int = 64, 
        preload: bool = False
    ):
        super().__init__()
        self.xlsx_path = xlsx_path
        self.labels_dir = Path(labels_dir)
        self.max_layout_length = max_layout_length 
        self.max_text_length = max_text_length
        self.num_classes = 9 

        print("Initializing Knowledge Graph...")
        self.pkg = PoetryKnowledgeGraph()
        
        self.location_gen = LocationSignalGenerator(grid_size=8)
        
        # 初始化视觉态势提取器
        self.gestalt_extractor = VisualGestaltExtractor()
        print("✅ Visual Gestalt Extractor (Pixel-Level Soft Moments) initialized.")
        
        # 加载 Excel
        df = pd.read_excel(xlsx_path)
        
        # [关键路径修复]
        # dataset_root 指向 .../dataset/
        self.dataset_root = os.path.dirname(os.path.abspath(xlsx_path))
        print(f"[Debug] Dataset Root set to: {self.dataset_root}")
        
        self.data = []

        print("Loading dataset index...")
        path_error_count = 0
        
        for _, row in df.iterrows():
            raw_img_path = str(row['image']).strip()
            poem = str(row['poem']).strip()
            
            # [核心修复] 智能路径解析 logic
            if os.path.isabs(raw_img_path):
                full_img_path = raw_img_path
            else:
                # 策略 A: 直接拼接 (针对文件名已包含目录的情况)
                path_a = os.path.join(self.dataset_root, raw_img_path)
                
                # 策略 B: 强制插入 '6800' 子目录 (针对文件名缺失目录的情况)
                path_b = os.path.join(self.dataset_root, '6800', raw_img_path)
                
                # 自动选择存在的路径
                if os.path.exists(path_a):
                    full_img_path = path_a
                elif os.path.exists(path_b):
                    full_img_path = path_b
                else:
                    # 如果都找不到，默认使用 path_b，以便报错信息能指向 dataset/6800
                    full_img_path = path_b
            
            # 记录找不到图片的数量，用于调试
            if not os.path.exists(full_img_path):
                path_error_count += 1
                if path_error_count <= 5: # 只打印前5个错误
                    print(f"❌ [Path Error] Image not found.\n   Tried A: {path_a}\n   Tried B: {path_b}")

            img_stem = Path(full_img_path).stem
            label_path = self.labels_dir / f"{img_stem}.txt"

            if not label_path.exists():
                continue

            # 读取标注
            boxes = []
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5: continue
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        if cls_id in VALID_CLASS_IDS and \
                           0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1:
                            boxes.append((float(cls_id), cx, cy, w, h)) 
            except Exception:
                continue

            if boxes:
                self.data.append({
                    'poem': poem,
                    'boxes': boxes,
                    'img_path': full_img_path
                })

        print(f"✅ PoegraphLayoutDataset 加载完成，共 {len(self.data)} 个样本")
        if path_error_count > 0:
            print(f"⚠️ 警告: 共 {path_error_count} 张图片未找到。请检查文件名是否正确。")
            
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        poem = sample['poem']
        gt_boxes = sample['boxes'] # List[(cls_id, cx, cy, w, h)]
        img_path = sample['img_path']

        # 1. 文本编码
        tokenized = self.tokenizer(
            poem,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )

        # 2. KG 提取
        kg_vector = self.pkg.extract_visual_feature_vector(poem)
        kg_spatial_matrix = self.pkg.extract_spatial_matrix(poem)
        existing_indices = torch.nonzero(kg_vector > 0).squeeze(1)
        raw_ids = (existing_indices + 2).tolist()
        kg_class_ids = self.pkg.expand_ids_with_quantity(raw_ids, poem)

        # 权重列表
        kg_class_weights = []
        for cid in kg_class_ids:
             idx = int(cid) - 2
             if 0 <= idx < self.num_classes:
                 kg_class_weights.append(kg_vector[idx].item())
             else:
                 kg_class_weights.append(1.0)

        if not kg_class_ids:
            kg_class_ids = [0]
            kg_class_weights = [0.0]

        # 位置信号生成
        current_occupancy = torch.zeros((8, 8), dtype=torch.float32) 
        location_grids_list = [] 
        for i, cls_id in enumerate(kg_class_ids):
            cls_id = int(cls_id)
            if cls_id == 0:
                location_grids_list.append(torch.zeros((8, 8), dtype=torch.float32))
                continue
            matrix_idx = cls_id - 2 if 0 <= cls_id - 2 < self.num_classes else 0
            spatial_row = kg_spatial_matrix[matrix_idx]  
            spatial_col = kg_spatial_matrix[:, matrix_idx] 
            signal, current_occupancy = self.location_gen.infer_stateful_signal(
                i, spatial_row, spatial_col, current_occupancy,
                mode='sample', top_k=3 
            )
            if random.random() < 0.7: 
                shift = random.randint(-2, 2) 
                signal = torch.roll(signal, shifts=shift, dims=1) 
            location_grids_list.append(signal)

        # 3. GT 对齐与 [NEW] 视觉特征提取
        target_boxes_8d = [] 
        loss_mask = []
        gestalt_mask = [] # 用于指示哪些样本的 Gestalt 是有效的

        gt_dict = {}
        for item in gt_boxes:
            cid, cx, cy, w, h = item
            cid = int(cid)
            if cid not in gt_dict: gt_dict[cid] = []
            gt_dict[cid].append([cx, cy, w, h])

        do_flip = random.random() < 0.5
        
        for k_cls in kg_class_ids:
            k_cls = int(k_cls)
            if k_cls == 0: 
                target_boxes_8d.append([0.0] * 8)
                loss_mask.append(0.0)
                gestalt_mask.append(0.0)
                continue

            if k_cls in gt_dict and len(gt_dict[k_cls]) > 0:
                box = gt_dict[k_cls].pop(0) # [cx, cy, w, h]
                
                # 脏数据过滤
                if box[2] * box[3] > 0.90 or box[2]/(box[3]+1e-6) > 10.0 or box[2]/(box[3]+1e-6) < 0.1:
                    target_boxes_8d.append([0.0] * 8)
                    loss_mask.append(0.0)
                    gestalt_mask.append(0.0)
                    continue
                
                # [V8.1 核心] 提取视觉态势特征和有效性掩码
                gestalt_features, g_valid = self.gestalt_extractor.extract(img_path, box) # ([features], valid)
                
                # 几何增强应用 (Flip)
                if do_flip:
                    box[0] = 1.0 - box[0] # Flip cx
                    # 翻转时，水平偏移反向，旋转角度反向，垂直偏移和Flow不变
                    gestalt_features[0] = -gestalt_features[0] # Flip bias_x 
                    gestalt_features[2] = -gestalt_features[2] # Flip rotation 
                
                # Jitter (仅对坐标增强，保持 Pixel 提取的 Gestalt 不变)
                noise = np.random.uniform(-0.02, 0.02, size=4)
                box_aug = [
                    np.clip(box[0] + noise[0], 0.0, 1.0),
                    np.clip(box[1] + noise[1], 0.0, 1.0),
                    np.clip(box[2] + noise[2], 0.01, 1.0),
                    np.clip(box[3] + noise[3], 0.01, 1.0)
                ]
                
                # 合并 4维坐标 + 4维态势 = 8维 Target
                target_boxes_8d.append(box_aug + gestalt_features)
                loss_mask.append(1.0)
                gestalt_mask.append(g_valid) # 记录该样本的态势是否有效
            else:
                target_boxes_8d.append([0.0] * 8)
                loss_mask.append(0.0)
                gestalt_mask.append(0.0)

        # 截断
        if len(kg_class_ids) > self.max_layout_length:
            kg_class_ids = kg_class_ids[:self.max_layout_length]
            kg_class_weights = kg_class_weights[:self.max_layout_length] 
            target_boxes_8d = target_boxes_8d[:self.max_layout_length]
            loss_mask = loss_mask[:self.max_layout_length]
            gestalt_mask = gestalt_mask[:self.max_layout_length]
            location_grids_list = location_grids_list[:self.max_layout_length]

        location_grids = torch.stack(location_grids_list)
        if do_flip:
            location_grids = torch.flip(location_grids, dims=[2])

        return {
            'input_ids': tokenized['input_ids'].squeeze(0), 
            'attention_mask': tokenized['attention_mask'].squeeze(0), 
            'kg_class_ids': torch.tensor(kg_class_ids, dtype=torch.long),
            'kg_class_weights': torch.tensor(kg_class_weights, dtype=torch.float32), 
            'target_boxes': torch.tensor(target_boxes_8d, dtype=torch.float32), 
            'loss_mask': torch.tensor(loss_mask, dtype=torch.float32),
            'gestalt_mask': torch.tensor(gestalt_mask, dtype=torch.float32), 
            'kg_spatial_matrix': kg_spatial_matrix,
            'kg_vector': kg_vector,
            'num_boxes': torch.tensor(len(gt_boxes), dtype=torch.long),
            'location_grids': location_grids 
        }

def layout_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function 适配 8 维 target_boxes 和 gestalt_mask"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    kg_spatial_matrices = torch.stack([item['kg_spatial_matrix'] for item in batch])
    kg_vectors = torch.stack([item['kg_vector'] for item in batch])
    num_boxes = torch.stack([item['num_boxes'] for item in batch])

    lengths = [len(item['kg_class_ids']) for item in batch]
    max_len = max(lengths)
    if max_len == 0: max_len = 1

    batched_class_ids = []
    batched_class_weights = [] 
    batched_target_boxes = []
    batched_loss_mask = []
    batched_gestalt_mask = [] # [NEW]
    batched_padding_mask = [] 
    batched_location_grids = []

    for item in batch:
        cur_len = len(item['kg_class_ids'])
        pad_len = max_len - cur_len
        
        # 1. IDs
        padded_ids = torch.cat([
            item['kg_class_ids'], 
            torch.zeros(pad_len, dtype=torch.long)
        ])
        batched_class_ids.append(padded_ids)
        
        # 2. Weights 
        padded_weights = torch.cat([
            item['kg_class_weights'],
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        batched_class_weights.append(padded_weights)
        
        # 3. Boxes (8-dim)
        padded_boxes = torch.cat([
            item['target_boxes'], 
            torch.zeros((pad_len, 8), dtype=torch.float32)
        ])
        batched_target_boxes.append(padded_boxes)
        
        # 4. Loss Mask
        padded_loss_mask = torch.cat([
            item['loss_mask'], 
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        batched_loss_mask.append(padded_loss_mask)
        
        # 5. [NEW] Gestalt Mask
        padded_gestalt_mask = torch.cat([
            item['gestalt_mask'],
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        batched_gestalt_mask.append(padded_gestalt_mask)

        # 6. Location Grids
        padded_grids = torch.cat([
            item['location_grids'],
            torch.zeros((pad_len, 8, 8), dtype=torch.float32)
        ])
        batched_location_grids.append(padded_grids)
        
        # 7. Pad Mask (Transformer Attention Mask)
        pad_mask = torch.zeros(max_len, dtype=torch.bool)
        if pad_len > 0:
            pad_mask[cur_len:] = True
        batched_padding_mask.append(pad_mask)

    return {
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'kg_class_ids': torch.stack(batched_class_ids),      
        'kg_class_weights': torch.stack(batched_class_weights), 
        'target_boxes': torch.stack(batched_target_boxes),   
        'loss_mask': torch.stack(batched_loss_mask),   
        'gestalt_mask': torch.stack(batched_gestalt_mask), # [NEW]      
        'padding_mask': torch.stack(batched_padding_mask),   
        'kg_spatial_matrix': kg_spatial_matrices,
        'kg_vector': kg_vectors,
        'num_boxes': num_boxes,
        'location_grids': torch.stack(batched_location_grids) 
    }

if __name__ == "__main__":
    pass