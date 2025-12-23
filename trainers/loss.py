# trainers/loss.py
import torch
import torch.nn.functional as F

def layout_loss(pred_cls, pred_coord, target_layout_seq, target_layout_mask, coord_loss_weight=1.0):
    """
    计算布局生成的混合损失 (Supervised Reconstruction Loss)。
    适用于基础几何布局 (Class + 4D Box)。
    
    Args:
        pred_cls: [B, num_elements, num_classes] 预测的类别 logits
        pred_coord: [B, num_elements, 4] 预测的坐标 (cx, cy, w, h)
        target_layout_seq: [B, S] (S = 5 * num_elements) 真实布局序列
        target_layout_mask: [B, S] 真实布局掩码
        coord_loss_weight: 坐标损失权重
    Returns:
        total_loss, cls_loss, coord_loss
    """
    batch_size, seq_len = target_layout_seq.shape
    num_elements = seq_len // 5

    reshaped_target = target_layout_seq.view(batch_size, num_elements, 5)
    target_cls_ids = reshaped_target[:, :, 0].long() # [B, num_elements]
    target_coords = reshaped_target[:, :, 1:5].float() # [B, num_elements, 4]

    reshaped_mask = target_layout_mask.view(batch_size, num_elements, 5)
    cls_mask = reshaped_mask[:, :, 0].bool() # [B, num_elements]

    # Classification loss
    # reduction='none' 允许我们手动应用 mask
    cls_loss = F.cross_entropy(pred_cls, target_cls_ids, reduction='none') # [B, num_elements]
    cls_loss = cls_loss * cls_mask.float()
    cls_loss = cls_loss.sum() / cls_mask.sum().clamp(min=1)

    # Coordinate loss
    coord_loss = F.smooth_l1_loss(pred_coord, target_coords, reduction='none') # [B, num_elements, 4]
    coord_loss = coord_loss * cls_mask.unsqueeze(-1).float()
    coord_loss = coord_loss.sum() / (cls_mask.sum().clamp(min=1) * 4)

    total_loss = cls_loss + coord_loss_weight * coord_loss
    return total_loss, cls_loss, coord_loss

def compute_gestalt_loss(
    pred_gestalt, 
    target_gestalt, 
    loss_mask, 
    gestalt_mask=None, 
    kg_class_ids=None, 
    pred_coords=None
):
    """
    [NEW V8.0] 计算视觉态势 (Gestalt) 的综合损失。
    结合了：
    1. 像素级监督 (Pixel-Level Supervision): 仅在提取置信度高 (gestalt_mask=1) 的区域计算回归损失。
    2. 语义先验 (Semantic Prior): 自监督，根据物体类别约束物理属性 (如太阳静止、水流动)。
    3. 空间平滑 (Spatial Smoothness): 自监督，相邻物体态势应趋于一致。

    Args:
        pred_gestalt: [B, N, 4] 预测态势 (bias_x, bias_y, rotation, flow)
                      前3维建议 Tanh (-1~1), 第4维建议 Sigmoid (0~1)
        target_gestalt: [B, N, 4] GT 态势 (从 OpenCV 提取)
        loss_mask: [B, N] 基础布局掩码 (Padding Mask)
        gestalt_mask: [B, N] (Optional) 态势有效性掩码 (1=Valid Extraction, 0=Failed/Noise)
                      用于过滤掉原图中提取失败的样本，防止污染模型。
        kg_class_ids: [B, N] (Optional) 物体类别 ID，用于语义先验。
        pred_coords: [B, N, 4] (Optional) 预测坐标，用于计算空间邻居关系。
        
    Returns:
        total_gestalt_loss: scalar
    """
    # 1. 基础监督损失 (Supervised Regression)
    # ---------------------------------------------------
    # 只有当 (是有效物体 loss_mask=1) AND (CV提取成功 gestalt_mask=1) 时才计算回归
    if gestalt_mask is not None:
        valid_mask = loss_mask * gestalt_mask
    else:
        valid_mask = loss_mask

    num_valid = valid_mask.sum().clamp(min=1)
    
    # Smooth L1 Loss
    # 注意: target_gestalt 应该已经在 Dataset 中归一化好
    reg_loss = F.smooth_l1_loss(pred_gestalt, target_gestalt, reduction='none') # [B, N, 4]
    loss_supervised = (reg_loss.mean(dim=-1) * valid_mask).sum() / num_valid

    # 2. 语义先验损失 (Semantic Prior - Self-Supervised)
    # ---------------------------------------------------
    loss_semantic = torch.tensor(0.0, device=pred_gestalt.device)
    if kg_class_ids is not None:
        # 定义静态物体 ID (需根据实际 Dataset 修改，假设 2=Sun, 3=Moon, 6=Building)
        # 这些物体通常没有显著的 Rotation 或 Flow
        static_ids = torch.tensor([2, 3, 6], device=pred_gestalt.device)
        is_static = torch.isin(kg_class_ids, static_ids) & (loss_mask > 0)
        
        if is_static.any():
            # 惩罚 static 物体的 rotation 和 flow
            # pred_gestalt: [..., 2] is rotation, [..., 3] is flow
            rot_flow_energy = pred_gestalt[..., 2].pow(2) + pred_gestalt[..., 3].pow(2)
            loss_semantic += (rot_flow_energy * is_static.float()).sum() / is_static.sum().clamp(min=1)

        # 定义流体 ID (假设 5=Water/River)
        # 这些物体应该有一定的 Flow
        water_id = 5
        is_water = (kg_class_ids == water_id) & (loss_mask > 0)
        
        if is_water.any():
            # Hinge Loss: 鼓励 flow > 0.3
            flow_pred = pred_gestalt[..., 3]
            loss_flow_incentive = F.relu(0.3 - flow_pred) # 如果 < 0.3 则产生 Loss
            loss_semantic += (loss_flow_incentive * is_water.float()).sum() / is_water.sum().clamp(min=1)

    # 3. 空间平滑损失 (Spatial Smoothness - Self-Supervised)
    # ---------------------------------------------------
    loss_smoothness = torch.tensor(0.0, device=pred_gestalt.device)
    if pred_coords is not None:
        B, N, _ = pred_coords.shape
        # 计算两两距离 [B, N, N]
        centers = pred_coords[..., :2]
        dist_mat = torch.cdist(centers, centers, p=2)
        
        # 定义邻居: 距离 < 0.2 且不是自己
        is_neighbor = (dist_mat < 0.2) & (dist_mat > 1e-4)
        # 确保只计算有效物体
        valid_objs = loss_mask.unsqueeze(1) * loss_mask.unsqueeze(2)
        neighbor_mask = is_neighbor & (valid_objs > 0)
        
        if neighbor_mask.sum() > 0:
            # 计算态势特征的差异 (Bias, Rot, Flow)
            # [B, N, 1, 4] - [B, 1, N, 4] -> [B, N, N, 4]
            g_diff = (pred_gestalt.unsqueeze(2) - pred_gestalt.unsqueeze(1)).pow(2).sum(dim=-1)
            loss_smoothness = (g_diff * neighbor_mask.float()).sum() / neighbor_mask.sum().clamp(min=1)

    # 4. 总损失加权
    # ---------------------------------------------------
    # 权重配置 (可根据实验调整)
    w_sup = 1.0
    w_sem = 0.5
    w_smooth = 0.5
    
    total_gestalt_loss = w_sup * loss_supervised + w_sem * loss_semantic + w_smooth * loss_smoothness
    return total_gestalt_loss

def compute_kl_loss(mu, logvar, free_bits=0.0):
    """
    [New] 健壮的 KL 散度计算函数，支持 Free Bits 策略。
    
    Args:
        mu: [B, latent_dim] 均值
        logvar: [B, latent_dim] 对数方差
        free_bits: float, KL 散度的最小阈值（Hinge Loss）。
                   如果 KL < free_bits，则 Loss 为 0。
                   这可以防止后验分布过早坍缩到先验，保留一定的编码信息。
                   
    Returns:
        kl_loss: scalar (averaged over batch)
    """
    # 1. 计算每个样本的 KL 散度
    # 公式: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    # 注意: 我们需要在 latent_dim 维度 (dim=1) 求和，表示单个样本的总 KL 信息量
    kld_element = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_per_sample = torch.sum(kld_element, dim=1) # [B]
    
    # 2. 应用 Free Bits (Hinge Loss)
    # 允许每个样本保留 free_bits 的信息量而不受惩罚
    if free_bits > 0.0:
        free_bits_tensor = torch.tensor(free_bits, device=mu.device)
        kld_per_sample = torch.max(kld_per_sample, free_bits_tensor)
        
    # 3. 对 Batch 求平均
    # 这是关键：必须使用 mean，否则 Loss 会随 Batch Size 变大而变大，压倒重建 Loss
    kl_loss = torch.mean(kld_per_sample)
    
    return kl_loss