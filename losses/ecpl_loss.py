# -*- coding: utf-8 -*-
"""
================================================================================
CVPR Innovation: ECPL (Elevation-Conditioned Prototype Learning)
高程条件化原型学习损失函数

================================================================================
创新性详解
================================================================================

【理论创新】高程条件化的表示学习
- 首次提出利用高程信息作为条件来学习类别原型
- 打破"单一原型"的传统范式
- 实现"高程感知的类别表示"

【方法创新】高程一致性对比损失
- 强制模型学习：Low Veg应该在低高程区域
- 强制模型学习：Trees应该在高高程区域
- 惩罚高程不一致的预测

【架构创新】零参数增加
- 不修改任何模型架构
- 只增加一个新的损失函数
- 与S3F完全兼容

================================================================================
核心公式
================================================================================

1. 高程一致性损失:
   L_elev = -log(P(correct_class | elevation))
   
   对于Low Veg (class 2):
   - elevation < 1m: 正常，无惩罚
   - elevation > 2m: 异常，强惩罚
   
   对于Trees (class 3):
   - elevation > 3m: 正常，无惩罚
   - elevation < 1m: 异常，强惩罚

2. 原型对比损失:
   L_proto = -log(exp(sim(f, p_pos)/τ) / Σexp(sim(f, p_neg)/τ))

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ElevationConsistencyLoss(nn.Module):
    """
    高程一致性损失
    
    核心思想：惩罚高程不一致的预测
    - Low Veg (class 2) 应该在低高程区域 (< 1m)
    - Trees (class 3) 应该在高高程区域 (> 3m)
    """
    def __init__(self, num_classes=6, low_veg_class=2, trees_class=3):
        super().__init__()
        self.num_classes = num_classes
        self.low_veg_class = low_veg_class
        self.trees_class = trees_class
        
        # 高程阈值 (归一化后的值，需要根据实际数据调整)
        # 假设nDSM已经归一化到0-1，其中0.1对应约1m，0.3对应约3m
        self.low_veg_max_elev = 0.15  # Low Veg最大高程阈值
        self.trees_min_elev = 0.25    # Trees最小高程阈值
        
    def forward(self, pred_logits, ndsm, target):
        """
        Args:
            pred_logits: [B, C, H, W] 预测logits
            ndsm: [B, 1, H, W] 归一化高程图 (0-1)
            target: [B, H, W] 真实标签
        Returns:
            loss: 高程一致性损失
        """
        B, C, H, W = pred_logits.shape
        
        # 调整ndsm尺寸
        if ndsm.shape[2:] != (H, W):
            ndsm = F.interpolate(ndsm, size=(H, W), mode='bilinear', align_corners=False)
        ndsm = ndsm.squeeze(1)  # [B, H, W]
        
        # 获取预测概率
        pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, H, W]
        
        # Low Veg一致性损失
        # 当预测为Low Veg但高程很高时，惩罚
        low_veg_prob = pred_probs[:, self.low_veg_class]  # [B, H, W]
        low_veg_penalty = torch.clamp(ndsm - self.low_veg_max_elev, min=0)  # 超过阈值的部分
        loss_low_veg = (low_veg_prob * low_veg_penalty).mean()
        
        # Trees一致性损失
        # 当预测为Trees但高程很低时，惩罚
        trees_prob = pred_probs[:, self.trees_class]  # [B, H, W]
        trees_penalty = torch.clamp(self.trees_min_elev - ndsm, min=0)  # 低于阈值的部分
        loss_trees = (trees_prob * trees_penalty).mean()
        
        # 总损失
        loss = loss_low_veg + loss_trees
        
        return loss


class PrototypeContrastiveLoss(nn.Module):
    """
    原型对比损失
    
    核心思想：
    - 拉近同类别特征与其原型的距离
    - 推远不同类别特征与其原型的距离
    - 特别关注Low Veg和Trees的区分
    """
    def __init__(self, num_classes=6, feat_dim=256, temperature=0.1, momentum=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.momentum = momentum
        
        # 注册原型缓冲区 (不参与梯度计算)
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))
        self.initialized = False
        
    def update_prototypes(self, features, labels):
        """
        使用动量更新原型
        
        Args:
            features: [N, D] 特征向量
            labels: [N] 标签
        """
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_features = features[mask]
                    class_mean = class_features.mean(dim=0)
                    
                    if not self.initialized or self.prototype_counts[c] == 0:
                        self.prototypes[c] = class_mean
                    else:
                        self.prototypes[c] = (self.momentum * self.prototypes[c] + 
                                             (1 - self.momentum) * class_mean)
                    self.prototype_counts[c] += mask.sum()
            
            self.initialized = True
    
    def forward(self, features, labels):
        """
        Args:
            features: [B, D, H, W] 特征图
            labels: [B, H, W] 标签
        Returns:
            loss: 原型对比损失
        """
        B, D, H, W = features.shape
        
        # 展平
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, D)  # [BHW, D]
        labels_flat = labels.reshape(-1)  # [BHW]
        
        # 过滤无效标签
        valid_mask = (labels_flat >= 0) & (labels_flat < self.num_classes)
        features_valid = features_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]
        
        if features_valid.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 更新原型
        self.update_prototypes(features_valid.detach(), labels_valid)
        
        if not self.initialized:
            return torch.tensor(0.0, device=features.device)
        
        # 计算特征与所有原型的相似度
        features_norm = F.normalize(features_valid, dim=1)
        prototypes_norm = F.normalize(self.prototypes, dim=1)
        
        similarities = torch.mm(features_norm, prototypes_norm.t()) / self.temperature  # [N, C]
        
        # 对比损失
        loss = F.cross_entropy(similarities, labels_valid)
        
        return loss


class ElevationConditionedPrototypeLoss(nn.Module):
    """
    高程条件化原型损失 (完整版)
    
    结合高程一致性损失和原型对比损失
    """
    def __init__(self, num_classes=6, feat_dim=256, 
                 lambda_elev=1.0, lambda_proto=0.5,
                 low_veg_class=2, trees_class=3):
        super().__init__()
        self.lambda_elev = lambda_elev
        self.lambda_proto = lambda_proto
        
        self.elev_loss = ElevationConsistencyLoss(
            num_classes=num_classes,
            low_veg_class=low_veg_class,
            trees_class=trees_class
        )
        self.proto_loss = PrototypeContrastiveLoss(
            num_classes=num_classes,
            feat_dim=feat_dim
        )
        
    def forward(self, pred_logits, features, ndsm, target):
        """
        Args:
            pred_logits: [B, C, H, W] 预测logits
            features: [B, D, H, W] 中间特征 (用于原型学习)
            ndsm: [B, 1, H, W] 归一化高程图
            target: [B, H, W] 真实标签
        Returns:
            loss: 总损失
            loss_elev: 高程一致性损失
            loss_proto: 原型对比损失
        """
        # 高程一致性损失
        loss_elev = self.elev_loss(pred_logits, ndsm, target)
        
        # 原型对比损失
        # 调整特征尺寸以匹配标签
        H, W = target.shape[1:]
        if features.shape[2:] != (H, W):
            features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)
        loss_proto = self.proto_loss(features, target)
        
        # 总损失
        loss = self.lambda_elev * loss_elev + self.lambda_proto * loss_proto
        
        return loss, loss_elev, loss_proto


class HardConfusionMiningLoss(nn.Module):
    """
    困难混淆样本挖掘损失
    
    核心思想：
    - 专门针对Low Veg和Trees的混淆区域
    - 在这些区域增加损失权重
    - 利用高程信息辅助判断
    """
    def __init__(self, num_classes=6, low_veg_class=2, trees_class=3):
        super().__init__()
        self.num_classes = num_classes
        self.low_veg_class = low_veg_class
        self.trees_class = trees_class
        
    def forward(self, pred_logits, ndsm, target):
        """
        Args:
            pred_logits: [B, C, H, W] 预测logits
            ndsm: [B, 1, H, W] 归一化高程图
            target: [B, H, W] 真实标签
        Returns:
            loss: 困难样本损失
        """
        B, C, H, W = pred_logits.shape
        
        # 调整ndsm尺寸
        if ndsm.shape[2:] != (H, W):
            ndsm = F.interpolate(ndsm, size=(H, W), mode='bilinear', align_corners=False)
        ndsm = ndsm.squeeze(1)  # [B, H, W]
        
        # 获取预测
        pred = pred_logits.argmax(dim=1)  # [B, H, W]
        pred_probs = F.softmax(pred_logits, dim=1)
        
        # 找到混淆区域：真实是Low Veg但预测为Trees，或反之
        confusion_mask_1 = (target == self.low_veg_class) & (pred == self.trees_class)
        confusion_mask_2 = (target == self.trees_class) & (pred == self.low_veg_class)
        confusion_mask = confusion_mask_1 | confusion_mask_2
        
        if confusion_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_logits.device)
        
        # 在混淆区域计算额外损失
        # 使用高程信息作为软标签
        # Low Veg区域（低高程）应该预测Low Veg
        # Trees区域（高高程）应该预测Trees
        
        # 高程引导的软标签
        elev_weight_low_veg = torch.clamp(1.0 - ndsm * 3, min=0, max=1)  # 低高程→高权重
        elev_weight_trees = torch.clamp(ndsm * 3 - 0.5, min=0, max=1)    # 高高程→高权重
        
        # 在混淆区域，根据高程调整损失
        loss_low_veg = -torch.log(pred_probs[:, self.low_veg_class] + 1e-6) * elev_weight_low_veg
        loss_trees = -torch.log(pred_probs[:, self.trees_class] + 1e-6) * elev_weight_trees
        
        # 只在混淆区域计算
        loss = (loss_low_veg * confusion_mask.float() * (target == self.low_veg_class).float() +
                loss_trees * confusion_mask.float() * (target == self.trees_class).float())
        
        return loss.mean()


class ECPLFullLoss(nn.Module):
    """
    ECPL完整损失函数
    
    整合所有组件：
    1. 高程一致性损失
    2. 原型对比损失
    3. 困难混淆样本挖掘损失
    """
    def __init__(self, num_classes=6, feat_dim=256,
                 lambda_elev=2.0, lambda_proto=0.5, lambda_hard=1.0,
                 low_veg_class=2, trees_class=3):
        super().__init__()
        self.lambda_elev = lambda_elev
        self.lambda_proto = lambda_proto
        self.lambda_hard = lambda_hard
        
        self.elev_loss = ElevationConsistencyLoss(
            num_classes=num_classes,
            low_veg_class=low_veg_class,
            trees_class=trees_class
        )
        self.proto_loss = PrototypeContrastiveLoss(
            num_classes=num_classes,
            feat_dim=feat_dim
        )
        self.hard_loss = HardConfusionMiningLoss(
            num_classes=num_classes,
            low_veg_class=low_veg_class,
            trees_class=trees_class
        )
        
    def forward(self, pred_logits, features, ndsm, target):
        """
        Args:
            pred_logits: [B, C, H, W] 预测logits
            features: [B, D, H, W] 中间特征
            ndsm: [B, 1, H, W] 归一化高程图
            target: [B, H, W] 真实标签
        Returns:
            loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 高程一致性损失
        loss_elev = self.elev_loss(pred_logits, ndsm, target)
        
        # 原型对比损失
        H, W = target.shape[1:]
        if features.shape[2:] != (H, W):
            features_resized = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)
        else:
            features_resized = features
        loss_proto = self.proto_loss(features_resized, target)
        
        # 困难样本损失
        loss_hard = self.hard_loss(pred_logits, ndsm, target)
        
        # 总损失
        loss = (self.lambda_elev * loss_elev + 
                self.lambda_proto * loss_proto + 
                self.lambda_hard * loss_hard)
        
        loss_dict = {
            'loss_elev': loss_elev.item(),
            'loss_proto': loss_proto.item(),
            'loss_hard': loss_hard.item()
        }
        
        return loss, loss_dict


# =============================================================================
# 简化版：只使用高程一致性损失（最轻量）
# =============================================================================

class SimpleElevationLoss(nn.Module):
    """
    简化版高程一致性损失
    
    最轻量的实现，只惩罚高程不一致的预测
    """
    def __init__(self, low_veg_class=2, trees_class=3):
        super().__init__()
        self.low_veg_class = low_veg_class
        self.trees_class = trees_class
        
    def forward(self, pred_logits, ndsm, target=None):
        """
        Args:
            pred_logits: [B, C, H, W] 预测logits
            ndsm: [B, 1, H, W] 归一化高程图 (0-1)
            target: [B, H, W] 真实标签 (可选，用于监督)
        Returns:
            loss: 高程一致性损失
        """
        B, C, H, W = pred_logits.shape
        
        # 调整ndsm尺寸
        if ndsm.shape[2:] != (H, W):
            ndsm = F.interpolate(ndsm, size=(H, W), mode='bilinear', align_corners=False)
        ndsm = ndsm.squeeze(1)  # [B, H, W]
        
        # 获取预测概率
        pred_probs = F.softmax(pred_logits, dim=1)
        
        # Low Veg: 高程高时惩罚
        low_veg_prob = pred_probs[:, self.low_veg_class]
        low_veg_penalty = ndsm ** 2  # 高程越高，惩罚越大
        loss_low_veg = (low_veg_prob * low_veg_penalty).mean()
        
        # Trees: 高程低时惩罚
        trees_prob = pred_probs[:, self.trees_class]
        trees_penalty = (1 - ndsm) ** 2  # 高程越低，惩罚越大
        loss_trees = (trees_prob * trees_penalty).mean()
        
        return loss_low_veg + loss_trees
