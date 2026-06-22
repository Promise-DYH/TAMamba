import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.morphology import erosion, dilation


class DynamicBoundaryLoss(nn.Module):
    """
    动态边界损失函数，采用在线难例挖掘策略。
    灵感来源: "node-guided resampling" 思想，专注于模型预测困难的边界区域。
    """

    def __init__(self, num_classes, beta=5.0, zone_kernel_size=5):
        """
        初始化动态边界损失.

        参数:
            num_classes (int): 数据集中的类别总数。
            beta (float): 难例权重放大因子。beta越大，对预测错误的像素惩罚越重。
            zone_kernel_size (int): 定义边界区域的膨胀核大小，必须是奇数。
                                   例如，5表示以真实边界为中心向外扩展2个像素的区域。
        """
        super(DynamicBoundaryLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        # kornia需要一个2D的核
        self.kernel = torch.ones(zone_kernel_size, zone_kernel_size)
        print(f"Initialized DynamicBoundaryLoss with beta={beta} and zone_kernel_size={zone_kernel_size}")

    def forward(self, pred_boundary_logits, seg_labels):
        """
        计算损失.

        参数:
            pred_boundary_logits (torch.Tensor): 模型的原始边界预测输出 (B, 1, H, W)。
            seg_labels (torch.Tensor): 分割的真值标签 (B, H, W)，值为 0, 1, ..., num_classes-1。

        返回:
            torch.Tensor: 一个标量的损失值。
        """
        # 确保核在正确的设备上
        self.kernel = self.kernel.to(pred_boundary_logits.device)

        # --- 步骤 1: 从分割标签生成边界真值和边界区域 ---
        with torch.no_grad():
            # kornia需要 (B, C, H, W) 的浮点数输入，所以先进行one-hot编码
            labels_one_hot = F.one_hot(seg_labels, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

            # 腐蚀操作找到每个类别区域的内部
            eroded_labels = erosion(labels_one_hot, self.kernel)

            # 边界 = 原始区域 - 腐蚀后的内部区域
            boundary_gt_multi_class = labels_one_hot - eroded_labels

            # 将多类别的边界图合并为单通道的二值边界图
            boundary_gt = (boundary_gt_multi_class.sum(dim=1, keepdim=True) > 0).float()

            # 膨胀操作得到我们关注的“边界区域 (boundary_zone)”
            boundary_zone = dilation(boundary_gt, self.kernel)

        # --- 步骤 2: 计算动态权重 ---
        with torch.no_grad():
            # 将logits转换为概率
            pred_boundary_prob = torch.sigmoid(pred_boundary_logits)

            # 计算预测概率与真值之间的误差
            boundary_error = torch.abs(pred_boundary_prob - boundary_gt)

            # 核心：只在边界区域内应用难例挖掘权重
            # 基础权重为1，在边界区域内，根据误差大小增加权重
            loss_weights = 1.0 + self.beta * boundary_error * boundary_zone

        # --- 步骤 3: 应用加权损失 ---
        # 使用 F.binary_cross_entropy_with_logits 函数，它可以直接接收权重参数
        # 'weight' 参数会与输入的每个像素点相乘
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_boundary_logits,
            boundary_gt,
            weight=loss_weights,
            reduction='mean'  # 对加权后的损失求平均
        )

        return bce_loss