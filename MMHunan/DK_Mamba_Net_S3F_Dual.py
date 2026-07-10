# --- START OF FILE dk_mamba_net_s3f_dual.py ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ===================================================================================
# SECTION 1: 基础工具模块 (保持不变)
# ===================================================================================
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("Warning: mamba_ssm is not installed. Falling back to a dummy Mamba layer.")


    class Mamba(nn.Module):
        def __init__(self, d_model, d_state, d_conv, expand):
            super().__init__()
            self.d_model = d_model
            self.conv = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv // 2, groups=d_model)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)
            return x


class DeterministicPyramidPooling(nn.Module):
    def __init__(self, pool_scale):
        super().__init__()
        self.pool_scale = pool_scale

    def forward(self, x):
        h, w = x.shape[2:]
        if h < self.pool_scale or w < self.pool_scale:
            return F.adaptive_avg_pool2d(x, (self.pool_scale, self.pool_scale))
        stride_h = h // self.pool_scale
        stride_w = w // self.pool_scale
        kernel_h = h - (self.pool_scale - 1) * stride_h
        kernel_w = w - (self.pool_scale - 1) * stride_w
        return F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))


class ConcatFusion(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ConcatFusion, self).__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(channels_in * 2, channels_in, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels_in),
            activation
        )

    def forward(self, feat1, feat2):
        return self.conv_fuse(torch.cat([feat1, feat2], dim=1))


class ConvFFN(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super(ConvFFN, self).__init__()
        self.fc1 = nn.Conv2d(in_ch, hidden_ch, 1, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ===================================================================================
# SECTION 2: 核心创新模块 - S3F Mechanism (保留并增强)
# ===================================================================================

class SingularityDetector(nn.Module):
    """
    [Core Innovation] 奇异点探测器
    从深层特征中提取拓扑断裂概率图 (Probability Map of Topological Fracture).
    """

    def __init__(self, in_dim):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.detector(x)


class FracturedMambaBlock(nn.Module):
    """
    [Core Innovation] 断裂式 Mamba 块
    支持 Delta-Hacking 注入机制。
    """

    def __init__(self, in_chs, dim, out_chs, pool_scales, context_dim=None):
        super().__init__()
        self.has_guidance = context_dim is not None

        if self.has_guidance:
            self.singularity_detector = SingularityDetector(in_dim=context_dim)

        # 基础特征提取
        self.main_branch = nn.Sequential(
            nn.Conv2d(in_chs, dim, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True)
        )
        self.pool_branches = nn.ModuleList([
            nn.Sequential(
                DeterministicPyramidPooling(p),
                nn.Conv2d(in_chs, dim, 1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for p in pool_scales
        ])

        # Mamba 输入维度
        self.mamba_d_model = dim * (len(pool_scales) + 1)

        # 注入适配器 (Delta-Hacking Injection Adapter)
        if self.has_guidance:
            self.injector = nn.Sequential(
                nn.Conv2d(self.mamba_d_model + 1, self.mamba_d_model, 1, bias=False),
                nn.BatchNorm2d(self.mamba_d_model),
                nn.ReLU(inplace=True)
            )

        # Mamba Core
        self.mamba = Mamba(d_model=self.mamba_d_model, d_state=16, d_conv=4, expand=2)

        # 后处理
        self.proj = nn.Conv2d(self.mamba_d_model, in_chs, 1)
        self.ffn = ConvFFN(in_ch=in_chs, hidden_ch=in_chs * 4, out_ch=out_chs)

    def forward(self, x, context=None):
        B, C, H, W = x.shape

        # Multi-scale aggregation
        pooled_feats = [self.main_branch(x)]
        for branch in self.pool_branches:
            pooled_feats.append(F.interpolate(branch(x), size=(H, W), mode='bilinear', align_corners=False))
        fused = torch.cat(pooled_feats, dim=1)

        # Singularity Injection (S3F)
        singularity_map = None
        if self.has_guidance and context is not None:
            context_up = F.interpolate(context, size=(H, W), mode='bilinear', align_corners=False)
            singularity_map = self.singularity_detector(context_up)  # [B, 1, H, W]

            # Delta-Hacking: Inject singularity into feature space
            combined = torch.cat([fused, singularity_map], dim=1)
            fused_injected = self.injector(combined)
        else:
            fused_injected = fused

        # Mamba Sequence Modeling
        fused_seq = fused_injected.flatten(2).transpose(1, 2)
        mamba_out = self.mamba(fused_seq).transpose(1, 2).reshape(B, -1, H, W)

        # Residual & Output
        x_mamba = self.proj(mamba_out)
        x_intermediate = x + x_mamba
        out = self.ffn(x_intermediate)

        # Return:
        # 1. x_intermediate: Enhanced features for next stage context
        # 2. out: Classification logits for this stage
        # 3. singularity_map: For visualization or boundary loss
        return x_intermediate, out, singularity_map


# ===================================================================================
# SECTION 3: 颠覆性双解码器架构 (Dual-Decoder Disruptive Architecture)
# ===================================================================================

class HyperFusionBridge(nn.Module):
    """
    [Performance Booster] 超级融合桥接
    用于在两个解码器之间进行高维特征对齐。
    不仅仅是 Upsample，而是先对齐通道再融合，保证语义不丢失。
    """

    def __init__(self, deep_dim, shallow_dim, out_dim):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(deep_dim, deep_dim, 3, padding=1, groups=deep_dim),  # Depthwise
            nn.Conv2d(deep_dim, out_dim, 1),  # Pointwise
            nn.BatchNorm2d(out_dim),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(shallow_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, deep, shallow):
        d = self.up_conv(deep)
        s = self.shallow_conv(shallow)
        return self.fusion(torch.cat([d, s], dim=1))


class DK_Mamba_Net_S3F_Dual(nn.Module):
    def __init__(self, num_classes=6, backbone_name='convnext_base', pretrained=True):
        super(DK_Mamba_Net_S3F_Dual, self).__init__()

        # --- 1. Dual-Stream Encoder (保持不变) ---
        in_chans_spectral = 4
        in_chans_elevation = 2
        self.encoder_spectral = timm.create_model(backbone_name, pretrained=pretrained, features_only=True,
                                                  out_indices=(0, 1, 2, 3), in_chans=in_chans_spectral)
        self.encoder_elevation = timm.create_model(backbone_name, pretrained=pretrained, features_only=True,
                                                   out_indices=(0, 1, 2, 3), in_chans=in_chans_elevation)

        # d_s4=128, d_s8=256, d_s16=512, d_s32=1024 (for ConvNeXt-Base)
        self.encoder_dims = self.encoder_spectral.feature_info.channels()
        d_s4, d_s8, d_s16, d_s32 = self.encoder_dims

        # SFF (Spatial-Frequency Fusion) Modules
        self.sff_modules = nn.ModuleList([
            ConcatFusion(channels_in=d_s4),
            ConcatFusion(channels_in=d_s8),
            ConcatFusion(channels_in=d_s16),
            ConcatFusion(channels_in=d_s32)
        ])

        # --- 2. Dual-Decoder Structure (重构部分) ---

        # [Strategy] 宽通道策略：减少Stage数量，增加单个Stage的厚度
        # 我们定义两个主要阶段：Semantic Stage (S) 和 Detail Stage (D)

        # --- DECODER 1: Semantic Anchor (处理 S32 + S16) ---
        # 融合 S32 和 S16，形成强大的语义基座
        self.semantic_bridge = HyperFusionBridge(deep_dim=d_s32, shallow_dim=d_s16, out_dim=512)

        # 这里的 dim 从原来的 128 提升到 256，大幅增加 Mamba 的容量
        self.decoder_semantic = FracturedMambaBlock(
            in_chs=512, dim=256, out_chs=num_classes,
            pool_scales=[1, 2, 4, 8], context_dim=None  # 语义层无需外部引导，它是源头
        )

        # --- DECODER 2: Detail Refiner (处理 S8 + Semantic) ---
        # 融合 Semantic Output 和 S8，专注于边界恢复
        self.detail_bridge = HyperFusionBridge(deep_dim=512, shallow_dim=d_s8, out_dim=256)

        # 这里的 context_dim=512 (来自 decoder_semantic 的中间特征)
        # S3F 机制在这里生效：利用深层语义生成的连续性地图，去切断浅层的纹理噪音
        self.decoder_detail = FracturedMambaBlock(
            in_chs=256, dim=128, out_chs=num_classes,
            pool_scales=[1, 2, 4], context_dim=512
        )

        # --- 3. Heads & Final Fusion ---

        # 边界辅助头 (基于 Detail Decoder 的上下文)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        # 双流注意力融合 (Dual-Stream Attention Fusion)
        # 只需要融合两个结果，收敛更快，干扰更小
        self.attention_head = nn.Sequential(
            nn.Conv2d(num_classes * 2, num_classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.PReLU(),
            nn.Conv2d(num_classes, 2, 3, padding=1, bias=False),  # Output 2 weights
        )

    def forward(self, x, ndsm, dsm, ndvi):
        spectral_input = torch.cat([x, ndvi], dim=1)
        elevation_input = torch.cat([dsm, ndsm], dim=1)
        h, w = spectral_input.shape[2:]

        # --- Encoding ---
        feats_spectral = self.encoder_spectral(spectral_input)
        feats_elevation = self.encoder_elevation(elevation_input)

        # 融合双模态特征
        fused_feats = [sff(s, e) for sff, s, e in zip(self.sff_modules, feats_spectral, feats_elevation)]
        _, feat_s8, feat_s16, feat_s32 = fused_feats

        # --- Decoding Stage 1: Semantic Anchor ---
        # 将 S32 和 S16 提前融合，构建更强的语义上下文
        feat_semantic_in = self.semantic_bridge(feat_s32, feat_s16)

        # Mamba 处理 (无 Singularity 注入，因为它是源头)
        # feat_sem_ctx: [B, 512, H/16, W/16]
        # out_sem: [B, num_classes, H/16, W/16]
        feat_sem_ctx, out_sem, _ = self.decoder_semantic(feat_semantic_in, context=None)

        # --- Decoding Stage 2: Detail Refiner ---
        # 将语义上下文与 S8 细节特征融合
        feat_detail_in = self.detail_bridge(feat_sem_ctx, feat_s8)

        # Mamba 处理 (注入 Semantic Context 作为 Singularity 指导)
        # 这里 S3F 机制发挥关键作用：
        # feat_sem_ctx 包含了全局地物分布，SingularityDetector 从中提取边界
        # 在边界处强行切断 feat_detail_in 的状态传递，防止S8中的纹理噪音蔓延
        feat_det_ctx, out_det, singularity_map = self.decoder_detail(feat_detail_in, context=feat_sem_ctx)

        # --- Output & Fusion ---

        # 1. 边界预测 (辅助损失)
        pred_boundary = self.boundary_head(feat_det_ctx)
        pred_boundary = F.interpolate(pred_boundary, size=(h, w), mode='bilinear', align_corners=False)

        # 2. 结果上采样
        seg_sem = F.interpolate(out_sem, size=(h, w), mode='bilinear', align_corners=False)
        seg_det = F.interpolate(out_det, size=(h, w), mode='bilinear', align_corners=False)

        # 3. 动态加权融合
        seg_concat = torch.cat([seg_sem, seg_det], dim=1)
        atten = F.softmax(self.attention_head(seg_concat), dim=1)  # [B, 2, H, W]

        seg_final = atten[:, 0:1] * seg_sem + atten[:, 1:2] * seg_det

        return seg_final, [seg_sem, seg_det], pred_boundary, singularity_map


if __name__ == '__main__':
    print("Building DK_Mamba_Net_S3F_Dual (Enhanced Performance Version)...")
    # 使用 ConvNeXt-Base 保证骨干能力
    model = DK_Mamba_Net_S3F_Dual(num_classes=6, backbone_name='convnext_base').cuda()
    model.eval()

    # 模拟输入 (ISPRS Vaihingen/Potsdam 典型尺寸)
    x = torch.randn(2, 3, 512, 512).cuda()
    ndsm = torch.randn(2, 1, 512, 512).cuda()
    dsm = torch.randn(2, 1, 512, 512).cuda()
    ndvi = torch.randn(2, 1, 512, 512).cuda()

    with torch.no_grad():
        seg_final, seg_list, boundary, s_map = model(x, ndsm, dsm, ndvi)

    print(f"Output Shape: {seg_final.shape}")
    if s_map is not None:
        print(f"Singularity Map Shape: {s_map.shape}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    print("Architecture Optimization: 3-Stage Cascaded -> 2-Stage Wide-Channel Interactive.")

# --- END OF FILE dk_mamba_net_s3f_dual.py ---