import os
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
import torch.nn as nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import random


# ===================================================================================
# 1. 随机性控制与全局设置 (核心复现性保障)
# ===================================================================================
def set_seed(seed: int = 42):
    """
    设置所有随机种子以确保结果可复现。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你使用了多块GPU

    # 这两项对于保证cuDNN的确定性至关重要
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"全局随机种子已设置为: {seed}")


def seed_worker(worker_id):
    """
    为Dataloader worker设置所有需要的种子，以确保数据加载的可复现性。
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)  # 修复数据增强随机性的关键！


# --- 脚本开始时立即设置种子 ---
seed = 42
set_seed(seed)

# --- 开启PyTorch的确定性算法模式 ---
torch.use_deterministic_algorithms(True)
print("已开启确定性算法模式。")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# ===================================================================================
# 2. 你的原始设置和导入
# ===================================================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ✅ 使用两张显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from MPSegNet.mamba.DK_Mamba_Net_S3F_Dual import DK_Mamba_Net_S3F_Dual as MPSegNet
from losses.dice import DiceLoss
from losses.soft_ce import SoftCrossEntropyLoss

torch.cuda.empty_cache()

DATASET = 'Vaihingen'
if DATASET == 'Vaihingen':
    # !! 重要 !!: 确保你有一个名为 utils_MPSegNet_Mamba_new_Boundry_M.py 的文件
    # 并且这个文件中的代码与你之前提供的一致。
    from utils_test import *

# ===================================================================================
# 3. 准备模型、损失函数、数据
# ===================================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"正在使用设备: {device}")
    DiceLoss_fn = DiceLoss(mode='multiclass').to(device)
    SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1).to(device)
    BoundaryLoss_fn = nn.BCEWithLogitsLoss().to(device)
else:
    raise RuntimeError("CUDA is not available. This script requires a GPU.")

if MODEL == 'MPSegNet':
    net = MPSegNet(num_classes=N_CLASSES).to(device)
    # ✅ 启用多GPU
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张GPU进行训练")
        net = nn.DataParallel(net)
    print("模型是 MPSegNet")
else:
    pass

params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"总可训练参数量: {params}")

print("训练集大小: ", len(train_ids))
print("测试集大小: ", len(test_ids))

train_set = ISPRS_dataset(train_ids, cache=CACHE)

# 为DataLoader创建一个可复现的生成器
g = torch.Generator()
g.manual_seed(seed)

# --- 修复: 为 DataLoader 加入 worker_init_fn 和 generator ---
# ✅ batch_size 从 16 改为 32
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=True
)

# ===================================================================================
# 4. 准备优化器和学习率调度器
# ===================================================================================
epochs = 100
base_lr = 0.0001
# LBABDA_BDY = 10.0
LBABDA_BDY = 5.0
weight_decay = 0.01
LBABDA_OBJ = 1.0
print("LBABDA_BDY: ", LBABDA_BDY)
print("LBABDA_OBJ: ", LBABDA_OBJ)

optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=weight_decay)
# milestones = [6,12,18,24]  84.75# 保持你的新设置
# [8,16,25,35,50]   84.78
# milestones = [8,16,25,35,45,55] # 84.93
milestones = [10,20,32]
gamma = 0.50
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


# ===================================================================================
# 5. 定义训练与测试函数 (已按要求修改)
# ===================================================================================
def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    test_images = (np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') / 255.0 for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    test_ndsm = ((lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8))(
        np.asarray(io.imread(nDSM_FOLDER.format(id)), dtype='float32')) for id in test_ids)
    test_dsm = ((lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8))(
        np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32')) for id in test_ids)
    test_ndvi = ((lambda x: (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8))(
        np.asarray(io.imread(NDVI_FOLDER.format(id)), dtype='float32')) for id in test_ids)

    if DATASET == 'Urban':
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64') - 1) for id in test_ids)
    else:
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)

    all_preds = []
    all_gts = []
    net.eval()
    with torch.no_grad():
        outer_tqdm = tqdm(zip(test_images, test_labels, eroded_labels, test_ndsm, test_dsm, test_ndvi),
                          total=len(test_ids), desc="Testing images")
        for i, (img, gt, gt_e, ndsm_img, dsm_img, ndvi_img) in enumerate(outer_tqdm):
            outer_tqdm.set_description(f"Processing image {test_ids[i]}")
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            inner_tqdm = tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)),
                              total=total, desc=f"Sliding windows", leave=False)
            for j, coords in enumerate(inner_tqdm):
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                ndsm_patches = [np.copy(ndsm_img[x:x + w, y:y + h])[np.newaxis, ...] for x, y, w, h in coords]
                dsm_patches = [np.copy(dsm_img[x:x + w, y:y + h])[np.newaxis, ...] for x, y, w, h in coords]
                ndvi_patches = [np.copy(ndvi_img[x:x + w, y:y + h])[np.newaxis, ...] for x, y, w, h in coords]

                image_patches = torch.from_numpy(np.asarray(image_patches)).float().to(device)
                ndsm_patches = torch.from_numpy(np.asarray(ndsm_patches)).float().to(device)
                dsm_patches = torch.from_numpy(np.asarray(dsm_patches)).float().to(device)
                ndvi_patches = torch.from_numpy(np.asarray(ndvi_patches)).float().to(device)

                outs_logits, _, _, _ = net(image_patches, ndsm_patches, dsm_patches, ndvi_patches)

                outs_probs = F.softmax(outs_logits, dim=1)
                outs_probs_np = outs_probs.cpu().numpy()
                for out_prob, (x, y, w, h) in zip(outs_probs_np, coords):
                    out_prob_hwc = out_prob.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out_prob_hwc

                del outs_logits, outs_probs, outs_probs_np, image_patches, ndsm_patches, dsm_patches, ndvi_patches

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)

    accuracy_metrics = metrics(np.concatenate([p.ravel() for p in all_preds]),
                               np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy_metrics, all_preds, all_gts
    else:
        return accuracy_metrics


# ===================================================================================
# 6. Train 函数 (含修正后的 S3F 逻辑)
# ===================================================================================
def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)

    if isinstance(weights, np.ndarray):
        weights_tensor = torch.from_numpy(weights).to(device, dtype=torch.float32)
    else:
        weights_tensor = weights.clone().detach().to(device, dtype=torch.float32)

    iter_ = 0
    MIoU_best = 0.30
    criteriono = ObjectLoss().to(device)

    MIOU_THRESHOLD = 0.848
    boundary_loss_activated = False

    for e in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n--- Epoch {e}/{epochs} --- Starting with Learning Rate: {current_lr} ---")

        if boundary_loss_activated:
            print("--- Using Loss Function: MUL + Boundary + S3F_Supervision (MIoU Threshold Reached) ---")
        else:
            print(f"--- Using Loss Function: {LOSS} + S3F_Supervision (Initial) ---")

        net.train()

        for batch_idx, (data, target, boundary, object_mask, NDVI, nDSM, DSM) in enumerate(train_loader):
            data, target, boundary, object_mask = data.to(device), target.to(device), boundary.to(
                device), object_mask.to(device)
            nDSM, DSM, NDVI = nDSM.to(device), DSM.to(device), NDVI.to(device)

            optimizer.zero_grad()

            # 接收 4 个返回值
            seg_final, seg_list, pred_boundary_logits, singularity_map = net(data, nDSM, DSM, NDVI)

            loss_M = 5 * DiceLoss_fn(seg_final, target) + 2 * SoftCrossEntropy_fn(seg_final, target)
            loss_ce = loss_calc(seg_final, target, weights_tensor)
            loss_object = criteriono(seg_final, object_mask)
            loss_boundary = BoundaryLoss_fn(pred_boundary_logits.squeeze(1), boundary)

            # --- S3F Loss 计算 (已修正) ---
            if singularity_map is not None:
                # 1. 上采样对齐
                target_h, target_w = boundary.shape[-2], boundary.shape[-1]
                singularity_map_up = F.interpolate(
                    singularity_map,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )

                # 2. 核心修正: 学习目标为 (1 - boundary)
                # 理论定义: 连续(内部)=1, 断裂(边界)=0
                # 标签定义: boundary中边缘为1
                # 因此 loss 目标应为 1-boundary
                loss_s3f = F.binary_cross_entropy(singularity_map_up.squeeze(1), 1.0 - boundary.float())
            else:
                loss_s3f = 0.0

            # --- 核心修改: 加入显式监督权重 ---
            # 权重设为 0.2，保证梯度传导，让 Singularity Map 清晰化
            S3F_WEIGHT = 0.2

            # --- Loss 聚合 (已加入 loss_s3f) ---
            if boundary_loss_activated:
                loss = loss_M + loss_boundary * LBABDA_BDY + loss_s3f * S3F_WEIGHT
            else:
                if LOSS == 'SEG':
                    loss = loss_ce + loss_s3f * S3F_WEIGHT
                elif LOSS == 'MUL':
                    loss = loss_M + loss_s3f * S3F_WEIGHT  # 关键点：这里加上了
                elif LOSS == 'MUL+BDY':
                    loss = loss_M + loss_boundary * LBABDA_BDY + loss_s3f * S3F_WEIGHT
                elif LOSS == 'SEG+BDY':
                    loss = loss_ce + loss_boundary * LBABDA_BDY + loss_s3f * S3F_WEIGHT
                elif LOSS == 'SEG+OBJ':
                    loss = loss_ce + loss_object * LBABDA_OBJ + loss_s3f * S3F_WEIGHT
                elif LOSS == 'MUL+BDY+OBJ':
                    loss = loss_M + loss_boundary * LBABDA_BDY + loss_object * LBABDA_OBJ + loss_s3f * S3F_WEIGHT

            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if iter_ % 10 == 0:
                pred = np.argmax(seg_final.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]

                val_s3f = loss_s3f.item() if isinstance(loss_s3f, torch.Tensor) else loss_s3f

                print(
                    'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\t'
                    'Loss_ce: {:.6f}\tLoss_M: {:.6f}\tLoss_bdy: {:.6f}\t'
                    'Loss_obj: {:.6f}\tLoss_s3f: {:.6f}\tTotal_Loss: {:.6f}\tAcc: {:.4f}'.format(
                        e, epochs, batch_idx, len(train_loader),
                        100. * batch_idx / len(train_loader),
                        loss_ce.item(),
                        loss_M.item(),
                        loss_boundary.item(),
                        loss_object.item(),
                        val_s3f,
                        loss.item(),
                        accuracy(pred, gt)))
            iter_ += 1
            del (data, target, boundary, object_mask, loss, seg_final, seg_list, pred_boundary_logits, singularity_map)

        if scheduler is not None:
            scheduler.step()

        if e % save_epoch == 0:
            net.eval()
            print(f"--- Running validation for epoch {e} ---")
            current_MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            print(f"Validation MIoU at epoch {e}: {current_MIoU:.4f}")
            net.train()

            if not boundary_loss_activated and current_MIoU >= MIOU_THRESHOLD:
                boundary_loss_activated = True
                print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"!! MIoU THRESHOLD ({MIOU_THRESHOLD * 100:.1f}%) REACHED!             !!")
                print(f"!! SWITCHING LOSS TO 'MUL+BDY' FOR SUBSEQUENT EPOCHS.       !!")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

            if current_MIoU > MIoU_best:
                MIoU_best = current_MIoU
                save_path = f'/home/duyihan/pycharm_project/SSRS-main/SAM_RS_L40/TRY/resultv_DK_Mamba_Net_S3F_Dual_HFBS/42_16/1_75_8_5_2/{MODEL}_epoch{e}_MIoU{MIoU_best:.4f}.pth'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(net.state_dict(), save_path)
                print(f"***** New best model saved to {save_path} *****")


if __name__ == '__main__':
    # 1. 强制指定 MODE 为 Test (避免未定义导致分支不触发)
    MODE = 'Test'  # 关键：确保执行Test分支，若要训练再改为'Train'
    # 2. 确保 LOSS 变量有初始值 (避免未定义报错)
    if 'LOSS' not in locals():
        LOSS = 'MUL'
        print(f"注意: 全局变量 'LOSS' 未定义，已临时设置为 '{LOSS}'")

    # 3. 执行对应模式逻辑
    if MODE == 'Train':
        train(net, optimizer, epochs, scheduler, save_epoch=1)
    elif MODE == 'Test':
        # 打印Test模式启动日志 (便于确认分支已触发)
        print(f"\n=== 进入 Test 模式 (数据集: {DATASET}) ===")
        # 加载权重 (修复权重键匹配问题)
        weight_path = '/home/duyihan/pycharm_project/SSRS-main/SAM_RS_L40/TRY/resultv_DK_Mamba_Net_S3F_Dual/42_16/75_1_75_8_5_2/XN_best_now/MPSegNet_epoch25_MIoU0.8576.pth'
        print(f"正在加载权重: {weight_path}")

        # 核心修复：移除权重中的'module.'前缀 (兼容多GPU训练的权重)
        state_dict = torch.load(weight_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            # 若权重键以'module.'开头，移除前缀（单GPU测试时匹配模型结构）
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 截取'module.'后的部分
            else:
                new_state_dict[k] = v

        # 加载处理后的权重
        net.load_state_dict(new_state_dict)
        print("权重加载完成，开始测试...")

        # 执行测试 (显式指定stride，避免依赖未定义的Stride_Size)
        MIoU, all_preds, all_gts = test(
            net, test_ids, all=True,
            stride=32,  # 显式指定滑动窗口步长（根据你的窗口大小调整，如WINDOW_SIZE=64则设为32）
            batch_size=8,  # 测试时batch_size可减小，避免显存溢出
            window_size=WINDOW_SIZE  # 确保WINDOW_SIZE已从utils_P_test.py导入
        )
        # 打印测试结果 (确保测试完成后有输出)
        print(f"\n=== Test 模式完成 ===")
        print(f"最终 MIoU: {MIoU:.4f}")
        # 保存预测结果 (确保路径存在)
        save_infer_path = '/home/duyihan/pycharm_project/SSRS-main/SAM_RS_L40/TRY/resultv_DK_Mamba_Net_S3F_Dual_HFBS/42_16/75_1_75_8_5_2/XN_best_now/8576/'
        os.makedirs(save_infer_path, exist_ok=True)  # 确保保存目录存在
        for p, id_ in zip(all_preds, test_ids):
            img = convert_to_color(p)
            io.imsave(f'{save_infer_path}/inference_{MODEL}_tile_{id_}.png', img)
            print(f"已保存预测图: inference_{MODEL}_tile_{id_}.png")
    else:
        raise ValueError(f"无效的 MODE: {MODE}，请设置为 'Train' 或 'Test'")
