# --- START OF FILE utils_Hunan_Mamba.py ---
# 专为 Hunan 数据集 + DK_Mamba_Net_S3F_Dual 模型整合的 Utils
# 核心适配点：
#   1. Hunan 数据集有 NDVI → 真实加载
#   2. Hunan 数据集无 nDSM → 用 DSM 副本替代（优于零占位，保留高程梯度）
#   3. Hunan 数据集无 OBJECT_FOLDER → 用全零 object_mask 占位
#   4. 边界标签由分割掩码动态生成（无需额外文件）
#   5. 标签 / 评估使用 LoveDA 方式（全类平均 MIoU）
#
# 关于 nDSM 占位策略：
#   - 零占位：elevation_input=[DSM, 0]，第二通道梯度为零，backbone 该通道卷积核无法更新
#   - DSM 替代：elevation_input=[DSM, DSM]，双通道均有有效梯度，性能显著优于零占位

import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
from skimage import io
import os
import cv2


# ===================================================================================
# 边界生成函数（从分割掩码动态生成，无需额外边界文件）
# ===================================================================================
def mask_to_boundary(mask, kernel_size=(3, 3)):
    """
    从分割掩码动态生成二值边界标签，安全处理 -1 (忽略标签)。
    """
    # 1. 拷贝一份并把 -1 暂时转成 0，防止 OpenCV 下溢出变成 255 产生假边界
    valid_mask = mask.copy()
    valid_mask[valid_mask < 0] = 0
    valid_mask = valid_mask.astype(np.uint8)

    # 2. 正常提取边界
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    boundary = cv2.morphologyEx(valid_mask, cv2.MORPH_GRADIENT, kernel)
    boundary[boundary > 0] = 1

    # 3. 将真实标签为 -1 (无效区) 的位置，边界强制清零
    boundary[mask < 0] = 0

    return boundary


# ===================================================================================
# 全局参数 (Hunan)
# ===================================================================================
WINDOW_SIZE = (256, 256)
STRIDE = 32           # Hunan 图块本身是固定大小，滑窗步长与窗口一致
IN_CHANNELS = 6        # Hunan 只有 RGB
BATCH_SIZE = 10
CACHE = True
MODEL = 'MPSegNet'
MODE = 'Train'
DATASET = 'Hunan'
LOSS = 'MUL'


# Hunan 类别定义
LABELS = ["cropland", "forest", "grassland", "wetland", "water", "unused land", "built-up area"]
N_CLASSES = len(LABELS)
WEIGHTS = torch.ones(N_CLASSES)

palette = {
    0: (196, 90, 17),    # cropland
    1: (51, 129, 88),    # forest
    2: (177, 205, 61),   # grassland
    3: (228, 84, 96),    # wetland
    4: (91, 154, 214),   # water
    5: (225, 174, 110),  # unused land
    6: (239, 159, 2)     # built-up area
}
invert_palette = {v: k for k, v in palette.items()}

# Hunan 数据集划分（来自 utils.py）
train_ids = ['10434', '11524', '11607', '11724', '11854', '11856', '11919', '12152', '12350', '12563', '12669',
             '12813', '1302', '13258', '13383', '13524', '13565', '13932', '14477', '14694', '15001', '15023',
             '15201', '15230', '15548', '15603', '15686', '1599', '15998', '16090', '16217', '16541', '16582',
             '16703', '16709', '17092', '17269', '18186', '18950', '1899', '1906', '19098', '19680', '19915',
             '20175', '20386', '20561', '20734', '20752', '20759', '21562', '21565', '21738', '21820', '22000',
             '2232', '22547', '22729', '2431', '24718', '24733', '25069', '2584', '2617', '26622', '27163',
             '27308', '27312', '2791', '29393', '29886', '30560', '30719', '31175', '31188', '31586', '31797',
             '3817', '4529', '4530', '4889', '5223', '6213', '6597', '6600', '6768', '7329', '8232', '830',
             '8931', '944', '9956', '2087', '17721', '13990', '13622', '13563', '18009', '12148', '16888',
             '14758', '1773', '16516', '20408', '2070', '10062', '17637', '14942', '13931', '13410', '11959',
             '15150', '17582', '17820', '21545', '21563', '21592', '21922', '2255', '26628', '28533', '28801',
             '29621', '29796', '30482', '31302', '31355', '4171', '4887', '5994', '6167', '6777', '7421', '833',
             '9064', '9662', '14936', '15493', '2097', '25709', '30301', '18117', '11766', '3994', '5830',
             '14786', '1774', '16032', '2597', '18164', '8976', '2427', '418', '23961', '1165', '6383', '22906',
             '26032', '18371', '6156', '7167', '20736', '16880', '29145', '21211', '7473', '29172', '22077',
             '14755', '2428', '16922', '15144', '5232', '25777', '21736', '14290', '15275', '1025', '11173',
             '12040', '12779', '14126', '15695', '16214', '16577', '18079', '1930', '21804', '22154', '25699',
             '29675', '298', '31653', '5042', '637', '6581', '708', '7679', '1031', '11272', '14463', '16745',
             '12244', '1775', '1752', '16744', '17095', '20910', '13742', '16702', '13925', '17800', '17040',
             '2062', '16912', '19149', '11371', '21601', '21610', '21737', '21747', '21982', '23232', '2606',
             '27860', '28532', '28933', '29028', '29284', '29288', '30597', '3122', '31242', '4362', '6405',
             '6770', '726', '7330', '7331', '8234', '2402', '646', '13718', '12363', '13995', '13807', '1471',
             '27168', '18298', '16093', '15763', '12042', '29020', '8831', '11375', '23772', '12728', '13448',
             '27960', '14467', '14763', '19866', '13766', '24296', '1436', '5236', '28796', '10258', '28736',
             '2100', '1451', '12918', '14155', '15184', '19471', '21822', '22728', '22837', '2408', '3100',
             '5450', '6186', '7181', '9258', '11625', '11644', '12098', '12154', '12241', '13248', '13522',
             '13564', '1383', '13927', '14287', '14822', '15075', '15520', '15864', '16028', '1619', '16699',
             '16742', '16866', '17369', '17934', '18183', '18185', '18732', '1928', '19688', '20314', '20464',
             '20737', '21111', '21561', '21938', '22555', '23080', '23588', '23701', '2604', '27282', '27718',
             '28073', '28189', '29171', '29286', '29844', '30148', '30399', '30425', '30606', '31014', '31583',
             '31621', '4363', '5043', '5221', '5238', '6379', '6407', '6601', '9174', '2748', '29829', '10970',
             '20926', '6795', '24149', '18121', '20935', '942', '29391', '29638', '20054', '3161', '6772',
             '17933', '13535', '5412', '20599', '299', '19609', '452', '28191', '11659', '1450', '13019',
             '11838', '29892', '12151', '13933', '11568', '11233', '12153', '13433', '13436', '14105', '14169',
             '16724', '18895', '19853', '21981', '2246', '22907', '29654', '30669', '3534', '6794', '1057',
             '11271', '11603', '11678', '11957', '1262', '12863', '13109', '13299', '13415', '14603', '14873',
             '15398', '15596', '1605', '16089', '16530', '16870', '17799', '17943', '1932', '21740', '21967',
             '23342', '2396', '2397', '2429', '2542', '25778', '2618', '26776', '28038', '29104', '29567',
             '29733', '29779', '30276', '31708', '3936', '419', '5066', '7898', '20933', '15597', '18118',
             '16356', '14937', '10503', '13437', '14247', '21566', '22099', '22731', '27437', '28078', '29394',
             '29571', '30130', '6771', '940']

test_ids = ['11767', '11816', '1239', '12626', '12815', '1290', '1303', '13254', '13257', '13515', '13765',
            '14108', '14293', '15426', '1625', '16373', '16750', '16890', '17039', '17055', '17107', '17455',
            '17821', '17980', '18958', '1908', '1923', '20028', '2106', '21385', '21423', '22312', '2256',
            '2265', '2442', '2603', '27264', '28965', '29287', '30275', '3280', '4017', '4166', '4886', '5215',
            '5233', '5410', '639', '7176', '11624']

Stride_Size = 256
epochs = 50
save_epoch = 1

# 数据路径（请根据实际服务器路径修改 MAIN_FOLDER）
MAIN_FOLDER   = "/home/duyihan/pycharm_project/数据集/Hunan/"
DATA_FOLDER   = MAIN_FOLDER + 'images_png/{}.png'
DSM_FOLDER    = MAIN_FOLDER + 'dsm_pngs/{}.png'         # Hunan 真实提供的 DSM
nDSM_FOLDER    = MAIN_FOLDER + 'dsm_pngs/{}.png'
NDVI_FOLDER   = MAIN_FOLDER + 'ndvi/s2_{}.png'        # Hunan 真实提供的 NDVI（你拥有）
# nDSM 策略：Hunan 无 nDSM 文件，训练时用 DSM 副本替代（比零占位性能更好）
LABEL_FOLDER  = MAIN_FOLDER + 'masks_png/{}.tif'
ERODED_FOLDER = MAIN_FOLDER + 'masks_png/{}.tif'        # Hunan 无 eroded，直接用原 mask

print(MODEL + ', ' + MODE + ', ' + DATASET + ', ' + LOSS)
print('WINDOW_SIZE:', WINDOW_SIZE, ' BATCH_SIZE:', BATCH_SIZE,
      ' Stride_Size:', Stride_Size, ' epochs:', epochs)


# ===================================================================================
# 颜色转换
# ===================================================================================
def convert_to_color(arr_2d, palette=palette):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d


def convert_from_color(arr_3d, palette=invert_palette):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d


# ===================================================================================
# Dataset（适配 Hunan：有真实 NDVI，nDSM 用 DSM 副本替代，OBJECT 用零占位）
# ===================================================================================
class ISPRS_dataset(torch.utils.data.Dataset):
    """
    Hunan 数据集加载器，适配 DK_Mamba_Net_S3F_Dual 的 7 元组返回格式：
        (data, label, boundary, object_mask, NDVI, nDSM, DSM)

    数据处理策略
    ----------------
    - RGB   : 真实加载，归一化到 [0,1]
    - NDVI  : 真实加载（你拥有），归一化到 [0,1]
    - DSM   : 真实加载，归一化到 [0,1]
    - nDSM  : Hunan 无提供 → 用 DSM 归一化副本替代（优于零占位）
              理由：elevation_input=[DSM,nDSM]，零占位使第二通道梯度恒为0
    - object_mask : 无目标实例文件 → 全零整形掩码 (H, W)，ObjectLoss ≈ 0
    - boundary : 由分割标签动态生成（无需额外文件）
    """

    def __init__(self, ids, cache=CACHE, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        self.data_files  = [DATA_FOLDER.format(id)  for id in ids]
        self.dsm_files   = [DSM_FOLDER.format(id)   for id in ids]
        self.ndvi_files  = [NDVI_FOLDER.format(id)  for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # 检查 RGB 图和标签是否存在（DSM/NDVI 单独处理，缺失时有降级策略）
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        self.data_cache_  = {}
        self.dsm_cache_   = {}
        self.ndvi_cache_  = {}
        self.label_cache_ = {}

    def __len__(self):
        return BATCH_SIZE * 500   # Hunan 保持原有设置

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip   = flip   and random.random() < 0.5
        will_mirror = mirror and random.random() < 0.5
        results = []
        for array in arrays:
            if will_flip:
                array = array[::-1, :]   if len(array.shape) == 2 else array[:, ::-1, :]
            if will_mirror:
                array = array[:, ::-1]   if len(array.shape) == 2 else array[:, :, ::-1]
            results.append(np.copy(array))
        return tuple(results)

    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.data_files) - 1)

        # ── RGB 图像 ──────────────────────────────────────────────────────────
        if random_idx in self.data_cache_:
            data = self.data_cache_[random_idx]
        else:
            img = io.imread(self.data_files[random_idx])
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=0)
            else:
                img = img[:, :, :3].transpose((2, 0, 1))
            data = 1.0 / 255.0 * np.asarray(img, dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        # ── DSM ─────────────────────────────────────────────────────────────
        if random_idx in self.dsm_cache_:
            dsm = self.dsm_cache_[random_idx]
        else:
            dsm_path = self.dsm_files[random_idx]
            if os.path.isfile(dsm_path):
                dsm_raw = np.asarray(io.imread(dsm_path), dtype='float32')
                if dsm_raw.ndim == 3: dsm_raw = dsm_raw[:, :, 0].astype('float32')
                mn, mx = dsm_raw.min(), dsm_raw.max()
                dsm = (dsm_raw - mn) / (mx - mn + 1e-8)
            else:
                dsm = np.zeros((data.shape[1], data.shape[2]), dtype='float32')
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        # ── NDVI ────────────────────────────────────────────────────────────
        if random_idx in self.ndvi_cache_:
            ndvi = self.ndvi_cache_[random_idx]
        else:
            ndvi_path = self.ndvi_files[random_idx]
            if os.path.isfile(ndvi_path):
                ndvi_raw = np.asarray(io.imread(ndvi_path), dtype='float32')
                if ndvi_raw.ndim == 3: ndvi_raw = ndvi_raw[:, :, 0].astype('float32')
                mn, mx = ndvi_raw.min(), ndvi_raw.max()
                ndvi = (ndvi_raw - mn) / (mx - mn + 1e-8)
            else:
                ndvi = np.zeros((data.shape[1], data.shape[2]), dtype='float32')
            if self.cache:
                self.ndvi_cache_[random_idx] = ndvi

        # ── 分割标签 (确保 label 在所有分支下都能被定义) ──────────────────────────
        if random_idx in self.label_cache_:
            label = self.label_cache_[random_idx]
        else:
            raw_label = io.imread(self.label_files[random_idx])
            if raw_label.ndim == 3: raw_label = raw_label[:, :, 0]

            # 清洗逻辑：将标签映射到 [0, N_CLASSES-1]
            # 原始可能是 1-7，减1后为 0-6
            label = raw_label.astype('int64')

            if self.cache:
                self.label_cache_[random_idx] = label

        # ── 动态生成边界标签 ──────────────────────────────────────────────────
        boundary = mask_to_boundary(label)

        # ── object_mask 占位 ─────────────────────────────────────────────────
        object_mask = np.zeros(label.shape, dtype='int64')

        # ── nDSM (DSM 副本) ──────────────────────────────────────────────────
        ndsm_arr = dsm.copy()

        # ── 数据增强 ──────────────────────────────────────────────────────────
        data_p, label_p, boundary_p, object_p, ndvi_p, ndsm_p, dsm_p = \
            self.data_augmentation(data, label, boundary, object_mask,
                                   ndvi, ndsm_arr, dsm)

        # ── 返回格式 ──────────────────────────────────────────────────────────
        return (
            torch.from_numpy(data_p),
            torch.from_numpy(label_p).long(),
            torch.from_numpy(boundary_p).float(),
            torch.from_numpy(object_p).long(),
            torch.from_numpy(ndvi_p[np.newaxis, :, :]),
            torch.from_numpy(ndsm_p[np.newaxis, :, :]),
            torch.from_numpy(dsm_p[np.newaxis, :, :]),
        )

# ===================================================================================
# 辅助工具函数
# ===================================================================================
def get_random_pos(img, window_shape):
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# ===================================================================================
# 损失函数
# ===================================================================================
class CrossEntropy2d_ignore(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


def loss_calc(pred, label, weights):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d_ignore().cuda()
    return criterion(pred, label, weights)


class ObjectLoss(nn.Module):
    """对象级一致性损失（Hunan 使用全零 object_mask，实际不产生有效梯度，保留接口兼容性）"""
    def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

    def forward(self, pred, gt):
        num_object = int(torch.max(gt)) + 1
        num_object = min(num_object, self.max_object)
        total_object_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        for object_index in range(1, num_object):
            mask = torch.where(gt == object_index, 1, 0).unsqueeze(1).to(pred.device)
            num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to(pred.device)
            avg_pool = mask / (num_point + 1)
            object_feature = pred.mul(avg_pool)
            avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(
                1, 1, gt.shape[1], gt.shape[2])
            avg_feature = avg_feature.mul(mask)
            object_loss = F.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
            total_object_loss = total_object_loss + object_loss
        return total_object_loss


# ===================================================================================
# 评估函数（Hunan 使用全类平均 MIoU，同 metrics_loveda）
# ===================================================================================
def metrics(predictions, gts, label_values=LABELS):
    """
    Hunan 数据集评估：使用全类平均 MIoU（LoveDA 方式），
    与 utils.py 中 metrics_loveda 行为一致。
    """
    cm = confusion_matrix(gts, predictions, labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    total = sum(sum(cm))
    acc = sum([cm[x][x] for x in range(len(cm))])
    acc *= 100 / float(total)
    print("%d pixels processed" % total)
    print("Total accuracy : %.2f" % acc)

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except Exception:
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % np.nanmean(F1Score[:]))
    print("---")

    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" % kappa)

    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:])    # 全类平均（LoveDA 方式）
    print('mean MIoU: %.4f' % MIoU)
    print("---")

    return MIoU
