# --- START OF FILE utils_MPSegNet_Mamba_new_Boundry_M.py ---

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
import cv2  # <--- NEW: 导入OpenCV


# ===================================================================================
# START: 新增边界生成函数
# ===================================================================================
def mask_to_boundary(mask, kernel_size=(3, 3)):
    """
    从分割掩码动态生成二值边界标签。
    Args:
        mask (np.array): 输入的分割掩码，维度为 (H, W)，值为类别索引 (0, 1, 2...).
        kernel_size (tuple): 用于形态学操作的核大小。
    Returns:
        np.array: 生成的边界标签，维度 (H, W)，值为 0 或 1。
    """
    mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    boundary[boundary > 0] = 1
    return boundary


# ===================================================================================
# END: 新增边界生成函数
# ===================================================================================


# Parameters
WINDOW_SIZE = (256, 256)
STRIDE = 32
IN_CHANNELS = 6
FOLDER = "/home/duyihan/pycharm_project/数据集/"
BATCH_SIZE = 16
LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
N_CLASSES = len(LABELS)
WEIGHTS = torch.ones(N_CLASSES)
CACHE = True

palette = {0: (255, 255, 255), 1: (0, 0, 255), 2: (0, 255, 255), 3: (0, 255, 0), 4: (255, 255, 0), 5: (255, 0, 0),
           6: (0, 0, 0)}
invert_palette = {v: k for k, v in palette.items()}

MODEL = 'MPSegNet'
MODE = 'Test'
DATASET = 'Vaihingen'
LOSS = 'MUL'  # <--- MODIFIED: 设置为使用边界损失的模式
print(MODEL + ', ' + MODE + ', ' + DATASET + ', ' + LOSS)

if DATASET == 'Vaihingen':
    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '21', '15', '30']
    # test_ids1 = ['2', '4', '6', '8', '10', '12', '14', '16', '20', '22', '24', '27', '29', '31', '33', '35', '38']
    Stride_Size = 32
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    NDVI_FOLDER = MAIN_FOLDER + 'NDVI/top_mosaic_09cm_area{}.tif'
    nDSM_FOLDER = MAIN_FOLDER + 'nDSM/ndsm_mosaic_09cm_area{}.jpg'
    DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
    OBJECT_FOLDER = '/home/duyihan/pycharm_project/SSRS-main/checkpoint/sam/Vaihingen_objects/' + 'top_mosaic_09cm_area{}_objects.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
elif DATASET == 'Urban':
    # ... Urban dataset paths (unchanged)
    pass


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


def object_process(object):
    ids = np.unique(object)
    new_id = 1
    for id in ids[1:]:
        object = np.where(object == id, new_id, object)
        new_id += 1
    return object


class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER, NDVI_files=NDVI_FOLDER,
                 nDSM_files=nDSM_FOLDER, DSM_files=DSM_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.NDVI_files = [NDVI_FOLDER.format(id) for id in ids]
        self.nDSM_files = [nDSM_FOLDER.format(id) for id in ids]
        self.DSM_files = [DSM_FOLDER.format(id) for id in ids]
        # self.boundary_files = [BOUNDARY_FOLDER.format(id) for id in ids] # <--- REMOVED:不再需要加载边界文件
        self.object_files = [OBJECT_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        self.NDVI_cache_ = {}
        self.nDSM_cache_ = {}
        self.DSM_cache_ = {}
        self.data_cache_ = {}
        # self.boundary_cache_ = {} # <--- REMOVED
        self.object_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        return BATCH_SIZE * 1000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
        return tuple(results)

    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.data_files) - 1)

        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            if DATASET == 'Potsdam':
                data = io.imread(self.data_files[random_idx])[:, :, :3].transpose((2, 0, 1))
                data = 1 / 255 * np.asarray(data, dtype='float32')
            else:
                data = io.imread(self.data_files[random_idx])
                data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        # 加载分割标签
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            if DATASET == 'Urban':
                label = np.asarray(io.imread(self.label_files[random_idx]), dtype='int64') - 1
            else:
                label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # <--- NEW: 动态生成边界标签 ---
        boundary = mask_to_boundary(label)
        # --------------------------------

        if random_idx in self.object_cache_.keys():
            object = self.object_cache_[random_idx]
        else:
            object = np.asarray(io.imread(self.object_files[random_idx]))
            object = object.astype(np.int64)
            if self.cache:
                self.object_cache_[random_idx] = object

        # 加载辅助数据... (逻辑不变)
        if random_idx in self.NDVI_cache_.keys():
            NDVI = self.NDVI_cache_[random_idx]
        else:
            NDVI = np.asarray(io.imread(self.NDVI_files[random_idx])).astype(np.float32)
            NDVI = (NDVI - NDVI.min()) / (NDVI.max() - NDVI.min() + 1e-8)
            if self.cache:
                self.NDVI_cache_[random_idx] = NDVI
        if random_idx in self.nDSM_cache_.keys():
            nDSM = self.nDSM_cache_[random_idx]
        else:
            nDSM = np.asarray(io.imread(self.nDSM_files[random_idx])).astype(np.float32)
            nDSM = (nDSM - nDSM.min()) / (nDSM.max() - nDSM.min() + 1e-8)
            if self.cache:
                self.nDSM_cache_[random_idx] = nDSM
        if random_idx in self.DSM_cache_.keys():
            DSM = self.DSM_cache_[random_idx]
        else:
            DSM = np.asarray(io.imread(self.DSM_files[random_idx])).astype(np.float32)
            DSM = (DSM - DSM.min()) / (DSM.max() - DSM.min() + 1e-8)
            if self.cache:
                self.DSM_cache_[random_idx] = DSM

        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]
        boundary_p = boundary[x1:x2, y1:y2]  # <--- NEW: 裁剪边界
        object_p = object[x1:x2, y1:y2]
        NDVI_p = NDVI[x1:x2, y1:y2][np.newaxis, :, :]
        nDSM_p = nDSM[x1:x2, y1:y2][np.newaxis, :, :]
        DSM_p = DSM[x1:x2, y1:y2][np.newaxis, :, :]

        # <--- MODIFIED: 将boundary_p也加入数据增强
        data_p, label_p, boundary_p, object_p, NDVI_p, nDSM_p, DSM_p = self.data_augmentation(data_p, label_p,
                                                                                              boundary_p, object_p,
                                                                                              NDVI_p, nDSM_p, DSM_p)
        object_p = object_process(object_p)

        # <--- MODIFIED: 返回值顺序调整，并确保数据类型正确
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p).long(),  # 分割标签
                torch.from_numpy(boundary_p).float(),  # 边界标签
                torch.from_numpy(object_p).long(),  # 对象标签
                torch.from_numpy(NDVI_p),
                torch.from_numpy(nDSM_p),
                torch.from_numpy(DSM_p))


def get_random_pos(img, window_shape):
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def loss_calc(pred, label, weights):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d_ignore().cuda()
    return criterion(pred, label, weights)


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


class BoundaryLoss(nn.Module):
    """
    Computes the Dice Loss for the boundary prediction.
    The input logits are single-channel, and the target is a binary mask.
    """

    def __init__(self, smooth=1e-5):
        super(BoundaryLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (N, 1, H, W), targets: (N, H, W)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class ObjectLoss(nn.Module):
    def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

    def forward(self, pred, gt):
        num_object = int(torch.max(gt)) + 1
        num_object = min(num_object, self.max_object)
        total_object_loss = 0
        for object_index in range(1, num_object):
            mask = torch.where(gt == object_index, 1, 0).unsqueeze(1).to('cuda')
            num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to('cuda')
            avg_pool = mask / (num_point + 1)
            object_feature = pred.mul(avg_pool)
            avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(1, 1, gt.shape[1], gt.shape[2])
            avg_feature = avg_feature.mul(mask)
            object_loss = torch.nn.functional.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
            total_object_loss = total_object_loss + object_loss
        return total_object_loss


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def sliding_window(top, step=10, window_size=(20, 20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]: x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]: y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]: x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]: y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk: return
        yield chunk


def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(gts, predictions, labels=range(len(label_values)))
    print("Confusion matrix :\n", cm)
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))
    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc): print("%s: %.4f" % (label_values[l_id], score))
    print("---")
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score): print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:5])))
    print("---")
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" % (kappa))
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:5])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")
    return MIoU