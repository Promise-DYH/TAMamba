# TAMamba
TAMamba
# Reference Repository
If you have any questions about our training and testing template, including model selection, dataset download and preprocessing, please check this open-source repository:
https://github.com/sstary/SSRS
Our experimental template is derived from this project.

# Environment & Training Configuration Modification Guide
## 1. Modify `train_DK_Mamba_Net_S3F_Dual42_16.py`
### 1.1 Specify Single GPU (GPU 0) for Training
Add the following code snippet at the very top of the script **before all library imports** to bind the training task to GPU 0 only:
```python
import os
# Assign training task to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### 1.2 Modify the Save Path of the Best Model Weights
#### Original Code Segment
```python
if current_MIoU > MIoU_best:
    MIoU_best = current_MIoU
    save_path = f'/home/duyihan/pycharm_project/SSRS-main/SAM_RS_L40/TRY/resultv_DK_Mamba_Net_S3F_Dual/42_16/0k/1_5_8_5/{MODEL}_epoch{e}_MIoU{MIoU_best:.4f}.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(net.state_dict(), save_path)
    print(f"***** New best model saved to {save_path} *****")
```

#### Modified Code (Linux System)
Replace the username placeholder with your local Linux username:
```python
if current_MIoU > MIoU_best:
    MIoU_best = current_MIoU
    # Custom local path for saving optimal model weights
    save_path = f'xxxxxxxxxxxxxxxx'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(net.state_dict(), save_path)
    print(f"***** New best model saved to {save_path} *****")
```

## 2. Modify Dataset Root Path in `utils_Mamba_new_Boundry_best_16.py`
### Original Code
```python
FOLDER = "/home/duyihan/pycharm_project/数据集/"
```

### Modified Code (Linux System)
```python
# Root directory of the local remote sensing dataset
FOLDER = "/home/your_username/pycharm_project/数据集/"
```

## Brief Explanation
1. GPU configuration restricts the training process to GPU 0 to avoid GPU resource conflicts in multi-GPU servers.
2. The program automatically saves model weights only when the validation MIoU reaches a new historical maximum, and automatically creates target folders without manual creation.
3. The `FOLDER` variable is the global dataset root path. An incorrect path will cause dataset loading failures during training and validation.
4. Model naming rule: `{Model Name}_epoch{Training Epoch}_MIoU{Best Mean Intersection over Union}.pth`
