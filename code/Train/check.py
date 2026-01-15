import cv2
import numpy as np
import os

# 随便找一张验证集的 Mask
mask_path = "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/weedsgalore/Weeds_Full_Standard/val/masks"
sample_mask = os.path.join(mask_path, os.listdir(mask_path)[0])
mask = cv2.imread(sample_mask, 0)

print(f"检查 Mask 文件: {sample_mask}")
print(f"图中包含的唯一像素值有: {np.unique(mask)}")
