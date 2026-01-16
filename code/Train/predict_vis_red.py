import os
import sys

# --- 1. 核心路径修正 (必须放在最前面) ---
# 获取当前文件 (predict_vis_red.py) 的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取其所在的 Train 目录
current_dir = os.path.dirname(current_file_path)
# 获取 Train 的父目录，即 code 目录
project_root = os.path.dirname(current_dir)

# 将 code 目录加入系统路径，这样 Python 就能看见 MODEL 和 data 了
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 2. 现在再进行原本的 import ---
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from MODEL.model import MSFusionUNet as MSFusionModel
from data.rededge_dataset import EschikonDataset

# --- 1. 配置路径 ---
# 指向刚才保存的 best_model.pth
best_model_path = "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/U-Net-v2/code/checkpoints/Eschikon_loss/best_model.pth"
data_root = "/media/cclsol/Chen/Lawin/LWViTs-for-weedmapping/dataset/processed"
test_split = "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/MSU-Net/code/splits/val.txt"
save_dir = "vis_results"

os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 颜色定义 (与你的 Label 一致) ---
# 0:黑色 (背景), 1:绿色 (作物), 2:红色 (杂草)
PALETTE = np.array([
    [0, 0, 0],      # Background
    [0, 255, 0],    # Crop (Green)
    [255, 0, 0]     # Weed (Red)
], dtype=np.uint8)

def visualize():
    # 加载模型
    # 注意：如果训练时用了包装类 MSFusion，这里加载需要对应
    model = MSFusionModel(in_channels=5, num_classes=3)
    state_dict = torch.load(best_model_path, map_location=device)
    
    # 移除 state_dict 中的 'model.' 前缀 (如果你的包装类里有这个)
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 加载数据
    dataset = EschikonDataset(data_root, test_split)
    print(f"🎨 正在从测试集选取图片进行可视化...")

    # 随机选 10 张
    for i in range(10):
        img_tensor, mask_tensor = dataset[i]
        
        # 推理
        input_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # 准备图像显示
        # 1. 原始 RGB (前3通道)
        rgb = img_tensor[:3, :, :].permute(1, 2, 0).numpy() * 255
        rgb = rgb.astype(np.uint8)
        
        # 2. 真实标签上色
        gt_color = PALETTE[mask_tensor.numpy()]
        
        # 3. 预测结果上色
        pred_color = PALETTE[pred]

        # 4. 叠加对比图 (Overlay)
        overlay = cv2.addWeighted(rgb, 0.7, pred_color, 0.3, 0)

        # 绘图
        plt.figure(figsize=(20, 5))
        images = [rgb, gt_color, pred_color, overlay]
        titles = ['Original RGB', 'Ground Truth', 'Prediction', 'Overlay']
        
        for j in range(4):
            plt.subplot(1, 4, j+1)
            plt.imshow(images[j])
            plt.title(titles[j])
            plt.axis('off')

        save_path = os.path.join(save_dir, f"sample_{i}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_path}")

if __name__ == '__main__':
    visualize()
