import os
import torch
import numpy as np
import time
import datetime
import sys
from sklearn.metrics import f1_score, jaccard_score
from tqdm import tqdm

# --- 1. 环境路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from data.my_dataset import MyDatasetInterface
from MODEL.model import MSFusionUNet as MSFusionModel

# 统一模型包装器
class MSFusion(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = MSFusionModel(in_channels=in_channels, num_classes=num_classes, norm_type='bn', dilation=2)
    def forward(self, x):
        return self.model(x)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_test():
    # ================= 配置区 =================
    model_path = "best_model.pth"
    img_size = 512
    num_classes = 2
    # 通道配置必须与训练一致
    channels_config = ['rgb', 'nir', 're', 'red', 'green']
    
    test_params = {
        'root': "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/dataset",
        'batch_size': 1, # 测试建议设为 1，最准确
        'num_workers': 4,
        'channels': [channels_config],
        'num_classes': [num_classes],
        'img_size': img_size
    }
    # =========================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据 (这里我们跑验证集 val 作为测试)
    data_interface = MyDatasetInterface(test_params)
    data_interface.build_data_loaders()
    test_loader = data_interface.val_loader 
    num_masks = len(test_loader.dataset)

    # 2. 初始化模型
    in_channels = 0
    for c in channels_config:
        in_channels += 3 if c == 'rgb' else 1
    
    model = MSFusion(in_channels=in_channels, num_classes=num_classes)
    if not os.path.exists(model_path):
        print(f"{get_timestamp()} - ❌ Error: Cannot find {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 3. 推理评估
    all_preds = []
    all_masks = []

    print(f"{get_timestamp()} - 🚀 Starting Evaluation...")
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy().flatten())
            all_masks.append(masks.numpy().flatten())

    # 合并数据
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_masks)

    # 4. 计算指标 (按照你的要求计算 mIoU 和 Dice/F1)
    # average='macro' 计算所有类别的算术平均值
    miou = jaccard_score(y_true, y_pred, average='macro')
    dice_f1 = f1_score(y_true, y_pred, average='macro')

    # 5. 按照要求的格式输出日志
    print("-" * 30)
    print(f"Processed Masks: {num_masks}")
    print(f"{get_timestamp()} -")
    print(f"mIoU: {miou:.4f}")
    print(f"{get_timestamp()} -")
    print(f"F1:   {dice_f1:.4f}")
    print("-" * 30)

    # 额外补充：打印每个类别的具体 F1
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    for i, score in enumerate(per_class_f1):
        label = "Background" if i == 0 else "Target"
        print(f"Class {i} ({label}) F1: {score:.4f}")

if __name__ == '__main__':
    run_test()
