import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import f1_score

# --- 工具函数：计算 IoU ---
def calculate_iou(y_true, y_pred, num_classes):
    iou_list = []
    for c in range(num_classes):
        intersection = np.logical_and(y_true == c, y_pred == c).sum()
        union = np.logical_or(y_true == c, y_pred == c).sum()
        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append(intersection / union)
    return iou_list

# --- 1. 路径导航 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from data.gal_dataset import MyDatasetInterface
from MODEL.model import MSFusionUNet as MSFusionModel

# --- 2. 工具函数：解析参数 ---
def get_param(p):
    return p[0] if isinstance(p, list) else p

# --- 3. 模型包装类 ---
class MSFusion(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = MSFusionModel(in_channels=in_channels, num_classes=num_classes, norm_type='bn', dilation=2)
    def forward(self, x):
        return self.model(x)

# --- 4. 训练主函数 ---
def train():
    # --- 1. 加载配置 ---
    config_path = os.path.join(project_root, "Params/Gal/gal_unet_v2.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['parameters']

    ds_config = config['dataset']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 2. 目录准备 ---
    dataset_root_path = get_param(ds_config['root'])
    dataset_name = os.path.basename(dataset_root_path)
    log_base_dir = os.path.join(project_root, "LOG", dataset_name)
    ckpt_base_dir = os.path.join(project_root, "Checkpoints", dataset_name)
    
    for folder in [log_base_dir, ckpt_base_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 修改表头以包含 mIoU 和各类 IoU
    log_file = os.path.join(log_base_dir, "training_history.log")
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding='utf-8') as f:
            f.write("Time\tEpoch\tTr_Loss\tVal_Loss\tmF1\tmIoU\tF1_0\tF1_1\tF1_2\tIoU_0\tIoU_1\tIoU_2")

    # --- 3. 数据与模型准备 ---
    data_interface = MyDatasetInterface(ds_config)
    data_interface.build_data_loaders()
    
    channels = get_param(ds_config['channels'])
    in_channels = sum([3 if c.lower() == 'rgb' else 1 for c in channels])
    num_classes = get_param(ds_config['num_classes'])
    
    model = MSFusion(in_channels=in_channels, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=get_param(config['train_params']['initial_lr']))
    
    epochs = get_param(config['train_params']['max_epochs'])
    best_mf1 = 0.0

    # --- 4. 训练循环 ---
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(data_interface.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- 5. 验证与指标计算 (修复重点) ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for imgs, masks in data_interface.val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
                
                # 收集预测和真实标签
                pred = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
                all_preds.append(pred)
                all_masks.append(masks.cpu().numpy().flatten())
        
        y_true = np.concatenate(all_masks)
        y_pred = np.concatenate(all_preds)
        
        # 计算指标
        f1_list = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])
        mf1 = f1_score(y_true, y_pred, average='macro')
        
        iou_list = calculate_iou(y_true, y_pred, num_classes=3)
        miou = np.nanmean(iou_list)

        # --- 6. 记录日志 ---
        log_str = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{epoch+1}\t"
                   f"{train_loss/len(data_interface.train_loader):.4f}\t"
                   f"{val_loss/len(data_interface.val_loader):.4f}\t"
                   f"{mf1:.4f}\t{miou:.4f}\t"
                   f"{f1_list[0]:.4f}\t{f1_list[1]:.4f}\t{f1_list[2]:.4f}\t"
                   f"{iou_list[0]:.4f}\t{iou_list[1]:.4f}\t{iou_list[2]:.4f}")
        
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(log_str + "\n")

        print(f"📉 Epoch {epoch+1}: mF1={mf1:.4f}, mIoU={miou:.4f}")

        # --- 7. 保存逻辑 ---
        if mf1 > best_mf1:
            best_mf1 = mf1
            torch.save(model.state_dict(), os.path.join(ckpt_base_dir, "best_model_gal.pth"))
            print(f"🌟 更新最优模型 (mF1: {best_mf1:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_base_dir, f"checkpoint_epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    train()
