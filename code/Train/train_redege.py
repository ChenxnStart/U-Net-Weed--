import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

# --- 1. 路径导航 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from data.rededge_dataset import MyDatasetInterface
from MODEL.model import MSFusionUNet as MSFusionModel

class MSFusion(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = MSFusionModel(in_channels=in_channels, num_classes=num_classes, norm_type='bn', dilation=2)
    def forward(self, x):
        return self.model(x)

def get_param(p):
    return p[0] if isinstance(p, list) else p

def train():
    # --- 加载配置 ---
    config_path = os.path.join(os.path.dirname(current_dir), "Params/Eschikon/Esc_unet_v2.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['parameters']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 使用设备: {device}")

    # --- 数据准备 ---
    ds_config = config['dataset']
    data_interface = MyDatasetInterface(ds_config)
    data_interface.build_data_loaders()
    
    # --- 自动适配通道与类别 ---
    channels = get_param(ds_config['channels'])
    in_channels = 0
    for c in channels:
        in_channels += 3 if c.lower() == 'rgb' else 1
    
    num_classes = get_param(ds_config['num_classes'])
    print(f"📊 模型配置: 输入 {in_channels} 通道 | 输出 {num_classes} 分类")

    # --- 模型与优化器 ---
    model = MSFusion(in_channels=in_channels, num_classes=num_classes).to(device)
    
    # 针对类别不平衡，可以考虑为 CrossEntropy 增加权重，目前使用默认
    criterion = nn.CrossEntropyLoss()
    
    tr_config = config['train_params']
    optimizer = optim.Adam(model.parameters(), lr=get_param(tr_config['initial_lr']))
    
    epochs = get_param(tr_config['max_epochs'])
    best_val_loss = float('inf')

    # --- 训练循环 ---
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(data_interface.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            if torch.isnan(loss):
                print("❌ 警告: Loss 为 NaN，跳过该 Batch")
                continue
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train = train_loss / len(data_interface.train_loader)
        
        # 验证逻辑
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in data_interface.val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
        
        avg_val = val_loss / len(data_interface.val_loader)
        print(f"📉 Epoch {epoch+1}: Train Loss {avg_train:.4f} | Val Loss {avg_val:.4f}")

        # 保存最优模型
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = "best_model_suger.pth"
            torch.save(model.state_dict(), save_path)
            print(f"🌟 已更新最优模型: {save_path}")

if __name__ == '__main__':
    train()
