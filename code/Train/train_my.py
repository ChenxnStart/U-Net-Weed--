import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- 1. 路径导航 (自动找到 code 目录) ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path) # .../code/Train
code_dir = os.path.dirname(current_dir)          # .../code
sys.path.append(code_dir)
# --------------------------------------

# --- 2. 导入模块 (不再尝试导入 wd) ---
from data.my_dataset import MyDatasetInterface
# 直接从 MODEL 文件夹导入
from MODEL.model import MSFusionUNet as MSFusionModel

# 定义一个简单的 Wrapper，统一接口
class MSFusion(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, norm_type='bn', dilation=2, **kwargs):
        super().__init__()
        # 直接调用核心模型
        self.model = MSFusionModel(in_channels=in_channels, num_classes=num_classes, norm_type=norm_type, dilation=dilation)
    def forward(self, x):
        return self.model(x)

# --- 3. 辅助函数: 解包列表参数 ---
# 专门解决 YAML 里写成 [3] 这种格式的问题
def get_clean_param(param_value):
    if isinstance(param_value, list):
        return param_value[0]
    return param_value

def train():
    # --- 配置路径 ---
    # 请确保这个路径正确
    config_path = "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/U-Net-v2/code/Params/mydatasets/Myd_unet_v2.yaml"
    
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 处理嵌套结构 (parameters -> dataset)
    if 'parameters' in config:
        params_root = config['parameters']
    else:
        params_root = config

    # --- 准备设备 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    # --- 加载数据集 ---
    print("正在加载数据集...")
    dataset_params = params_root['dataset']
    data_interface = MyDatasetInterface(dataset_params)
    data_interface.build_data_loaders()
    
    train_loader = data_interface.train_loader
    val_loader = data_interface.val_loader

    # --- 加载模型 ---
    print("正在加载模型...")
    
    # 1. 计算输入通道数
    channels_list = dataset_params.get('channels', ['rgb'])
    # 再次确保 channels_list 是一层列表 (防止 yaml 双层嵌套)
    if len(channels_list) > 0 and isinstance(channels_list[0], list):
        channels_list = channels_list[0]

    in_channels = 0
    for c in channels_list:
        if c.lower() == 'rgb': in_channels += 3
        else: in_channels += 1 
    
    # 2. 获取类别数 (关键修复：解包列表)
    num_classes = get_clean_param(dataset_params['num_classes'])
    
    print(f"  - 输入通道数: {in_channels}")
    print(f"  - 类别数: {num_classes}")

    # 3. 初始化
    model = MSFusion(in_channels=in_channels, num_classes=num_classes, norm_type='bn', dilation=2)
    model = model.to(device)

    # --- 定义 Loss 和 优化器 ---
    criterion = nn.CrossEntropyLoss()
    
    train_config = params_root['train_params']
    
    # 修复：解包 LR 和 Epochs
    initial_lr = get_clean_param(train_config['initial_lr'])
    max_epochs = get_clean_param(train_config['max_epochs'])

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)

    # --- 训练循环 ---
    print(f"开始训练，总 Epochs: {max_epochs}, LR: {initial_lr}")
    best_loss = float('inf')

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        # 如果数据集太小导致 tqdm 闪退，可以把 tqdm 去掉直接用 train_loader
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(current_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存: {save_path}")

if __name__ == '__main__':
    train()
