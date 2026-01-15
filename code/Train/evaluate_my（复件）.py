import os
import torch
import numpy as np
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score

# --- 1. 路径与模型配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from data.my_dataset import MyDatasetInterface
from MODEL.model import MSFusionUNet as MSFusionModel

class MSFusion(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = MSFusionModel(in_channels=in_channels, num_classes=num_classes, norm_type='bn', dilation=2)
    def forward(self, x):
        return self.model(x)

def evaluate_suger():
    # ================= 配置区 =================
    model_path = "best_model_suger.pth" 
    num_classes = 3
    channels_config = ['rgb', 'nir', 'rededge'] # 5通道
    
    val_params = {
        'root': "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/weedsgalore/Weeds_Full_Standard",
        'batch_size': 1,
        'num_workers': 4,
        'channels': [channels_config],
        'num_classes': [num_classes],
        'img_size': 512
    }
    # =========================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    data_interface = MyDatasetInterface(val_params)
    data_interface.build_data_loaders()
    val_loader = data_interface.val_loader

    # 加载模型
    in_channels = 5 # 3(rgb) + 1(nir) + 1(rededge)
    model = MSFusion(in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_masks = []
    
    print(f"🚀 开始评估 3 分类模型...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy().flatten())
            all_masks.append(masks.numpy().flatten())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_masks)

    # 计算指标
    miou = jaccard_score(y_true, y_pred, average='macro')
    mf1 = f1_score(y_true, y_pred, average='macro')
    class_f1 = f1_score(y_true, y_pred, average=None)

    print("\n" + "="*30)
    print(f"✅ Overall mIoU:     {miou:.4f}")
    print(f"✅ Overall mF1:       {mf1:.4f}")
    print("-" * 30)
    
    labels = ["Background (背景)", "Crop (作物)", "Weed (杂草)"]
    for i, score in enumerate(class_f1):
        if i < len(labels):
            print(f"🔸 {labels[i]} F1: {score:.4f}")
    print("="*30)

if __name__ == '__main__':
    evaluate_suger()
