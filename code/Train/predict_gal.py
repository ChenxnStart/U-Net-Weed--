import os
import torch
import cv2
import numpy as np
import sys

# --- 路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))

from MODEL.model import MSFusionUNet as MSFusionModel

class MSFusion(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = MSFusionModel(in_channels=in_channels, num_classes=num_classes, norm_type='bn', dilation=2)
    def forward(self, x):
        return self.model(x)

def predict_color():
    # ================= 配置区 =================
    model_path = "best_model_suger.pth" 
    img_size = 512
    num_classes = 3
    channels_config = ['rgb', 'nir', 'rededge'] # 5通道
    # 选一张验证集里的图片来测试
    test_dir = "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/weedsgalore/Weeds_Full_Standard/val"
    # =========================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 初始化模型并加载权重
    model = MSFusion(in_channels=5, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. 读取多通道测试图 (找第一张图)
    rgb_path = os.path.join(test_dir, 'rgb')
    test_file = sorted(os.listdir(rgb_path))[0]
    file_stem = os.path.splitext(test_file)[0]
    
    data_list = []
    for c in channels_config:
        folder = os.path.join(test_dir, c.lower())
        img = cv2.imread(os.path.join(folder, test_file), cv2.IMREAD_UNCHANGED)
        if c == 'rgb': img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.ndim == 2: img = img[:, :, np.newaxis]
        data_list.append(img)
    
    combined = np.concatenate(data_list, axis=2)
    h, w = combined.shape[:2]
    combined = cv2.resize(combined, (img_size, img_size)).astype('float32') / 255.0
    input_t = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        output = model(input_t)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 4. 上色 (BGR格式)
    # 0:黑色, 1:绿色(作物), 2:红色(杂草)
    color_mask = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    color_mask[pred == 1] = [0, 255, 0]   # Green for Crop
    color_mask[pred == 2] = [0, 0, 255]   # Red for Weed
    
    # 放大回原始尺寸
    color_mask = cv2.resize(color_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    save_path = f"color_pred_{file_stem}.png"
    cv2.imwrite(save_path, color_mask)
    print(f"🎉 彩色预测图已保存: {save_path}")

if __name__ == '__main__':
    predict_color()
