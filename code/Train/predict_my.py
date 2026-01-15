import os
import torch
import cv2
import numpy as np
import sys

# --- 1. 路径配置 ---
# 自动找到上级目录以便导入模型
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.dirname(current_dir))

from MODEL.model import MSFusionUNet as MSFusionModel

# 简单的模型包装器 (必须和训练时一致)
class MSFusion(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = MSFusionModel(in_channels=in_channels, num_classes=num_classes, norm_type='bn', dilation=2)
    def forward(self, x):
        return self.model(x)

def predict():
    # ================= 配置区域 =================
    # 1. 模型路径
    model_path = "best_model.pth" 
    
    # 2. 测试图片的基础路径 (RGB文件夹)
    # 请修改为您硬盘上的真实路径，例如 dataset/test/rgb/xxx.png
    # 这里我们自动去 dataset/val/rgb 里随便找一张来测
    base_dataset_dir = "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/dataset/val"
    
    # 3. 参数配置 (必须和训练时完全一致)
    img_size = 512
    num_classes = 2
    # 您的训练用了7通道: rgb, nir, re, red, green
    channels_config = ['rgb', 'nir', 're', 'red', 'green'] 
    # ===========================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用设备: {device}")

    # --- 1. 加载模型 ---
    in_channels = 0
    for c in channels_config:
        in_channels += 3 if c == 'rgb' else 1
        
    print(f"初始化模型 (输入通道: {in_channels}, 类别: {num_classes})...")
    model = MSFusion(in_channels=in_channels, num_classes=num_classes)
    
    # 加载权重
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型文件 {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ 模型加载成功！")

    # --- 2. 寻找测试图片 ---
    rgb_dir = os.path.join(base_dataset_dir, 'rgb')
    if not os.path.exists(rgb_dir):
        print(f"❌ 找不到测试目录: {rgb_dir}")
        return

    # 随便拿第一张图来测
    test_files = [f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.tif', '.jpg'))]
    if not test_files:
        print("❌ 测试目录下没有图片")
        return
    
    filename = test_files[0] # 取第一张
    file_stem = os.path.splitext(filename)[0]
    print(f"正在预测图片: {filename} ...")

    # --- 3. 读取并组合多通道数据 ---
    data_list = []
    try:
        for channel in channels_config:
            # 自动寻找对应文件夹 (rgb, nir, re...)
            # 注意：这里假设所有文件夹都在 val 目录下，且文件名(stem)一致
            target_folder = os.path.join(base_dataset_dir, channel.lower())
            
            # 尝试找 .tif 或 .png
            found_path = None
            for ext in ['.tif', '.png', '.jpg']:
                path = os.path.join(target_folder, file_stem + ext)
                if os.path.exists(path):
                    found_path = path
                    break
            
            if not found_path:
                print(f"❌ 缺少通道文件: {channel}/{file_stem}")
                return

            img = cv2.imread(found_path, cv2.IMREAD_UNCHANGED)
            
            if channel == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img.ndim == 2: img = img[:, :, np.newaxis]
            data_list.append(img)
            
    except Exception as e:
        print(f"读取图片出错: {e}")
        return

    # 组合
    combined_img = np.concatenate(data_list, axis=2)
    
    # Resize & Normalize
    original_h, original_w = combined_img.shape[:2]
    combined_img = cv2.resize(combined_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    if combined_img.ndim == 2: combined_img = combined_img[:, :, np.newaxis]
    
    combined_img = combined_img.astype('float32')
    if combined_img.max() > 255: combined_img /= 65535.0
    else: combined_img /= 255.0

    # 转 Tensor
    input_tensor = torch.from_numpy(combined_img).permute(2, 0, 1).unsqueeze(0).to(device)

    # --- 4. 预测 ---
    with torch.no_grad():
        output = model(input_tensor)
        # 取最大概率的类别 (Argmax)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # --- 5. 保存结果 ---
    # 把预测结果放大回原始尺寸
    pred_mask = pred_mask.astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # 为了可视化，把 0/1 变成 0/255 (黑白)
    pred_mask = pred_mask * 255

    save_name = f"pred_{file_stem}.png"
    cv2.imwrite(save_name, pred_mask)
    print(f"🎉 预测完成！结果已保存为: {save_name}")
    print("请打开这个图片看看效果如何！(白色是目标，黑色是背景)")

if __name__ == '__main__':
    predict()
