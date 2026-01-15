import os
import torch
import numpy as np
import cv2
import yaml
import sys

# 路径导航
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from data.gal_dataset import MyDatasetInterface
from train_gal import MSFusion  # 确保能引用到你的模型包装类

def visualize():
    # 1. 配置与模型加载
    config_path = os.path.join(project_root, "Params/Gal/gal_unet_v2.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['parameters']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = config['dataset']['num_classes'][0]
    
    # 自动定位最优权重
    dataset_name = os.path.basename(config['dataset']['root'][0])
    model_path = os.path.join(project_root, "Checkpoints", dataset_name, "best_model_gal.pth")
    
    # 初始化模型 (确保输入通道数正确，这里根据配置动态计算)
    channels = config['dataset']['channels'][0]
    in_channels = sum([3 if c.lower() == 'rgb' else 1 for c in channels])
    
    model = MSFusion(in_channels=in_channels, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 准备数据
    data_interface = MyDatasetInterface(config['dataset'])
    data_interface.build_data_loaders()
    
    # 创建结果保存目录
    vis_save_dir = os.path.join(project_root, "VIS_Results", dataset_name)
    os.makedirs(vis_save_dir, exist_ok=True)

    # 定义颜色映射 (BGR格式): [0:黑色(背景), 1:绿色(作物), 2:红色(杂草)]
    colors = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 255]], dtype='uint8')

    print(f"🧐 正在从验证集中提取图片进行可视化...")

    with torch.no_grad():
        # 取 5 张图看效果
        for i, (imgs, masks) in enumerate(data_interface.val_loader):
            if i >= 5: break 
            
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            for j in range(imgs.size(0)):
                # 1. 提取并转换 RGB 图像
                # 注意：imgs 是 [B, C, H, W]，取出的单张是 [C, H, W]
                img_tensor = imgs[j, :3, :, :]
                img_rgb = img_tensor.permute(1, 2, 0).cpu().numpy() # 变为 [H, W, 3]
                
                # 反归一化：假设训练时除以了 255
                img_rgb = (img_rgb * 255).astype('uint8')
                # OpenCV 默认使用 BGR，所以要转换一下颜色空间
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # 2. 准备 GT 和 Pred 的彩色掩膜
                gt_np = masks[j].cpu().numpy().astype('int')
                pred_np = preds[j].cpu().numpy().astype('int')
                
                gt_color = colors[gt_np]
                pred_color = colors[pred_np]

                # 3. 横向拼接并保存
                combined = np.hstack([img_rgb, gt_color, pred_color])
                # 画上文字标注
                cv2.putText(combined, "Original RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, "Ground Truth", (512 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined, "Prediction", (1024 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                save_path = os.path.join(vis_save_dir, f"result_batch{i}_idx{j}.png")
                cv2.imwrite(save_path, combined)

    print(f"✅ 可视化完成！结果保存在: {vis_save_dir}")

if __name__ == '__main__':
    visualize()
