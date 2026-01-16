import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# --- 1. 路径修复 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from data.rededge_dataset import EschikonDataset 
from MODEL.model import MSFusionUNet as MSFusionModel

# ==========================================
# 📊 1. 评估与损失函数
# ==========================================
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def mean_intersection_over_union(self):
        miou = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + 
            np.sum(self.confusion_matrix, axis=0) - 
            np.diag(self.confusion_matrix) + 1e-6
        )
        return np.nanmean(miou), miou

    def f1_score(self):
        TP = np.diag(self.confusion_matrix)
        FP = np.sum(self.confusion_matrix, axis=0) - TP
        FN = np.sum(self.confusion_matrix, axis=1) - TP
        f1 = 2 * TP / (2 * TP + FP + FN + 1e-6)
        return np.nanmean(f1), f1

    def add_batch(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        self.confusion_matrix += count.reshape(self.num_class, self.num_class)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax: inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None: weight = [1.0] * self.n_classes
        
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes

# ==========================================
# 📝 2. 日志系统与模型包装
# ==========================================
def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    logger = logging.getLogger("MSUNet_Train")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file); fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    ch = logging.StreamHandler(); ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

class MSFusion(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.model = MSFusionModel(in_channels=in_channels, num_classes=num_classes, norm_type='bn', dilation=2)
    def forward(self, x): return self.model(x)

# ==========================================
# 🔥 4. 训练主程序
# ==========================================
def train():
    cfg = {
        'data_root': "/media/cclsol/Chen/Lawin/LWViTs-for-weedmapping/dataset/processed",
        'train_split': "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/MSU-Net/code/splits/train.txt",
        'val_split': "/media/cclsol/df07c0f4-31b8-4090-8a4a-8c254d91c123/ch/MSU-Net/MSU-Net/code/splits/test.txt",
        'num_classes': 3,
        'batch_size': 4,
        'lr': 0.00001,
        'epochs': 100,
        'save_dir': os.path.join(project_root, "checkpoints", "Eschikon_loss_2")
    }

    logger = setup_logger(cfg['save_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(EschikonDataset(cfg['data_root'], cfg['train_split']), batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(EschikonDataset(cfg['data_root'], cfg['val_split']), batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    
    model = MSFusion(in_channels=5, num_classes=cfg['num_classes']).to(device)

    # --- 损失函数组合 ---
    weights = torch.tensor([0.0638, 1.0, 1.6817]).to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    criterion_dice = DiceLoss(n_classes=cfg['num_classes'])
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    evaluator = Evaluator(cfg['num_classes'])
    
    best_f1 = 0.0

    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]")
        
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            masks[(masks > 2) & (masks != 255)] = 255

            outputs = model(images)
            
            # 🟢 组合损失计算
            loss_ce = criterion_ce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks, weight=weights.tolist())
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # --- 验证 ---
        model.eval()
        val_loss = 0
        evaluator.reset()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                masks[(masks > 2) & (masks != 255)] = 255
                outputs = model(images)
                
                v_ce = criterion_ce(outputs, masks)
                v_dice = criterion_dice(outputs, masks, weight=weights.tolist())
                val_loss += (0.5 * v_ce + 0.5 * v_dice).item()

                preds = torch.argmax(outputs, dim=1)
                evaluator.add_batch(masks.cpu().numpy(), preds.cpu().numpy())

        mIoU, class_iou = evaluator.mean_intersection_over_union()
        mF1, class_f1 = evaluator.f1_score()
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        logger.info(f"\n📊 Ep {epoch+1} | Val Loss: {avg_val:.4f} | mIoU: {mIoU:.2%} | mF1: {mF1:.2%} | Weed: {class_f1[2]:.2%}")

        if mF1 > best_f1:
            best_f1 = mF1
            torch.save(model.state_dict(), os.path.join(cfg['save_dir'], "best_model.pth"))
            logger.info(f"🌟 新高! 模型已保存")

if __name__ == '__main__':
    train()
