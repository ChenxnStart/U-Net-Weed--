import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

class MyDatasetInterface:
    def __init__(self, dataset_params, name=None):
        # --- 1. 参数解析 (兼容列表和单值) ---
        self.root = dataset_params['root']
        if isinstance(self.root, list): self.root = self.root[0]

        self.batch_size = dataset_params['batch_size']
        if isinstance(self.batch_size, list): self.batch_size = self.batch_size[0]

        self.num_workers = dataset_params.get('num_workers', 4)
        if isinstance(self.num_workers, list): self.num_workers = self.num_workers[0]

        # --- 2. 通道配置 ---
        self.channels_config = dataset_params.get('channels', ['rgb'])
        # 处理嵌套列表 [['rgb', ...]]
        if len(self.channels_config) > 0 and isinstance(self.channels_config[0], list):
            self.channels_config = self.channels_config[0]
        
        # 确保 rgb 存在
        lower_channels = [c.lower() for c in self.channels_config]
        if 'rgb' not in lower_channels:
            print("警告: 自动添加 'rgb' 通道")
            self.channels_config.insert(0, 'rgb')
            
        # --- 3. 图片尺寸 ---
        self.img_size = dataset_params.get('img_size', 512)
        if isinstance(self.img_size, list): self.img_size = self.img_size[0]
        
        print(f"ℹ️ 数据集配置完成: 尺寸 {self.img_size}x{self.img_size}, 批量 {self.batch_size}")

        # --- 4. 初始化数据集 ---
        self.trainset = ChannelSeparatedDataset(self.root, split='train', channels=self.channels_config, img_size=self.img_size)
        
        # 验证集
        if os.path.exists(os.path.join(self.root, 'val')):
            self.valset = ChannelSeparatedDataset(self.root, split='val', channels=self.channels_config, img_size=self.img_size)
        else:
            self.valset = self.trainset

    def build_data_loaders(self, **kwargs):
        self.train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

class ChannelSeparatedDataset(Dataset):
    def __init__(self, root, split='train', channels=['rgb'], img_size=512):
        self.split_dir = os.path.join(root, split)
        self.masks_dir = os.path.join(self.split_dir, 'masks')
        self.channels = channels
        self.img_size = img_size
        
        if os.path.exists(self.masks_dir):
            all_masks = sorted([f for f in os.listdir(self.masks_dir) if f.endswith(('.png', '.jpg', '.tif', '.bmp'))])
        else:
            all_masks = []
            print(f"错误: Masks 目录不存在: {self.masks_dir}")
            
        self.filenames = []
        self.valid_paths_cache = {} 
        
        # --- 建立索引 (自动匹配后缀) ---
        print(f"正在索引 {split} 集...")
        for mask_name in all_masks:
            is_valid = True
            file_stem = os.path.splitext(mask_name)[0]
            current_paths = {}

            for channel_name in self.channels:
                folder_name = channel_name.lower()
                target_folder = os.path.join(self.split_dir, folder_name)
                
                found_path = None
                # 尝试多种后缀
                for ext in ['.tif', '.png', '.jpg', '.TIF', '.PNG', '.JPG']:
                    potential_path = os.path.join(target_folder, file_stem + ext)
                    if os.path.exists(potential_path):
                        found_path = potential_path
                        break
                
                if found_path is None:
                    is_valid = False
                    break
                else:
                    current_paths[channel_name] = found_path
            
            if is_valid:
                self.filenames.append(mask_name)
                self.valid_paths_cache[mask_name] = current_paths
        
        print(f"✅ {split} 集有效样本: {len(self.filenames)}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        mask_name = self.filenames[idx]
        paths_map = self.valid_paths_cache[mask_name]
        
        data_list = []
        
        # --- 1. 读取多通道图片 ---
        for channel_name in self.channels:
            img_path = paths_map[channel_name]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                # 极少数情况文件可能损坏
                raise ValueError(f"无法读取图片: {img_path}")
            
            # RGB 转码
            if channel_name.lower() == 'rgb':
                if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # 补齐维度 (H, W) -> (H, W, 1)
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            
            data_list.append(img)

        # 合并
        combined_img = np.concatenate(data_list, axis=2)
        
        # --- 2. Resize 图片 (线性插值) ---
        orig_h, orig_w = combined_img.shape[:2]
        if orig_h != self.img_size or orig_w != self.img_size:
            combined_img = cv2.resize(combined_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            # Resize 后单通道会变回 (H, W)，需要再补齐
            if combined_img.ndim == 2:
                combined_img = combined_img[:, :, np.newaxis]

        # 归一化
        combined_img = combined_img.astype('float32')
        if combined_img.max() > 255:
            combined_img /= 65535.0
        else:
            combined_img /= 255.0
            
        image_tensor = torch.from_numpy(combined_img).permute(2, 0, 1)

        # --- 3. 读取 Mask ---
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, 0) # 读取为灰度
        
        # --- 4. Resize Mask (最近邻插值) ---
        if mask.shape[0] != self.img_size or mask.shape[1] != self.img_size:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
        # --- 🔴 5. 关键修复：清洗标签值 ---
        # 您的 Mask 可能是 0(黑) 和 255(白)
        # 这里的代码把所有大于 0 的值都变成 1
        mask[mask > 0] = 1
        
        # 再次保险：如果还有大于等于 2 的值，强制变成 0
        mask[mask >= 2] = 0
        # -------------------------------

        mask_tensor = torch.from_numpy(mask).long()

        return image_tensor, mask_tensor
