import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from ezdl.data import DatasetInterface

class SimpleDatasetInterface(DatasetInterface):
    def __init__(self, dataset_params, name=None):
        super().__init__(dataset_params)
        self.root = dataset_params['root']
        self.batch_size = dataset_params['batch_size']
        self.num_workers = dataset_params.get('num_workers', 4)
        
        # 初始化数据集
        self.trainset = SimpleFolderDataset(self.root, split='train')
        self.valset = SimpleFolderDataset(self.root, split='val')
        # 如果没有test文件夹，就用val代替，或者留空
        test_path = os.path.join(self.root, 'test')
        if os.path.exists(test_path):
            self.testset = SimpleFolderDataset(self.root, split='test')
        else:
            self.testset = self.valset

    def build_data_loaders(self, batch_size_factor=1, num_workers=4, **kwargs):
        self.train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.val_loader = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

class SimpleFolderDataset(Dataset):
    def __init__(self, root, split='train'):
        self.images_dir = os.path.join(root, split, 'images')
        self.masks_dir = os.path.join(root, split, 'masks')
        
        # 过滤文件，只保留图片
        if os.path.exists(self.images_dir):
            self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])
        else:
            print(f"警告: 未找到目录 {self.images_dir}")
            self.images = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        # 1. 读取图像 (支持多通道 .tif)
        # cv2.IMREAD_UNCHANGED 会保留 tif 的所有通道和深度(16bit)
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"无法读取图片: {img_path}")

        # 处理通道顺序: OpenCV 读 RGB 时是 BGR，但多光谱通常顺序是固定的
        # 如果是普通的3通道 jpg/png，转 RGB
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # 归一化和转 Tensor
        image = image.astype('float32')
        if image.max() > 255:
            image = image / 65535.0 # 16位图
        else:
            image = image / 255.0   # 8位图
            
        # [H, W, C] -> [C, H, W]
        if image.ndim == 2: # 灰度图
            image = image[np.newaxis, :, :]
        else:
            image = np.transpose(image, (2, 0, 1))
            
        image = torch.from_numpy(image)

        # 2. 读取 Mask
        # 假设 mask 是单通道 png，像素值代表类别索引 (0, 1, 2...)
        mask = cv2.imread(mask_path, 0) 
        mask = torch.from_numpy(mask).long()

        return image, mask
