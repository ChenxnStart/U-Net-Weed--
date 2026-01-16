# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# 随便找一张 iMap 图片的路径 (确保路径存在)
imap_path = "/media/cclsol/Chen/Lawin/LWViTs-for-weedmapping/dataset/processed/000/groundtruth/000_frame0000_GroundTruth_iMap.png"

if not os.path.exists(imap_path):
    print("❌ 错误：找不到文件，请检查路径是否正确！")
else:
    # 以灰度模式读取（非常重要，必须是灰度）
    img = cv2.imread(imap_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ 错误：文件存在但无法读取（可能是图片损坏）")
    else:
        # 打印里面所有的唯一数值
        unique_values = np.unique(img)
        print("✅ 读取成功！")
        print("这张图里的数值有: " + str(unique_values))
        
        # 简单判断
        if np.array_equal(unique_values, [0, 1, 2]) or np.array_equal(unique_values, [0, 255]): 
             print("--> 这是一个标准的 Mask 文件。")
        else:
             print("--> 注意：数值看起来有些特别，请确认这是否是你想要的标签。")
