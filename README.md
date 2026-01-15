### 📂 MSU-Net Weed Segmentation (U-Net-v2) 项目说明文档

# MSU-Net 农田杂草/作物多光谱语义分割

本项目基于 **MSFusionUNet** 模型，专门用于处理多光谱遥感影像的语义分割任务。针对大尺寸无人机影像（4000x3000）、类别不平衡以及数据读取瓶颈进行了针对性优化。

## 🌟 核心特性

* **多通道融合**: 自动对齐并拼接 RGB(3通道)、NIR、RE、Red、Green 等多个波段。
* **内存优化**: 支持 512x512 自动缩放，降低显存压力，防止 CUDA Out of Memory。
* **标签清洗**: 自动修复 Mask 像素值（将 255/128 等转为 0/1 逻辑标签），彻底解决 `CUDA error: device-side assert triggered`。
* **评估体系**: 完整的像素级指标统计，包括 mIoU、mF1-Score 以及各类别细分 F1 分数。

## 📊 当前模型性能 (Val Set)

| 指标 | 数值 | 备注 |
| --- | --- | --- |
| **mIoU** | **0.6956** | 平均交并比 |
| **mF1-Score** | **0.7820** | Dice 系数 |
| **Background F1** | **0.9984** | 背景识别精度 |
| **Target F1** | **0.5656** | 目标（杂草/作物）识别精度 |

---

## 🚀 快速开始

### 1. 环境准备

确保已安装 PyTorch 及相关评估库：

```bash
pip install torch torchvision scikit-learn opencv-python tqdm

```

### 2. 训练模型 (Training)

运行训练脚本。程序会自动处理 YAML 配置文件中的列表参数，并进行 512x512 实时缩放：

```bash
cd code/Train
python train_my.py

```

* **输入**: `/dataset/train/` 目录下的多通道图与 Mask。
* **输出**: 训练日志及 `best_model.pth` 权重文件。

### 3. 模型预测与可视化 (Inference)

加载训练好的模型，对图片进行推理并生成可视化的黑白分割图：

```bash
python predict_my.py

```

* **结果**: 生成 `pred_xxxx.png`，白色为识别出的目标，黑色为背景。

### 4. 全量指标评估 (Evaluation)

计算验证集上每个像素的分类准确度，得出 mIoU 和 F1 指标：

```bash
python evaluate_my.py

```

### 5. 标准测试日志导出 (Testing)

生成符合学术/工程要求的带时间戳的测试报告：

```bash
python test_my.py

```

---

## 📂 文件说明

### 数据处理层 (`code/data/`)

* **`my_dataset.py`**:
* 实现了 `ChannelSeparatedDataset` 类。
* **智能匹配**: 支持 `.tif` 和 `.png` 后缀混合存在时的自动搜索。
* **数据清洗**: `mask[mask > 127] = 1` 确保标签符合分类要求。



### 运行层 (`code/Train/`)

* **`train_my.py`**: 包含 Adam 优化器、交叉熵损失函数以及模型保存逻辑。
* **`test_my.py`**: 格式化输出测试结果，输出包含 `Processed Masks` 总数。
* **`predict_my.py`**: 自动将预测结果 Resize 回原始图片尺寸，方便肉眼对比。

---

## 🛠 开发备注

* **权重加载**: 使用 `weights_only=True` 参数以符合最新的 PyTorch 安全标准。
* **缩放策略**: 图片使用 `INTER_LINEAR`（线性插值），Mask 使用 `INTER_NEAREST`（最近邻插值）以保证标签不失真。

---

### 📤 提交到 Git

完成 README 整合后，请运行以下命令同步到 GitHub：

```bash
git add README.md
git commit -m "docs: 整合完整README文档与运行指南"
git push origin main

```


