# PatchCore Defect - 缺陷检测与分类系统

基于 [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection) 仓库构建的改造项目，在原有异常检测能力基础上增加了缺陷特征库和分类功能。

## 🎯 项目简介

### 原始 PatchCore
原始 PatchCore 只能进行**异常检测**，即判断"这个样本离正常有多远"。

### 改造后的 PatchCore Defect
新增**缺陷特征库**和**分类功能**，支持：
- ✅ 像素级缺陷分割（保留原有功能）
- ✅ 缺陷类型分类（新增）
- ✅ 同时输出分割图和分类结果

## 🏗️ 整体架构

```
训练阶段:
┌──────────────┐     ┌─────────────────┐
│  好品图片    │────▶│  好品特征库     │
│  (good)      │     │  (FAISS)        │
└──────────────┘     └─────────────────┘
          │
          ▼
┌──────────────────────┐     ┌─────────────────┐
│  缺陷图 + mask       │────▶│  缺陷特征库     │
│  (defect + mask)     │     │  (FAISS + 标签) │
└──────────────────────┘     └─────────────────┘

推理阶段:
┌──────────────┐     ┌──────────────────────────────┐
│  测试图片     │────▶│  同时查询两个库，比距离      │
└──────────────┘     │  ┌─────────┐  ┌─────────┐  │
                     │  │好品库   │  │缺陷库   │  │
                     │  │        │  │         │  │
                     │  └─────────┘  └─────────┘  │
                     └──────────────────────────────┘
                              │
                              ▼
                     ┌──────────────────────────────┐
                     │  输出:                       │
                     │  - 分割图 (segmentation)     │
                     │  - 分类结果 (defect_type)   │
                     │  - 异常分数 (anomaly_score)  │
                     └──────────────────────────────┘
```

## 📁 项目结构

```
./patchcore_defect/
├── src/
│   └── patchcore/
│       ├── __init__.py           # 包初始化
│       ├── backbones.py           # 骨干网络（从原仓库复制）
│       ├── common.py              # 通用工具（基于原仓库修改）
│       ├── sampler.py             # 采样器（从原仓库复制）
│       ├── patchcore.py           # 核心类（核心修改）
│       ├── defect_library.py      # 缺陷特征库（新增）
│       ├── dual_scorer.py         # 双库评分器（新增）
│       └── datasets.py            # 数据集类（新增）
├── bin/
│   ├── run_patchcore.py          # 原始训练脚本（保留兼容）
│   └── run_patchcore_defect.py   # 带缺陷库的训练/推理入口
├── test/
│   └── test_dual_scorer.py       # 单元测试
├── requirements.txt               # 依赖列表
├── setup.py                      # 安装配置
└── README.md                     # 项目文档
```

## 🔧 安装方法

### 方法一：从源码安装

```bash
cd patchcore_defect
pip install -e .
```

### 方法二：直接安装依赖

```bash
pip install -r requirements.txt
```

### GPU 支持（可选）

```bash
pip install faiss-gpu>=1.7.0
```

## 📊 数据目录结构

### 支持 MVTec AD 格式

```
your_dataset/
├── train/
│   └── good/              # 好品图片（用于构建好品库）
├── defect/               # 缺陷数据（用于构建缺陷库）
│   ├── crack/            # 缺陷类别1
│   │   ├── images/        # 缺陷图片
│   │   └── masks/         # 缺陷mask
│   ├── scratch/           # 缺陷类别2
│   │   ├── images/
│   │   └── masks/
│   └── dent/              # 缺陷类别3
│       ├── images/
│       └── masks/
└── test/
    ├── good/              # 测试用好品图片
    └── defect/            # 测试用缺陷图片
        ├── crack/
        ├── scratch/
        └── dent/
```

### Mask 文件说明

- Mask 格式：灰度图，白色区域（>127）为缺陷，黑色区域为正常
- Mask 文件名应与对应图片相同（扩展名可为 .png, .jpg）
- 也支持 `{image_name}_mask.png` 命名格式

## 🚀 使用示例

### 1. 训练缺陷检测模型

```bash
python bin/run_patchcore_defect.py train_defect \
    --data_path /path/to/mvtec_ad \
    --category bottle \
    --defect_types crack scratch dent \
    --backbone resnet50 \
    --layers layer2 layer3 \
    --classification_strategy distance_ratio \
    --save_path ./models/bottle_defect \
    --device cuda
```

### 2. 评估模型

```bash
python bin/run_patchcore_defect.py evaluate_defect \
    --model_path ./models/bottle_defect \
    --test_data /path/to/mvtec_ad/bottle \
    --output_dir ./results \
    --save_visualization \
    --device cuda
```

### 3. 原始 PatchCore 训练（保持兼容）

```bash
python bin/run_patchcore.py \
    --data_path /path/to/mvtec_ad \
    --category bottle \
    --backbone resnet50 \
    --layers layer2 layer3 \
    --save_path ./models/bottle_original \
    --device cuda
```

## 📝 分类策略说明

### 1. nearest_wins（最近邻获胜）

```python
# 如果到缺陷库的距离 < 到好品库的距离，则判定为缺陷
if distance_to_defect < distance_to_normal:
    predict = defect_type
else:
    predict = "normal"
```

**特点**：简单直接，适用于缺陷和正常样本差异明显的情况。

### 2. distance_ratio（距离比值）- 默认

```python
# 距离比 < 1 表示更像缺陷
ratio = distance_to_defect / distance_to_normal

# 结合异常阈值二次过滤
if ratio < 1.0 and anomaly_score > threshold:
    predict = defect_type
else:
    predict = "normal"
```

**特点**：结合绝对距离和相对距离，更加鲁棒。阈值可从好品样本自动计算。

### 3. softmax_prob（softmax概率）

```python
# 将距离转为概率
prob_normal = softmax(-distance_to_normal)
prob_defect = softmax(-distance_to_defect)

# 缺陷概率 > 0.5 判定为缺陷
if prob_defect > 0.5:
    predict = defect_type
else:
    predict = "normal"
```

**特点**：概率化输出，便于后续决策。

## 🔬 API 使用示例

### Python 代码调用

```python
import torch
import numpy as np
from patchcore_defect.src.patchcore import (
    PatchCore, DefectFeatureLibrary, DualLibraryScorer, MVTecDefectDataset
)
from patchcore_defect.src.patchcore import backbones, common, sampler

# 1. 加载骨干网络
backbone = backbones.load("resnet50")
backbone.name = "resnet50"

# 2. 创建模型
device = torch.device("cuda")
model = PatchCore(device)
model.load(
    backbone=backbone,
    layers_to_extract_from=["layer2", "layer3"],
    device=device,
    input_shape=(3, 224, 224),
    pretrain_embed_dimension=1536,
    target_embed_dimension=1536,
    patchsize=3,
    nn_method=common.FaissNN(False, 4),
    classification_strategy="distance_ratio",
)

# 3. 训练好品库
good_loader = ...  # 好品数据加载器
model.fit(good_loader)

# 4. 构建缺陷库
defect_loader = ...  # 缺陷数据加载器
model.fit_defect_library(
    defect_loader,
    classification_strategy="distance_ratio",
    percentile=95.0,
)

# 5. 推理
test_loader = ...  # 测试数据加载器
scores, masks, labels_gt, masks_gt, defect_types = model.predict(test_loader)

# 6. 保存模型
model.save_to_path("./models/my_model")
```

## ⚠️ 注意事项

1. **Mask 下采样**：使用最近邻插值（nearest）而非双线性插值，避免边界模糊
2. **类别编号**：0 预留给"正常"，缺陷类型从 1 开始编号
3. **向后兼容**：没有缺陷库时自动退化到原始 PatchCore 功能
4. **返回值变化**：`_predict` 返回 3 元组 `(scores, masks, defect_type_maps)`

## 📈 性能考虑

- 特征采样：`--sampler_percentage` 控制好品库采样比例，建议 0.1-0.2
- 缺陷采样：`--sampling_ratio` 控制缺陷库采样比例，建议 1.0（全量）
- GPU 加速：使用 `--device cuda` 启用 GPU 加速

## 🧪 运行测试

```bash
cd test
python test_dual_scorer.py
```

## 📚 引用

如果你在研究中使用了这个项目，请引用原始 PatchCore：

```bibtex
@article{patchcore2022,
  title={PatchCore: Towards Total Recall in Industrial Anomaly Detection},
  author={Roth, Karsten and Pemula, Lena and Zepeda, Joaquin and Schölkopf, Bernhard and Brox, Thomas and Gehler, Peter},
  journal={CVPR},
  year={2022}
}
```

## 📄 许可证

本项目继承原 PatchCore 的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
