#!/usr/bin/env python3
"""原始 PatchCore 训练脚本 - 保留向后兼容

此脚本用于训练原始 PatchCore 模型，仅使用好品特征库进行异常检测。
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import patchcore
import patchcore.backbones as backbones
import patchcore.common as common
import patchcore.sampler as sampler

LOGGER = logging.getLogger(__name__)


class MVTecDataset(torch.utils.data.Dataset):
    """MVTec AD 数据集加载器"""

    def __init__(self, root, category, transform=None, target_transform=None):
        from pathlib import Path
        self.root = Path(root)
        self.category = category
        self.transform = transform
        self.target_transform = target_transform
        
        self.image_paths = []
        self.labels = []
        
        # 加载训练数据（好品）
        train_good_dir = self.root / category / "train" / "good"
        if train_good_dir.exists():
            for img_path in sorted(train_good_dir.glob("*.png")):
                self.image_paths.append(img_path)
                self.labels.append(0)  # 0 表示正常
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return {"image": image, "label": self.labels[idx]}


def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore 训练脚本")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="MVTec 数据集路径")
    parser.add_argument("--category", type=str, required=True,
                        help="MVTec 类别名称")
    
    # 模型参数
    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="骨干网络名称")
    parser.add_argument("--layers_to_extract_from", type=str, nargs="+",
                        default=["layer2", "layer3"],
                        help="特征提取层列表")
    parser.add_argument("--pretrain_embed_dimension", type=int, default=1536,
                        help="预训练嵌入维度")
    parser.add_argument("--target_embed_dimension", type=int, default=1536,
                        help="目标嵌入维度")
    parser.add_argument("--patchsize", type=int, default=3,
                        help="Patch 大小")
    
    # 采样参数
    parser.add_argument("--sampler", type=str, default="greedy_coreset",
                        choices=["identity", "greedy_coreset", "random"],
                        help="特征采样器类型")
    parser.add_argument("--sampler_percentage", type=float, default=0.1,
                        help="采样比例")
    
    # 推理参数
    parser.add_argument("--anomaly_score_num_nn", type=int, default=1,
                        help="用于计算异常分数的最近邻数量")
    
    # 输出参数
    parser.add_argument("--save_path", type=str, required=True,
                        help="模型保存路径")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载工作进程数")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载骨干网络
    LOGGER.info(f"加载骨干网络: {args.backbone}")
    backbone = backbones.load(args.backbone)
    backbone.name = args.backbone
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 加载训练数据
    LOGGER.info(f"加载训练数据: {args.data_path}/{args.category}")
    train_dataset = MVTecDataset(
        root=args.data_path,
        category=args.category,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # 创建 PatchCore 模型
    LOGGER.info("创建 PatchCore 模型...")
    patchcore_model = patchcore.PatchCore(device)
    
    # 配置模型
    nn_method = common.FaissNN(False, 4)
    
    # 配置采样器
    if args.sampler == "identity":
        featuresampler = sampler.IdentitySampler()
    elif args.sampler == "greedy_coreset":
        featuresampler = sampler.GreedyCoresetSampler(
            percentage=args.sampler_percentage,
            device=device,
        )
    elif args.sampler == "random":
        featuresampler = sampler.RandomSampler(percentage=args.sampler_percentage)
    else:
        featuresampler = sampler.IdentitySampler()
    
    # 加载模型配置
    patchcore_model.load(
        backbone=backbone,
        layers_to_extract_from=args.layers_to_extract_from,
        device=device,
        input_shape=(3, 224, 224),
        pretrain_embed_dimension=args.pretrain_embed_dimension,
        target_embed_dimension=args.target_embed_dimension,
        patchsize=args.patchsize,
        patchstride=1,
        anomaly_score_num_nn=args.anomaly_score_num_nn,
        featuresampler=featuresampler,
        nn_method=nn_method,
    )
    
    # 训练模型
    LOGGER.info("开始训练...")
    patchcore_model.fit(train_loader)
    
    # 保存模型
    LOGGER.info(f"保存模型到: {args.save_path}")
    patchcore_model.save_to_path(args.save_path)
    
    LOGGER.info("训练完成!")


if __name__ == "__main__":
    main()
