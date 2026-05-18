#!/usr/bin/env python3
"""原始 PatchCore 训练和推理脚本 - 保留向后兼容

此脚本用于训练原始 PatchCore 模型，仅使用好品特征库进行异常检测。
支持训练和推理两种模式。
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import patchcore
import patchcore.backbones as backbones
import patchcore.common as common
import patchcore.sampler as sampler
import patchcore.datasets as datasets

LOGGER = logging.getLogger(__name__)


class MVTecDataset(torch.utils.data.Dataset):
    """MVTec AD 数据集加载器"""

    def __init__(self, root, category, transform=None, target_transform=None, split="train"):
        from pathlib import Path
        self.root = Path(root)
        self.category = category
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        
        if split == "train":
            # 加载训练数据（好品）
            train_good_dir = self.root / category / "train" / "good"
            if train_good_dir.exists():
                for img_path in sorted(train_good_dir.glob("*.png")):
                    self.image_paths.append(img_path)
                    self.labels.append(0)  # 0 表示正常
                    self.mask_paths.append(None)
        else:
            # 加载测试数据（包含正常和异常）
            test_dir = self.root / category / "test"
            if test_dir.exists():
                # 正常样本
                good_dir = test_dir / "good"
                if good_dir.exists():
                    for img_path in sorted(good_dir.glob("*.png")):
                        self.image_paths.append(img_path)
                        self.labels.append(0)
                        self.mask_paths.append(None)
                
                # 异常样本
                for defect_type in test_dir.iterdir():
                    if defect_type.is_dir() and defect_type.name != "good":
                        for img_path in sorted(defect_type.glob("*.png")):
                            self.image_paths.append(img_path)
                            self.labels.append(1)
                            # 对应的mask路径
                            mask_path = self.root / category / "ground_truth" / defect_type.name / img_path.name
                            self.mask_paths.append(mask_path if mask_path.exists() else None)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        item = {"image": image, "label": self.labels[idx], "filename": img_path.name}
        
        # 只在测试模式下添加mask信息
        if self.split != "train":
            if self.mask_paths[idx] is not None:
                mask = Image.open(self.mask_paths[idx]).convert("L")
                if self.target_transform:
                    mask = self.target_transform(mask)
                item["mask"] = mask
            else:
                # 如果没有mask，创建全零mask张量
                from torchvision import transforms
                mask = torch.zeros((1, 224, 224), dtype=torch.float32)
                item["mask"] = mask
        
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore 训练和推理脚本")
    
    # 操作模式
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train",
                        help="操作模式：train（训练）或 inference（推理）")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, 
                        help="MVTec 数据集路径（批量推理时必需）")
    parser.add_argument("--category", type=str, 
                        help="MVTec 类别名称（批量推理时必需）")
    parser.add_argument("--image_path", type=str,
                        help="单张图片路径（单张图片推理时必需）")
    
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
    parser.add_argument("--sampler", type=str, default="approximate_greedy_coreset",
                        choices=["identity", "greedy_coreset", "approximate_greedy_coreset", "random"],
                        help="特征采样器类型")
    parser.add_argument("--sampler_percentage", type=float, default=0.1,
                        help="采样比例")
    
    # 推理参数
    parser.add_argument("--anomaly_score_num_nn", type=int, default=1,
                        help="用于计算异常分数的最近邻数量")
    
    # 路径参数
    parser.add_argument("--save_path", type=str, 
                        help="模型保存路径（训练模式必需）")
    parser.add_argument("--load_path", type=str, required=True,
                        help="模型加载路径（推理模式必需）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="推理结果输出目录（推理模式可选）")
    parser.add_argument("--save_visualization", action="store_true",
                        help="保存可视化结果")
    
    # 运行参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载工作进程数")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    
    return parser.parse_args()


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def train_mode(args):
    """训练模式"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"使用设备: {device}")
    
    # 验证必要参数
    if not args.save_path:
        LOGGER.error("--save_path 是训练模式的必需参数")
        sys.exit(1)
    
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
        split="train",
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
    elif args.sampler == "approximate_greedy_coreset":
        featuresampler = sampler.ApproximateGreedyCoresetSampler(
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


def inference_mode(args):
    """推理模式"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"使用设备: {device}")
    
    # 验证必要参数
    if not args.load_path:
        LOGGER.error("--load_path 是推理模式的必需参数")
        sys.exit(1)
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    LOGGER.info(f"从 {args.load_path} 加载预训练模型")
    nn_method = common.FaissNN(False, 4)
    
    patchcore_model = patchcore.PatchCore(device)
    patchcore_model.load_from_path(
        load_path=args.load_path,
        device=device,
        nn_method=nn_method,
        load_defect_library=False,  # 原始PatchCore不需要缺陷库
    )
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    
    # 加载测试数据
    LOGGER.info(f"加载测试数据: {args.data_path}/{args.category}")
    test_dataset = MVTecDataset(
        root=args.data_path,
        category=args.category,
        transform=transform,
        target_transform=mask_transform,
        split="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=datasets.custom_collate_fn,
    )
    
    # 推理
    LOGGER.info("开始推理...")
    scores, masks, labels_gt, masks_gt, defect_types_pred = patchcore_model.predict(test_loader)
    
    # 输出结果
    LOGGER.info(f"推理完成，共 {len(scores)} 个样本")
    
    # 保存结果
    if args.output_dir:
        save_inference_results(
            test_dataset=test_loader.dataset,
            scores=scores,
            masks=masks,
            labels_gt=labels_gt,
            masks_gt=masks_gt,
            output_dir=args.output_dir,
        )
        
        if args.save_visualization:
            visualize_inference_results(
                test_dataset=test_loader.dataset,
                scores=scores,
                masks=masks,
                output_dir=args.output_dir,
            )
    
    # 打印统计信息
    normal_scores = [scores[i] for i in range(len(scores)) if labels_gt[i] == 0]
    anomaly_scores = [scores[i] for i in range(len(scores)) if labels_gt[i] == 1]
    
    if normal_scores:
        LOGGER.info(f"正常样本平均分数: {np.mean(normal_scores):.4f} ± {np.std(normal_scores):.4f}")
    if anomaly_scores:
        LOGGER.info(f"异常样本平均分数: {np.mean(anomaly_scores):.4f} ± {np.std(anomaly_scores):.4f}")


def single_image_inference(args):
    """单张图片推理模式"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"使用设备: {device}")
    
    # 验证必要参数
    if not args.image_path:
        LOGGER.error("--image_path 是单张图片推理的必需参数")
        sys.exit(1)
    
    if not os.path.exists(args.image_path):
        LOGGER.error(f"图片路径不存在: {args.image_path}")
        sys.exit(1)
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    LOGGER.info(f"从 {args.load_path} 加载预训练模型")
    nn_method = common.FaissNN(False, 4)
    
    patchcore_model = patchcore.PatchCore(device)
    patchcore_model.load_from_path(
        load_path=args.load_path,
        device=device,
        nn_method=nn_method,
        load_defect_library=False,  # 原始PatchCore不需要缺陷库
    )
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 加载单张图片
    LOGGER.info(f"加载图片: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    # 推理
    LOGGER.info("开始推理...")
    scores, masks, defect_types_pred = patchcore_model._predict(image_tensor)
    
    # 输出结果
    score = scores[0]
    mask = masks[0]
    
    LOGGER.info(f"推理完成!")
    LOGGER.info(f"异常分数: {score:.4f}")
    
    # 保存结果
    if args.output_dir:
        save_single_image_result(
            image_path=args.image_path,
            score=score,
            mask=mask,
            output_dir=args.output_dir,
        )
        
        if args.save_visualization:
            visualize_single_image_result(
                original_image=image,
                score=score,
                mask=mask,
                output_dir=args.output_dir,
                filename=os.path.basename(args.image_path),
            )
    
    return score, mask


def save_inference_results(test_dataset, scores, masks, labels_gt, masks_gt, output_dir):
    """保存推理结果"""
    output_path = Path(output_dir)
    
    # 保存汇总
    summary_path = output_path / "inference_summary.txt"
    with open(summary_path, "w") as f:
        f.write("PatchCore Inference Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Samples: {len(scores)}\n")
        
        anomalies = sum(labels_gt)
        f.write(f"Anomalies: {anomalies}\n")
        f.write(f"Normal: {len(scores) - anomalies}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(scores)):
            sample = test_dataset[i]
            filename = sample.get("filename", f"sample_{i}")
            label = labels_gt[i]
            
            f.write(f"{filename}: score={scores[i]:.4f}, label={label}\n")
    
    LOGGER.info(f"推理结果汇总已保存到: {summary_path}")


def visualize_inference_results(test_dataset, scores, masks, output_dir):
    """保存可视化结果"""
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    vis_dir = output_path / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    for i in range(len(scores)):
        sample = test_dataset[i]
        
        # 获取图片
        img = sample["image"]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
            # 反归一化
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
        
        # 获取分割图
        mask = masks[i]
        
        # 创建图像
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title(f"Input Image\nScore: {scores[i]:.4f}")
        axes[0].axis("off")
        
        axes[1].imshow(mask, cmap="hot")
        axes[1].set_title("Anomaly Map")
        axes[1].axis("off")
        
        # 显示真实标签
        label = sample.get("label", 0)
        label_text = "Normal" if label == 0 else "Anomaly"
        axes[2].text(0.5, 0.5, f"Ground Truth:\n{label_text}",
                    ha="center", va="center", fontsize=16,
                    transform=axes[2].transAxes)
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # 保存
        filename = sample.get("filename", f"result_{i}.png")
        save_path = vis_dir / filename.replace(".png", "_result.png")
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    
    LOGGER.info(f"可视化结果已保存到: {vis_dir}")


def save_single_image_result(image_path, score, mask, output_dir):
    """保存单张图片推理结果"""
    output_path = Path(output_dir)
    
    # 保存汇总
    summary_path = output_path / "single_image_result.txt"
    with open(summary_path, "w") as f:
        f.write("PatchCore Single Image Inference Result\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Anomaly Score: {score:.4f}\n")
    
    # 保存分割图
    import matplotlib.pyplot as plt
    mask_path = output_path / "anomaly_mask.png"
    plt.imsave(str(mask_path), mask, cmap="hot")
    
    LOGGER.info(f"单张图片结果已保存到: {summary_path}")
    LOGGER.info(f"分割图已保存到: {mask_path}")


def visualize_single_image_result(original_image, score, mask, output_dir, filename):
    """保存单张图片可视化结果"""
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    vis_dir = output_path / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # 创建图像
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title(f"Input Image\nScore: {score:.4f}")
    axes[0].axis("off")
    
    axes[1].imshow(mask, cmap="hot")
    axes[1].set_title("Anomaly Map")
    axes[1].axis("off")
    
    plt.tight_layout()
    
    # 保存
    save_path = vis_dir / filename.replace(".png", "_result.png").replace(".jpg", "_result.jpg")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    LOGGER.info(f"单张图片可视化结果已保存到: {save_path}")


def main():
    args = parse_args()
    setup_logging()
    
    if args.mode == "train":
        # 验证训练模式必需参数
        if not args.save_path:
            LOGGER.error("训练模式需要指定 --save_path 参数")
            sys.exit(1)
        train_mode(args)
    elif args.mode == "inference":
        # 验证推理模式必需参数
        if not args.load_path:
            LOGGER.error("推理模式需要指定 --load_path 参数")
            sys.exit(1)
        
        # 判断是单张图片推理还是批量推理
        if args.image_path:
            # 单张图片推理
            single_image_inference(args)
        else:
            # 批量推理（原有逻辑）
            if not args.data_path or not args.category:
                LOGGER.error("批量推理需要指定 --data_path 和 --category 参数")
                sys.exit(1)
            inference_mode(args)
    else:
        LOGGER.error(f"未知模式: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()