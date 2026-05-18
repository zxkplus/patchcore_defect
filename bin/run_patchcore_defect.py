#!/usr/bin/env python3
"""带缺陷库的 PatchCore 训练和推理脚本

支持：
1. train_defect: 先构建好品库，再构建缺陷库
2. evaluate_defect: 加载模型，推理，输出分割+分类结果
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patchcore import PatchCore
import patchcore.backbones as backbones
import patchcore.common as common
import patchcore.sampler as sampler
import patchcore.datasets as datasets

LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="PatchCore Defect 训练和推理脚本"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # train_defect 子命令
    train_parser = subparsers.add_parser("train_defect", help="训练缺陷检测模型")
    
    # 数据参数
    train_parser.add_argument("--data_path", type=str, required=True,
                              help="数据集根目录（支持 MVTec 格式）")
    train_parser.add_argument("--category", type=str, default=None,
                              help="MVTec 类别名称（可选）")
    
    # 好品数据
    train_parser.add_argument("--good_train_dir", type=str, default=None,
                              help="好品训练图片目录")
    train_parser.add_argument("--good_mask_dir", type=str, default=None,
                              help="好品 mask 目录（可选）")
    
    # 缺陷数据
    train_parser.add_argument("--defect_data", type=str, nargs="+", default=None,
                              help="缺陷数据目录列表，格式: type:path")
    train_parser.add_argument("--defect_types", type=str, nargs="+",
                              default=None,
                              help="缺陷类型列表")
    
    # 模型参数
    train_parser.add_argument("--backbone", type=str, default="resnet50",
                              help="骨干网络名称")
    train_parser.add_argument("--layers", type=str, nargs="+",
                              default=["layer2", "layer3"],
                              help="特征提取层列表")
    train_parser.add_argument("--pretrain_embed_dim", type=int, default=1536,
                              help="预训练嵌入维度")
    train_parser.add_argument("--target_embed_dim", type=int, default=1536,
                              help="目标嵌入维度")
    train_parser.add_argument("--patchsize", type=int, default=3,
                              help="Patch 大小")
    
    # 采样参数
    train_parser.add_argument("--sampler", type=str, default="approximate_greedy_coreset",
                              choices=["identity", "greedy_coreset", "approximate_greedy_coreset", "random"],
                              help="特征采样器类型")
    train_parser.add_argument("--sampler_percentage", type=float, default=0.1,
                              help="采样比例")
    
    # 分类策略
    train_parser.add_argument("--classification_strategy", type=str,
                              default="distance_ratio",
                              choices=["nearest_wins", "distance_ratio", "softmax_prob"],
                              help="分类策略")
    train_parser.add_argument("--threshold_percentile", type=float, default=95.0,
                              help="异常阈值百分位数")
    train_parser.add_argument("--sampling_ratio", type=float, default=1.0,
                              help="缺陷特征采样比例")
    
    # 输出参数
    train_parser.add_argument("--save_path", type=str, required=True,
                              help="模型保存路径")
    train_parser.add_argument("--batch_size", type=int, default=2,
                              help="批次大小")
    train_parser.add_argument("--num_workers", type=int, default=2,
                              help="数据加载工作进程数")
    
    # 设备参数
    train_parser.add_argument("--device", type=str, default="cuda",
                              help="计算设备")
    train_parser.add_argument("--verbose", action="store_true",
                              help="详细输出")
    
    # evaluate_defect 子命令
    eval_parser = subparsers.add_parser("evaluate_defect", help="评估缺陷检测模型")
    
    # 模型参数
    eval_parser.add_argument("--model_path", type=str, required=True,
                             help="模型路径")
    
    # 数据参数
    eval_parser.add_argument("--test_data", type=str, required=True,
                             help="测试数据目录")
    eval_parser.add_argument("--defect_types", type=str, nargs="+",
                             default=None,
                             help="缺陷类型列表")
    
    # 输出参数
    eval_parser.add_argument("--output_dir", type=str, default=None,
                              help="结果输出目录")
    eval_parser.add_argument("--save_visualization", action="store_true",
                             help="保存可视化结果")
    eval_parser.add_argument("--batch_size", type=int, default=2,
                             help="批次大小")
    eval_parser.add_argument("--num_workers", type=int, default=2,
                             help="数据加载工作进程数")
    
    # 设备参数
    eval_parser.add_argument("--device", type=str, default="cuda",
                             help="计算设备")
    eval_parser.add_argument("--verbose", action="store_true",
                             help="详细输出")
    
    return parser.parse_args()


def train_defect_command(args):
    """训练缺陷检测模型"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 加载骨干网络
    LOGGER.info(f"加载骨干网络: {args.backbone}")
    backbone = backbones.load(args.backbone)
    backbone.name = args.backbone
    
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
    
    # 创建 PatchCore 模型
    LOGGER.info("创建 PatchCore 模型...")
    nn_method = common.FaissNN(False, 4)
    
    patchcore_model = PatchCore(device)
    patchcore_model.load(
        backbone=backbone,
        layers_to_extract_from=args.layers,
        device=device,
        input_shape=(3, 224, 224),
        pretrain_embed_dimension=args.pretrain_embed_dim,
        target_embed_dimension=args.target_embed_dim,
        patchsize=args.patchsize,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=featuresampler,
        nn_method=nn_method,
        classification_strategy=args.classification_strategy,
    )
    
    # 准备数据加载器
    if args.category:
        # MVTec 格式
        data_path = Path(args.data_path)
        
        # 好品训练数据
        good_train_dir = data_path / args.category / "train" / "good"
        if good_train_dir.exists():
            LOGGER.info(f"加载好品训练数据: {good_train_dir}")
            good_dataset = datasets.MVTecDefectDataset(
                root_dir=str(data_path / args.category),
                defect_types=[],  # 不加载缺陷数据
                split="train",
                include_good=True,
            )
            
            good_loader = torch.utils.data.DataLoader(
                good_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=datasets.custom_collate_fn,
            )
            
            # 训练好品库
            LOGGER.info("构建好品特征库...")
            patchcore_model.fit(good_loader)
        
        # 缺陷数据
        if args.defect_types:
            LOGGER.info(f"加载缺陷训练数据...")
            
            # 创建缺陷数据集
            defect_samples = []
            defect_dir = data_path / args.category / "defect"
            
            for defect_type in args.defect_types:
                type_dir = defect_dir / defect_type
                if not type_dir.exists():
                    LOGGER.warning(f"缺陷目录不存在: {type_dir}")
                    continue
                
                images_dir = type_dir / "images"
                masks_dir = type_dir / "masks"
                
                if not images_dir.exists():
                    LOGGER.warning(f"缺陷图片目录不存在: {images_dir}")
                    continue
                
                LOGGER.info(f"  {defect_type}: {images_dir}")
                
                defect_dataset = datasets.DefectDataset(
                    image_dir=str(images_dir),
                    mask_dir=str(masks_dir) if masks_dir.exists() else None,
                    defect_type=defect_type,
                    image_size=(224, 224),
                )
                defect_samples.extend(defect_dataset)
            
            if defect_samples:
                defect_loader = torch.utils.data.DataLoader(
                    defect_samples,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    collate_fn=datasets.custom_collate_fn,
                )
                
                # 构建缺陷库
                LOGGER.info("构建缺陷特征库...")
                patchcore_model.fit_defect_library(
                    defect_loader,
                    classification_strategy=args.classification_strategy,
                    sampling_ratio=args.sampling_ratio,
                    percentile=args.threshold_percentile,
                )
    else:
        LOGGER.error("需要指定 --category 或提供 --good_train_dir")
        return
    
    # 保存模型
    LOGGER.info(f"保存模型到: {args.save_path}")
    patchcore_model.save_to_path(args.save_path)
    
    LOGGER.info("训练完成!")


def evaluate_defect_command(args):
    """评估缺陷检测模型"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"使用设备: {device}")
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    LOGGER.info(f"加载模型: {args.model_path}")
    nn_method = common.FaissNN(False, 4)
    
    patchcore_model = PatchCore(device)
    patchcore_model.load_from_path(
        load_path=args.model_path,
        device=device,
        nn_method=nn_method,
        load_defect_library=True,
    )
    
    # 准备测试数据
    test_data_path = Path(args.test_data)
    
    if (test_data_path / "test").exists():
        # MVTec 格式
        test_dataset = datasets.MVTecDefectDataset(
            root_dir=str(test_data_path),
            split="test",
            include_good=True,
        )
    else:
        # 通用格式
        test_dataset = datasets.DefectDataset(
            image_dir=str(test_data_path),
            defect_type="unknown",
            image_size=(224, 224),
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=datasets.custom_collate_fn,
    )
    
    # 推理
    LOGGER.info("开始推理...")
    scores, masks, labels_gt, masks_gt, defect_types_pred = patchcore_model.predict(
        test_loader
    )
    
    # 输出结果
    LOGGER.info(f"推理完成，共 {len(scores)} 个样本")
    
    # 统计结果
    if hasattr(patchcore_model, 'defect_library') and patchcore_model.defect_library is not None:
        defect_types = patchcore_model.defect_library.get_defect_types()
    else:
        defect_types = []
    LOGGER.info(f"检测到的缺陷类型: {defect_types}")
    
    # 保存可视化结果
    if args.save_visualization and args.output_dir:
        LOGGER.info("保存可视化结果...")
        visualize_results(
            test_dataset=test_loader.dataset if hasattr(test_loader, 'dataset') else test_dataset,
            scores=scores,
            masks=masks,
            defect_types_pred=defect_types_pred,
            output_dir=args.output_dir,
            defect_library=patchcore_model.defect_library,
        )
    
    # 保存文本结果
    if args.output_dir:
        save_results(
            test_dataset=test_loader.dataset if hasattr(test_loader, 'dataset') else test_dataset,
            scores=scores,
            masks=masks,
            labels_gt=labels_gt,
            defect_types_pred=defect_types_pred,
            output_dir=args.output_dir,
        )


def visualize_results(test_dataset, scores, masks, defect_types_pred, output_dir, defect_library):
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
        
        # 获取预测的缺陷类型
        defect_type = defect_types_pred[i]
        defect_type_name = "Normal"
        # 处理defect_type可能是数组的情况
        if defect_library and defect_type is not None:
            if isinstance(defect_type, np.ndarray):
                # 如果是数组，找到最频繁的非零缺陷类型
                non_zero_types = defect_type[defect_type > 0]
                if len(non_zero_types) > 0:
                    # 找到出现次数最多的缺陷类型
                    unique_types, counts = np.unique(non_zero_types, return_counts=True)
                    most_frequent_type = unique_types[np.argmax(counts)]
                    defect_type_name = defect_library.defect_id_to_type.get(most_frequent_type, f"Defect_{most_frequent_type}")
            elif defect_type > 0:  # 如果是标量
                defect_type_name = defect_library.defect_id_to_type.get(defect_type, f"Defect_{defect_type}")
        
        # 创建图像
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title(f"Input Image\nScore: {scores[i]:.4f}")
        axes[0].axis("off")
        
        axes[1].imshow(mask, cmap="hot")
        axes[1].set_title("Anomaly Map")
        axes[1].axis("off")
        
        axes[2].text(0.5, 0.5, f"Predicted:\n{defect_type_name}",
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


def save_results(test_dataset, scores, masks, labels_gt, defect_types_pred, output_dir):
    """保存文本结果"""
    output_path = Path(output_dir)
    
    # 保存汇总
    summary_path = output_path / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("PatchCore Defect Detection Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Samples: {len(scores)}\n")
        
        if labels_gt:
            anomalies = sum(labels_gt)
            f.write(f"Anomalies: {anomalies}\n")
            f.write(f"Normal: {len(scores) - anomalies}\n\n")
        
        f.write("Detailed Results:\n")
        f.write("-" * 50 + "\n")
        
        for i in range(len(scores)):
            sample = test_dataset[i]
            filename = sample.get("filename", f"sample_{i}")
            label = labels_gt[i] if i < len(labels_gt) else "N/A"
            defect_type = defect_types_pred[i] if i < len(defect_types_pred) else 0
            
            f.write(f"{filename}: score={scores[i]:.4f}, "
                   f"label={label}, defect_type={defect_type}\n")
    
    LOGGER.info(f"结果汇总已保存到: {summary_path}")


def main():
    args = parse_args()
    setup_logging(args.verbose if hasattr(args, 'verbose') else False)
    
    if args.command == "train_defect":
        train_defect_command(args)
    elif args.command == "evaluate_defect":
        evaluate_defect_command(args)
    else:
        LOGGER.error("请指定命令: train_defect 或 evaluate_defect")
        sys.exit(1)


if __name__ == "__main__":
    main()