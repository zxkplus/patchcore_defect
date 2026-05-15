#!/usr/bin/env python3
"""PatchCore Defect 简易使用脚本

直接从文件夹读取图片：
- 好品图片文件夹
- 缺陷图片+mask文件夹  
- 测试图片文件夹

自动构建特征库并进行测试和可视化。
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
import matplotlib.pyplot as plt
import cv2

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import patchcore
import patchcore.backbones as backbones
import patchcore.common as common
import patchcore.sampler as sampler
from patchcore.patchcore import PatchCore


LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_images_from_folder(folder_path: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
    """从文件夹加载所有图片"""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")
    
    image_files = []
    for ext in extensions:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    if not image_files:
        raise ValueError(f"文件夹中没有找到图片: {folder_path}")
    
    LOGGER.info(f"从 {folder_path} 加载了 {len(image_files)} 张图片")
    return image_files


def load_image_pairs(defect_folder: str, mask_folder: str = None, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')):
    """加载缺陷图片和对应的mask
    
    如果mask_folder为None，则假设mask文件与图片文件在同一目录，文件名以_mask结尾
    """
    defect_path = Path(defect_folder)
    if not defect_path.exists():
        raise FileNotFoundError(f"缺陷图片文件夹不存在: {defect_folder}")
    
    # 获取所有缺陷图片
    defect_files = []
    for ext in extensions:
        defect_files.extend(defect_path.glob(f"*{ext}"))
        defect_files.extend(defect_path.glob(f"*{ext.upper()}"))
    
    defect_files = sorted(defect_files)
    if not defect_files:
        raise ValueError(f"缺陷图片文件夹中没有找到图片: {defect_folder}")
    
    # 加载对应的mask
    mask_files = []
    if mask_folder is None:
        # 假设mask在同一目录，文件名以_mask结尾
        for defect_file in defect_files:
            mask_file = defect_file.parent / f"{defect_file.stem}_mask{defect_file.suffix}"
            if mask_file.exists():
                mask_files.append(mask_file)
            else:
                # 尝试其他可能的mask命名方式
                mask_file_alt = defect_file.parent / f"{defect_file.stem}_gt{defect_file.suffix}"
                if mask_file_alt.exists():
                    mask_files.append(mask_file_alt)
                else:
                    raise FileNotFoundError(f"未找到对应的mask文件: {defect_file}")
    else:
        mask_path = Path(mask_folder)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask文件夹不存在: {mask_folder}")
        
        for defect_file in defect_files:
            mask_file = mask_path / defect_file.name
            if mask_file.exists():
                mask_files.append(mask_file)
            else:
                raise FileNotFoundError(f"未找到对应的mask文件: {mask_file}")
    
    LOGGER.info(f"加载了 {len(defect_files)} 对缺陷图片和mask")
    return defect_files, mask_files


def create_patchcore_model(
    backbone_name: str = "resnet50",
    layers: list = ["layer2", "layer3"],
    pretrain_embed_dim: int = 1536,
    target_embed_dim: int = 1536,
    patchsize: int = 3,
    device: str = "cuda"
):
    """创建PatchCore模型"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 创建骨干网络
    backbone = backbones.load(backbone_name)
    backbone.to(device)
    backbone.eval()
    
    # 创建特征提取器
    feature_extractor = PatchCore(device)
    feature_extractor.load(
        backbone=backbone,
        device=device,
        input_shape=(3, 224, 224),
        layers_to_extract_from=layers,
        pretrain_embed_dimension=pretrain_embed_dim,
        target_embed_dimension=target_embed_dim,
        patchsize=patchsize,
        featuresampler=sampler.GreedyCoresetSampler(percentage=0.1,device=device),
        anomaly_scorer_num_nn=1,
    )
    
    return feature_extractor, device


def process_images_to_features(model, image_files, device, batch_size=16):
    """将图片处理为特征"""
    def load_and_preprocess_image(image_path):
        """加载并预处理单张图片"""
        image = Image.open(image_path).convert("RGB")  # 改为RGB格式，保持3通道
        # 调整大小到224x224（标准输入尺寸）
        image = image.resize((224, 224))
        # 转换为tensor
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        return image_tensor
    
    all_features = []
    all_images = []
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_tensors = []
        
        for image_file in batch_files:
            img_tensor = load_and_preprocess_image(image_file)
            batch_tensors.append(img_tensor)
            all_images.append(Image.open(image_file).convert("RGB"))  # 改为RGB
        
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            features = model._embed(batch_tensor)
            all_features.extend(features)
    
    return all_features, all_images


def process_defect_data_to_features(model, defect_files, mask_files, device, batch_size=16):
    """将缺陷图片和mask处理为特征和mask"""
    def load_and_preprocess_defect_pair(defect_path, mask_path):
        """加载并预处理缺陷图片和mask"""
        # 加载缺陷图片
        defect_image = Image.open(defect_path).convert("RGB")  # 改为RGB格式，保持3通道
        defect_image = defect_image.resize((224, 224))
        defect_tensor = torch.tensor(np.array(defect_image)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        defect_tensor = (defect_tensor - mean) / std
        
        # 加载mask并转换为二值mask
        mask_image = Image.open(mask_path).convert("L")
        mask_image = mask_image.resize((224, 224))
        mask_array = np.array(mask_image)
        # 转换为二值mask (0或1)
        binary_mask = (mask_array > 127).astype(np.float32)
        
        return defect_tensor, binary_mask
    
    all_defect_features = []
    all_defect_masks = []
    all_defect_images = []
    
    for i in range(0, len(defect_files), batch_size):
        batch_defect_files = defect_files[i:i+batch_size]
        batch_mask_files = mask_files[i:i+batch_size]
        batch_defect_tensors = []
        batch_mask_arrays = []
        
        for defect_file, mask_file in zip(batch_defect_files, batch_mask_files):
            defect_tensor, mask_array = load_and_preprocess_defect_pair(defect_file, mask_file)
            batch_defect_tensors.append(defect_tensor)
            batch_mask_arrays.append(mask_array)
            all_defect_images.append(Image.open(defect_file).convert("RGB"))
        
        batch_defect_tensor = torch.stack(batch_defect_tensors).to(device)
        
        with torch.no_grad():
            features = model._embed(batch_defect_tensor)
            all_defect_features.extend(features)
            all_defect_masks.extend(batch_mask_arrays)
    
    return all_defect_features, all_defect_masks, all_defect_images


def visualize_results(test_images, anomaly_maps, defect_type_maps, output_dir, defect_types):
    """可视化结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, (test_img, anomaly_map, defect_type_map) in enumerate(zip(test_images, anomaly_maps, defect_type_maps)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图片
        axes[0].imshow(test_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 异常分割图
        if anomaly_map is not None:
            anomaly_map_resized = cv2.resize(anomaly_map, (224, 224))
            axes[1].imshow(anomaly_map_resized, cmap='hot')
            axes[1].set_title("Anomaly Map")
        else:
            axes[1].imshow(np.zeros((224, 224)), cmap='hot')
            axes[1].set_title("Anomaly Map (None)")
        axes[1].axis('off')
        
        # 缺陷类型图
        if defect_type_map is not None:
            defect_type_map_resized = cv2.resize(defect_type_map.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
            # 创建颜色映射
            colored_map = np.zeros((224, 224, 3), dtype=np.uint8)
            for defect_id, defect_name in enumerate(defect_types):
                if defect_id == 0:  # 正常类别用绿色
                    color = [0, 255, 0]
                elif defect_id == 1:  # 第一种缺陷用红色
                    color = [255, 0, 0]
                elif defect_id == 2:  # 第二种缺陷用蓝色
                    color = [0, 0, 255]
                else:  # 其他缺陷用黄色
                    color = [255, 255, 0]
                mask = (defect_type_map_resized == defect_id)
                colored_map[mask] = color
            axes[2].imshow(colored_map)
            axes[2].set_title("Defect Type Map")
        else:
            axes[2].imshow(np.zeros((224, 224, 3), dtype=np.uint8))
            axes[2].set_title("Defect Type Map (None)")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / f"result_{i:03d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        LOGGER.info(f"保存结果图片: result_{i:03d}.png")


def main():
    parser = argparse.ArgumentParser(description="PatchCore Defect 简易使用脚本")
    
    # 输入目录参数
    parser.add_argument("--good_dir", type=str, required=True,
                        help="好品图片文件夹路径")
    parser.add_argument("--defect_dir", type=str, required=True,
                        help="缺陷图片文件夹路径")
    parser.add_argument("--defect_mask_dir", type=str, default=None,
                        help="缺陷mask文件夹路径（可选，如果为None则自动查找_mask文件）")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="测试图片文件夹路径")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="结果输出目录")
    
    # 模型参数
    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="骨干网络名称")
    parser.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"],
                        help="特征提取层")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    
    # 缺陷类型
    parser.add_argument("--defect_types", type=str, nargs="+", default=["defect"],
                        help="缺陷类型列表")
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        # 创建输出目录
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建模型
        LOGGER.info("创建PatchCore模型...")
        model, device = create_patchcore_model(
            backbone_name=args.backbone,
            layers=args.layers,
            device=args.device
        )
        
        # 加载好品图片
        LOGGER.info("加载好品图片...")
        good_files = load_images_from_folder(args.good_dir)
        good_features, _ = process_images_to_features(model, good_files, device, args.batch_size)
        
        # 构建好品特征库
        LOGGER.info("构建好品特征库...")
        model.anomaly_scorer.fit(good_features)
        
        # 加载缺陷数据
        LOGGER.info("加载缺陷数据...")
        defect_files, mask_files = load_image_pairs(args.defect_dir, args.defect_mask_dir)
        defect_features, defect_masks, _ = process_defect_data_to_features(
            model, defect_files, mask_files, device, args.batch_size
        )
        
        # 构建缺陷特征库
        LOGGER.info("构建缺陷特征库...")
        from patchcore.defect_library import DefectFeatureLibrary
        
        defect_library = DefectFeatureLibrary(feature_dimension=model.target_embed_dimension)
        
        # 准备缺陷类型列表（重复每个样本的缺陷类型）
        defect_type_list = []
        for i in range(len(defect_features)):
            defect_type_list.append(args.defect_types[i % len(args.defect_types)])
        
        defect_library.build_from_defect_data(
            features=np.array([f.cpu().numpy() for f in defect_features]),
            masks=np.array(defect_masks),
            defect_types=defect_type_list,
            patch_shapes=(7, 7),  # 根据实际特征图大小调整
            target_image_size=(224, 224),
            sampling_ratio=1.0
        )
        
        # 创建双库评分器
        from patchcore.dual_scorer import DualLibraryScorer
        
        dual_scorer = DualLibraryScorer(
            normal_scorer=model.anomaly_scorer,
            defect_library=defect_library,
            classification_strategy="distance_ratio"
        )
        
        # 加载测试图片
        LOGGER.info("加载测试图片...")
        test_files = load_images_from_folder(args.test_dir)
        test_features, test_images = process_images_to_features(model, test_files, device, args.batch_size)
        
        # 进行预测
        LOGGER.info("进行预测...")
        image_scores, segmentation_maps, defect_type_maps = dual_scorer.predict(
            query_features=[np.array([f.cpu().numpy() for f in test_features])],
            patch_shapes=[(7, 7)],
            target_size=(224, 224)
        )
        
        # 可视化结果
        LOGGER.info("可视化结果...")
        visualize_results(
            test_images=test_images,
            anomaly_maps=segmentation_maps,
            defect_type_maps=defect_type_maps,
            output_dir=args.output_dir,
            defect_types=["normal"] + args.defect_types
        )
        
        LOGGER.info(f"处理完成！结果保存在: {args.output_dir}")
        
    except Exception as e:
        LOGGER.error(f"处理过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()