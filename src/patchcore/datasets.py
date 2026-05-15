"""缺陷数据集模块 - 用于加载缺陷检测数据"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

LOGGER = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """自定义 collate 函数，用于处理包含 None 值的批次
    
    当数据集中某些字段（如 mask）可能为 None 时，此函数将这些字段作为列表保留，
    而不是尝试将它们堆叠成 tensor。
    
    Args:
        batch: 包含字典的批次列表
        
    Returns:
        处理后的批次字典
    """
    # 提取所有键
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # 检查是否所有值都是 None
        if all(v is None for v in values):
            result[key] = None
        # 检查是否所有值都不是 None 且是 tensor
        elif all(v is not None for v in values) and all(isinstance(v, torch.Tensor) for v in values):
            try:
                result[key] = torch.stack(values)
            except RuntimeError:
                # 如果无法堆叠，保持为列表
                result[key] = values
        else:
            # 包含 None 值或其他类型，保持为列表
            result[key] = values
    
    return result


class DefectDataset(Dataset):
    """缺陷数据集类
    
    用于加载缺陷图片、mask和缺陷类型信息。
    
    Attributes:
        image_dir: 图片目录路径
        mask_dir: mask目录路径（可选）
        defect_type: 缺陷类型名称
        image_transform: 图片预处理变换
        mask_transform: mask预处理变换
        filenames: 图片文件名列表
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        defect_type: str = "unknown",
        image_size: Tuple[int, int] = (224, 224),
    ):
        """初始化缺陷数据集
        
        Args:
            image_dir: 图片目录路径
            mask_dir: mask目录路径（可选）
            defect_type: 缺陷类型名称
            image_size: 图片大小
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.defect_type = defect_type
        self.image_size = image_size
        
        # 设置图片变换
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 设置mask变换
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
        # 收集图片文件
        self.filenames = self._collect_images()
        
        LOGGER.info(
            f"初始化缺陷数据集: {image_dir}, "
            f"找到 {len(self.filenames)} 张图片, 类型: {defect_type}"
        )
    
    def _collect_images(self) -> List[str]:
        """收集图片目录中的所有图片文件"""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        
        if not self.image_dir.exists():
            LOGGER.warning(f"图片目录不存在: {self.image_dir}")
            return []
        
        filenames = []
        for f in sorted(self.image_dir.iterdir()):
            if f.suffix.lower() in valid_extensions:
                filenames.append(f.name)
        
        return filenames
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据集中的一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含以下键的字典:
                - image: 图片tensor
                - mask: mask numpy数组（如果有）
                - defect_type: 缺陷类型名称
                - filename: 文件名
        """
        filename = self.filenames[idx]
        
        # 加载图片
        image_path = self.image_dir / filename
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        
        # 加载mask（如果有）
        mask = None
        if self.mask_dir is not None and self.mask_dir.exists():
            # 尝试多种mask文件名格式
            mask_path = None
            stem = Path(filename).stem
            
            possible_masks = [
                self.mask_dir / filename,
                self.mask_dir / f"{stem}.png",
                self.mask_dir / f"{stem}_mask.png",
                self.mask_dir / f"{stem}_mask.jpg",
            ]
            
            for mp in possible_masks:
                if mp.exists():
                    mask_path = mp
                    break
            
            if mask_path is None:
                LOGGER.warning(f"未找到mask文件: {filename}")
            else:
                mask_img = Image.open(mask_path).convert("L")
                mask_tensor = self.mask_transform(mask_img)
                # 二值化处理 (>0.5 -> 1)
                mask = (mask_tensor > 0.5).float().numpy().squeeze()
        
        return {
            "image": image_tensor,
            "mask": mask,
            "defect_type": self.defect_type,
            "filename": filename,
        }


class MVTecDefectDataset(Dataset):
    """MVTec AD 格式的缺陷数据集
    
    支持 MVTec AD 数据集的目录结构。
    
    目录结构:
    ├── train/
    │   └── good/              # 好品图片
    ├── defect/                # 缺陷数据（用户自定义目录结构）
    │   ├── crack/
    │   │   ├── images/
    │   │   └── masks/
    │   └── scratch/
    │       ├── images/
    │       └── masks/
    └── test/
        ├── good/
        └── defect/
    
    Attributes:
        root_dir: 数据集根目录
        defect_types: 缺陷类型列表
        split: 数据集划分（train/test）
    """
    
    def __init__(
        self,
        root_dir: str,
        defect_types: Optional[List[str]] = None,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        include_good: bool = True,
    ):
        """初始化 MVTec AD 格式数据集
        
        Args:
            root_dir: 数据集根目录
            defect_types: 缺陷类型列表（如果为None，自动从目录结构检测）
            split: 数据集划分，可选 "train" 或 "test"
            image_size: 图片大小
            include_good: 是否包含好品类别
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.split = split
        
        # 设置图片变换
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 设置mask变换（使用nearest插值保持二值性）
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
        # 检测缺陷类型
        if defect_types is None:
            defect_types = self._detect_defect_types()
        
        self.defect_types = defect_types
        
        # 构建数据集
        self.samples = self._build_dataset(include_good)
        
        LOGGER.info(
            f"初始化 MVTecDefectDataset: {root_dir}, "
            f"划分: {split}, 样本数: {len(self.samples)}"
        )
    
    def _detect_defect_types(self) -> List[str]:
        """自动检测缺陷类型"""
        defect_dir = self.root_dir / "defect"
        
        if not defect_dir.exists():
            LOGGER.warning(f"缺陷目录不存在: {defect_dir}")
            return []
        
        defect_types = []
        for d in sorted(defect_dir.iterdir()):
            if d.is_dir():
                defect_types.append(d.name)
        
        LOGGER.info(f"自动检测到缺陷类型: {defect_types}")
        return defect_types
    
    def _build_dataset(self, include_good: bool) -> List[Dict]:
        """构建数据集样本列表"""
        samples = []
        
        if self.split == "train":
            # 训练集：好品图片
            if include_good:
                good_dir = self.root_dir / "train" / "good"
                if good_dir.exists():
                    for img_file in sorted(good_dir.glob("*")):
                        if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                            samples.append({
                                "image_path": str(img_file),
                                "mask_path": None,
                                "defect_type": "good",
                                "is_anomaly": 0,
                            })
                LOGGER.info(f"好品样本数: {len([s for s in samples if s['defect_type'] == 'good'])}")
            
            # 训练集：缺陷图片（用于构建缺陷库）
            defect_dir = self.root_dir / "defect"
            if defect_dir.exists():
                for defect_type in self.defect_types:
                    defect_type_dir = defect_dir / defect_type
                    if not defect_type_dir.exists():
                        continue
                    
                    images_dir = defect_type_dir / "images"
                    masks_dir = defect_type_dir / "masks"
                    
                    if not images_dir.exists():
                        continue
                    
                    for img_file in sorted(images_dir.glob("*")):
                        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                            continue
                        
                        # 查找对应的mask
                        mask_path = None
                        stem = img_file.stem
                        if masks_dir.exists():
                            for ext in [".png", "_mask.png", "_mask.jpg"]:
                                potential_mask = masks_dir / f"{stem}{ext}"
                                if potential_mask.exists():
                                    mask_path = str(potential_mask)
                                    break
                        
                        samples.append({
                            "image_path": str(img_file),
                            "mask_path": mask_path,
                            "defect_type": defect_type,
                            "is_anomaly": 1,
                        })
                
                defect_sample_count = len([s for s in samples if s["defect_type"] != "good"])
                LOGGER.info(f"缺陷样本数: {defect_sample_count}")
        
        else:
            # 测试集
            test_dir = self.root_dir / "test"
            
            # 好品测试
            if include_good:
                good_dir = test_dir / "good"
                if good_dir.exists():
                    for img_file in sorted(good_dir.glob("*")):
                        if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                            samples.append({
                                "image_path": str(img_file),
                                "mask_path": None,
                                "defect_type": "good",
                                "is_anomaly": 0,
                            })
            
            # 缺陷测试
            for defect_type in self.defect_types:
                defect_type_dir = test_dir / "defect" / defect_type
                if not defect_type_dir.exists():
                    continue
                
                for img_file in sorted(defect_type_dir.glob("*")):
                    if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                        continue
                    
                    samples.append({
                        "image_path": str(img_file),
                        "mask_path": None,  # 测试集通常没有mask
                        "defect_type": defect_type,
                        "is_anomaly": 1,
                    })
        
        return samples
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据集中的一个样本"""
        sample_info = self.samples[idx]
        
        # 加载图片
        image = Image.open(sample_info["image_path"]).convert("RGB")
        image_tensor = self.image_transform(image)
        
        # 加载mask
        mask = None
        if sample_info["mask_path"] is not None:
            mask_img = Image.open(sample_info["mask_path"]).convert("L")
            mask_tensor = self.mask_transform(mask_img)
            # 二值化处理 (>0.5 -> 1)
            mask = (mask_tensor > 0.5).float().numpy().squeeze()
        
        return {
            "image": image_tensor,
            "mask": mask,
            "defect_type": sample_info["defect_type"],
            "filename": Path(sample_info["image_path"]).name,
            "is_anomaly": sample_info["is_anomaly"],
        }
    
    def get_samples_by_defect_type(self, defect_type: str) -> List[Dict]:
        """获取指定缺陷类型的所有样本"""
        return [s for s in self.samples if s["defect_type"] == defect_type]
    
    def get_good_samples(self) -> List[Dict]:
        """获取所有好品样本"""
        return [s for s in self.samples if s["defect_type"] == "good"]
    
    def get_defect_samples(self) -> List[Dict]:
        """获取所有缺陷样本"""
        return [s for s in self.samples if s["defect_type"] != "good"]


class DefectDataLoader:
    """缺陷数据加载器辅助类
    
    提供方便的数据加载和批处理功能。
    """
    
    @staticmethod
    def create_defect_dataloader(
        defect_dataset: MVTecDefectDataset,
        defect_types: Optional[List[str]] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 4,
    ) -> torch.utils.data.DataLoader:
        """为指定缺陷类型创建数据加载器
        
        Args:
            defect_dataset: MVTecDefectDataset 实例
            defect_types: 要加载的缺陷类型列表（None表示所有）
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: 工作进程数
            
        Returns:
            DataLoader 实例
        """
        if defect_types is not None:
            # 筛选指定类型的样本
            filtered_samples = []
            for st in defect_types:
                filtered_samples.extend(defect_dataset.get_samples_by_defect_type(st))
            
            # 创建新的数据集
            filtered_dataset = _FilteredDataset(defect_dataset, filtered_samples)
            return torch.utils.data.DataLoader(
                filtered_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=custom_collate_fn,
            )
        
        return torch.utils.data.DataLoader(
            defect_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
        )


class _FilteredDataset(Dataset):
    """筛选后的数据集"""
    
    def __init__(self, base_dataset: MVTecDefectDataset, samples: List[Dict]):
        self.base_dataset = base_dataset
        self.sample_indices = []
        
        # 构建索引映射
        for sample in samples:
            for idx, s in enumerate(base_dataset.samples):
                if s["image_path"] == sample["image_path"]:
                    self.sample_indices.append(idx)
                    break
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.base_dataset[self.sample_indices[idx]]