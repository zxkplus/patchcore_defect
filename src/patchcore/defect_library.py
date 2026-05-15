"""缺陷特征库模块 - 存储和管理缺陷样本的特征"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class DefectFeatureLibrary:
    """缺陷特征库类
    
    用于存储和管理缺陷样本的特征，支持按缺陷类型分类存储。
    使用 FAISS 索引加速相似度搜索。
    
    Attributes:
        defect_type_to_id: 缺陷类型名称到ID的映射
        defect_id_to_type: 缺陷ID到类型名称的映射
        features: 存储的所有缺陷特征
        labels: 对应特征的真实标签
        search_index: FAISS 搜索索引
        feature_dimension: 特征维度
    """
    
    # 缺陷类别0预留给"正常"，缺陷类型从1开始编号
    NORMAL_CLASS_ID = 0
    
    def __init__(self, feature_dimension: int):
        """初始化缺陷特征库
        
        Args:
            feature_dimension: 特征的维度
        """
        self.defect_type_to_id: Dict[str, int] = {"normal": self.NORMAL_CLASS_ID}
        self.defect_id_to_type: Dict[int, str] = {self.NORMAL_CLASS_ID: "normal"}
        self.next_defect_id = 1
        
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.search_index: Optional[faiss.Index] = None
        self.feature_dimension = feature_dimension
        
        LOGGER.info(f"初始化缺陷特征库，特征维度: {feature_dimension}")
    
    def add_defect_type(self, defect_type: str) -> int:
        """注册新的缺陷类型
        
        Args:
            defect_type: 缺陷类型名称
            
        Returns:
            缺陷类型对应的ID
        """
        if defect_type not in self.defect_type_to_id:
            self.defect_type_to_id[defect_type] = self.next_defect_id
            self.defect_id_to_type[self.next_defect_id] = defect_type
            self.next_defect_id += 1
            LOGGER.info(f"注册新缺陷类型: {defect_type} -> ID {self.defect_type_to_id[defect_type]}")
        return self.defect_type_to_id[defect_type]
    
    def get_defect_type_id(self, defect_type: str) -> int:
        """获取缺陷类型对应的ID
        
        Args:
            defect_type: 缺陷类型名称
            
        Returns:
            缺陷类型ID，如果不存在则返回-1
        """
        return self.defect_type_to_id.get(defect_type, -1)
    
    def build_from_defect_data(
        self,
        features: np.ndarray,
        masks: np.ndarray,
        defect_types: List[str],
        patch_shapes: Tuple[int, int],
        target_image_size: Tuple[int, int],
        sampling_ratio: float = 1.0,
    ) -> None:
        """从缺陷图+mask构建缺陷特征库
        
        将mask下采样到patch网格尺寸（用nearest插值保持二值性），
        筛选mask>0.5的patch的特征，并记录每个特征的缺陷类别。
        
        Args:
            features: 缺陷图片的特征，形状为 [N, D] 或 [N, D, H, W]
            masks: 缺陷区域的mask，形状为 [N, H, W] 或 [N, 1, H, W]
            defect_types: 每个样本的缺陷类型列表
            patch_shapes: patch网格的形状 (H_patches, W_patches)
            target_image_size: 目标图像大小 (H, W)
            sampling_ratio: 采样比例，用于减少特征数量
        """
        # 处理特征形状
        if features.ndim == 4:
            # [N, D, H, W] -> keep as is for now
            n_samples, dim, h_patches, w_patches = features.shape
            if dim != self.feature_dimension:
                raise ValueError(f"特征维度不匹配: 期望 {self.feature_dimension}, 实际 {dim}")
        elif features.ndim == 2:
            # [N, D] 已经是flatten的
            n_samples = features.shape[0]
            h_patches, w_patches = patch_shapes
            if features.shape[1] != self.feature_dimension:
                raise ValueError(f"特征维度不匹配: 期望 {self.feature_dimension}, 实际 {features.shape[1]}")
        else:
            raise ValueError(f"不支持的特征形状: {features.shape}")
        
        # 处理mask形状和下采样
        if masks.ndim == 4:
            masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]
        
        # 验证输入长度一致
        if len(masks) != n_samples or len(defect_types) != n_samples:
            raise ValueError(f"输入长度不一致: features={n_samples}, masks={len(masks)}, defect_types={len(defect_types)}")
        
        # 将mask下采样到patch网格尺寸（使用nearest插值保持二值性）
        masks_downsampled = []
        for i, mask in enumerate(masks):
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            mask_down = F.interpolate(
                mask_tensor,
                size=(h_patches, w_patches),
                mode="nearest"
            )
            masks_downsampled.append(mask_down.squeeze().numpy())
        masks_downsampled = np.array(masks_downsampled)
        
        # 筛选缺陷区域特征
        selected_features = []
        selected_labels = []
        
        if features.ndim == 4:
            # 处理4D特征 [N, D, H, W]
            for i in range(n_samples):
                feature_sample = features[i]  # [D, H, W]
                mask_sample = masks_downsampled[i]  # [H, W]
                defect_type = defect_types[i]
                
                # 获取该样本的patch mask并展平
                sample_mask = mask_sample.flatten()  # [H*W]
                # 筛选mask > 0.5的区域
                defect_indices = np.where(sample_mask > 0.5)[0]
                
                if len(defect_indices) == 0:
                    LOGGER.warning(f"样本 {i} ({defect_type}) 没有有效的缺陷区域")
                    continue
                
                # 将特征转为 [H*W, D] 并选择缺陷区域
                feature_flat = feature_sample.reshape(self.feature_dimension, -1).T  # [H*W, D]
                defect_features = feature_flat[defect_indices]  # [num_defect_patches, D]
                
                # 采样（如果需要）
                if sampling_ratio < 1.0 and len(defect_features) > 1:
                    n_samples_defect = max(1, int(len(defect_features) * sampling_ratio))
                    sample_indices = np.random.choice(
                        len(defect_features), n_samples_defect, replace=False
                    )
                    defect_features = defect_features[sample_indices]
                
                # 获取或创建缺陷类型ID
                defect_id = self.add_defect_type(defect_type)
                
                selected_features.append(defect_features)
                selected_labels.extend([defect_id] * len(defect_features))
        else:
            # 处理2D特征 [N, D]
            for i in range(n_samples):
                feature_sample = features[i:i+1]  # [1, D]
                mask_sample = masks_downsampled[i]  # [H, W]
                defect_type = defect_types[i]
                
                # 对于2D特征，我们假设整个样本都是缺陷区域
                # 或者根据mask来决定是否包含
                sample_mask = mask_sample.flatten()
                if np.any(sample_mask > 0.5):
                    defect_id = self.add_defect_type(defect_type)
                    selected_features.append(feature_sample)
                    selected_labels.append(defect_id)
                else:
                    LOGGER.warning(f"样本 {i} ({defect_type}) 没有有效的缺陷区域")
        
        if len(selected_features) == 0:
            raise ValueError("没有找到有效的缺陷特征")
        
        # 合并所有特征
        self.features = np.vstack(selected_features).astype(np.float32)
        self.labels = np.array(selected_labels, dtype=np.int32)
        
        LOGGER.info(
            f"构建缺陷特征库完成: {len(self.features)} 个特征, "
            f"{len(self.defect_type_to_id)} 种缺陷类型"
        )
        
        # 构建FAISS索引
        self._build_index()
    
    def _build_index(self) -> None:
        """构建FAISS索引"""
        if self.features is None or len(self.features) == 0:
            raise ValueError("没有可索引的特征")
        
        # 验证特征维度
        if self.features.shape[1] != self.feature_dimension:
            raise ValueError(f"特征维度不匹配: 期望 {self.feature_dimension}, 实际 {self.features.shape[1]}")
        
        # 使用 L2 距离的 Flat 索引
        self.search_index = faiss.IndexFlatL2(self.feature_dimension)
        self.search_index.add(self.features)
        
        LOGGER.info(f"FAISS索引构建完成: {self.search_index.ntotal} 个向量")
    
    def query(
        self,
        query_features: np.ndarray,
        k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """查询缺陷库
        
        Args:
            query_features: 查询特征，形状为 [N, D] 或 [N, D, H, W]
            k: 返回的最近邻数量
            
        Returns:
            distances: 距离数组
            indices: 索引数组
            labels: 类别标签数组
        """
        if self.search_index is None:
            raise ValueError("缺陷库尚未构建，请先调用 build_from_defect_data")
        
        if query_features.ndim == 4:
            # [N, D, H, W] -> [N*H*W, D]
            n_samples, dim, h_patches, w_patches = query_features.shape
            if dim != self.feature_dimension:
                raise ValueError(f"查询特征维度不匹配: 期望 {self.feature_dimension}, 实际 {dim}")
            query_features = query_features.transpose(0, 2, 3, 1).reshape(-1, dim)
        elif query_features.ndim == 2:
            if query_features.shape[1] != self.feature_dimension:
                raise ValueError(f"查询特征维度不匹配: 期望 {self.feature_dimension}, 实际 {query_features.shape[1]}")
        else:
            raise ValueError(f"不支持的查询特征形状: {query_features.shape}")
        
        query_features = query_features.astype(np.float32)
        distances, indices = self.search_index.search(query_features, k)
        
        # 获取对应的标签
        if k == 1:
            labels = self.labels[indices.flatten()].reshape(-1, 1)
        else:
            # 对于多个最近邻，使用加权投票确定类别
            labels = np.zeros((len(indices), self.next_defect_id), dtype=np.float32)
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                # 距离越小，权重越大
                weights = 1.0 / (dist + 1e-8)
                for d, j in zip(weights, idx):
                    labels[i, self.labels[j]] += d
            labels = np.argmax(labels, axis=1).reshape(-1, 1)
        
        return distances, indices, labels
    
    def get_defect_types(self) -> List[str]:
        """获取所有已注册的缺陷类型"""
        return [
            self.defect_id_to_type[i] 
            for i in range(self.NORMAL_CLASS_ID, self.next_defect_id)
        ]
    
    def save(self, save_path: str) -> None:
        """保存缺陷特征库到文件
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS索引
        index_path = os.path.join(save_path, "defect_library.faiss")
        if self.search_index is not None:
            faiss.write_index(
                faiss.index_gpu_to_cpu(self.search_index)
                if hasattr(self.search_index, 'index_handle')
                else self.search_index,
                index_path
            )
        
        # 保存元数据
        metadata = {
            "defect_type_to_id": self.defect_type_to_id,
            "defect_id_to_type": self.defect_id_to_type,
            "next_defect_id": self.next_defect_id,
            "labels": self.labels,
            "feature_dimension": self.feature_dimension,
        }
        metadata_path = os.path.join(save_path, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        
        LOGGER.info(f"缺陷特征库已保存到: {save_path}")
    
    @classmethod
    def load(cls, load_path: str) -> "DefectFeatureLibrary":
        """从文件加载缺陷特征库
        
        Args:
            load_path: 加载路径
            
        Returns:
            加载的缺陷特征库实例
        """
        # 加载FAISS索引
        index_path = os.path.join(load_path, "defect_library.faiss")
        search_index = faiss.read_index(index_path)
        
        # 加载元数据
        metadata_path = os.path.join(load_path, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        # 创建实例
        library = cls(feature_dimension=metadata["feature_dimension"])
        library.defect_type_to_id = metadata["defect_type_to_id"]
        library.defect_id_to_type = metadata["defect_id_to_type"]
        library.next_defect_id = metadata["next_defect_id"]
        library.labels = metadata["labels"]
        library.search_index = search_index
        library.features = search_index.reconstruct_n(0, search_index.ntotal)
        
        LOGGER.info(f"缺陷特征库已从 {load_path} 加载")
        return library
    
    def __len__(self) -> int:
        """返回特征库中的特征数量"""
        return len(self.features) if self.features is not None else 0
    
    def is_empty(self) -> bool:
        """检查特征库是否为空"""
        return self.features is None or len(self.features) == 0
