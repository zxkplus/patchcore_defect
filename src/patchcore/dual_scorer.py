"""双库评分器模块 - 同时使用好品库和缺陷库进行评分和分类"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F

from patchcore.defect_library import DefectFeatureLibrary

LOGGER = logging.getLogger(__name__)


class ClassificationStrategy(Enum):
    """分类策略枚举"""
    NEAREST_WINS = "nearest_wins"  # 离谁最近归谁
    DISTANCE_RATIO = "distance_ratio"  # d_defect/d_normal < 1 -> 缺陷
    SOFTMAX_PROB = "softmax_prob"  # 距离转概率


class DualLibraryScorer:
    """双库评分器类
    
    同时查询好品特征库和缺陷特征库，结合距离信息进行异常检测和缺陷分类。
    
    Attributes:
        normal_library: 好品特征库（NearestNeighbourScorer）
        defect_library: 缺陷特征库（DefectFeatureLibrary）
        classification_strategy: 分类策略
        anomaly_threshold: 异常阈值
        k_nearest_neighbours: 用于计算距离的最近邻数量
    """
    
    def __init__(
        self,
        normal_scorer,  # NearestNeighbourScorer
        defect_library: Optional[DefectFeatureLibrary] = None,
        classification_strategy: str = "distance_ratio",
        anomaly_threshold: Optional[float] = None,
        k_nearest_neighbours: int = 1,
    ):
        """初始化双库评分器
        
        Args:
            normal_scorer: 好品库评分器（NearestNeighbourScorer实例）
            defect_library: 缺陷特征库（可选）
            classification_strategy: 分类策略，可选值：
                - "nearest_wins": 离谁最近归谁
                - "distance_ratio": d_defect/d_normal < 1 -> 缺陷
                - "softmax_prob": 距离转概率
            anomaly_threshold: 异常阈值（可选）
            k_nearest_neighbours: 计算距离的最近邻数量
        """
        self.normal_scorer = normal_scorer
        self.defect_library = defect_library
        self.k_nearest_neighbours = k_nearest_neighbours
        
        # 设置分类策略
        try:
            self.classification_strategy = ClassificationStrategy(classification_strategy)
        except ValueError:
            LOGGER.warning(
                f"未知的分类策略: {classification_strategy}, "
                f"使用默认值: {ClassificationStrategy.DISTANCE_RATIO.value}"
            )
            self.classification_strategy = ClassificationStrategy.DISTANCE_RATIO
        
        self.anomaly_threshold = anomaly_threshold
        self.device = None
        
        LOGGER.info(f"初始化双库评分器，分类策略: {self.classification_strategy.value}")
        if defect_library is not None:
            LOGGER.info(f"缺陷库包含 {len(defect_library)} 个特征")
    
    def compute_anomaly_threshold(
        self,
        normal_features: List[np.ndarray],
        percentile: float = 95.0,
    ) -> float:
        """从好品特征自动计算异常阈值
        
        Args:
            normal_features: 好品图片的特征列表
            percentile: 用于计算阈值的百分位数
            
        Returns:
            计算得到的异常阈值
        """
        from patchcore.common import ConcatMerger
        
        merger = ConcatMerger()
        merged_features = merger.merge(normal_features)
        
        # 查询好品库，计算每个特征的距离
        distances, _ = self.normal_scorer.imagelevel_nn(merged_features)
        
        # 使用百分位数作为阈值
        self.anomaly_threshold = np.percentile(distances, percentile)
        
        LOGGER.info(
            f"计算得到异常阈值: {self.anomaly_threshold:.4f} "
            f"(基于 {percentile}% 百分位数)"
        )
        return self.anomaly_threshold
    
    def predict(
        self,
        query_features: List[np.ndarray],
        patch_shapes: Optional[List[Tuple[int, int]]] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], Optional[List[np.ndarray]]]:
        """同时查好品库和缺陷库，返回分割和分类结果
        
        Args:
            query_features: 查询特征列表
            patch_shapes: patch形状列表（可选）
            target_size: 目标输出大小（可选）
            
        Returns:
            image_scores: 图像级异常分数
            segmentation_maps: 分割图列表
            defect_type_maps: 缺陷类型图列表（如果没有缺陷库则为None）
        """
        from patchcore.common import ConcatMerger
        
        merger = ConcatMerger()
        merged_features = merger.merge(query_features)
        
        # 查询好品库 - 获取正常距离
        normal_distances, _, _ = self.normal_scorer.predict(query_features)
        
        # 处理分割图
        if patch_shapes is not None:
            segmentation_maps = self._compute_segmentation_from_scores(
                normal_distances, patch_shapes, target_size
            )
        else:
            segmentation_maps = [normal_distances] if isinstance(normal_distances, np.ndarray) else normal_distances
        
        # 计算图像级分数
        image_scores = np.array([
            np.max(seg_map) for seg_map in segmentation_maps
        ])
        
        # 如果没有缺陷库，只返回异常检测结果
        if self.defect_library is None or self.defect_library.is_empty():
            return image_scores, segmentation_maps, None
        
        # 查询缺陷库
        defect_distances, _, defect_labels = self._query_defect_library(merged_features)
        
        # 计算缺陷类型图
        if patch_shapes is not None:
            defect_type_maps = self._compute_defect_type_map(
                defect_labels, patch_shapes, target_size
            )
        else:
            defect_type_maps = [defect_labels.squeeze()]
        
        # 根据分类策略生成最终分类结果
        # 现在 normal_distances 和 defect_distances 都是展平的数组，形状应该一致
        defect_classifications = self._classify_defect(
            normal_distances, defect_distances, image_scores
        )
        
        # 将分类结果应用到类型图
        defect_type_maps = self._apply_classification(
            defect_type_maps, defect_classifications
        )
        
        return image_scores, segmentation_maps, defect_type_maps
    
    def _query_defect_library(
        self,
        query_features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """查询缺陷库
        
        Args:
            query_features: 查询特征
            
        Returns:
            min_distances: 每个查询的最小距离
            indices: 最近邻索引
            labels: 最近邻的标签
        """
        distances, indices, labels = self.defect_library.query(
            query_features, k=self.k_nearest_neighbours
        )
        
        # 计算最小距离
        min_distances = distances.min(axis=1)
        
        return min_distances, indices, labels
    
    def _compute_segmentation_from_scores(
        self,
        patch_scores: np.ndarray,
        patch_shapes: List[Tuple[int, int]],
        target_size: Optional[Tuple[int, int]],
    ) -> List[np.ndarray]:
        """从已有的patch分数计算分割图
        
        Args:
            patch_scores: 每个patch的异常分数 (N, H*W) 或 (H*W,)
            patch_shapes: patch形状列表
            target_size: 目标大小
            
        Returns:
            分割图列表
        """
        # 确定batch size
        if patch_scores.ndim == 1:
            batch_size = 1
            patch_scores = patch_scores.reshape(1, -1)
        else:
            batch_size = patch_scores.shape[0]
            
        # 计算实际的patch数量
        total_patches = patch_scores.shape[1]
        
        # 尝试使用提供的patch_shapes[0]进行重塑
        patch_shape = patch_shapes[0] if patch_shapes else (1, 1)
        h, w = patch_shape
        
        # 重塑为patch网格 (N, H, W)
        try:
            patch_scores_reshaped = patch_scores.reshape(batch_size, h, w)
        except ValueError:
            LOGGER.warning(f"无法将分数重塑为形状 ({batch_size}, {h}, {w})，当前形状 {patch_scores.shape}")
            # fallback: 根据实际patch数量计算合理的网格尺寸
            # 假设网格是正方形或接近正方形
            import math
            grid_size = int(math.sqrt(total_patches))
            if grid_size * grid_size == total_patches:
                # 完美正方形
                patch_scores_reshaped = patch_scores.reshape(batch_size, grid_size, grid_size)
                h, w = grid_size, grid_size
            elif (grid_size + 1) * grid_size == total_patches:
                # 矩形网格
                patch_scores_reshaped = patch_scores.reshape(batch_size, grid_size + 1, grid_size)
                h, w = grid_size + 1, grid_size
            elif grid_size * (grid_size + 1) == total_patches:
                # 矩形网格
                patch_scores_reshaped = patch_scores.reshape(batch_size, grid_size, grid_size + 1)
                h, w = grid_size, grid_size + 1
            else:
                # 最后的fallback：使用1D reshape，然后在插值时处理
                patch_scores_reshaped = patch_scores.reshape(batch_size, 1, total_patches)
                h, w = 1, total_patches

        # 上采样到目标大小（双线性插值）
        segmentation_maps = []
        for i in range(batch_size):
            ps = patch_scores_reshaped[i]
            if isinstance(ps, np.ndarray):
                ps_tensor = torch.from_numpy(ps)
            else:
                ps_tensor = ps
            
            # 确保tensor有正确的维度用于插值
            if ps_tensor.dim() == 1:
                ps_tensor = ps_tensor.unsqueeze(0).unsqueeze(0).float()
            elif ps_tensor.dim() == 2:
                ps_tensor = ps_tensor.unsqueeze(0).unsqueeze(0).float()
            else:
                ps_tensor = ps_tensor.unsqueeze(0).float()
            
            if target_size is not None:
                ps_up = F.interpolate(ps_tensor, size=target_size, mode="bilinear", align_corners=False)
            else:
                ps_up = ps_tensor
                
            ps_np = ps_up.squeeze().numpy()
            
            # 高斯平滑
            if target_size is not None or True: # 总是平滑
                ps_np = ndimage.gaussian_filter(ps_np, sigma=4)
                
            segmentation_maps.append(ps_np)
            
        return segmentation_maps
    
    def _compute_defect_type_map(
        self,
        defect_labels: np.ndarray,
        patch_shapes: List[Tuple[int, int]],
        target_size: Optional[Tuple[int, int]],
    ) -> List[np.ndarray]:
        """计算缺陷类型图
        
        Args:
            defect_labels: 缺陷标签
            patch_shapes: patch形状列表
            target_size: 目标大小
            
        Returns:
            缺陷类型图列表
        """
        # 确定batch size - 从defect_labels推断而不是patch_shapes长度
        if defect_labels.ndim == 1:
            # 单个样本的情况
            batch_size = 1
            total_elements = defect_labels.shape[0]
        else:
            batch_size = defect_labels.shape[0]
            total_elements = defect_labels.shape[1] if defect_labels.ndim == 2 else defect_labels.shape[0]
        
        # 使用第一个patch_shape作为参考
        patch_shape = patch_shapes[0] if patch_shapes else (1, 1)
        h, w = patch_shape
        
        # 验证元素数量是否匹配
        expected_elements = h * w
        if total_elements != expected_elements:
            # 如果不匹配，尝试根据实际元素数量计算合理的网格尺寸
            import math
            grid_size = int(math.sqrt(total_elements))
            if grid_size * grid_size == total_elements:
                h, w = grid_size, grid_size
            elif (grid_size + 1) * grid_size == total_elements:
                h, w = grid_size + 1, grid_size
            elif grid_size * (grid_size + 1) == total_elements:
                h, w = grid_size, grid_size + 1
            else:
                # 最后的fallback：使用1D reshape
                h, w = 1, total_elements
        
        # 重塑标签
        if defect_labels.ndim > 1 and defect_labels.shape[0] == 1:
            defect_labels = defect_labels.squeeze(0)
        elif defect_labels.ndim > 1:
            # 多batch情况
            pass
        elif defect_labels.ndim == 1 and batch_size == 1:
            # 确保是正确的形状
            defect_labels = defect_labels.reshape(h, w)
            labels_reshaped = defect_labels[np.newaxis, :, :]  # 添加batch维度
        else:
            defect_labels = defect_labels.reshape(batch_size, h, w)
            labels_reshaped = defect_labels
        
        if defect_labels.ndim == 2:
            labels_reshaped = defect_labels[np.newaxis, :, :]
        else:
            labels_reshaped = defect_labels
        
        # 上采样到目标大小（最近邻插值，保持类别标签）
        if target_size is not None:
            labels = []
            for lb in labels_reshaped:
                lb_tensor = torch.from_numpy(lb).unsqueeze(0).unsqueeze(0).float()
                lb_up = F.interpolate(lb_tensor, size=target_size, mode="nearest")
                labels.append(lb_up.squeeze().numpy().astype(np.int32))
            return labels
        
        return [lb.astype(np.int32) for lb in labels_reshaped]
    
    def _classify_defect(
        self,
        normal_distances: np.ndarray,
        defect_distances: np.ndarray,
        image_scores: np.ndarray,
    ) -> np.ndarray:
        """根据分类策略进行缺陷分类
        
        Args:
            normal_distances: 到好品库的距离
            defect_distances: 到缺陷库的距离
            image_scores: 图像级异常分数
            
        Returns:
            分类结果掩码，0表示正常，1表示缺陷（具体类型由 defect_type_maps 提供）
        """
        mask = np.zeros_like(normal_distances, dtype=np.int32)
        
        if self.classification_strategy == ClassificationStrategy.NEAREST_WINS:
            # 离谁最近归谁
            mask = (defect_distances < normal_distances).astype(np.int32)
        
        elif self.classification_strategy == ClassificationStrategy.DISTANCE_RATIO:
            # d_defect/d_normal < 1 -> 缺陷
            ratios = defect_distances / (normal_distances + 1e-8)
            
            # 第一步：距离比小于1表示更像缺陷
            is_defect = ratios < 1.0
            
            # 第二步：如果设置了阈值，检查异常分数是否超过阈值
            if self.anomaly_threshold is not None:
                is_anomaly = image_scores > self.anomaly_threshold
                is_defect = is_defect & is_anomaly
            
            mask = is_defect.astype(np.int32)
        
        elif self.classification_strategy == ClassificationStrategy.SOFTMAX_PROB:
            # 距离转概率
            combined = np.stack([normal_distances, defect_distances], axis=1)
            neg_distances = -combined
            exp_neg_dist = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
            probs = exp_neg_dist / np.sum(exp_neg_dist, axis=1, keepdims=True)
            
            # 如果缺陷概率大于0.5，判定为缺陷
            defect_probs = probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(probs))
            is_defect = defect_probs > 0.5
            
            mask = is_defect.astype(np.int32)
            
        return mask
    
    def _get_defect_type_from_distance(self, distances: np.ndarray, original_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """根据距离获取缺陷类型
        
        这是一个简化实现，实际使用时应该根据具体需求调整
        
        Args:
            distances: 到缺陷库的距离
            original_indices: 可选，如果在外部已经查询过，可以直接使用对应的标签
            
        Returns:
            缺陷类型标签
        """
        # 这个函数在 _classify_defect 中被调用，但那里只传了 distances。
        # 实际上，缺陷类型应该来自于 _query_defect_library 返回的 labels。
        # 由于 _classify_defect 目前是 image-level 的，而 labels 是 patch-level 的，
        # 这里存在逻辑断层。
        
        # 为了修复 predict 中的调用，我们最好直接在 predict 中使用从 _query_defect_library 获得的 labels
        # 而不是在这里重新查询。
        
        # 鉴于 _classify_defect 签名限制，我们暂时保留此函数但标记为 deprecated 或仅作为 fallback
        # 在 predict 中我们将不再依赖此函数进行主要的 label 映射，而是直接使用 defect_labels
        
        if self.defect_library is not None and hasattr(self.defect_library, 'labels') and self.defect_library.labels is not None:
             # 尝试通过距离反查，但这通常不准确，因为距离是标量
             # 正确的做法是使用 indices 从 library 中获取 label
             pass
             
        # 默认返回 1
        return np.ones_like(distances, dtype=np.int32)
    
    def _apply_classification(
        self,
        defect_type_maps: List[np.ndarray],
        classifications: np.ndarray,
    ) -> List[np.ndarray]:
        """将分类结果应用到类型图
        
        Args:
            defect_type_maps: 缺陷类型图列表
            classifications: 分类结果
            
        Returns:
            应用分类后的类型图
        """
        result = []
        for i, type_map in enumerate(defect_type_maps):
            # 如果该样本被判定为正常，将类型图设为0
            if i < len(classifications) and classifications[i] == 0:
                result.append(np.zeros_like(type_map))
            else:
                result.append(type_map)
        return result
    
    def set_anomaly_threshold(self, threshold: float) -> None:
        """设置异常阈值
        
        Args:
            threshold: 异常阈值
        """
        self.anomaly_threshold = threshold
        LOGGER.info(f"设置异常阈值: {threshold}")
    
    def get_defect_types(self) -> List[str]:
        """获取所有已注册的缺陷类型"""
        if self.defect_library is None:
            return ["normal"]
        return self.defect_library.get_defect_types()
