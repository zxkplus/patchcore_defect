"""PatchCore 核心模块 - 基于 amazon-science/patchcore-inspection 修改

此文件在原始 PatchCore 基础上增加了：
1. 缺陷特征库支持 (DefectFeatureLibrary)
2. 双库评分器支持 (DualLibraryScorer)
3. 分割+分类功能

"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from patchcore import backbones
from patchcore import common
from patchcore import sampler
from patchcore.defect_library import DefectFeatureLibrary
from patchcore.dual_scorer import DualLibraryScorer

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    """PatchCore 异常检测类
    
    在原始 PatchCore 基础上增加了缺陷特征库支持，可以同时进行：
    1. 像素级缺陷分割（和原始一样）
    2. 缺陷类型分类（新增）
    
    Attributes:
        device: 计算设备
        backbone: 骨干网络
        layers_to_extract_from: 特征提取层列表
        defect_library: 缺陷特征库（可选）
        dual_scorer: 双库评分器（可选）
        classification_strategy: 分类策略
        anomaly_threshold: 异常阈值
    """
    
    def __init__(self, device):
        """PatchCore 异常检测类初始化"""
        super(PatchCore, self).__init__()
        self.device = device
        
        # 新增：缺陷库相关属性
        self.defect_library: Optional[DefectFeatureLibrary] = None
        self.dual_scorer: Optional[DualLibraryScorer] = None
        self.classification_strategy: str = "distance_ratio"
        self.anomaly_threshold: Optional[float] = None
        
        # 原始属性初始化（保持向后兼容）
        self.backbone = None
        self.layers_to_extract_from = None
        self.input_shape = None
        self.patch_maker = None
        self.forward_modules = None
        self.anomaly_scorer = None
        self.anomaly_segmentor = None
        self.featuresampler = None
    
    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=sampler.IdentitySampler(),
        nn_method=common.FaissNN(False, 4),
        classification_strategy: str = "distance_ratio",
        **kwargs,
    ):
        """加载 PatchCore 模型
        
        Args:
            backbone: 骨干网络模型
            layers_to_extract_from: 特征提取层列表
            device: 计算设备
            input_shape: 输入图片形状
            pretrain_embed_dimension: 预训练嵌入维度
            target_embed_dimension: 目标嵌入维度
            patchsize: patch大小
            patchstride: patch步长
            anomaly_score_num_nn: 异常评分使用的最近邻数量
            featuresampler: 特征采样器
            nn_method: 最近邻搜索方法
            classification_strategy: 分类策略
        """
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.classification_strategy = classification_strategy
        
        # 创建 PatchMaker
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        
        # 初始化前向模块
        self.forward_modules = torch.nn.ModuleDict({})
        
        # 特征聚合器
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator
        
        # 预处理模块
        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing
        
        # 目标嵌入维度
        self.target_embed_dimension = target_embed_dimension
        
        # 预适配聚合器
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
        
        # 异常评分器（原始最近邻评分器）
        self.anomaly_scorer = common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )
        
        # 异常分割器
        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )
        
        # 特征采样器
        self.featuresampler = featuresampler
        
        LOGGER.info(
            f"PatchCore 加载完成 "
            f"layers={layers_to_extract_from}, "
            f"classification_strategy={classification_strategy}"
        )
    
    def embed(self, data):
        """提取图片特征"""
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)
    
    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""
        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features
        
        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        
        features = [features[layer] for layer in self.layers_to_extract_from]
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        
        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        # 预处理和聚合
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        
        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)
    
    def fit(self, training_data):
        """PatchCore 训练
        
        计算训练数据的嵌入并填充记忆库
        """
        self._fill_memory_bank(training_data)
    
    def _fill_memory_bank(self, input_data):
        """计算并设置支持特征"""
        _ = self.forward_modules.eval()
        
        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)
        
        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))
        
        features = np.concatenate(features, axis=0)
        features = self.featuresampler.run(features)
        self.anomaly_scorer.fit(detection_features=[features])
    
    def fit_defect_library(
        self,
        defect_dataloader,
        classification_strategy: str = "distance_ratio",
        sampling_ratio: float = 1.0,
        percentile: float = 95.0,
    ) -> None:
        """构建缺陷特征库
        
        从缺陷图+mask构建缺陷库，并创建双库评分器。
        
        Args:
            defect_dataloader: 缺陷数据加载器，返回 {"image": tensor, "mask": tensor, "defect_type": str}
            classification_strategy: 分类策略
            sampling_ratio: 特征采样比例
            percentile: 用于计算异常阈值的百分位数
        """
        LOGGER.info("开始构建缺陷特征库...")
        
        _ = self.forward_modules.eval()
        all_features = []
        all_masks = []
        all_defect_types = []
        all_patch_shapes = None
        
        with tqdm.tqdm(
            defect_dataloader, desc="提取缺陷特征...", position=1, leave=False
        ) as data_iterator:
            for batch in data_iterator:
                images = batch["image"].to(self.device)
                masks = batch["mask"]
                defect_types = batch["defect_type"]
                
                with torch.no_grad():
                    features, patch_shapes = self._embed(
                        images, provide_patch_shapes=True
                    )
                    features = np.asarray(features)
                
                all_patch_shapes = patch_shapes
                
                # 处理每个样本
                for i in range(len(images)):
                    feat = features[i]  # [D, H, W] 或 [D]
                    mask = masks[i] if isinstance(masks, list) else masks[i].numpy()
                    defect_type = (
                        defect_types[i] if isinstance(defect_types, list) 
                        else defect_types
                    )
                    
                    all_features.append(feat)
                    all_masks.append(mask)
                    all_defect_types.append(defect_type)
        
        if len(all_features) == 0:
            LOGGER.warning("没有找到缺陷样本，跳过缺陷库构建")
            return
        
        # 获取特征维度
        feature = all_features[0]
        if feature.ndim == 3:
            feature_dimension = feature.shape[0]
        else:
            feature_dimension = feature.shape[-1]
        
        # 创建缺陷特征库
        self.defect_library = DefectFeatureLibrary(feature_dimension)
        
        # 堆叠特征
        if all_features[0].ndim == 3:
            # [N, D, H, W] 格式
            stacked_features = np.stack(all_features, axis=0)
        else:
            # [N, D] 格式已经是flatten的
            stacked_features = np.stack(all_features, axis=0)
        
        # 堆叠mask
        stacked_masks = np.stack(all_masks, axis=0)
        
        # 构建缺陷库
        self.defect_library.build_from_defect_data(
            features=stacked_features,
            masks=stacked_masks,
            defect_types=all_defect_types,
            patch_shapes=all_patch_shapes[0] if all_patch_shapes else (1, 1),
            target_image_size=self.input_shape[-2:],
            sampling_ratio=sampling_ratio,
        )
        
        # 获取好品特征用于计算阈值
        LOGGER.info("计算异常阈值...")
        normal_features = self._get_normal_features_for_threshold()
        
        # 创建双库评分器
        self.dual_scorer = DualLibraryScorer(
            normal_scorer=self.anomaly_scorer,
            defect_library=self.defect_library,
            classification_strategy=classification_strategy,
            anomaly_threshold=None,
            k_nearest_neighbours=1,
        )
        
        # 计算异常阈值
        if normal_features is not None and len(normal_features) > 0:
            self.anomaly_threshold = self.dual_scorer.compute_anomaly_threshold(
                normal_features, percentile=percentile
            )
        
        # 修复日志格式问题 - 按照安全日志规范处理
        if self.anomaly_threshold is not None:
            LOGGER.info(f"缺陷特征库构建完成: {len(self.defect_library)} 个特征, 异常阈值: {self.anomaly_threshold:.4f}")
        else:
            LOGGER.info(f"缺陷特征库构建完成: {len(self.defect_library)} 个特征, 异常阈值: N/A")
    
    def _get_normal_features_for_threshold(self) -> Optional[List[np.ndarray]]:
        """获取好品特征用于计算阈值
        
        这个方法需要在调用前先用好品数据训练过模型
        """
        if not hasattr(self.anomaly_scorer, "detection_features"):
            return None
        if self.anomaly_scorer.detection_features is None:
            return None
        return [self.anomaly_scorer.detection_features]
    
    def predict(self, data):
        """预测异常和缺陷类型"""
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)
    
    def _predict_dataloader(self, dataloader):
        """对整个dataloader进行预测"""
        _ = self.forward_modules.eval()
        
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        defect_types_pred = []
        
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image.get("is_anomaly", [0] * len(image["image"])))
                    masks_gt.extend(
                        image.get("mask", [None] * len(image["image"]))
                        if isinstance(image.get("mask"), list)
                        else image["mask"].numpy().tolist()
                        if image.get("mask") is not None
                        else [None] * len(image["image"])
                    )
                    image = image["image"]
                
                _scores, _masks, _defect_types = self._predict(image)
                
                for score, mask, defect_type in zip(_scores, _masks, _defect_types):
                    scores.append(score)
                    masks.append(mask)
                    defect_types_pred.append(defect_type)
        
        return scores, masks, labels_gt, masks_gt, defect_types_pred
    
    def _predict(self, images):
        """Infer score and mask for a batch of images.
        
        如果 dual_scorer 存在，使用双库推理；否则退化到原始逻辑。
        
        Returns:
            image_scores: 图像级异常分数
            segmentation_maps: 分割图列表
            defect_type_maps: 缺陷类型图列表（可能为None）
        """
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()
        
        batchsize = images.shape[0]
        
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)
            
            # 判断使用哪种推理方式
            if self.dual_scorer is not None and self.defect_library is not None:
                # 使用双库推理
                image_scores, segmentation_maps, defect_type_maps = self.dual_scorer.predict(
                    query_features=[features],
                    patch_shapes=patch_shapes,
                    target_size=self.input_shape[-2:],
                )
                
                return (
                    [score for score in image_scores],
                    segmentation_maps if isinstance(segmentation_maps, list) else [segmentation_maps],
                    defect_type_maps if defect_type_maps is not None else [None] * batchsize,
                )
            
            # 原始推理逻辑
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)
            
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
            
            return (
                [score for score in image_scores],
                [mask for mask in masks],
                [None] * batchsize,
            )
    
    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")
    
    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        """保存 PatchCore 模型到指定路径
        
        Args:
            save_path: 保存路径
            prepend: 文件名前缀
        """
        LOGGER.info("Saving PatchCore data.")
        
        # 保存原始的异常评分器
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        
        # 保存模型参数
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
            "classification_strategy": self.classification_strategy,
        }
        
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)
        
        # 保存缺陷库（如果有）
        if self.defect_library is not None and not self.defect_library.is_empty():
            defect_library_path = os.path.join(save_path, "defect_library")
            self.defect_library.save(defect_library_path)
            LOGGER.info(f"缺陷特征库已保存到: {defect_library_path}")
        
        # 保存异常阈值
        if self.anomaly_threshold is not None:
            threshold_path = os.path.join(save_path, prepend + "anomaly_threshold.pkl")
            with open(threshold_path, "wb") as f:
                pickle.dump(self.anomaly_threshold, f)
    
    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method=None,
        prepend: str = "",
        load_defect_library: bool = True,
    ) -> None:
        """从指定路径加载 PatchCore 模型
        
        Args:
            load_path: 加载路径
            device: 计算设备
            nn_method: 最近邻搜索方法
            prepend: 文件名前缀
            load_defect_library: 是否加载缺陷库
        """
        LOGGER.info("Loading and initializing PatchCore.")
        
        if nn_method is None:
            nn_method = common.FaissNN(False, 4)
        
        # 加载模型参数
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        
        # 加载骨干网络
        patchcore_params["backbone"] = backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        
        # 保存分类策略
        self.classification_strategy = patchcore_params.pop(
            "classification_strategy", "distance_ratio"
        )
        
        # 加载模型
        self.load(**patchcore_params, device=device, nn_method=nn_method)
        
        # 加载异常评分器
        self.anomaly_scorer.load(load_path, prepend)
        
        # 加载异常阈值
        threshold_path = os.path.join(load_path, prepend + "anomaly_threshold.pkl")
        if os.path.exists(threshold_path):
            with open(threshold_path, "rb") as f:
                self.anomaly_threshold = pickle.load(f)
            LOGGER.info(f"加载异常阈值: {self.anomaly_threshold}")
        
        # 加载缺陷库（如果存在）
        defect_library_path = os.path.join(load_path, "defect_library")
        if load_defect_library and os.path.exists(defect_library_path):
            try:
                self.defect_library = DefectFeatureLibrary.load(defect_library_path)
                self.dual_scorer = DualLibraryScorer(
                    normal_scorer=self.anomaly_scorer,
                    defect_library=self.defect_library,
                    classification_strategy=self.classification_strategy,
                    anomaly_threshold=self.anomaly_threshold,
                    k_nearest_neighbours=1,
                )
                LOGGER.info("缺陷库加载成功")
            except Exception as e:
                LOGGER.warning(f"加载缺陷库失败: {e}")
                self.defect_library = None
                self.dual_scorer = None
        else:
            LOGGER.info("未找到缺陷库，仅使用原始 PatchCore 功能")


# Image handling classes.
class PatchMaker:
    """Patch 制作器"""
    
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride
    
    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.

        Args:
            x: [torch.Tensor, bs x c x w x h]

        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features
    
    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])
    
    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
