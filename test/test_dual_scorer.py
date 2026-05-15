#!/usr/bin/env python3
"""双库评分器的单元测试

使用合成数据测试 DefectFeatureLibrary 和 DualLibraryScorer 的功能。
不依赖真实数据集。
"""

import unittest
import numpy as np
import torch
import tempfile
import os
import shutil

# 添加 src 目录到路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patchcore.defect_library import DefectFeatureLibrary
from patchcore.dual_scorer import DualLibraryScorer, ClassificationStrategy


class TestDefectFeatureLibrary(unittest.TestCase):
    """测试 DefectFeatureLibrary 类"""
    
    def setUp(self):
        """设置测试环境"""
        self.feature_dim = 128
        self.n_samples = 50
        self.patch_h, self.patch_w = 7, 7
        
        # 创建测试特征数据
        np.random.seed(42)
        self.features = np.random.randn(self.n_samples, self.feature_dim, 
                                        self.patch_h, self.patch_w).astype(np.float32)
        
        # 创建测试 mask（随机二值）
        self.masks = np.random.randint(0, 2, (self.n_samples, 224, 224)).astype(np.float32)
        
        # 创建测试缺陷类型
        self.defect_types = ["crack"] * 20 + ["scratch"] * 20 + ["dent"] * 10
    
    def test_initialization(self):
        """测试初始化"""
        library = DefectFeatureLibrary(feature_dimension=self.feature_dim)
        
        self.assertEqual(library.feature_dimension, self.feature_dim)
        self.assertTrue(library.is_empty())
        self.assertIn("normal", library.defect_type_to_id)
        self.assertEqual(library.defect_type_to_id["normal"], 0)
    
    def test_add_defect_type(self):
        """测试添加缺陷类型"""
        library = DefectFeatureLibrary(feature_dimension=self.feature_dim)
        
        # 添加新类型
        crack_id = library.add_defect_type("crack")
        self.assertEqual(crack_id, 1)
        
        # 添加同类型应返回相同ID
        crack_id_again = library.add_defect_type("crack")
        self.assertEqual(crack_id_again, crack_id)
        
        # 添加新类型
        scratch_id = library.add_defect_type("scratch")
        self.assertEqual(scratch_id, 2)
    
    def test_build_from_defect_data(self):
        """测试从缺陷数据构建特征库"""
        library = DefectFeatureLibrary(feature_dimension=self.feature_dim)
        
        # 准备更简单的测试数据
        features = np.random.randn(10, self.feature_dim, 3, 3).astype(np.float32)
        masks = np.ones((10, 224, 224), dtype=np.float32)
        defect_types = ["crack"] * 10
        
        library.build_from_defect_data(
            features=features,
            masks=masks,
            defect_types=defect_types,
            patch_shapes=(3, 3),
            target_image_size=(224, 224),
        )
        
        self.assertFalse(library.is_empty())
        self.assertGreater(len(library), 0)
        self.assertIn("crack", library.defect_type_to_id)
    
    def test_query(self):
        """测试查询功能"""
        library = DefectFeatureLibrary(feature_dimension=self.feature_dim)
        
        # 先构建库
        features = np.random.randn(10, self.feature_dim, 3, 3).astype(np.float32)
        masks = np.ones((10, 224, 224), dtype=np.float32)
        defect_types = ["crack"] * 10
        
        library.build_from_defect_data(
            features=features,
            masks=masks,
            defect_types=defect_types,
            patch_shapes=(3, 3),
            target_image_size=(224, 224),
        )
        
        # 查询 - 注意：查询返回的是展平后的结果
        query_features = np.random.randn(5, self.feature_dim, 3, 3).astype(np.float32)
        distances, indices, labels = library.query(query_features, k=1)
        
        # 展平后的查询特征数量应该是 5*3*3 = 45
        expected_query_count = 5 * 3 * 3
        self.assertEqual(distances.shape[0], expected_query_count)
        self.assertEqual(labels.shape[0], expected_query_count)
    
    def test_save_and_load(self):
        """测试保存和加载"""
        library = DefectFeatureLibrary(feature_dimension=self.feature_dim)
        
        # 构建库
        features = np.random.randn(10, self.feature_dim, 3, 3).astype(np.float32)
        masks = np.ones((10, 224, 224), dtype=np.float32)
        defect_types = ["crack"] * 5 + ["scratch"] * 5
        
        library.build_from_defect_data(
            features=features,
            masks=masks,
            defect_types=defect_types,
            patch_shapes=(3, 3),
            target_image_size=(224, 224),
        )
        
        # 保存
        temp_dir = tempfile.mkdtemp()
        try:
            library.save(temp_dir)
            
            # 加载
            loaded_library = DefectFeatureLibrary.load(temp_dir)
            
            self.assertEqual(len(loaded_library), len(library))
            self.assertEqual(loaded_library.next_defect_id, library.next_defect_id)
            self.assertEqual(
                set(loaded_library.defect_type_to_id.keys()),
                set(library.defect_type_to_id.keys())
            )
        finally:
            shutil.rmtree(temp_dir)


class TestDualLibraryScorer(unittest.TestCase):
    """测试 DualLibraryScorer 类"""
    
    def setUp(self):
        """设置测试环境"""
        self.feature_dim = 128
        
        # 创建模拟的好品评分器
        self.normal_scorer = self._create_mock_normal_scorer()
        
        # 创建缺陷库
        self.defect_library = DefectFeatureLibrary(feature_dimension=self.feature_dim)
        
        # 构建缺陷库
        np.random.seed(42)
        features = np.random.randn(20, self.feature_dim, 3, 3).astype(np.float32)
        masks = np.ones((20, 224, 224), dtype=np.float32)
        defect_types = ["crack"] * 10 + ["scratch"] * 10
        
        self.defect_library.build_from_defect_data(
            features=features,
            masks=masks,
            defect_types=defect_types,
            patch_shapes=(3, 3),
            target_image_size=(224, 224),
        )
    
    def _create_mock_normal_scorer(self):
        """创建模拟的好品评分器"""
        class MockNormalScorer:
            def __init__(self):
                self.n_nearest_neighbours = 1
                self.detection_features = np.random.randn(100, 128).astype(np.float32)
            
            def predict(self, query_features):
                # 返回模拟的距离（模仿 NearestNeighbourScorer 的返回格式）
                if isinstance(query_features, list):
                    query = query_features[0]
                else:
                    query = query_features
                
                # 合并特征以获取正确的查询数量
                from patchcore.common import ConcatMerger
                merger = ConcatMerger()
                merged_query = merger.merge([query]) if isinstance(query, np.ndarray) else merger.merge(query_features)
                
                n_queries = merged_query.shape[0]  # N*H*W
                
                # 模拟距离：每个查询点都有k个最近邻距离
                k = self.n_nearest_neighbours
                query_distances = np.random.rand(n_queries, k) * 0.5
                query_nns = np.zeros((n_queries, k), dtype=np.int64)
                anomaly_scores = np.mean(query_distances, axis=-1)  # (N*H*W,)
                
                return anomaly_scores, query_distances, query_nns
            
            def imagelevel_nn(self, query):
                n = query.shape[0] if query.ndim > 1 else 1
                distances = np.random.rand(n, 1) * 0.5
                indices = np.zeros((n, 1), dtype=np.int64)
                return distances, indices
        
        return MockNormalScorer()
    
    def test_initialization(self):
        """测试初始化"""
        scorer = DualLibraryScorer(
            normal_scorer=self.normal_scorer,
            defect_library=self.defect_library,
            classification_strategy="distance_ratio",
        )
        
        self.assertEqual(
            scorer.classification_strategy,
            ClassificationStrategy.DISTANCE_RATIO
        )
        self.assertIsNotNone(scorer.defect_library)
    
    def test_classification_strategies(self):
        """测试不同的分类策略"""
        for strategy in ["nearest_wins", "distance_ratio", "softmax_prob"]:
            scorer = DualLibraryScorer(
                normal_scorer=self.normal_scorer,
                defect_library=self.defect_library,
                classification_strategy=strategy,
            )
            self.assertEqual(
                scorer.classification_strategy.value,
                strategy
            )
    
    def test_predict(self):
        """测试预测功能"""
        scorer = DualLibraryScorer(
            normal_scorer=self.normal_scorer,
            defect_library=self.defect_library,
            classification_strategy="distance_ratio",
        )
        
        # 准备查询特征
        batch_size = 5
        query_features = [np.random.randn(batch_size, self.feature_dim, 3, 3).astype(np.float32)]
        # patch_shapes 应该与 batch_size 匹配
        patch_shapes = [(3, 3)] * batch_size
        target_size = (224, 224)
        
        image_scores, segmentation_maps, defect_type_maps = scorer.predict(
            query_features=query_features,
            patch_shapes=patch_shapes,
            target_size=target_size,
        )
        
        self.assertEqual(len(image_scores), batch_size)
        self.assertIsNotNone(segmentation_maps)
        self.assertIsNotNone(defect_type_maps)
        self.assertEqual(len(segmentation_maps), batch_size)
        self.assertEqual(len(defect_type_maps), batch_size)
    
    def test_compute_anomaly_threshold(self):
        """测试异常阈值计算"""
        scorer = DualLibraryScorer(
            normal_scorer=self.normal_scorer,
            defect_library=self.defect_library,
            classification_strategy="distance_ratio",
        )
        
        # 准备正常特征
        normal_features = [self.normal_scorer.detection_features[:50]]
        
        threshold = scorer.compute_anomaly_threshold(
            normal_features, percentile=95.0
        )
        
        self.assertIsNotNone(threshold)
        self.assertGreater(threshold, 0)


class TestMaskDownsampling(unittest.TestCase):
    """测试 mask 下采样功能"""
    
    def test_nearest_interpolation(self):
        """测试最近邻插值保持二值性"""
        import torch.nn.functional as F
        
        # 创建随机二值 mask
        mask = torch.randint(0, 2, (1, 1, 224, 224)).float()
        
        # 下采样到目标大小
        target_h, target_w = 7, 7
        mask_down = F.interpolate(
            mask,
            size=(target_h, target_w),
            mode="nearest"
        )
        
        # 检查是否仍然保持二值性
        unique_values = torch.unique(mask_down)
        self.assertTrue(
            all(v in [0.0, 1.0] for v in unique_values),
            f"Mask should be binary, but got: {unique_values}"
        )
        
        # 检查形状
        self.assertEqual(mask_down.shape, (1, 1, target_h, target_w))


def run_tests():
    """运行所有测试"""
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestDefectFeatureLibrary))
    suite.addTests(loader.loadTestsFromTestCase(TestDualLibraryScorer))
    suite.addTests(loader.loadTestsFromTestCase(TestMaskDownsampling))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回是否成功
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)