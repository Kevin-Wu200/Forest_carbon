"""
森林非监督分类脚本
功能：对多光谱遥感影像进行非监督分类，计算森林覆盖率、乔木林覆盖率和面积
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.cluster import KMeans
from scipy import ndimage
from skimage.segmentation import slic
import os
import json
import pandas as pd
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from colorama import init, Fore, Back, Style
import warnings
warnings.filterwarnings('ignore')

# 并行处理相关导入
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import gc
import sys
import signal

# 初始化 colorama
init(autoreset=True)

# ========== 全局中断管理 ==========
# 全局变量用于跟踪中断状态
_interrupt_flag = False
_current_executor = None

def signal_handler(signum, frame):
    """
    信号处理器，处理 SIGINT (Ctrl+C) 和 SIGTERM

    Parameters:
    -----------
    signum : int
        信号编号
    frame : frame
        当前栈帧
    """
    global _interrupt_flag, _current_executor
    _interrupt_flag = True

    print(f"\n{Fore.YELLOW}⚠ 收到中断信号，正在停止程序...{Style.RESET_ALL}")

    # 如果有正在运行的进程池，尝试关闭它
    if _current_executor is not None:
        try:
            _current_executor.shutdown(wait=False, cancel_futures=True)
            print(f"{Fore.YELLOW}⚠ 已请求取消所有正在进行的任务{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ 关闭进程池时出错: {e}{Style.RESET_ALL}")

    sys.exit(1)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

from config import (CLASSIFICATION_CONFIG, SLIC_CONFIG, NDVI_THRESHOLDS,
                    BAND_INDICES, POST_PROCESSING, OUTPUT_CONFIG, AREA_CONFIG,
                    PARALLEL_CONFIG, get_memory_info)


# ========== 彩色输出辅助函数 ==========

def print_success(message):
    """打印成功信息（绿色）"""
    print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")

def print_info(message):
    """打印信息（蓝色）"""
    print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")

def print_warning(message):
    """打印警告信息（黄色）"""
    print(f"{Fore.YELLOW}⚠ {message}{Style.RESET_ALL}")

def print_error(message):
    """打印错误信息（红色）"""
    print(f"{Fore.RED}✗ {message}{Style.RESET_ALL}")

def print_step(step_number, total_steps, step_name):
    """打印步骤信息"""
    print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}步骤 {step_number}/{total_steps}: {step_name}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")

def print_header(text):
    """打印标题"""
    print(f"\n{Fore.BLUE}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}{text.center(60)}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")


class ForestClassifier:
    """森林分类器类"""

    def __init__(self, config=None):
        """
        初始化分类器

        Parameters:
        -----------
        config : dict
            分类配置参数
        """
        self.config = config if config else CLASSIFICATION_CONFIG
        self.data = None
        self.profile = None
        self.ndvi = None
        self.labels = None
        self.n_classes = self.config['n_clusters']

        # 并行处理配置
        self.parallel_config = PARALLEL_CONFIG

    def get_memory_usage(self):
        """
        获取当前进程内存使用情况

        Returns:
        --------
        dict
            内存使用信息
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            return {
                'rss_mb': mem_info.rss / (1024 * 1024),
                'vms_mb': mem_info.vms / (1024 * 1024)
            }
        except ImportError:
            return {'rss_mb': 0, 'vms_mb': 0}

    def check_memory_limit(self, verbose=True):
        """
        检查内存使用是否超过限制

        Parameters:
        -----------
        verbose : bool
            是否打印详细信息

        Returns:
        --------
        bool
            是否超过内存限制
        """
        mem_info = self.get_memory_usage()
        memory_info = get_memory_info()
        limit_mb = self.parallel_config['max_memory_mb']

        if verbose:
            print_info(f"当前内存使用: {mem_info['rss_mb']:.2f} MB / {limit_mb:.2f} MB")

        if mem_info['rss_mb'] > limit_mb:
            if verbose:
                print_warning(f"内存使用超过限制 ({mem_info['rss_mb']:.2f} MB > {limit_mb:.2f} MB)")
            return True
        return False

    def force_garbage_collection(self):
        """强制垃圾回收，释放内存"""
        gc.collect()
        if 'torch' in sys.modules:
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
        if 'tensorflow' in sys.modules:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
            except:
                pass

    def calculate_chunk_size(self, height, width, n_bands, dtype=np.float32):
        """
        计算分块大小

        Parameters:
        -----------
        height : int
            影像高度
        width : int
            影像宽度
        n_bands : int
            波段数
        dtype : numpy dtype
            数据类型

        Returns:
        --------
        tuple
            (chunk_height, chunk_width)
        """
        # 计算单像素内存占用（字节）
        bytes_per_pixel = np.dtype(dtype).itemsize * n_bands

        # 计算目标像素数（基于内存限制）
        max_bytes = self.parallel_config['chunk_size_mb'] * 1024 * 1024
        target_pixels = max_bytes / bytes_per_pixel

        # 计算分块尺寸（保持近似正方形）
        chunk_size = int(np.sqrt(target_pixels))

        # 限制最小和最大分块大小
        chunk_size = max(256, min(chunk_size, 2048))

        # 计算需要多少块
        n_chunks_h = int(np.ceil(height / chunk_size))
        n_chunks_w = int(np.ceil(width / chunk_size))

        # 调整分块大小以均匀分布
        chunk_height = int(np.ceil(height / n_chunks_h))
        chunk_width = int(np.ceil(width / n_chunks_w))

        # 添加重叠区域
        overlap = self.parallel_config['chunk_overlap']

        return chunk_height, chunk_width, overlap

    def read_tif(self, tif_path):
        """
        读取TIF文件

        Parameters:
        -----------
        tif_path : str
            TIF文件路径

        Returns:
        --------
        bool
            是否成功读取
        """
        try:
            print_info(f"正在读取文件: {tif_path}")
            with rasterio.open(tif_path) as src:
                self.profile = src.profile
                # 读取所有波段
                self.data = src.read()
                print_success(f"成功读取TIF文件")
                print_info(f"影像尺寸: {self.data.shape[1]} x {self.data.shape[2]}")
                print_info(f"波段数量: {self.data.shape[0]}")
                return True
        except Exception as e:
            print_error(f"读取TIF文件失败: {e}")
            return False

    def calculate_ndvi(self):
        """计算归一化植被指数（NDVI）"""
        print_info("正在计算NDVI植被指数...")

        red_band = self.data[BAND_INDICES['red']]
        nir_band = self.data[BAND_INDICES['nir']]

        # 避免除以零（使用numpy的向量化操作）
        denominator = nir_band + red_band
        denominator = np.where(denominator == 0, 1e-10, denominator)

        # 使用numpy的向量化操作计算NDVI（已自动利用多核CPU）
        self.ndvi = (nir_band - red_band) / denominator

        print_success(f"NDVI计算完成")
        print_info(f"NDVI范围: [{self.ndvi.min():.3f}, {self.ndvi.max():.3f}]")
        print_info(f"NDVI均值: {self.ndvi.mean():.3f}")

        # 检查内存使用
        if self.check_memory_limit():
            print_warning("NDVI计算后内存使用较高，建议启用分块处理")
            self.force_garbage_collection()

    def preprocess_data(self):
        """
        数据预处理

        Returns:
        --------
        np.ndarray
            预处理后的特征矩阵
        """
        # 将数据重塑为 (n_pixels, n_bands) 的二维数组
        n_bands, height, width = self.data.shape
        features = self.data.reshape(n_bands, -1).T

        # 移除无效值（NaN或Inf）
        valid_mask = np.all(np.isfinite(features), axis=1)
        features_clean = features[valid_mask]

        # 归一化处理
        features_min = features_clean.min(axis=0)
        features_max = features_clean.max(axis=0)
        features_normalized = (features_clean - features_min) / (features_max - features_min + 1e-10)

        print(f"预处理完成，有效像素数: {len(features_normalized)}")
        return features_normalized, valid_mask

    def classify(self):
        """
        执行分类（根据配置选择SLIC或K-means）

        Returns:
        --------
        bool
            是否成功分类
        """
        if self.data is None:
            print("错误: 未读取TIF文件")
            return False

        algorithm = self.config.get('algorithm', 'slic')

        if algorithm == 'slic':
            return self.classify_slic()
        elif algorithm == 'kmeans':
            return self.classify_kmeans()
        else:
            print(f"错误: 不支持的算法 '{algorithm}'")
            return False

    def classify_kmeans(self):
        """
        执行K-means非监督分类

        Returns:
        --------
        bool
            是否成功分类
        """
        print_info("开始K-means聚类分类...")

        # 数据预处理
        features_normalized, valid_mask = self.preprocess_data()

        # 执行K-means聚类（添加verbose=True显示迭代信息）
        print_info(f"聚类参数: K={self.config['n_clusters']}, n_init={self.config['n_init']}, max_iter={self.config['max_iter']}")

        kmeans = KMeans(
            n_clusters=self.config['n_clusters'],
            random_state=self.config['random_state'],
            n_init=self.config['n_init'],
            max_iter=self.config['max_iter'],
            verbose=1
        )

        print_info("正在进行聚类训练...")
        labels = kmeans.fit_predict(features_normalized)

        # 重构标签矩阵
        n_bands, height, width = self.data.shape
        full_labels = np.zeros(height * width, dtype=np.int32)
        full_labels[:] = -1  # 无效值标记为-1
        full_labels[valid_mask] = labels
        self.labels = full_labels.reshape(height, width)

        print_success(f"K-means分类完成，共 {self.n_classes} 个类别")
        print_info(f"聚类收敛迭代次数: {kmeans.n_iter_}")

        # 释放内存
        del features_normalized
        self.force_garbage_collection()

        return True

    def classify_slic(self):
        """
        执行SLIC超像素分割

        Returns:
        --------
        bool
            是否成功分类
        """
        print_info("开始SLIC超像素分割...")

        # 准备RGB图像用于SLIC
        n_bands, height, width = self.data.shape

        # 使用前三个波段（RGB）进行超像素分割
        # 如果波段不足3个，使用可用的波段
        if n_bands >= 3:
            rgb_image = np.stack([self.data[0], self.data[1], self.data[2]], axis=2)
        else:
            rgb_image = np.stack([self.data[i] for i in range(n_bands)], axis=2)

        # 归一化到0-255范围（使用向量化操作）
        print_info("正在归一化图像数据...")
        rgb_min = rgb_image.min(axis=(0, 1), keepdims=True)
        rgb_max = rgb_image.max(axis=(0, 1), keepdims=True)
        rgb_range = rgb_max - rgb_min
        rgb_range[rgb_range == 0] = 1  # 避免除以零

        rgb_normalized = np.clip((rgb_image - rgb_min) / rgb_range * 255, 0, 255).astype(np.uint8)

        # 执行SLIC超像素分割
        print_info(f"SLIC参数: n_segments={SLIC_CONFIG['n_segments']}, compactness={SLIC_CONFIG['compactness']}")
        print_info("正在执行超像素分割...")
        segments = slic(
            rgb_normalized,
            n_segments=SLIC_CONFIG['n_segments'],
            compactness=SLIC_CONFIG['compactness'],
            max_num_iter=SLIC_CONFIG['max_num_iter'],
            sigma=SLIC_CONFIG['sigma'],
            min_size_factor=SLIC_CONFIG['min_size_factor'],
            max_size_factor=SLIC_CONFIG['max_size_factor'],
            enforce_connectivity=SLIC_CONFIG['enforce_connectivity'],
            start_label=0
        )

        num_superpixels = len(np.unique(segments))
        print_success(f"SLIC分割完成，共生成 {num_superpixels} 个超像素")

        # 释放归一化后的图像内存
        del rgb_normalized
        self.force_garbage_collection()

        # 基于超像素进行分类
        self.labels = self.classify_superpixels(segments)

        # 更新类别数量
        self.n_classes = len(np.unique(self.labels[self.labels >= 0]))

        print_success(f"超像素分类完成，共 {self.n_classes} 个类别")
        return True

    def classify_superpixels(self, segments):
        """
        基于超像素特征进行分类

        Parameters:
        -----------
        segments : np.ndarray
            超像素标签矩阵

        Returns:
        --------
        np.ndarray
            分类结果标签矩阵
        """
        height, width = segments.shape
        n_bands = self.data.shape[0]

        # 获取所有超像素ID
        superpixel_ids = np.unique(segments)
        num_superpixels = len(superpixel_ids)

        # 提取每个超像素的特征（使用并行处理）
        print_info(f"正在提取 {num_superpixels} 个超像素的特征...")
        print_info(f"并行进程数: {self.parallel_config['feature_extraction_n_jobs']}")

        # 定义特征提取函数
        def extract_single_superpixel_features(sp_id, data, ndvi, n_bands):
            mask = (segments == sp_id)
            features = []

            # 计算每个波段的平均值
            for band_idx in range(n_bands):
                band_data = data[band_idx][mask]
                mean_value = np.mean(band_data) if len(band_data) > 0 else 0
                features.append(mean_value)

            # 添加NDVI特征
            ndvi_values = ndvi[mask]
            mean_ndvi = np.mean(ndvi_values) if len(ndvi_values) > 0 else 0
            features.append(mean_ndvi)

            return sp_id, features

        # 使用并行处理提取特征
        n_jobs = self.parallel_config['feature_extraction_n_jobs']

        if n_jobs > 1 and num_superpixels > 100:
            # 使用多进程处理大量超像素
            global _current_executor
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                _current_executor = executor
                futures = []
                for sp_id in superpixel_ids:
                    # 检查是否被中断
                    if _interrupt_flag:
                        raise KeyboardInterrupt("用户中断了程序执行")
                    future = executor.submit(
                        extract_single_superpixel_features,
                        sp_id, self.data, self.ndvi, n_bands
                    )
                    futures.append(future)

                # 收集结果
                results = []
                for future in tqdm(futures, desc="提取超像素特征", unit="超像素"):
                    # 检查是否被中断
                    if _interrupt_flag:
                        raise KeyboardInterrupt("用户中断了程序执行")
                    sp_id, features = future.result()
                    results.append((sp_id, features))

                _current_executor = None

                # 按原始顺序排序
                results.sort(key=lambda x: x[0])
                superpixel_features = np.array([f[1] for f in results])
        else:
            # 单进程处理（超像素数量较少时）
            superpixel_features = []
            for sp_id in tqdm(superpixel_ids, desc="提取超像素特征", unit="超像素"):
                mask = (segments == sp_id)

                # 计算每个波段的平均值
                features = []
                for band_idx in range(n_bands):
                    band_data = self.data[band_idx][mask]
                    mean_value = np.mean(band_data) if len(band_data) > 0 else 0
                    features.append(mean_value)

                # 添加NDVI特征
                ndvi_values = self.ndvi[mask]
                mean_ndvi = np.mean(ndvi_values) if len(ndvi_values) > 0 else 0
                features.append(mean_ndvi)

                superpixel_features.append(features)

            superpixel_features = np.array(superpixel_features)

        # 归一化特征
        print_info("正在归一化超像素特征...")
        features_min = superpixel_features.min(axis=0)
        features_max = superpixel_features.max(axis=0)
        features_normalized = (superpixel_features - features_min) / (features_max - features_min + 1e-10)

        # 使用K-means对超像素进行聚类
        print_info("正在进行超像素聚类...")

        kmeans = KMeans(
            n_clusters=self.config['n_clusters'],
            random_state=self.config['random_state'],
            n_init=self.config['n_init'],
            max_iter=self.config['max_iter'],
            verbose=1
        )

        superpixel_labels = kmeans.fit_predict(features_normalized)

        # 将超像素标签映射回像素级别
        print_info("正在将超像素标签映射到像素级别...")
        pixel_labels = np.zeros((height, width), dtype=np.int32)
        for i, sp_id in tqdm(enumerate(superpixel_ids), desc="映射标签", unit="超像素"):
            pixel_labels[segments == sp_id] = superpixel_labels[i]

        # 释放内存
        del superpixel_features, features_normalized
        self.force_garbage_collection()

        return pixel_labels

    def post_process(self):
        """后处理：过滤小斑块"""
        if POST_PROCESSING['use_majority_filter']:
            # 使用众数滤波去除噪声
            min_patch = POST_PROCESSING['min_patch_size']
            print_info(f"正在进行后处理，过滤小于 {min_patch} 像素的小斑块...")

            filtered_labels = self.labels.copy()

            # 使用并行处理标记小斑块
            print_info("正在标记小斑块...")
            print_info(f"并行进程数: {self.parallel_config['post_process_n_jobs']}")

            def process_single_class(label, labels, min_patch):
                """处理单个类别的小斑块"""
                mask = (labels == label).astype(np.uint8)
                labeled_array, num_features = ndimage.label(mask)

                small_patches = []
                for i in range(1, num_features + 1):
                    if np.sum(labeled_array == i) < min_patch:
                        small_patches.append(labeled_array == i)

                return label, small_patches

            n_jobs = self.parallel_config['post_process_n_jobs']

            if n_jobs > 1 and self.n_classes > 2:
                # 使用多进程并行处理类别
                global _current_executor
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    _current_executor = executor
                    futures = []
                    for label in range(self.n_classes):
                        # 检查是否被中断
                        if _interrupt_flag:
                            raise KeyboardInterrupt("用户中断了程序执行")
                        future = executor.submit(
                            process_single_class,
                            label, self.labels, min_patch
                        )
                        futures.append(future)

                    # 收集结果
                    for future in tqdm(futures, desc="处理类别", unit="类别"):
                        # 检查是否被中断
                        if _interrupt_flag:
                            raise KeyboardInterrupt("用户中断了程序执行")
                        label, small_patches = future.result()
                        for patch_mask in small_patches:
                            filtered_labels[patch_mask] = -1

                    _current_executor = None
            else:
                # 单进程处理
                for label in tqdm(range(self.n_classes), desc="处理类别", unit="类别"):
                    mask = (self.labels == label).astype(np.uint8)
                    labeled_array, num_features = ndimage.label(mask)
                    for i in range(1, num_features + 1):
                        if np.sum(labeled_array == i) < min_patch:
                            # 将小斑块标记为需要处理
                            filtered_labels[labeled_array == i] = -1

            # 用周围主要类别填充
            print_info("正在填充小斑块...")
            from scipy.ndimage import mode
            for i in range(self.n_classes):
                mask = (filtered_labels == -1)
                if np.any(mask):
                    # 使用卷积找到周围的主要类别
                    kernel = np.ones((3, 3))
                    for iteration in tqdm(range(2), desc="填充迭代", unit="次"):
                        filled = filtered_labels.copy()
                        for y in range(1, self.labels.shape[0] - 1):
                            for x in range(1, self.labels.shape[1] - 1):
                                if filtered_labels[y, x] == -1:
                                    neighbors = filtered_labels[y-1:y+2, x-1:x+2]
                                    neighbors = neighbors[neighbors != -1]
                                    if len(neighbors) > 0:
                                        filled[y, x] = mode(neighbors)[0][0]
                        filtered_labels = filled

            self.labels = filtered_labels
            print_success("后处理完成：已过滤小斑块")
        else:
            print_warning("后处理已禁用")

    def identify_forest_classes(self):
        """
        根据NDVI值识别森林类别

        Returns:
        --------
        dict
            包含各类别信息的字典
        """
        print_info("正在识别森林类别...")

        # 计算每个类别的平均NDVI值
        class_stats = {}
        for i in range(self.n_classes):
            mask = (self.labels == i)
            if np.any(mask):
                class_ndvi = self.ndvi[mask]
                class_stats[i] = {
                    'mean_ndvi': np.mean(class_ndvi),
                    'pixel_count': np.sum(mask)
                }

        # 根据NDVI阈值识别森林类别
        forest_classes = []
        arbor_forest_classes = []

        for class_id, stats in class_stats.items():
            mean_ndvi = stats['mean_ndvi']
            if mean_ndvi >= NDVI_THRESHOLDS['arbor_forest_min']:
                arbor_forest_classes.append(class_id)
                forest_classes.append(class_id)
            elif mean_ndvi >= NDVI_THRESHOLDS['forest_min']:
                forest_classes.append(class_id)

        print_success(f"森林类别（NDVI >= {NDVI_THRESHOLDS['forest_min']}）: {forest_classes}")
        print_success(f"乔木林类别（NDVI >= {NDVI_THRESHOLDS['arbor_forest_min']}）: {arbor_forest_classes}")

        return {
            'forest_classes': forest_classes,
            'arbor_forest_classes': arbor_forest_classes,
            'class_stats': class_stats
        }

    def calculate_statistics(self, class_info):
        """
        计算森林覆盖率和面积

        Parameters:
        -----------
        class_info : dict
            类别信息字典

        Returns:
        --------
        dict
            统计结果
        """
        print_info("正在计算森林覆盖率...")

        # 计算总像素数（排除无效值）
        total_pixels = np.sum(self.labels >= 0)

        # 计算森林像素数
        forest_mask = np.isin(self.labels, class_info['forest_classes'])
        forest_pixels = np.sum(forest_mask)

        # 计算乔木林像素数
        arbor_forest_mask = np.isin(self.labels, class_info['arbor_forest_classes'])
        arbor_forest_pixels = np.sum(arbor_forest_mask)

        # 计算像素面积
        if AREA_CONFIG['pixel_area']:
            pixel_area_m2 = AREA_CONFIG['pixel_area']
        else:
            # 从元数据中获取像素面积
            if 'transform' in self.profile:
                pixel_width = abs(self.profile['transform'][0])
                pixel_height = abs(self.profile['transform'][4])
                pixel_area_m2 = pixel_width * pixel_height
            else:
                print_warning("无法从元数据获取像素面积，假设为30m x 30m（Landsat）")
                pixel_area_m2 = 30 * 30

        # 转换为公顷
        pixel_area_ha = pixel_area_m2 / 10000

        # 计算面积
        total_area_ha = total_pixels * pixel_area_ha
        forest_area_ha = forest_pixels * pixel_area_ha
        arbor_forest_area_ha = arbor_forest_pixels * pixel_area_ha

        # 计算覆盖率
        forest_coverage = (forest_pixels / total_pixels) * 100
        arbor_forest_coverage = (arbor_forest_pixels / total_pixels) * 100

        stats = {
            'total_pixels': int(total_pixels),
            'total_area_hectares': round(total_area_ha, 2),
            'forest_pixels': int(forest_pixels),
            'forest_area_hectares': round(forest_area_ha, 2),
            'forest_coverage_percent': round(forest_coverage, 2),
            'arbor_forest_pixels': int(arbor_forest_pixels),
            'arbor_forest_area_hectares': round(arbor_forest_area_ha, 2),
            'arbor_forest_coverage_percent': round(arbor_forest_coverage, 2),
            'pixel_area_square_meters': round(pixel_area_m2, 2),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 使用彩色输出显示统计结果
        print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}统计结果{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}总面积: {Fore.GREEN}{stats['total_area_hectares']} 公顷{Style.RESET_ALL}")
        print(f"{Fore.WHITE}森林覆盖率: {Fore.GREEN}{stats['forest_coverage_percent']}%{Style.RESET_ALL}")
        print(f"{Fore.WHITE}森林面积: {Fore.GREEN}{stats['forest_area_hectares']} 公顷{Style.RESET_ALL}")
        print(f"{Fore.WHITE}乔木林覆盖率: {Fore.GREEN}{stats['arbor_forest_coverage_percent']}%{Style.RESET_ALL}")
        print(f"{Fore.WHITE}乔木林面积: {Fore.GREEN}{stats['arbor_forest_area_hectares']} 公顷{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}\n")

        return stats

    def save_classified_tif(self, output_path):
        """
        保存分类结果为TIF文件

        Parameters:
        -----------
        output_path : str
            输出文件路径
        """
        print_info(f"正在保存分类结果到: {output_path}")
        profile = self.profile.copy()
        profile.update({
            'count': 1,
            'dtype': 'int32',
            'nodata': -1
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(self.labels.astype(np.int32), 1)

        print_success(f"分类结果已保存: {output_path}")

    def save_ndvi_tif(self, output_path):
        """
        保存NDVI为TIF文件

        Parameters:
        -----------
        output_path : str
            输出文件路径
        """
        print_info(f"正在保存NDVI到: {output_path}")
        profile = self.profile.copy()
        profile.update({
            'count': 1,
            'dtype': 'float32'
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(self.ndvi.astype(np.float32), 1)

        print_success(f"NDVI已保存: {output_path}")

    def visualize_results(self, output_path, class_info=None):
        """
        可视化分类结果

        Parameters:
        -----------
        output_path : str
            输出图片路径
        class_info : dict
            类别信息（可选）
        """
        if not OUTPUT_CONFIG['visualization']:
            print_warning("可视化已禁用")
            return

        print_info(f"正在生成可视化结果...")
        algorithm = self.config.get('algorithm', 'slic')

        # 如果是SLIC算法，创建2x3布局；否则保持2x2布局
        if algorithm == 'slic':
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 原始影像（使用前三个波段显示为RGB）
        rgb_image = np.stack([
            self.data[2],  # Red
            self.data[1],  # Green
            self.data[0]   # Blue
        ], axis=2)

        # 归一化显示
        rgb_display = np.clip(rgb_image / rgb_image.max() * 255, 0, 255).astype(np.uint8)
        axes[0, 0].imshow(rgb_display)
        axes[0, 0].set_title('原始影像 (RGB)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # NDVI
        ndvi_display = self.ndvi
        axes[0, 1].imshow(ndvi_display, cmap='RdYlGn', vmin=-0.2, vmax=1)
        axes[0, 1].set_title('NDVI 植被指数', fontsize=14, fontweight='bold')
        plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046, pad=0.04)
        axes[0, 1].axis('off')

        if algorithm == 'slic':
            # 显示超像素边界（在RGB影像上）
            from skimage.segmentation import mark_boundaries
            rgb_float = rgb_display.astype(np.float32) / 255.0
            boundaries = mark_boundaries(rgb_float, self.labels, color=(1, 0, 0), mode='thick')
            axes[0, 2].imshow(boundaries)
            axes[0, 2].set_title('超像素边界', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')

        # 分类结果
        colors = plt.cm.tab20(np.linspace(0, 1, self.n_classes))
        cmap = ListedColormap(colors)

        if algorithm == 'slic':
            axes[1, 0].imshow(self.labels, cmap=cmap)
            axes[1, 0].set_title(f'SLIC超像素分类结果 (K={self.n_classes})', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')

            # 超像素统计
            unique_labels, counts = np.unique(self.labels[self.labels >= 0], return_counts=True)
            stats_text = f"类别数: {len(unique_labels)}\n"
            stats_text += f"超像素总数: {np.sum(counts)}\n"
            stats_text += f"平均超像素大小: {np.mean(counts):.0f} 像素"
            axes[1, 1].text(0.5, 0.5, stats_text, ha='center', va='center',
                            fontsize=12, transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('超像素统计', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')

            # 森林分布
            if class_info:
                forest_mask = np.isin(self.labels, class_info['forest_classes'])
                arbor_mask = np.isin(self.labels, class_info['arbor_forest_classes'])

                # 创建森林分布图
                forest_map = np.zeros_like(self.ndvi)
                forest_map[arbor_mask] = 2  # 乔木林
                forest_map[forest_mask & ~arbor_mask] = 1  # 其他森林

                forest_colors = ['#d3d3d3', '#90EE90', '#228B22']  # 非森林, 森林, 乔木林
                forest_cmap = ListedColormap(forest_colors)

                axes[1, 2].imshow(forest_map, cmap=forest_cmap, vmin=0, vmax=2)
                axes[1, 2].set_title('森林分布', fontsize=14, fontweight='bold')

                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=forest_colors[0], label='非森林'),
                    Patch(facecolor=forest_colors[1], label='森林'),
                    Patch(facecolor=forest_colors[2], label='乔木林')
                ]
                axes[1, 2].legend(handles=legend_elements, loc='upper right', fontsize=10)

            axes[1, 2].axis('off')
        else:
            axes[1, 0].imshow(self.labels, cmap=cmap)
            axes[1, 0].set_title(f'非监督分类结果 (K={self.n_classes})', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')

            # 森林分布
            if class_info:
                forest_mask = np.isin(self.labels, class_info['forest_classes'])
                arbor_mask = np.isin(self.labels, class_info['arbor_forest_classes'])

                # 创建森林分布图
                forest_map = np.zeros_like(self.ndvi)
                forest_map[arbor_mask] = 2  # 乔木林
                forest_map[forest_mask & ~arbor_mask] = 1  # 其他森林

                forest_colors = ['#d3d3d3', '#90EE90', '#228B22']  # 非森林, 森林, 乔木林
                forest_cmap = ListedColormap(forest_colors)

                axes[1, 1].imshow(forest_map, cmap=forest_cmap, vmin=0, vmax=2)
                axes[1, 1].set_title('森林分布', fontsize=14, fontweight='bold')

                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor=forest_colors[0], label='非森林'),
                    Patch(facecolor=forest_colors[1], label='森林'),
                    Patch(facecolor=forest_colors[2], label='乔木林')
                ]
                axes[1, 1].legend(handles=legend_elements, loc='upper right', fontsize=10)

            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()

        print_success(f"可视化结果已保存: {output_path}")


def parse_args():
    """
    解析命令行参数

    Returns:
    --------
    argparse.Namespace
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='森林非监督分类系统 - 基于K-means和SLIC算法的多光谱遥感影像分类',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 输入输出参数
    parser.add_argument('input', help='输入TIF文件路径')
    parser.add_argument('-o', '--output-dir', default=OUTPUT_CONFIG['output_dir'],
                       help='输出目录')
    parser.add_argument('--prefix', default='',
                       help='输出文件前缀')

    # 分类算法参数
    parser.add_argument('-a', '--algorithm', choices=['slic', 'kmeans'],
                       default=CLASSIFICATION_CONFIG['algorithm'],
                       help='分类算法')
    parser.add_argument('-k', '--n-clusters', type=int,
                       default=CLASSIFICATION_CONFIG['n_clusters'],
                       help='聚类数量')
    parser.add_argument('--random-state', type=int,
                       default=CLASSIFICATION_CONFIG['random_state'],
                       help='随机种子')
    parser.add_argument('--n-init', type=int,
                       default=CLASSIFICATION_CONFIG['n_init'],
                       help='K-means重复次数')
    parser.add_argument('--max-iter', type=int,
                       default=CLASSIFICATION_CONFIG['max_iter'],
                       help='最大迭代次数')

    # SLIC参数
    slic_group = parser.add_argument_group('SLIC超像素参数')
    slic_group.add_argument('--n-segments', type=int,
                           default=SLIC_CONFIG['n_segments'],
                           help='超像素数量')
    slic_group.add_argument('--compactness', type=float,
                           default=SLIC_CONFIG['compactness'],
                           help='紧凑度参数')
    slic_group.add_argument('--slic-max-iter', type=int,
                           default=SLIC_CONFIG['max_num_iter'],
                           help='SLIC最大迭代次数')
    slic_group.add_argument('--sigma', type=float,
                           default=SLIC_CONFIG['sigma'],
                           help='高斯平滑标准差')

    # NDVI阈值参数
    ndvi_group = parser.add_argument_group('NDVI阈值参数')
    ndvi_group.add_argument('--forest-ndvi', type=float,
                          default=NDVI_THRESHOLDS['forest_min'],
                          help='森林NDVI最小值')
    ndvi_group.add_argument('--arbor-forest-ndvi', type=float,
                          default=NDVI_THRESHOLDS['arbor_forest_min'],
                          help='乔木林NDVI最小值')

    # 波段配置参数
    band_group = parser.add_argument_group('波段配置')
    band_group.add_argument('--red-band', type=int,
                          default=BAND_INDICES['red'],
                          help='红光波段索引（从0开始）')
    band_group.add_argument('--nir-band', type=int,
                          default=BAND_INDICES['nir'],
                          help='近红外波段索引（从0开始）')

    # 后处理参数
    post_group = parser.add_argument_group('后处理参数')
    post_group.add_argument('--min-patch-size', type=int,
                           default=POST_PROCESSING['min_patch_size'],
                           help='最小斑块大小（像素数）')
    post_group.add_argument('--no-post-process', action='store_true',
                           help='禁用后处理')

    # 输出控制参数
    output_group = parser.add_argument_group('输出控制')
    output_group.add_argument('--no-visualization', action='store_true',
                            help='禁用可视化')
    output_group.add_argument('--no-ndvi', action='store_true',
                            help='不保存NDVI文件')

    # 并行处理参数
    parallel_group = parser.add_argument_group('并行处理')
    parallel_group.add_argument('-j', '--n-jobs', type=int,
                              default=PARALLEL_CONFIG['n_jobs'],
                              help='并行进程数（-1表示使用所有CPU核心）')
    parallel_group.add_argument('--max-memory', type=float,
                              default=PARALLEL_CONFIG['max_memory_mb'],
                              help='最大内存限制MB')

    return parser.parse_args()


def interactive_input():
    """
    交互式获取用户输入

    Returns:
    --------
    argparse.Namespace
        解析后的参数
    """
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}森林非监督分类系统 - 交互式配置{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

    # 创建一个 Namespace 对象来存储所有参数
    args = argparse.Namespace()

    # 输入输出参数
    print(f"{Fore.YELLOW}【输入输出配置】{Style.RESET_ALL}")
    while True:
        default_file = ""
        if os.path.exists('20201023.tif'):
            default_file = "20201023.tif"
        input_path = input(f"输入TIF文件路径 [{default_file}]: ").strip()
        if not input_path:
            input_path = default_file
        if input_path and os.path.exists(input_path):
            args.input = input_path
            break
        print(f"{Fore.RED}错误: 文件不存在，请重新输入{Style.RESET_ALL}")

    args.output_dir = input(f"输出目录 [{OUTPUT_CONFIG['output_dir']}]: ").strip()
    if not args.output_dir:
        args.output_dir = OUTPUT_CONFIG['output_dir']

    args.prefix = input("输出文件前缀 (可选): ").strip()

    # 分类算法参数
    print(f"\n{Fore.YELLOW}【分类算法配置】{Style.RESET_ALL}")
    print(f"  1. SLIC超像素分割 (推荐)")
    print(f"  2. K-means聚类")
    while True:
        choice = input(f"选择分类算法 [1-2] (默认: 1): ").strip()
        if not choice:
            args.algorithm = 'slic'
            break
        if choice == '1':
            args.algorithm = 'slic'
            break
        elif choice == '2':
            args.algorithm = 'kmeans'
            break
        print(f"{Fore.RED}错误: 请输入 1 或 2{Style.RESET_ALL}")

    args.n_clusters = input(f"聚类数量 [{CLASSIFICATION_CONFIG['n_clusters']}]: ").strip()
    if not args.n_clusters:
        args.n_clusters = CLASSIFICATION_CONFIG['n_clusters']
    else:
        args.n_clusters = int(args.n_clusters)

    args.random_state = input(f"随机种子 [{CLASSIFICATION_CONFIG['random_state']}]: ").strip()
    if not args.random_state:
        args.random_state = CLASSIFICATION_CONFIG['random_state']
    else:
        args.random_state = int(args.random_state)

    args.n_init = input(f"K-means重复次数 [{CLASSIFICATION_CONFIG['n_init']}]: ").strip()
    if not args.n_init:
        args.n_init = CLASSIFICATION_CONFIG['n_init']
    else:
        args.n_init = int(args.n_init)

    args.max_iter = input(f"最大迭代次数 [{CLASSIFICATION_CONFIG['max_iter']}]: ").strip()
    if not args.max_iter:
        args.max_iter = CLASSIFICATION_CONFIG['max_iter']
    else:
        args.max_iter = int(args.max_iter)

    # SLIC参数（仅在选择SLIC时询问）
    if args.algorithm == 'slic':
        print(f"\n{Fore.YELLOW}【SLIC超像素参数】{Style.RESET_ALL}")
        args.n_segments = input(f"超像素数量 [{SLIC_CONFIG['n_segments']}]: ").strip()
        if not args.n_segments:
            args.n_segments = SLIC_CONFIG['n_segments']
        else:
            args.n_segments = int(args.n_segments)

        args.compactness = input(f"紧凑度参数 [{SLIC_CONFIG['compactness']}]: ").strip()
        if not args.compactness:
            args.compactness = SLIC_CONFIG['compactness']
        else:
            args.compactness = float(args.compactness)

        args.slic_max_iter = input(f"SLIC最大迭代次数 [{SLIC_CONFIG['max_num_iter']}]: ").strip()
        if not args.slic_max_iter:
            args.slic_max_iter = SLIC_CONFIG['max_num_iter']
        else:
            args.slic_max_iter = int(args.slic_max_iter)

        args.sigma = input(f"高斯平滑标准差 [{SLIC_CONFIG['sigma']}]: ").strip()
        if not args.sigma:
            args.sigma = SLIC_CONFIG['sigma']
        else:
            args.sigma = float(args.sigma)
    else:
        # K-means时使用SLIC默认值
        args.n_segments = SLIC_CONFIG['n_segments']
        args.compactness = SLIC_CONFIG['compactness']
        args.slic_max_iter = SLIC_CONFIG['max_num_iter']
        args.sigma = SLIC_CONFIG['sigma']

    # NDVI阈值参数
    print(f"\n{Fore.YELLOW}【NDVI阈值参数】{Style.RESET_ALL}")
    args.forest_ndvi = input(f"森林NDVI最小值 [{NDVI_THRESHOLDS['forest_min']}]: ").strip()
    if not args.forest_ndvi:
        args.forest_ndvi = NDVI_THRESHOLDS['forest_min']
    else:
        args.forest_ndvi = float(args.forest_ndvi)

    args.arbor_forest_ndvi = input(f"乔木林NDVI最小值 [{NDVI_THRESHOLDS['arbor_forest_min']}]: ").strip()
    if not args.arbor_forest_ndvi:
        args.arbor_forest_ndvi = NDVI_THRESHOLDS['arbor_forest_min']
    else:
        args.arbor_forest_ndvi = float(args.arbor_forest_ndvi)

    # 波段配置参数
    print(f"\n{Fore.YELLOW}【波段配置】{Style.RESET_ALL}")
    print(f"  Landsat 8: Red=3, NIR=4")
    print(f"  Sentinel-2: Red=3, NIR=7")
    args.red_band = input(f"红光波段索引 [{BAND_INDICES['red']}]: ").strip()
    if not args.red_band:
        args.red_band = BAND_INDICES['red']
    else:
        args.red_band = int(args.red_band)

    args.nir_band = input(f"近红外波段索引 [{BAND_INDICES['nir']}]: ").strip()
    if not args.nir_band:
        args.nir_band = BAND_INDICES['nir']
    else:
        args.nir_band = int(args.nir_band)

    # 后处理参数
    print(f"\n{Fore.YELLOW}【后处理参数】{Style.RESET_ALL}")
    args.min_patch_size = input(f"最小斑块大小（像素数）[{POST_PROCESSING['min_patch_size']}]: ").strip()
    if not args.min_patch_size:
        args.min_patch_size = POST_PROCESSING['min_patch_size']
    else:
        args.min_patch_size = int(args.min_patch_size)

    use_post = input(f"启用后处理？ [Y/n] (默认: Y): ").strip().lower()
    args.no_post_process = (use_post == 'n' or use_post == 'no')

    # 输出控制参数
    print(f"\n{Fore.YELLOW}【输出控制】{Style.RESET_ALL}")
    use_vis = input(f"生成可视化结果？ [Y/n] (默认: Y): ").strip().lower()
    args.no_visualization = (use_vis == 'n' or use_vis == 'no')

    use_ndvi = input(f"保存NDVI文件？ [Y/n] (默认: Y): ").strip().lower()
    args.no_ndvi = (use_ndvi == 'n' or use_ndvi == 'no')

    # 并行处理参数
    print(f"\n{Fore.YELLOW}【并行处理】{Style.RESET_ALL}")
    print(f"  当前系统CPU核心数: {PARALLEL_CONFIG['n_jobs']}")
    args.n_jobs = input(f"并行进程数 (默认: 使用所有核心): ").strip()
    if not args.n_jobs:
        args.n_jobs = PARALLEL_CONFIG['n_jobs']
    else:
        args.n_jobs = int(args.n_jobs)

    args.max_memory = input(f"最大内存限制MB [{PARALLEL_CONFIG['max_memory_mb']:.0f}]: ").strip()
    if not args.max_memory:
        args.max_memory = PARALLEL_CONFIG['max_memory_mb']
    else:
        args.max_memory = float(args.max_memory)

    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}配置完成！即将开始处理...{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")

    return args


def main(input_tif, config=None):
    """
    主函数：执行完整的分类流程

    Parameters:
    -----------
    input_tif : str
        输入TIF文件路径
    config : dict, optional
        配置参数字典，如果提供则覆盖默认配置
    """
    global _current_executor

    # 如果提供了配置，则使用它来覆盖默认配置
    if config is not None:
        # 更新分类配置
        CLASSIFICATION_CONFIG.update(config.get('classification', {}))
        SLIC_CONFIG.update(config.get('slic', {}))
        NDVI_THRESHOLDS.update(config.get('ndvi', {}))
        BAND_INDICES.update(config.get('bands', {}))
        POST_PROCESSING.update(config.get('post_process', {}))
        OUTPUT_CONFIG.update(config.get('output', {}))
        PARALLEL_CONFIG.update(config.get('parallel', {}))

    try:
        print_header("森林非监督分类系统")

        # 显示当前使用的算法
        algorithm = CLASSIFICATION_CONFIG.get('algorithm', 'slic')
        if algorithm == 'slic':
            print_info(f"分类算法: SLIC超像素分割")
        elif algorithm == 'kmeans':
            print_info(f"分类算法: K-means聚类")

        print(f"\n{Fore.YELLOW}处理参数:{Style.RESET_ALL}")
        print(f"  聚类数量: {CLASSIFICATION_CONFIG['n_clusters']}")
        print(f"  森林NDVI阈值: {NDVI_THRESHOLDS['forest_min']}")
        print(f"  乔木林NDVI阈值: {NDVI_THRESHOLDS['arbor_forest_min']}")

        # 显示并行处理配置
        print(f"\n{Fore.YELLOW}并行处理配置:{Style.RESET_ALL}")
        print(f"  K-means并行进程数: {PARALLEL_CONFIG['kmeans_n_jobs']}")
        print(f"  SLIC并行进程数: {PARALLEL_CONFIG['slic_n_jobs']}")
        print(f"  超像素特征提取并行进程数: {PARALLEL_CONFIG['feature_extraction_n_jobs']}")
        print(f"  后处理并行进程数: {PARALLEL_CONFIG['post_process_n_jobs']}")
        print(f"  最大内存使用限制: {PARALLEL_CONFIG['max_memory_mb']:.2f} MB (30%)")
        print(f"  分块处理: {'启用' if PARALLEL_CONFIG['enable_chunking'] else '禁用'}")

        # 创建输出目录
        output_dir = OUTPUT_CONFIG['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        print_info(f"输出目录: {output_dir}")

        # 初始化分类器
        classifier = ForestClassifier()

        # 步骤1：读取TIF文件
        print_step(1, 7, "读取遥感影像")
        if not classifier.read_tif(input_tif):
            print_error("文件读取失败，程序终止")
            return

        # 步骤2：计算NDVI
        print_step(2, 7, "计算NDVI植被指数")
        classifier.calculate_ndvi()

        # 步骤3：执行分类
        print_step(3, 7, "执行非监督分类")
        if not classifier.classify():
            print_error("分类失败，程序终止")
            return

        # 步骤4：后处理
        print_step(4, 7, "后处理（过滤小斑块）")
        classifier.post_process()

        # 步骤5：识别森林类别
        print_step(5, 7, "识别森林类别")
        class_info = classifier.identify_forest_classes()

        # 步骤6：计算统计结果
        print_step(6, 7, "计算森林覆盖率")
        statistics = classifier.calculate_statistics(class_info)

        # 步骤7：保存结果
        print_step(7, 7, "保存结果文件")

        # 保存分类结果
        classified_output = os.path.join(output_dir, OUTPUT_CONFIG['classified_file'])
        classifier.save_classified_tif(classified_output)

        # 保存NDVI（如果配置允许）
        if OUTPUT_CONFIG.get('save_ndvi', True):
            ndvi_output = os.path.join(output_dir, OUTPUT_CONFIG['ndvi_file'])
            classifier.save_ndvi_tif(ndvi_output)
        else:
            print_warning("NDVI文件保存已禁用")

        # 保存统计结果
        stats_output = os.path.join(output_dir, OUTPUT_CONFIG['statistics_file'])
        with open(stats_output, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        print_success(f"统计结果已保存: {stats_output}")

        # 生成CSV报告
        report_output = os.path.join(output_dir, OUTPUT_CONFIG['report_file'])
        df = pd.DataFrame([statistics])
        df.to_csv(report_output, index=False, encoding='utf-8-sig')
        print_success(f"CSV报告已保存: {report_output}")

        # 可视化
        if OUTPUT_CONFIG['visualization']:
            print_info("正在生成可视化结果...")
            vis_output = os.path.join(output_dir, 'classification_results.png')
            classifier.visualize_results(vis_output, class_info)


        # 完成提示
        print(f"\n{Fore.GREEN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{Style.BRIGHT}✓ 处理完成！{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}所有结果已保存到: {Fore.CYAN}{output_dir}{Style.RESET_ALL}\n")

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠ 程序被用户中断{Style.RESET_ALL}")
        # 确保关闭进程池
        if _current_executor is not None:
            try:
                _current_executor.shutdown(wait=False, cancel_futures=True)
                print(f"{Fore.YELLOW}⚠ 已取消所有正在进行的任务{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}✗ 关闭进程池时出错: {e}{Style.RESET_ALL}")

        # 清理内存
        gc.collect()
        print(f"{Fore.YELLOW}⚠ 程序已停止，部分处理可能未完成{Style.RESET_ALL}")
        sys.exit(130)  # 标准的退出码 130 表示 Ctrl+C 中断

    except Exception as e:
        print(f"\n{Fore.RED}✗ 发生错误: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

        # 确保关闭进程池
        if _current_executor is not None:
            try:
                _current_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        # 清理内存
        gc.collect()
        sys.exit(1)

    finally:
        # 确保进程池被关闭
        if _current_executor is not None:
            try:
                _current_executor.shutdown(wait=True)
            except Exception:
                pass
            _current_executor = None

        # 强制垃圾回收
        gc.collect()


if __name__ == '__main__':
    import sys

    # 检测运行模式：只有当第一个参数是以 '-' 开头的选项时才使用命令行模式
    # 否则（包括只提供文件名或无参数）进入交互式模式
    if len(sys.argv) > 1 and sys.argv[1].startswith('-'):
        # 命令行模式：解析命令行参数
        args = parse_args()
    else:
        # 交互式模式：通过询问获取参数
        args = interactive_input()

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 - {args.input}")
        sys.exit(1)

    # 构建配置字典
    config = {
        'classification': {
            'algorithm': args.algorithm,
            'n_clusters': args.n_clusters,
            'random_state': args.random_state,
            'n_init': args.n_init,
            'max_iter': args.max_iter,
        },
        'slic': {
            'n_segments': args.n_segments,
            'compactness': args.compactness,
            'max_num_iter': args.slic_max_iter,
            'sigma': args.sigma,
        },
        'ndvi': {
            'forest_min': args.forest_ndvi,
            'arbor_forest_min': args.arbor_forest_ndvi,
        },
        'bands': {
            'red': args.red_band,
            'nir': args.nir_band,
        },
        'post_process': {
            'min_patch_size': args.min_patch_size,
            'use_majority_filter': not args.no_post_process,
        },
        'output': {
            'output_dir': args.output_dir,
            'visualization': not args.no_visualization,
            'save_ndvi': not args.no_ndvi,
        },
        'parallel': {
            'n_jobs': args.n_jobs,
            'kmeans_n_jobs': args.n_jobs,
            'slic_n_jobs': args.n_jobs,
            'feature_extraction_n_jobs': args.n_jobs,
            'post_process_n_jobs': args.n_jobs,
            'max_memory_mb': args.max_memory,
        }
    }

    # 如果有前缀，修改输出文件名
    if args.prefix:
        prefix = args.prefix
        config['output']['classified_file'] = f"{prefix}_classified.tif"
        config['output']['ndvi_file'] = f"{prefix}_ndvi.tif"
        config['output']['statistics_file'] = f"{prefix}_statistics.json"
        config['output']['report_file'] = f"{prefix}_report.csv"

    # 调用主函数
    main(args.input, config)