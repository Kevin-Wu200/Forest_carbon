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
                    PARALLEL_CONFIG, get_memory_info, TREE_SPECIES_CONFIG,
                    BLOCK_CONFIG, VECTOR_OUTPUT_CONFIG, CARBON_CALCULATION_CONFIG,
                    get_cpu_count)


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

    def extract_rgb_features(self):
        """
        提取RGB颜色特征

        用于基于432假彩色合成的树种判断

        Returns:
        --------
        dict
            每个类别的RGB特征字典
        """
        print_info("正在提取RGB颜色特征...")

        # 获取RGB合成波段索引
        rgb_config = TREE_SPECIES_CONFIG['rgb_composite']
        red_band_idx = rgb_config['red_band']
        green_band_idx = rgb_config['green_band']
        blue_band_idx = rgb_config['blue_band']

        # 检查波段是否有效
        if red_band_idx >= self.data.shape[0] or \
           green_band_idx >= self.data.shape[0] or \
           blue_band_idx >= self.data.shape[0]:
            print_error(f"RGB波段索引超出范围，数据只有{self.data.shape[0]}个波段")
            return None

        # 提取RGB波段
        red_band = self.data[red_band_idx]
        green_band = self.data[green_band_idx]
        blue_band = self.data[blue_band_idx]

        # 计算每个类别的RGB特征
        rgb_features = {}

        for class_id in range(self.n_classes):
            mask = (self.labels == class_id)
            if not np.any(mask):
                continue

            # 提取该类别的RGB值
            class_red = red_band[mask]
            class_green = green_band[mask]
            class_blue = blue_band[mask]

            # 计算RGB特征
            mean_red = np.mean(class_red)
            mean_green = np.mean(class_green)
            mean_blue = np.mean(class_blue)

            # 计算亮度（简单平均）
            brightness = (mean_red + mean_green + mean_blue) / 3.0

            # 计算饱和度（max - min）
            saturation = max(mean_red, mean_green, mean_blue) - min(mean_red, mean_green, mean_blue)

            # 计算红色占比
            total = mean_red + mean_green + mean_blue
            red_ratio = mean_red / total if total > 0 else 0

            # 计算红色主导性（R相对于G+B的倍数）
            gb_sum = mean_green + mean_blue
            red_dominance = mean_red / gb_sum if gb_sum > 0 else 1.0

            rgb_features[class_id] = {
                'mean_red': mean_red,
                'mean_green': mean_green,
                'mean_blue': mean_blue,
                'brightness': brightness,
                'saturation': saturation,
                'red_ratio': red_ratio,
                'red_dominance': red_dominance,
                'pixel_count': np.sum(mask)
            }

        print_success(f"RGB特征提取完成，共 {len(rgb_features)} 个类别")

        return rgb_features

    def map_tree_species(self):
        """
        将聚类类别映射为树种类别

        支持三种分类模式：
        - rgb: 仅基于RGB颜色特征判断（432假彩色合成）
        - ndvi: 仅基于NDVI特征判断
        - combined: 结合RGB和NDVI特征判断

        Returns:
        --------
        dict
            包含树种映射和统计信息的字典
        """
        if not TREE_SPECIES_CONFIG['enable']:
            print_warning("树种分类功能已禁用")
            return None

        print_info("正在进行树种分类...")

        # 初始化树种标签矩阵
        species_labels = np.zeros_like(self.labels, dtype=np.int32)
        species_labels[:] = -1  # -1 表示未分类

        # 获取分类模式
        classification_mode = TREE_SPECIES_CONFIG['classification_mode']
        print_info(f"分类模式: {classification_mode}")

        # 提取RGB特征（如果需要）
        rgb_features = None
        if classification_mode in ['rgb', 'combined']:
            rgb_features = self.extract_rgb_features()
            if rgb_features is None:
                print_error("RGB特征提取失败")
                return None

        # 提取NDVI特征（如果需要）
        ndvi_features = None
        if classification_mode in ['ndvi', 'combined']:
            # 计算每个类别的NDVI统计
            ndvi_features = {}
            for class_id in range(self.n_classes):
                mask = (self.labels == class_id)
                if np.any(mask):
                    class_ndvi = self.ndvi[mask]
                    ndvi_features[class_id] = {
                        'ndvi_mean': np.mean(class_ndvi),
                        'ndvi_std': np.std(class_ndvi),
                        'pixel_count': np.sum(mask)
                    }

        # 根据配置选择分类方式
        if TREE_SPECIES_CONFIG['auto_classification']:
            # 自动分类
            species_mapping = {}
            species_code = 0
            species_rules = TREE_SPECIES_CONFIG['species_rules']

            # 为每个聚类类别匹配树种
            for class_id in range(self.n_classes):
                if class_id not in rgb_features and classification_mode != 'ndvi':
                    continue

                if classification_mode == 'rgb':
                    # 仅使用RGB特征
                    rgb_stats = rgb_features[class_id]
                    matched_species, match_score = self.match_by_rgb(rgb_stats, species_rules)
                    match_info = f"R={rgb_stats['mean_red']:.0f}, G={rgb_stats['mean_green']:.0f}, B={rgb_stats['mean_blue']:.0f}"

                elif classification_mode == 'ndvi':
                    # 仅使用NDVI特征
                    ndvi_stats = ndvi_features[class_id]
                    matched_species, match_score = self.match_by_ndvi(ndvi_stats, species_rules)
                    match_info = f"NDVI={ndvi_stats['ndvi_mean']:.3f}"

                else:  # combined
                    # 结合RGB和NDVI特征
                    rgb_stats = rgb_features[class_id]
                    ndvi_stats = ndvi_features[class_id]
                    matched_species, match_score = self.match_by_combined(rgb_stats, ndvi_stats, species_rules)
                    match_info = f"NDVI={ndvi_stats['ndvi_mean']:.3f}, RGB=(R={rgb_stats['mean_red']:.0f}, G={rgb_stats['mean_green']:.0f}, B={rgb_stats['mean_blue']:.0f})"

                if matched_species:
                    if matched_species not in species_mapping:
                        species_mapping[matched_species] = species_code
                        species_code += 1

                    # 将该类别的像素映射到对应的树种
                    species_labels[self.labels == class_id] = species_mapping[matched_species]

                    print_info(f"类别 {class_id} → {species_rules[matched_species]['name']} "
                             f"({match_info}, 匹配度={match_score:.2f})")
                else:
                    # 如果没有匹配的树种，标记为未分类
                    if classification_mode == 'rgb':
                        rgb_stats = rgb_features[class_id]
                        print_warning(f"类别 {class_id} 无法匹配树种 (RGB: R={rgb_stats['mean_red']:.0f}, G={rgb_stats['mean_green']:.0f}, B={rgb_stats['mean_blue']:.0f})")
                    elif classification_mode == 'ndvi':
                        ndvi_stats = ndvi_features[class_id]
                        print_warning(f"类别 {class_id} 无法匹配树种 (NDVI={ndvi_stats['ndvi_mean']:.3f})")
                    else:
                        rgb_stats = rgb_features[class_id]
                        ndvi_stats = ndvi_features[class_id]
                        print_warning(f"类别 {class_id} 无法匹配树种 (NDVI={ndvi_stats['ndvi_mean']:.3f}, RGB: R={rgb_stats['mean_red']:.0f}, G={rgb_stats['mean_green']:.0f}, B={rgb_stats['mean_blue']:.0f})")

            self.species_labels = species_labels
            self.n_species = species_code

            # 打印树种统计
            print_success(f"树种分类完成，共识别 {self.n_species} 个树种类型")

            species_summary = {}
            for species_key, code in species_mapping.items():
                mask = (species_labels == code)
                species_summary[species_key] = {
                    'name': species_rules[species_key]['name'],
                    'code': code,
                    'pixel_count': np.sum(mask),
                    'carbon_factor': species_rules[species_key]['carbon_factor']
                }
                print_info(f"  - {species_rules[species_key]['name']}: {np.sum(mask)} 像素")

            return {
                'species_mapping': species_mapping,
                'species_labels': species_labels,
                'species_summary': species_summary,
                'species_rules': species_rules
            }

        else:
            # 手动映射模式
            print_info("使用手动映射模式")

            manual_mapping = TREE_SPECIES_CONFIG.get('manual_mapping')
            if manual_mapping is None:
                print_error("手动映射模式需要配置 manual_mapping")
                return None

            species_mapping = {}
            species_code = 0

            for class_id, species_key in manual_mapping.items():
                if species_key not in species_mapping:
                    species_mapping[species_key] = species_code
                    species_code += 1

                species_labels[self.labels == class_id] = species_mapping[species_key]

            self.species_labels = species_labels
            self.n_species = species_code

            print_success(f"手动映射完成，共 {self.n_species} 个树种类型")

            return {
                'species_mapping': species_mapping,
                'species_labels': species_labels,
                'n_species': self.n_species
            }

    def match_by_rgb(self, rgb_stats, species_rules):
        """
        基于RGB特征匹配树种

        Parameters:
        -----------
        rgb_stats : dict
            RGB特征统计
        species_rules : dict
            树种规则

        Returns:
        --------
        tuple
            (匹配的树种键, 匹配分数)
        """
        matched_species = None
        best_match_score = 0

        for species_key, rule in species_rules.items():
            rgb_rule = rule['rgb']

            # 检查所有RGB特征是否在范围内
            checks = [
                rgb_rule['red_ratio_min'] <= rgb_stats['red_ratio'] <= rgb_rule['red_ratio_max'],
                rgb_rule['brightness_min'] <= rgb_stats['brightness'] <= rgb_rule['brightness_max'],
                rgb_rule['saturation_min'] <= rgb_stats['saturation'] <= rgb_rule['saturation_max'],
                rgb_stats['red_dominance'] >= rgb_rule['red_dominance']
            ]

            # 如果所有条件都满足，计算综合匹配度
            if all(checks):
                # 计算各特征的匹配度（0-1）
                red_ratio_score = 1.0
                brightness_score = 1.0 - abs(rgb_stats['brightness'] - (rgb_rule['brightness_min'] + rgb_rule['brightness_max']) / 2) / ((rgb_rule['brightness_max'] - rgb_rule['brightness_min']) / 2)
                saturation_score = 1.0 - abs(rgb_stats['saturation'] - (rgb_rule['saturation_min'] + rgb_rule['saturation_max']) / 2) / ((rgb_rule['saturation_max'] - rgb_rule['saturation_min']) / 2)

                # 综合匹配度（加权平均）
                score = (red_ratio_score * 0.3 + brightness_score * 0.35 + saturation_score * 0.35)

                if score > best_match_score:
                    best_match_score = score
                    matched_species = species_key

        return matched_species, best_match_score

    def match_by_ndvi(self, ndvi_stats, species_rules):
        """
        基于NDVI特征匹配树种

        Parameters:
        -----------
        ndvi_stats : dict
            NDVI特征统计
        species_rules : dict
            树种规则

        Returns:
        --------
        tuple
            (匹配的树种键, 匹配分数)
        """
        matched_species = None
        best_match_score = 0

        for species_key, rule in species_rules.items():
            # 检查NDVI是否在范围内
            if 'ndvi' in rule:
                ndvi_min = rule['ndvi']['ndvi_min']
                ndvi_max = rule['ndvi']['ndvi_max']

                if ndvi_min <= ndvi_stats['ndvi_mean'] <= ndvi_max:
                    # 计算匹配度（越接近中心值，匹配度越高）
                    center = (ndvi_min + ndvi_max) / 2
                    range_width = ndvi_max - ndvi_min
                    distance = abs(ndvi_stats['ndvi_mean'] - center)
                    score = 1 - (distance / (range_width / 2)) if range_width > 0 else 1.0

                    if score > best_match_score:
                        best_match_score = score
                        matched_species = species_key

        return matched_species, best_match_score

    def match_by_combined(self, rgb_stats, ndvi_stats, species_rules):
        """
        结合RGB和NDVI特征匹配树种

        Parameters:
        -----------
        rgb_stats : dict
            RGB特征统计
        ndvi_stats : dict
            NDVI特征统计
        species_rules : dict
            树种规则

        Returns:
        --------
        tuple
            (匹配的树种键, 匹配分数)
        """
        matched_species = None
        best_match_score = 0

        rgb_weight = TREE_SPECIES_CONFIG.get('rgb_weight', 0.7)
        ndvi_weight = TREE_SPECIES_CONFIG.get('ndvi_weight', 0.3)

        for species_key, rule in species_rules.items():
            # RGB匹配
            rgb_score = 0
            if 'rgb' in rule:
                rgb_rule = rule['rgb']
                checks = [
                    rgb_rule['red_ratio_min'] <= rgb_stats['red_ratio'] <= rgb_rule['red_ratio_max'],
                    rgb_rule['brightness_min'] <= rgb_stats['brightness'] <= rgb_rule['brightness_max'],
                    rgb_rule['saturation_min'] <= rgb_stats['saturation'] <= rgb_rule['saturation_max'],
                    rgb_stats['red_dominance'] >= rgb_rule['red_dominance']
                ]
                if all(checks):
                    brightness_score = 1.0 - abs(rgb_stats['brightness'] - (rgb_rule['brightness_min'] + rgb_rule['brightness_max']) / 2) / ((rgb_rule['brightness_max'] - rgb_rule['brightness_min']) / 2)
                    saturation_score = 1.0 - abs(rgb_stats['saturation'] - (rgb_rule['saturation_min'] + rgb_rule['saturation_max']) / 2) / ((rgb_rule['saturation_max'] - rgb_rule['saturation_min']) / 2)
                    rgb_score = (brightness_score * 0.5 + saturation_score * 0.5)

            # NDVI匹配
            ndvi_score = 0
            if 'ndvi' in rule:
                ndvi_rule = rule['ndvi']
                if ndvi_rule['ndvi_min'] <= ndvi_stats['ndvi_mean'] <= ndvi_rule['ndvi_max']:
                    center = (ndvi_rule['ndvi_min'] + ndvi_rule['ndvi_max']) / 2
                    range_width = ndvi_rule['ndvi_max'] - ndvi_rule['ndvi_min']
                    distance = abs(ndvi_stats['ndvi_mean'] - center)
                    ndvi_score = 1 - (distance / (range_width / 2)) if range_width > 0 else 1.0

            # 综合匹配度
            if rgb_score > 0 or ndvi_score > 0:
                combined_score = rgb_score * rgb_weight + ndvi_score * ndvi_weight

                if combined_score > best_match_score:
                    best_match_score = combined_score
                    matched_species = species_key

        return matched_species, best_match_score

    def load_block_boundaries(self, block_file_path=None):
        """
        加载区块边界矢量文件

        Parameters:
        -----------
        block_file_path : str
            区块边界文件路径（.shp），如果为None则使用配置中的路径

        Returns:
        --------
        geopandas.GeoDataFrame
            区块边界数据
        """
        if not BLOCK_CONFIG['enable']:
            print_warning("区块处理功能已禁用")
            return None

        if block_file_path is None:
            block_file_path = BLOCK_CONFIG['vector']['file_path']

        if block_file_path is None:
            print_warning("未提供区块边界文件路径")
            return None

        try:
            import geopandas as gpd

            print_info(f"正在加载区块边界文件: {block_file_path}")

            # 读取shapefile
            blocks_gdf = gpd.read_file(block_file_path)

            print_success(f"成功加载 {len(blocks_gdf)} 个区块")

            # 检查CRS
            if blocks_gdf.crs is None:
                print_warning("区块文件没有坐标参考系统（CRS）")
            else:
                print_info(f"区块文件CRS: {blocks_gdf.crs}")

            # 检查ID字段
            id_field = BLOCK_CONFIG['vector']['id_field']
            if id_field in blocks_gdf.columns:
                print_info(f"使用ID字段: {id_field}")
            else:
                print_warning(f"ID字段 '{id_field}' 不存在，将使用索引")

            # 检查名称字段
            name_field = BLOCK_CONFIG['vector']['name_field']
            if name_field in blocks_gdf.columns:
                print_info(f"使用名称字段: {name_field}")
            else:
                print_warning(f"名称字段 '{name_field}' 不存在")

            return blocks_gdf

        except ImportError:
            print_error("需要安装 geopandas 库: pip install geopandas")
            return None
        except Exception as e:
            print_error(f"加载区块边界文件失败: {e}")
            return None

    def export_vector_shapefile(self, output_path, blocks_gdf=None):
        """
        导出分类结果为Shapefile

        Parameters:
        -----------
        output_path : str
            输出文件路径
        blocks_gdf : geopandas.GeoDataFrame, optional
            区块边界数据，如果提供则按区块分割

        Returns:
        --------
        bool
            是否成功导出
        """
        if not VECTOR_OUTPUT_CONFIG['enable']:
            print_warning("矢量输出功能已禁用")
            return False

        try:
            import geopandas as gpd
            from rasterio.features import shapes
            from shapely.geometry import shape, mapping
            import pandas as pd

            print_info(f"正在导出Shapefile: {output_path}")

            # 确定使用标签（如果有树种标签则使用树种标签，否则使用聚类标签）
            if hasattr(self, 'species_labels') and self.species_labels is not None:
                labels = self.species_labels
                label_type = 'species'
                n_labels = self.n_species
                print_info("使用树种分类结果")
            else:
                labels = self.labels
                label_type = 'class'
                n_labels = self.n_classes
                print_info("使用聚类分类结果")

            # 将栅格转换为矢量多边形
            print_info("正在将栅格转换为矢量多边形...")

            # 创建掩码（排除无效值）
            mask = (labels >= 0)

            # 使用rasterio.features.shapes生成多边形
            shapes_list = list(shapes(
                labels.astype(np.int32),
                mask=mask,
                transform=self.profile['transform']
            ))

            print_info(f"生成了 {len(shapes_list)} 个多边形")

            # 创建属性数据
            features = []
            fid = 0

            # 获取像素面积
            if 'transform' in self.profile:
                pixel_width = abs(self.profile['transform'][0])
                pixel_height = abs(self.profile['transform'][4])
                pixel_area_m2 = pixel_width * pixel_height
                pixel_area_ha = pixel_area_m2 / 10000
            else:
                pixel_area_m2 = 30 * 30  # 默认30m x 30m
                pixel_area_ha = pixel_area_m2 / 10000

            # 获取树种规则（如果存在）
            species_rules = None
            if label_type == 'species' and hasattr(self, 'species_info'):
                species_rules = self.species_info.get('species_rules', {})

            for geom, label_value in shapes_list:
                if label_value < 0:
                    continue

                # 创建shapely几何对象
                poly = shape(geom)

                # 简化多边形（减少顶点数）
                simplify_tolerance = VECTOR_OUTPUT_CONFIG['simplify_tolerance']
                if simplify_tolerance > 0:
                    poly = poly.simplify(simplify_tolerance, preserve_topology=True)

                # 计算面积
                area_m2 = poly.area * (pixel_area_m2 / (poly.area / (pixel_area_m2 if hasattr(poly, 'area') else 1)))
                area_ha = area_m2 / 10000

                # 过滤太小的斑块
                min_area = VECTOR_OUTPUT_CONFIG['min_area_hectares']
                if area_ha < min_area:
                    continue

                # 计算周长
                perimeter = poly.length * np.sqrt(pixel_area_m2)

                # 计算复杂度指数
                complexity = perimeter / (2 * np.pi * np.sqrt(area_m2 / np.pi)) if area_m2 > 0 else 0

                # 获取该类别的NDVI统计
                # 这里简化处理，实际应该从栅格中提取该多边形区域的NDVI
                ndvi_mean = self.ndvi[labels == label_value].mean() if hasattr(self, 'ndvi') else 0

                # 创建属性字典
                properties = {
                    'FID': fid,
                    'CLASS_ID': int(label_value),
                    'AREA_HA': round(area_ha, 4),
                    'AREA_M2': round(area_m2, 2),
                    'NDVI_MEAN': round(ndvi_mean, 4),
                    'NDVI_MIN': round(ndvi_mean, 4),  # 简化处理
                    'NDVI_MAX': round(ndvi_mean, 4),  # 简化处理
                    'PERIMETER': round(perimeter, 2),
                    'COMPLEXITY': round(complexity, 4)
                }

                # 如果是树种分类，添加树种相关属性
                if label_type == 'species' and species_rules:
                    # 反向查找树种
                    species_key = None
                    for key, code in self.species_info.get('species_mapping', {}).items():
                        if code == label_value:
                            species_key = key
                            break

                    if species_key and species_key in species_rules:
                        rule = species_rules[species_key]
                        properties['SPECIES'] = rule['name']
                        properties['SPECIES_CODE'] = species_key
                        properties['CARBON_FACTOR'] = rule['carbon_factor']
                        # 计算碳汇量
                        properties['ESTIMATED_CARBON'] = round(area_ha * rule['carbon_factor'], 2)
                    else:
                        properties['SPECIES'] = 'Unknown'
                        properties['SPECIES_CODE'] = 'unknown'
                        properties['CARBON_FACTOR'] = 0.0
                        properties['ESTIMATED_CARBON'] = 0.0
                else:
                    # 聚类类别
                    properties['SPECIES'] = f'Class_{label_value}'
                    properties['SPECIES_CODE'] = f'class_{label_value}'
                    properties['CARBON_FACTOR'] = 0.0
                    properties['ESTIMATED_CARBON'] = 0.0

                # 如果有区块边界，添加区块属性
                if blocks_gdf is not None:
                    # 找到包含该多边形质心的区块
                    centroid = poly.centroid
                    for idx, block in blocks_gdf.iterrows():
                        if block.geometry.contains(centroid):
                            properties['BLOCK_ID'] = idx
                            # 尝试获取区块名称
                            name_field = BLOCK_CONFIG['vector']['name_field']
                            if name_field in block:
                                properties['BLOCK_NAME'] = str(block[name_field])
                            else:
                                properties['BLOCK_NAME'] = f'Block_{idx}'
                            break
                    else:
                        properties['BLOCK_ID'] = -1
                        properties['BLOCK_NAME'] = 'Unknown'
                else:
                    properties['BLOCK_ID'] = 0
                    properties['BLOCK_NAME'] = 'Entire_Area'

                features.append({
                    'geometry': poly,
                    'properties': properties
                })

                fid += 1

            print_info(f"过滤后剩余 {len(features)} 个有效多边形")

            # 创建GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(features, crs=self.profile['crs'])

            # 按属性列排序
            column_order = VECTOR_OUTPUT_CONFIG['attribute_fields']
            # 只保留存在的列
            column_order = [col for col in column_order if col in gdf.columns]
            gdf = gdf[column_order]

            # 保存为Shapefile
            gdf.to_file(output_path, encoding='utf-8')

            print_success(f"Shapefile已导出: {output_path}")
            print_info(f"包含 {len(gdf)} 个要素")
            print_info(f"属性字段: {', '.join(gdf.columns)}")

            return True

        except ImportError as e:
            print_error(f"缺少必要的库: {e}")
            print_info("请安装: pip install geopandas shapely")
            return False
        except Exception as e:
            print_error(f"导出Shapefile失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def calculate_block_statistics(self, blocks_gdf):
        """
        按区块计算统计信息

        Parameters:
        -----------
        blocks_gdf : geopandas.GeoDataFrame
            区块边界数据

        Returns:
        --------
        dict
            按区块的统计信息
        """
        if not BLOCK_CONFIG['enable']:
            print_warning("区块统计功能已禁用")
            return None

        if blocks_gdf is None:
            print_warning("未提供区块边界数据")
            return None

        print_info("正在计算区块统计信息...")

        try:
            import geopandas as gpd
            from rasterio.features import geometry_mask
            import pandas as pd

            # 确定使用标签
            if hasattr(self, 'species_labels') and self.species_labels is not None:
                labels = self.species_labels
                label_type = 'species'
                n_labels = self.n_species
            else:
                labels = self.labels
                label_type = 'class'
                n_labels = self.n_classes

            # 获取像素面积
            if 'transform' in self.profile:
                pixel_width = abs(self.profile['transform'][0])
                pixel_height = abs(self.profile['transform'][4])
                pixel_area_m2 = pixel_width * pixel_height
                pixel_area_ha = pixel_area_m2 / 10000
            else:
                pixel_area_m2 = 30 * 30
                pixel_area_ha = pixel_area_m2 / 10000

            # 为每个区块计算统计
            block_stats = {}

            id_field = BLOCK_CONFIG['vector']['id_field']
            name_field = BLOCK_CONFIG['vector']['name_field']

            for idx, block in blocks_gdf.iterrows():
                block_id = block.get(id_field, idx)
                block_name = block.get(name_field, f'Block_{idx}')

                # 创建区块掩码
                block_mask = geometry_mask(
                    [block.geometry],
                    transform=self.profile['transform'],
                    invert=True,
                    out_shape=(self.profile['height'], self.profile['width'])
                )

                # 提取区块内的标签
                block_labels = labels[block_mask]
                block_ndvi = self.ndvi[block_mask]

                # 计算统计
                total_pixels = np.sum(block_labels >= 0)
                total_area_ha = total_pixels * pixel_area_ha

                # 按类别统计
                category_stats = {}
                for label in range(n_labels):
                    if label not in block_labels:
                        continue

                    mask = (block_labels == label)
                    pixel_count = np.sum(mask)
                    area_ha = pixel_count * pixel_area_ha
                    ndvi_mean = block_ndvi[mask].mean() if np.any(mask) else 0

                    category_stats[f'{label_type}_{label}'] = {
                        'pixel_count': int(pixel_count),
                        'area_ha': round(area_ha, 4),
                        'ndvi_mean': round(ndvi_mean, 4),
                        'coverage_percent': round((pixel_count / total_pixels * 100), 2) if total_pixels > 0 else 0
                    }

                # 计算碳汇量（如果是树种分类）
                total_carbon = 0.0
                if label_type == 'species' and hasattr(self, 'species_info'):
                    species_mapping = self.species_info.get('species_mapping', {})
                    species_rules = self.species_info.get('species_rules', {})

                    for species_key, code in species_mapping.items():
                        if f'species_{code}' in category_stats:
                            carbon_factor = species_rules[species_key]['carbon_factor']
                            area = category_stats[f'species_{code}']['area_ha']
                            carbon = area * carbon_factor
                            total_carbon += carbon

                            category_stats[f'species_{code}']['carbon_factor'] = carbon_factor
                            category_stats[f'species_{code}']['estimated_carbon'] = round(carbon, 2)

                # 整体统计
                block_stats[idx] = {
                    'block_id': block_id,
                    'block_name': block_name,
                    'total_area_ha': round(total_area_ha, 4),
                    'total_pixels': int(total_pixels),
                    'mean_ndvi': round(block_ndvi[block_labels >= 0].mean(), 4) if np.any(block_labels >= 0) else 0,
                    'category_stats': category_stats,
                    'total_carbon_tons': round(total_carbon, 2),
                    'carbon_density_tons_per_ha': round(total_carbon / total_area_ha, 2) if total_area_ha > 0 else 0
                }

                print_info(f"区块 {block_name}: 总面积={total_area_ha:.2f}公顷, "
                         f"碳汇量={total_carbon:.2f}吨")

            print_success(f"区块统计完成，共 {len(block_stats)} 个区块")

            return block_stats

        except Exception as e:
            print_error(f"计算区块统计失败: {e}")
            import traceback
            traceback.print_exc()
            return None

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

    # 树种分类参数
    species_group = parser.add_argument_group('树种分类参数')
    species_group.add_argument('--no-species', action='store_true',
                              help='禁用树种分类')
    species_group.add_argument('--species-manual', action='store_true',
                              help='使用手动树种映射模式')
    species_group.add_argument('--combined-ndvi', action='store_true',
                              help='使用综合模式（RGB颜色特征 + NDVI），默认仅使用RGB颜色特征')

    # 区块处理参数
    block_group = parser.add_argument_group('区块处理参数')
    block_group.add_argument('--no-block', action='store_true',
                            help='禁用区块处理')
    block_group.add_argument('--block-file', type=str,
                            help='区块边界文件路径（.shp）')
    block_group.add_argument('--block-id-field', type=str,
                            default=BLOCK_CONFIG['vector']['id_field'],
                            help='区块ID字段名')
    block_group.add_argument('--block-name-field', type=str,
                            default=BLOCK_CONFIG['vector']['name_field'],
                            help='区块名称字段名')

    # 矢量输出参数
    vector_group = parser.add_argument_group('矢量输出参数')
    vector_group.add_argument('--no-vector', action='store_true',
                             help='禁用矢量输出')
    vector_group.add_argument('--vector-file', type=str,
                             default=VECTOR_OUTPUT_CONFIG['output_file'],
                             help='输出Shapefile文件名')
    vector_group.add_argument('--simplify-tolerance', type=float,
                             default=VECTOR_OUTPUT_CONFIG['simplify_tolerance'],
                             help='多边形简化容差（米）')
    vector_group.add_argument('--min-area', type=float,
                             default=VECTOR_OUTPUT_CONFIG['min_area_hectares'],
                             help='最小面积阈值（公顷）')

    # 碳汇计算参数
    carbon_group = parser.add_argument_group('碳汇计算参数')
    carbon_group.add_argument('--no-carbon', action='store_true',
                             help='禁用碳汇计算')

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

    args.n_init = input(f"聚类重复次数（用于K-means算法和SLIC超像素聚类）[{CLASSIFICATION_CONFIG['n_init']}]: ").strip()
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

    # 树种分类参数
    print(f"\n{Fore.YELLOW}【树种分类】{Style.RESET_ALL}")
    use_species = input(f"启用树种分类？ [Y/n] (默认: Y): ").strip().lower()
    args.no_species = (use_species == 'n' or use_species == 'no')

    if not args.no_species:
        use_auto = input(f"使用自动分类模式？ [Y/n] (默认: Y): ").strip().lower()
        args.species_manual = (use_auto == 'n' or use_auto == 'no')

        if not args.species_manual:
            # 询问是否结合NDVI判断
            print(f"\n{Fore.CYAN}  分类模式说明:{Style.RESET_ALL}")
            print(f"  - 仅使用颜色特征: 基于432假彩色合成的RGB颜色特征判断（推荐）")
            print(f"  - 结合NDVI判断: 同时使用RGB颜色特征和NDVI植被指数判断")
            use_ndvi = input(f"  是否结合NDVI判断？ [y/N] (默认: N): ").strip().lower()
            if use_ndvi == 'y' or use_ndvi == 'yes':
                args.combined_ndvi = True
                print(f"  {Fore.GREEN}✓ 使用综合模式（RGB + NDVI）{Style.RESET_ALL}")
            else:
                args.combined_ndvi = False
                print(f"  {Fore.GREEN}✓ 使用RGB模式（仅颜色特征）{Style.RESET_ALL}")
        else:
            args.combined_ndvi = False

    # 区块处理参数
    print(f"\n{Fore.YELLOW}【区块处理】{Style.RESET_ALL}")
    use_block = input(f"启用区块处理？ [Y/n] (默认: Y): ").strip().lower()
    args.no_block = (use_block == 'n' or use_block == 'no')

    if not args.no_block:
        args.block_file = input(f"区块边界文件路径 (.shp，可选): ").strip()
        if not args.block_file:
            args.block_file = None
        else:
            args.block_id_field = input(f"区块ID字段名 [{BLOCK_CONFIG['vector']['id_field']}]: ").strip()
            if not args.block_id_field:
                args.block_id_field = BLOCK_CONFIG['vector']['id_field']

            args.block_name_field = input(f"区块名称字段名 [{BLOCK_CONFIG['vector']['name_field']}]: ").strip()
            if not args.block_name_field:
                args.block_name_field = BLOCK_CONFIG['vector']['name_field']

    # 矢量输出参数
    print(f"\n{Fore.YELLOW}【矢量输出】{Style.RESET_ALL}")
    use_vector = input(f"导出Shapefile？ [Y/n] (默认: Y): ").strip().lower()
    args.no_vector = (use_vector == 'n' or use_vector == 'no')

    if not args.no_vector:
        args.vector_file = input(f"输出文件名 [{VECTOR_OUTPUT_CONFIG['output_file']}]: ").strip()
        if not args.vector_file:
            args.vector_file = VECTOR_OUTPUT_CONFIG['output_file']

        simplify_tol = input(f"多边形简化容差（米）[{VECTOR_OUTPUT_CONFIG['simplify_tolerance']}]: ").strip()
        if not simplify_tol:
            args.simplify_tolerance = VECTOR_OUTPUT_CONFIG['simplify_tolerance']
        else:
            args.simplify_tolerance = float(simplify_tol)

        min_area = input(f"最小面积阈值（公顷）[{VECTOR_OUTPUT_CONFIG['min_area_hectares']}]: ").strip()
        if not min_area:
            args.min_area = VECTOR_OUTPUT_CONFIG['min_area_hectares']
        else:
            args.min_area = float(min_area)

    # 碳汇计算参数
    print(f"\n{Fore.YELLOW}【碳汇计算】{Style.RESET_ALL}")
    use_carbon = input(f"启用碳汇计算？ [Y/n] (默认: Y): ").strip().lower()
    args.no_carbon = (use_carbon == 'n' or use_carbon == 'no')

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
        print_step(5, 9, "识别森林类别")
        class_info = classifier.identify_forest_classes()

        # 步骤6：树种分类（如果启用）
        species_info = None
        blocks_gdf = None
        block_stats = None

        if TREE_SPECIES_CONFIG['enable']:
            print_step(6, 9, "树种分类")
            species_info = classifier.map_tree_species()
            if species_info:
                classifier.species_info = species_info
                print_success("树种分类完成")

        # 步骤7：加载区块边界（如果启用）
        if BLOCK_CONFIG['enable'] and BLOCK_CONFIG['vector']['file_path']:
            print_step(7, 9, "加载区块边界")
            blocks_gdf = classifier.load_block_boundaries()
            if blocks_gdf is not None:
                print_success(f"成功加载 {len(blocks_gdf)} 个区块")

        # 步骤8：计算统计结果
        print_step(8, 9, "计算森林覆盖率")
        statistics = classifier.calculate_statistics(class_info)

        # 如果有区块，计算区块统计
        if blocks_gdf is not None:
            print_info("正在计算区块统计...")
            block_stats = classifier.calculate_block_statistics(blocks_gdf)
            if block_stats:
                # 保存区块统计
                block_stats_file = os.path.join(output_dir, 'block_statistics.json')
                with open(block_stats_file, 'w', encoding='utf-8') as f:
                    json.dump(block_stats, f, indent=2, ensure_ascii=False)
                print_success(f"区块统计已保存: {block_stats_file}")

        # 步骤9：保存结果
        print_step(9, 9, "保存结果文件")

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

        # 导出Shapefile（如果启用）
        if VECTOR_OUTPUT_CONFIG['enable']:
            print_info("正在导出矢量数据...")
            vector_output = os.path.join(output_dir, VECTOR_OUTPUT_CONFIG['output_file'])
            classifier.export_vector_shapefile(vector_output, blocks_gdf)

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
        },
        'tree_species': {
            'enable': not args.no_species,
            'auto_classification': not getattr(args, 'species_manual', False),
            'classification_mode': 'combined' if getattr(args, 'combined_ndvi', False) else 'rgb',
        },
        'block': {
            'enable': not args.no_block,
            'vector': {
                'file_path': getattr(args, 'block_file', None),
                'id_field': getattr(args, 'block_id_field', BLOCK_CONFIG['vector']['id_field']),
                'name_field': getattr(args, 'block_name_field', BLOCK_CONFIG['vector']['name_field']),
            }
        },
        'vector_output': {
            'enable': not args.no_vector,
            'output_file': getattr(args, 'vector_file', VECTOR_OUTPUT_CONFIG['output_file']),
            'simplify_tolerance': getattr(args, 'simplify_tolerance', VECTOR_OUTPUT_CONFIG['simplify_tolerance']),
            'min_area_hectares': getattr(args, 'min_area', VECTOR_OUTPUT_CONFIG['min_area_hectares']),
        },
        'carbon': {
            'enable': not args.no_carbon,
        }
    }

    # 更新全局配置
    TREE_SPECIES_CONFIG['enable'] = config['tree_species']['enable']
    TREE_SPECIES_CONFIG['auto_classification'] = config['tree_species']['auto_classification']
    TREE_SPECIES_CONFIG['classification_mode'] = config['tree_species']['classification_mode']

    BLOCK_CONFIG['enable'] = config['block']['enable']
    if config['block']['vector']['file_path']:
        BLOCK_CONFIG['vector']['file_path'] = config['block']['vector']['file_path']
        BLOCK_CONFIG['vector']['id_field'] = config['block']['vector']['id_field']
        BLOCK_CONFIG['vector']['name_field'] = config['block']['vector']['name_field']

    VECTOR_OUTPUT_CONFIG['enable'] = config['vector_output']['enable']
    VECTOR_OUTPUT_CONFIG['output_file'] = config['vector_output']['output_file']
    VECTOR_OUTPUT_CONFIG['simplify_tolerance'] = config['vector_output']['simplify_tolerance']
    VECTOR_OUTPUT_CONFIG['min_area_hectares'] = config['vector_output']['min_area_hectares']

    CARBON_CALCULATION_CONFIG['enable'] = config['carbon']['enable']

    # 如果有前缀，修改输出文件名
    if args.prefix:
        prefix = args.prefix
        config['output']['classified_file'] = f"{prefix}_classified.tif"
        config['output']['ndvi_file'] = f"{prefix}_ndvi.tif"
        config['output']['statistics_file'] = f"{prefix}_statistics.json"
        config['output']['report_file'] = f"{prefix}_report.csv"

    # 调用主函数
    main(args.input, config)