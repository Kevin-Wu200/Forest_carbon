"""
配置文件 - 森林分类参数设置
"""

# 分类参数
CLASSIFICATION_CONFIG = {
    'algorithm': 'slic',          # 分类算法：'slic'（超像素）或 'kmeans'
    'n_clusters': 7,              # 聚类数量（K-means使用）
    'random_state': 42,           # 随机种子，保证结果可复现
    'n_init': 10,                 # K-means重复次数
    'max_iter': 300,              # 最大迭代次数
}

# SLIC超像素参数
SLIC_CONFIG = {
    'n_segments': 1000,           # 超像素数量（近似值，实际数量可能略有不同）
    'compactness': 10.0,          # 紧凑度参数，值越大超像素越紧凑
    'max_num_iter': 10,           # 最大迭代次数
    'sigma': 1.0,                 # 高斯平滑标准差，用于预处理
    'min_size_factor': 0.5,       # 最小超像素大小因子（相对于期望大小）
    'max_size_factor': 3.0,       # 最大超像素大小因子
    'enforce_connectivity': True, # 强制超像素连接
}

# 植被指数阈值（用于识别森林类别）
NDVI_THRESHOLDS = {
    'forest_min': 0.6,            # 森林NDVI最小值
    'arbor_forest_min': 0.7,     # 乔木林NDVI最小值
    'vegetation_min': 0.4,       # 植被NDVI最小值
}

# 波段索引（根据TIF文件的波段顺序调整）
# Landsat 8: B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1, B7=SWIR2
# Sentinel-2: B2=Blue, B3=Green, B4=Red, B8=NIR, B11=SWIR1, B12=SWIR2
BAND_INDICES = {
    'red': 2,      # 红光波段索引（从0开始）
    'nir': 3,      # 近红外波段索引
}

# 后处理参数
POST_PROCESSING = {
    'min_patch_size': 5,         # 最小斑块大小（像素数），小于此值的斑块将被过滤
    'use_majority_filter': True, # 是否使用众数滤波去除噪声
}

# 输出设置
OUTPUT_CONFIG = {
    'output_dir': 'output',
    'classified_file': 'classified.tif',
    'ndvi_file': 'ndvi.tif',
    'statistics_file': 'statistics.json',
    'report_file': 'report.csv',
    'visualization': True,        # 是否生成可视化结果
    'save_ndvi': True,            # 是否保存NDVI文件
}

# 单位面积（用于面积计算）
AREA_CONFIG = {
    'unit': 'hectare',            # 面积单位：hectare（公顷）
    'pixel_area': None,           # 单个像素的面积（平方米），None表示从TIF元数据中读取
}


# ========== 硬件资源自动检测 ==========
import os
import multiprocessing

def get_cpu_count():
    """
    获取CPU核心数

    Returns:
    --------
    int
        CPU核心数
    """
    try:
        return multiprocessing.cpu_count()
    except:
        return 4  # 默认值

def get_memory_info():
    """
    获取系统内存信息

    Returns:
    --------
    dict
        内存信息字典，包含 total_mb（总内存MB）和 available_mb（可用内存MB）
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total_mb': mem.total / (1024 * 1024),
            'available_mb': mem.available / (1024 * 1024)
        }
    except ImportError:
        # 如果没有psutil，使用os.sysconf作为备选方案
        try:
            total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            return {
                'total_mb': total_memory / (1024 * 1024),
                'available_mb': total_memory / (1024 * 1024)  # 假设全部可用
            }
        except:
            # 默认值：假设16GB内存
            return {
                'total_mb': 16384,
                'available_mb': 16384
            }


# ========== 并行处理配置（自动适配硬件资源） ==========
# 获取硬件资源
CPU_COUNT = get_cpu_count()
MEMORY_INFO = get_memory_info()

# 计算最大安全内存使用量（不超过总内存的50%）
MAX_SAFE_MEMORY_MB = MEMORY_INFO['total_mb'] * 0.5

# 并行处理配置
PARALLEL_CONFIG = {
    # CPU并行度配置（充分利用CPU）
    'n_jobs': CPU_COUNT,  # 使用所有CPU核心
    'max_workers': CPU_COUNT,  # 最大工作线程数

    # K-means并行配置
    'kmeans_n_jobs': CPU_COUNT,  # 使用所有核心

    # SLIC并行配置（skimage的slic不支持n_jobs，但其他操作可以并行）
    'slic_n_jobs': CPU_COUNT,

    # 后处理并行配置
    'post_process_n_jobs': CPU_COUNT,  # 使用所有核心

    # 超像素特征提取并行配置
    'feature_extraction_n_jobs': CPU_COUNT,  # 使用所有核心

    # 内存控制配置
    'max_memory_mb': MAX_SAFE_MEMORY_MB,
    'memory_limit_percent': 0.5,  # 最大使用50%内存

    # 分块处理配置
    'enable_chunking': True,  # 启用分块处理
    'chunk_size_mb': MAX_SAFE_MEMORY_MB / 6,  # 每块最大内存使用量（MB），分更多块以保持CPU忙碌
    'chunk_overlap': 100,  # 块间重叠像素数（避免边界效应）
}

# 打印硬件资源信息
print(f"\n{'='*60}")
print(f"硬件资源检测:")
print(f"{'='*60}")
print(f"CPU核心数: {CPU_COUNT}")
print(f"总内存: {MEMORY_INFO['total_mb']:.2f} MB")
print(f"最大安全内存使用: {MAX_SAFE_MEMORY_MB:.2f} MB (50%)")
print(f"并行进程数: {PARALLEL_CONFIG['n_jobs']}")
print(f"分块大小: {PARALLEL_CONFIG['chunk_size_mb']:.2f} MB")
print(f"{'='*60}\n")