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


# ========== 树种分类配置 ==========
TREE_SPECIES_CONFIG = {
    'enable': True,               # 是否启用树种分类
    'auto_classification': True,  # 是否自动分类

    # 分类模式
    'classification_mode': 'rgb',  # 'rgb'(仅RGB特征) / 'ndvi'(仅NDVI) / 'combined'(RGB+NDVI结合)

    # RGB合成波段配置（432假彩色合成）
    'rgb_composite': {
        'red_band': 3,      # 波段4 (NIR) -> 红色通道
        'green_band': 2,    # 波段3 (Red) -> 绿色通道
        'blue_band': 1,     # 波段2 (Green) -> 蓝色通道
    },

    # 树种类型定义（基于RGB颜色特征）
    # 适用于432假彩色合成影像
    'species_rules': {
        'bamboo_forest': {        # 竹林 - 粉色
            'name': '竹林',
            'carbon_factor': 0.8,         # 碳汇系数（吨/公顷）
            'description': '竹林，呈粉色，碳汇能力较低',
            # RGB特征规则
            'rgb': {
                'red_ratio_min': 0.4,     # R/(R+G+B) 最小值
                'red_ratio_max': 0.6,     # R/(R+G+B) 最大值
                'brightness_min': 80,     # 亮度最小值 (0-255)
                'brightness_max': 150,    # 亮度最大值
                'saturation_min': 20,     # 饱和度最小值
                'saturation_max': 80,     # 饱和度最大值
                'red_dominance': 1.2,     # R相对于(G+B)的最小倍数
            },
            # NDVI特征规则（可选，用于combined模式）
            'ndvi': {
                'ndvi_min': 0.55,
                'ndvi_max': 0.75,
            }
        },
        'broadleaf_forest': {     # 阔叶林 - 鲜红
            'name': '阔叶林',
            'carbon_factor': 1.5,         # 碳汇系数
            'description': '阔叶林，鲜红色，碳汇能力最强',
            'rgb': {
                'red_ratio_min': 0.5,
                'red_ratio_max': 0.7,
                'brightness_min': 120,
                'brightness_max': 200,
                'saturation_min': 60,
                'saturation_max': 120,
                'red_dominance': 1.5,
            },
            'ndvi': {
                'ndvi_min': 0.7,
                'ndvi_max': 0.9,
            }
        },
        'mixed_forest': {         # 混交林 - 中红
            'name': '混交林',
            'carbon_factor': 1.35,        # 碳汇系数
            'description': '混交林，中红色，碳汇能力中等',
            'rgb': {
                'red_ratio_min': 0.45,
                'red_ratio_max': 0.65,
                'brightness_min': 100,
                'brightness_max': 170,
                'saturation_min': 40,
                'saturation_max': 100,
                'red_dominance': 1.3,
            },
            'ndvi': {
                'ndvi_min': 0.6,
                'ndvi_max': 0.85,
            }
        },
        'coniferous_forest': {    # 针叶林 - 深红
            'name': '针叶林',
            'carbon_factor': 1.2,         # 碳汇系数
            'description': '针叶林，深红色，碳汇能力较强',
            'rgb': {
                'red_ratio_min': 0.5,
                'red_ratio_max': 0.7,
                'brightness_min': 60,
                'brightness_max': 130,
                'saturation_min': 50,
                'saturation_max': 110,
                'red_dominance': 1.4,
            },
            'ndvi': {
                'ndvi_min': 0.65,
                'ndvi_max': 0.85,
            }
        },
        'economic_forest': {      # 经济林 - 浅红
            'name': '经济林',
            'carbon_factor': 0.6,         # 碳汇系数
            'description': '经济林，浅红色，碳汇能力较低',
            'rgb': {
                'red_ratio_min': 0.4,
                'red_ratio_max': 0.6,
                'brightness_min': 130,
                'brightness_max': 200,
                'saturation_min': 20,
                'saturation_max': 60,
                'red_dominance': 1.1,
            },
            'ndvi': {
                'ndvi_min': 0.5,
                'ndvi_max': 0.7,
            }
        },
        'shrub_forest': {         # 灌木林 - 暗红
            'name': '灌木林',
            'carbon_factor': 0.4,         # 碳汇系数
            'description': '灌木林，暗红色，碳汇能力低',
            'rgb': {
                'red_ratio_min': 0.4,
                'red_ratio_max': 0.6,
                'brightness_min': 50,
                'brightness_max': 110,
                'saturation_min': 30,
                'saturation_max': 80,
                'red_dominance': 1.2,
            },
            'ndvi': {
                'ndvi_min': 0.35,
                'ndvi_max': 0.55,
            }
        },
        'non_vegetation': {       # 非植被 - 灰色
            'name': '非植被',
            'carbon_factor': 0.0,         # 碳汇系数
            'description': '非植被区域，灰色',
            'rgb': {
                'red_ratio_min': 0.3,
                'red_ratio_max': 0.4,
                'brightness_min': 30,
                'brightness_max': 120,
                'saturation_min': 0,
                'saturation_max': 30,
                'red_dominance': 1.0,
            },
            'ndvi': {
                'ndvi_min': -0.2,
                'ndvi_max': 0.35,
            }
        }
    },

    # 如果禁用自动分类，使用手动映射（聚类类别ID -> 树种）
    'manual_mapping': None,       # 示例: {0: 'coniferous_forest', 1: 'broadleaf_forest', ...}

    # NDVI结合判断参数（仅在combined模式下有效）
    'ndvi_weight': 0.3,           # NDVI权重（0-1）
    'rgb_weight': 0.7,            # RGB特征权重（0-1）
}


# ========== 区块处理配置 ==========
BLOCK_CONFIG = {
    'enable': True,               # 是否启用区块处理
    'method': 'vector',           # 区块划分方法: 'vector'(矢量边界) / 'grid'(规则网格)

    # 矢量边界配置
    'vector': {
        'file_path': None,        # 区块边界文件路径（.shp）
        'id_field': 'FID',        # 区块ID字段名
        'name_field': 'NAME',     # 区块名称字段名
    },

    # 规则网格配置（备用）
    'grid': {
        'size_meters': 1000,      # 网格大小（米）
    }
}


# ========== 矢量输出配置 ==========
VECTOR_OUTPUT_CONFIG = {
    'enable': True,               # 是否启用矢量输出
    'output_format': 'shapefile', # 输出格式: 'shapefile' / 'geojson'

    # 输出文件
    'output_file': 'forest_species_blocks.shp',

    # 多边形处理
    'simplify_tolerance': 5.0,    # 简化容差（米），用于减少多边形顶点数
    'min_area_hectares': 0.1,     # 最小面积阈值（公顷），小于此值的斑块将被过滤

    # 属性表字段
    'include_attributes': True,   # 是否包含完整属性表
    'attribute_fields': [
        'FID',                    # 要素ID
        'CLASS_ID',               # 聚类类别ID
        'SPECIES',                # 树种名称
        'SPECIES_CODE',           # 树种代码
        'BLOCK_ID',               # 区块ID
        'BLOCK_NAME',             # 区块名称
        'AREA_HA',                # 面积（公顷）
        'AREA_M2',                # 面积（平方米）
        'NDVI_MEAN',              # 平均NDVI
        'NDVI_MIN',               # 最小NDVI
        'NDVI_MAX',               # 最大NDVI
        'CARBON_FACTOR',          # 碳汇系数
        'ESTIMATED_CARBON',       # 估算碳汇量（吨）
        'PERIMETER',              # 周长（米）
        'COMPLEXITY'              # 复杂度指数（周长/面积比）
    ]
}


# ========== 碳汇计算配置 ==========
CARBON_CALCULATION_CONFIG = {
    'enable': True,               # 是否启用碳汇计算
    'unit': 'tons',               # 单位: 'tons'(吨) / 'kg'(千克)

    # 碳汇量计算公式
    # 碳汇量 = 面积(公顷) × 碳汇系数(吨/公顷)
    'formula': 'area_ha * carbon_factor',

    # 生长调整因子（可选）
    'growth_factor': 1.0,         # 生长因子，考虑树木年龄、生长阶段等
    'season_factor': 1.0,         # 季节因子，考虑季节变化

    # 碳汇报告
    'report_file': 'carbon_report.csv',  # 碳汇报告文件名
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