# 森林非监督分类系统

基于K-means和SLIC算法的多光谱遥感影像非监督分类工具，用于计算森林覆盖率、乔木林覆盖率和面积统计。

## 功能特点

- 支持多种分类算法（SLIC超像素分割、K-means聚类）
- 自动计算NDVI植被指数
- 智能识别森林和乔木林类别
- 生成分类结果TIF文件
- 计算森林覆盖率和面积统计
- 生成可视化结果图
- 导出JSON和CSV格式报告
- 支持并行处理，提升大文件处理效率
- 完整的命令行参数配置

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python forest_classifier.py input.tif
```

### 查看帮助信息

```bash
python forest_classifier.py --help
```

### 命令行参数

所有参数都支持命令行输入，如不指定则使用默认值。

#### 输入输出参数

- `input` (必需): 输入TIF文件路径
- `-o, --output-dir`: 输出目录 (默认: output)
- `--prefix`: 输出文件前缀 (默认: 空)

#### 分类算法参数

- `-a, --algorithm`: 分类算法，可选 slic/kmeans (默认: slic)
- `-k, --n-clusters`: 聚类数量 (默认: 7)
- `--random-state`: 随机种子 (默认: 42)
- `--n-init`: K-means重复次数 (默认: 10)
- `--max-iter`: 最大迭代次数 (默认: 300)

#### SLIC超像素参数（仅在使用 -a slic 时有效）

- `--n-segments`: 超像素数量 (默认: 1000)
- `--compactness`: 紧凑度参数 (默认: 10.0)
- `--slic-max-iter`: SLIC最大迭代次数 (默认: 10)
- `--sigma`: 高斯平滑标准差 (默认: 1.0)

#### NDVI阈值参数

- `--forest-ndvi`: 森林NDVI最小值 (默认: 0.6)
- `--arbor-forest-ndvi`: 乔木林NDVI最小值 (默认: 0.7)

#### 波段配置参数

- `--red-band`: 红光波段索引（从0开始） (默认: 2)
- `--nir-band`: 近红外波段索引（从0开始） (默认: 3)

#### 后处理参数

- `--min-patch-size`: 最小斑块大小（像素数） (默认: 5)
- `--no-post-process`: 禁用后处理

#### 输出控制参数

- `--no-visualization`: 禁用可视化生成
- `--no-ndvi`: 不保存NDVI文件

#### 并行处理参数

- `-j, --n-jobs`: 并行进程数 (默认: 所有CPU核心)
- `--max-memory`: 最大内存限制MB (默认: 自动)

### 参数配置文件

除了命令行参数，也可以编辑 `config.py` 文件来修改默认配置：

**分类参数：**
- `n_clusters`: 聚类数量（默认7）
- `random_state`: 随机种子
- `n_init`: K-means重复次数
- `max_iter`: 最大迭代次数

**NDVI阈值：**
- `forest_min`: 森林NDVI最小值（默认0.6）
- `arbor_forest_min`: 乔木林NDVI最小值（默认0.7）

**波段索引：**
- `red`: 红光波段索引
- `nir`: 近红外波段索引

**输出设置：**
- `output_dir`: 输出目录
- `visualization`: 是否生成可视化结果

## 输入要求

- **文件格式**: GeoTIFF (.tif)
- **数据类型**: 多光谱遥感影像
- **必需波段**: 至少包含红光和近红外波段
- **支持数据源**: Landsat、Sentinel-2等

## 输出结果

处理完成后，会在 `output/` 目录下生成以下文件：

1. **classified.tif** - 分类结果影像
2. **ndvi.tif** - NDVI植被指数影像
3. **statistics.json** - 详细统计结果（JSON格式）
4. **report.csv** - 统计报告（CSV格式）
5. **classification_results.png** - 可视化结果图

## 统计指标

- **森林覆盖率**: 森林像素占总像素的百分比
- **乔木林覆盖率**: 乔木林像素占总像素的百分比
- **森林面积**: 森林覆盖的总面积（公顷）
- **乔木林面积**: 乔木林覆盖的总面积（公顷）

## 算法对比

### SLIC超像素分割（推荐）

**优点：**
- 结果更平滑，边界更自然
- 更好的空间一致性
- 适合大面积遥感影像
- 对噪声更鲁棒

**适用场景：**
- 大范围森林覆盖调查
- 需要较高分类精度的场景
- 影像质量较好的情况

**建议参数：**
- `--n-segments 1000` - 超像素数量
- `--compactness 10.0` - 紧凑度参数

### K-means聚类

**优点：**
- 计算速度较快
- 参数简单，易于理解
- 适合快速分析

**适用场景：**
- 快速预览分类结果
- 小范围区域分析
- 计算资源有限的情况

**建议参数：**
- `--n-clusters 7` - 聚类数量
- 建议配合后处理使用

## 注意事项

1. **波段配置**: 确保TIF文件包含红光和近红外波段，并根据数据源正确设置波段索引
2. **阈值调整**: NDVI阈值可能需要根据研究区域、季节和植被类型进行调整
3. **大文件处理**: 处理大文件（>1GB）时可能需要较长时间，建议使用并行参数优化性能
4. **内存管理**: 处理超大影像时，建议限制内存使用：`--max-memory 8000`
5. **数据质量**: 建议使用Landsat或Sentinel-2等高质量数据源以获得最佳效果
6. **算法选择**: SLIC算法通常能产生更好的分类结果，但计算时间更长；K-means适合快速分析
7. **输出目录**: 确保输出目录有足够的磁盘空间

## 后续扩展

目前系统输出面积统计，活木蓄积量计算可以通过以下方式补充：

1. **固定系数法**: 根据经验公式计算
2. **回归模型**: 基于NDVI与蓄积量的统计关系
3. **机器学习模型**: 结合地面样地数据建立反演模型

## 示例

### 基本示例

```bash
# 使用默认参数处理影像
python forest_classifier.py input.tif
```

### 分类算法示例

```bash
# 使用K-means算法
python forest_classifier.py input.tif -a kmeans

# 使用SLIC算法并调整聚类数量
python forest_classifier.py input.tif -a slic -k 10
```

### 调整NDVI阈值

```bash
# 调整森林和乔木林的NDVI阈值
python forest_classifier.py input.tif --forest-ndvi 0.5 --arbor-forest-ndvi 0.65
```

### 处理不同卫星数据

```bash
# 处理Landsat影像（红光波段索引3，近红外波段索引4）
python forest_classifier.py landsat.tif --red-band 3 --nir-band 4

# 处理Sentinel-2影像（红光波段索引3，近红外波段索引7）
python forest_classifier.py sentinel2.tif --red-band 3 --nir-band 7
```

### 优化性能

```bash
# 禁用可视化和NDVI保存以加快速度
python forest_classifier.py input.tif --no-visualization --no-ndvi

# 调整并行进程数
python forest_classifier.py input.tif -j 4

# 限制内存使用
python forest_classifier.py input.tif --max-memory 8000
```

### 自定义输出

```bash
# 指定输出目录和文件前缀
python forest_classifier.py input.tif -o results --prefix region1
```

### 禁用后处理

```bash
# 禁用后处理（不过滤小斑块）
python forest_classifier.py input.tif --no-post-process
```

## 技术栈

- Python 3.7+
- numpy - 数值计算
- rasterio - 影像读写
- scikit-learn - K-means聚类
- scikit-image - SLIC超像素分割
- scipy - 图像处理
- matplotlib - 可视化
- pandas - 数据处理
- tqdm - 进度条显示
- colorama - 彩色终端输出
- psutil - 系统资源监控

## 许可证

MIT License