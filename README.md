# 森林非监督分类系统

基于K-means算法的多光谱遥感影像非监督分类工具，用于计算森林覆盖率、乔木林覆盖率和面积统计。

## 功能特点

- 非监督分类（K-means聚类）
- 自动计算NDVI植被指数
- 智能识别森林和乔木林类别
- 生成分类结果TIF文件
- 计算森林覆盖率和面积统计
- 生成可视化结果图
- 导出JSON和CSV格式报告

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
python forest_classifier.py input.tif
```

### 参数配置

编辑 `config.py` 文件可以调整以下参数：

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

## 注意事项

1. 确保TIF文件包含红光和近红外波段
2. 根据实际数据调整 `config.py` 中的波段索引
3. NDVI阈值可能需要根据研究区域进行调整
4. 大文件处理可能需要较长时间
5. 建议使用Landsat或Sentinel-2数据以获得最佳效果

## 后续扩展

目前系统输出面积统计，活木蓄积量计算可以通过以下方式补充：

1. **固定系数法**: 根据经验公式计算
2. **回归模型**: 基于NDVI与蓄积量的统计关系
3. **机器学习模型**: 结合地面样地数据建立反演模型

## 示例

```bash
# 处理Landsat影像
python forest_classifier.py landsat_image.tif

# 处理Sentinel-2影像（需先调整波段索引）
python forest_classifier.py sentinel2_image.tif
```

## 技术栈

- Python 3.7+
- numpy - 数值计算
- rasterio - 影像读写
- scikit-learn - K-means聚类
- scipy - 图像处理
- matplotlib - 可视化
- pandas - 数据处理

## 许可证

MIT License