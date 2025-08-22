import numpy as np
import rasterio
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point

# 加载分类结果 TIF 文件
file_path = 'E:\data\物种多样性计算\image\sj\\veg\SJ_0.03.tif'
with rasterio.open(file_path) as src:
    classification_data = src.read(1)  # 只读第一个波段
    transform = src.transform
    crs = src.crs

rows, cols = classification_data.shape

# 设置窗口大小
window_size = 333
pad_size = window_size // 2

# 对图像边缘进行填充，防止越界
padded_data = np.pad(classification_data, pad_size, mode='constant')

def shannon_index(window):
    unique, counts = np.unique(window, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    freq = counts / counts.sum()
    return -np.sum(freq * np.log(freq))

# 定义Simpson多样性指数函数
def simpson_index(window):
    unique, counts = np.unique(window, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    freq = counts / counts.sum()
    return 1.0 - np.sum(freq**2)

# 定义Margalef丰富度指数函数
def margalef_index(unique_species_count, total_count):
    if total_count == 1:  # 避免除以零或对1取对数
        return 0.0
    return (unique_species_count - 1) / np.log(total_count)

# 定义Pielou均匀度指数函数
def pielou_evenness(window):
    unique, counts = np.unique(window, return_counts=True)
    if len(counts) <= 1:
        return 0.0
    freq = counts / counts.sum()
    h_prime = -np.sum(freq * np.log(freq))
    h_max = np.log(len(unique))
    return h_prime / h_max  # 归一化到 [0,1]


# 加载点数据 Shapefile
shapefile_path = 'E:\data\物种多样性计算\\result_veg\合并point\\SJ1.shp'
gdf = gpd.read_file(shapefile_path)

# 创建一个列表存储每个点的多样性指标值
results = []

for index, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Processing Points"):
    point = row['geometry']
    x, y = point.x, point.y

    # 根据点坐标转换为像素坐标
    col, row = ~transform * (x, y)
    col = int(round(col))
    row = int(round(row))

    # 提取窗口区域
    win = padded_data[row:row + window_size, col:col + window_size]

    # 跳过无效窗口（例如超出边界的情况）
    if win.size != window_size * window_size:
        continue

    # 计算各种多样性指标
    shannon = shannon_index(win)
    simpson = simpson_index(win)
    unique_species_count = len(np.unique(win))
    margalef = margalef_index(unique_species_count, win.size)
    pielou = pielou_evenness(win)

    # 将结果保存到列表
    results.append({
        'Point_ID': index,
        'Shannon_Index': shannon,
        'Simpson_Index': simpson,
        'Margalef_Richness': margalef,
        'Pielou_Evenness': pielou
    })

# 输出结果到CSV文件
import pandas as pd

results_df = pd.DataFrame(results)
output_csv_file = "E:\\data\\物种多样性计算\\result_veg\SJ_a_really_merge.csv"
results_df.to_csv(output_csv_file, index=False)

print(f"✅ Diversity indices for points saved to: {output_csv_file}")