import numpy as np
import rasterio
from collections import Counter
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point

# 加载分类结果 TIF 文件
file_path = 'E:\data\物种多样性计算\image\sj\\veg\SJ_0.03.tif'
with rasterio.open(file_path) as src:
    classification_data = src.read(1)
    transform = src.transform
    crs = src.crs

rows, cols = classification_data.shape

# 设置窗口大小（奇数）
window_size = 333
pad_size = window_size // 2

# 对图像边缘进行填充
padded_data = np.pad(classification_data, pad_size, mode='constant', constant_values=-1)

def compute_bray_curtis(win1, win2):
    """计算两个窗口之间的 Braye-Curtis 相异性"""
    count1 = Counter(win1)
    count2 = Counter(win2)

    all_species = set(count1.keys()).union(set(count2.keys()))

    numerator = 0
    denominator = 0

    for sp in all_species:
        c1 = count1.get(sp, 0)
        c2 = count2.get(sp, 0)
        numerator += abs(c1 - c2)
        denominator += c1 + c2

    return numerator / denominator if denominator != 0 else 0.0


# 加载点数据 Shapefile
shapefile_path = 'E:\data\物种多样性计算\\result_veg\合并point\\SJ1.shp'
gdf = gpd.read_file(shapefile_path)

# 确保点数据与栅格数据在同一坐标系
if gdf.crs != crs:
    gdf = gdf.to_crs(crs)

# 存储每个点的 β 多样性值
results = []

for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Processing Points"):
    point: Point = row.geometry
    x, y = point.x, point.y

    # 地理坐标转像素坐标
    try:
        col, row_pixel = ~transform * (x, y)
        col = int(col)
        row_pixel = int(row_pixel)
    except Exception as e:
        print(f"⚠️ 坐标转换失败: {e}")
        continue

    # 判断是否越界
    if not (0 <= row_pixel < rows and 0 <= col < cols):
        continue

    # 当前点对应在 padded 图像中的坐标
    x = row_pixel + pad_size
    y = col + pad_size

    # 提取当前窗口
    current_win = padded_data[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1].flatten()
    current_win = current_win[current_win != -1]

    adjacent_wins = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in directions:
        adj_x_start = x - pad_size + dx * window_size
        adj_y_start = y - pad_size + dy * window_size
        adj_x_end = adj_x_start + window_size
        adj_y_end = adj_y_start + window_size

        if (0 <= adj_x_start < padded_data.shape[0] and
            0 <= adj_x_end <= padded_data.shape[0] and
            0 <= adj_y_start < padded_data.shape[1] and
            0 <= adj_y_end <= padded_data.shape[1]):

            adjacent_win = padded_data[adj_x_start:adj_x_end, adj_y_start:adj_y_end].flatten()
            adjacent_win = adjacent_win[adjacent_win != -1]
            if len(adjacent_win) > 0:
                adjacent_wins.append(adjacent_win)

    bc_values = []
    for adj_win in adjacent_wins:
        bc = compute_bray_curtis(current_win, adj_win)
        bc_values.append(bc)

    if bc_values:
        avg_bc = np.mean(bc_values)
    else:
        avg_bc = 0.0  # 没有有效相邻窗口时设为0

    results.append({
        "Point_ID": idx,
        "Bray_Curtis_Beta_Diversity": avg_bc
    })

# 保存结果为CSV文件
output_csv = "E:\\data\\物种多样性计算\\result_veg\\SJ_bray_really_merge.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)

print(f"✅ Beta diversity values saved to: {output_csv}")