import numpy as np
import rasterio
from tqdm import tqdm
from collections import Counter

# 加载分类结果 TIF 文件
file_path = 'E:\\data\\物种多样性计算\\image\\sj\\veg\\SJ_0.3.tif'
with rasterio.open(file_path) as src:
    classification_data = src.read(1)
    profile = src.profile

rows, cols = classification_data.shape

# 设置窗口大小（奇数）
window_size = 33
pad_size = window_size // 2

# 对图像边缘进行填充
padded_data = np.pad(classification_data, pad_size, mode='constant', constant_values=-1)

# 初始化输出数组
bray_curtis_map = np.zeros_like(classification_data, dtype=np.float32)


def compute_bray_curtis(win1, win2):
    """计算两个窗口之间的 Bray-Curtis 相异性"""
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


# 遍历每个像素作为窗口中心
for i in tqdm(range(rows), desc="Calculating Bray-Curtis Dissimilarity"):
    for j in range(cols):
        x = i + pad_size
        y = j + pad_size

        current_win = padded_data[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1].flatten()
        current_win = current_win[current_win != -1]

        adjacent_wins = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # 获取8个方向上的相邻窗口
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
            bray_curtis_map[i, j] = np.mean(bc_values)
        else:
            bray_curtis_map[i, j] = 0.0

# 更新profile以支持浮点型输出
profile.update(dtype=rasterio.float32, count=1, compress='lzw')

# 输出路径
output_file = "E:\\data\\物种多样性计算\\bray_curtis_SJ0.3.tif"
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(bray_curtis_map, 1)

print(f"✅ Bray-Curtis Dissimilarity saved to: {output_file}")