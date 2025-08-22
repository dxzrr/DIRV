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
padded_data = np.pad(classification_data, pad_size, mode='constant', constant_values=0)

# 初始化输出数组
jaccard_map = np.zeros_like(classification_data, dtype=np.float32)


def compute_jaccard(win1, win2):
    """计算两个窗口之间的 Jaccard 相似性指数"""
    set1 = set(win1)
    set2 = set(win2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union != 0 else 0.0


# 遍历每个像素作为窗口中心
for i in tqdm(range(rows), desc="Calculating Jaccard Similarity"):
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

        jaccard_values = []
        for adj_win in adjacent_wins:
            jaccard = compute_jaccard(current_win, adj_win)
            jaccard_values.append(jaccard)

        if jaccard_values:
            jaccard_map[i, j] = np.mean(jaccard_values)
        else:
            jaccard_map[i, j] = 0.0  # 没有相邻窗口时设为0

# 更新profile以支持浮点型输出
profile.update(dtype=rasterio.float32, count=1, compress='lzw')

# 输出路径
output_file = "E:\\data\\物种多样性计算\\jaccard_SJ0.3.tif"
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(jaccard_map, 1)

print(f"✅ Jaccard Similarity saved to: {output_file}")