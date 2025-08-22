import numpy as np
import rasterio
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# 加载分类结果 TIF 文件
file_path = 'E:\\data\\物种多样性计算\\image\\sj\\veg\\SJ2_3.3.tif'
with rasterio.open(file_path) as src:
    classification_data = src.read(1)  # 只读第一个波段
    transform = src.transform
    crs = src.crs
    profile = src.profile

rows, cols = classification_data.shape

# 设置窗口大小（奇数，如3、5、7）
window_size = 3
pad_size = window_size // 2

# 对图像边缘进行填充，防止越界
padded_data = np.pad(classification_data, pad_size, mode='constant')

# 初始化输出数组
shannon_map = np.zeros_like(classification_data, dtype=np.float32)
simpson_map = np.zeros_like(classification_data, dtype=np.float32)
margalef_map = np.zeros_like(classification_data, dtype=np.float32)
pielou_map = np.zeros_like(classification_data, dtype=np.float32)

# 定义香农多样性指数函数
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

# 带进度条地遍历每个像素
for i in tqdm(range(rows), desc="Processing Rows", total=rows):
    for j in range(cols):
        # 提取窗口区域
        win = padded_data[i:i + window_size, j:j + window_size]
        # 计算香农多样性指数
        shannon_map[i, j] = shannon_index(win)
        # 计算Simpson多样性指数
        simpson_map[i, j] = simpson_index(win)
        # 计算Margalef丰富度指数
        unique_species_count = len(np.unique(win))
        margalef_map[i, j] = margalef_index(unique_species_count, win.size)
        # 计算Pielou均匀度指数
        pielou_map[i, j] = pielou_evenness(win)


# 分别保存香农多样性和Simpson多样性指数图
output_shannon_file = "E:\\data\\物种多样性计算\\shannon_temp.tif"
profile.update(dtype=rasterio.float32, count=1, compress='lzw')
with rasterio.open(output_shannon_file, 'w', **profile) as dst:
    dst.write(shannon_map, 1)

output_simpson_file = "E:\\data\\物种多样性计算\\simpson_temp.tif"
with rasterio.open(output_simpson_file, 'w', **profile) as dst:
    dst.write(simpson_map, 1)

# 保存Margalef丰富度指数图
output_margalef_file = "E:\\data\\物种多样性计算\\margalef_temp.tif"
with rasterio.open(output_margalef_file, 'w', **profile) as dst:
    dst.write(margalef_map, 1)

# 保存Pielou均匀度指数图
output_pielou_file = "E:\\data\\物种多样性计算\\pielou_temp.tif"
with rasterio.open(output_pielou_file, 'w', **profile) as dst:
    dst.write(pielou_map, 1)