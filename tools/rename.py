import os
import tifffile as tiff
import numpy as np
from cellpose import io

# 图像文件所在目录
image_dir = "./datasets/OrganoID/OrganoIDProcessed"  # 输入文件夹路径

# 输出目录（用于保存分离后的图像）
output_dir = "./datasets/OrganoID/single_images_seg"  # 输出文件夹路径
os.makedirs(output_dir, exist_ok=True)

# 遍历目录中的所有.tif文件
for idx, image_file in enumerate(os.listdir(image_dir), start=1):
    if image_file.endswith(".tif"):
        # 为每个文件创建一个新文件夹，例如 01, 02, 03 等
        folder_name = f"{idx:02d}_SEG"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # 读取堆叠的图像
        image_path = os.path.join(image_dir, image_file)
        image_stack = tiff.imread(image_path)  # 读取堆叠的.tif文件

        # 确保堆叠的图像有19层
        if image_stack.shape[0] != 19:
            print(f"警告：{image_file} 并不是19层堆叠，跳过该文件。")
            continue

        # 遍历每一层，保存为单独的图像文件
        for i in range(image_stack.shape[0]):
            # 获取当前层
            single_image = image_stack[i, :, :]

            # 创建图像文件名，例如 t001.tif, t002.tif, ...
            output_filename = f"seg_t{i:03d}.tif"
            output_path = os.path.join(folder_path, output_filename)

            # 保存当前层图像
            tiff.imwrite(output_path, single_image)

            # img_path="datasets/OrganoID/single_images_seg/01_SEG/seg_t000.tif"
            # print(f"Shape of first training image: {io.imread(img_path).shape}")
            print(f"文件 {image_file} 的第 {i+1} 层已保存为 {output_filename}")
