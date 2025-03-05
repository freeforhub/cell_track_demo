import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage import io
import matplotlib.colors as mcolors


def create_tracking_animation_colored(track_file, image_dir, segmentation_dir, output_gif, duration=0.1):
    """
    创建细胞追踪动画，不同细胞用不同颜色区分，分裂细胞用同一种颜色标识，并保留细胞ID。

    Args:
        track_file (str): res_track.txt 文件的路径。
        image_dir (str): 原始图像所在的目录。
        segmentation_dir (str): 分割图像所在的目录。
        output_gif (str): 输出 GIF 文件的路径。
        duration (float): 每一帧的显示时间（秒）。默认为 0.1。
    """

    # 1. 读取追踪数据和构建父细胞关系字典
    tracks = {}
    parent_child_map = {}
    with open(track_file, 'r') as f:
        for line in f:
            L, B, E, P = map(int, line.strip().split())
            tracks[L] = (B, E, P)
            if P != 0:
                if P not in parent_child_map:
                    parent_child_map[P] = []
                parent_child_map[P].append(L)

    print(f"parent_child_map: {parent_child_map}")  # 调试
    print(f"keys of tracks: {tracks.keys()}")

    # 2. 为每个细胞分配颜色
    num_colors = len(tracks)
    color_list = list(mcolors.CSS4_COLORS.keys())
    np.random.shuffle(color_list)
    cell_colors = {}

    def assign_color(cell_id, avoid_recursion=set()):  # 增加避免递归的集合
        """递归地为细胞及其所有子细胞分配相同的颜色，避免无限递归"""
        if cell_id in cell_colors:
            return

        if cell_id in avoid_recursion:  # 避免循环递归
            print(f"Warning: 循环递归 detected for cell {cell_id}")
            return

        avoid_recursion.add(cell_id)  # 标记当前细胞已访问

        if cell_id in parent_child_map:  # 如果是父细胞，先确保自己有颜色
            if cell_id not in cell_colors:
                cell_colors[cell_id] = color_list[len(cell_colors) % len(color_list)]
            for child_id in parent_child_map[cell_id]:
                if child_id not in cell_colors:
                    cell_colors[child_id] = cell_colors[cell_id]  # 继承父细胞的颜色
                    assign_color(child_id, avoid_recursion)  # 递归地为子细胞着色
        else:
            # 如果不是父细胞，则分配新颜色
            cell_colors[cell_id] = color_list[len(cell_colors) % len(color_list)]

    # 3.  初始化细胞颜色 (从根细胞开始)
    for cell_id in tracks.keys():
        if cell_id not in cell_colors:
            assign_color(cell_id)

    # 4. 获取图像文件列表并排序
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
    segmentation_files = sorted(
        [f for f in os.listdir(segmentation_dir) if f.startswith('mask') and f.endswith('.tif')])

    num_frames = len(image_files)
    if num_frames != len(segmentation_files):
        print("The number of images:", num_frames, "The number of segmentation images:", len(segmentation_files),
              "Warning: The number of raw images and segmentation images do not match.")
        return

    # segmentation_path="data/CTC/Fluo-N2DH-SIM+/Test/01_SEG/seg_t000.tif"
    # segmentation = io.imread(segmentation_path)
    # print(f"分割图像的数据类型：{segmentation.dtype}")

    # 5. 创建动画帧
    frames = []
    for frame_index in range(num_frames):
        raw_image_path = os.path.join(image_dir, image_files[frame_index])
        segmentation_path = os.path.join(segmentation_dir, segmentation_files[frame_index])

        try:
            raw_image = io.imread(raw_image_path)  # 使用 skimage.io 读取
            segmentation = io.imread(segmentation_path)
        except FileNotFoundError as e:
            print(f"Error: Image not found: {e}")
            return

        # 创建 Matplotlib 图
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(raw_image, cmap='gray')  # 显示原始图像，灰度图
        ax.set_title(f"Cell Tracking - Frame {frame_index}")
        ax.axis('off')

        # 绘制细胞轮廓并上色
        for cell_id, (B, E, P) in tracks.items():
            if B <= frame_index <= E:
                try:
                    # 获取细胞轮廓
                    contours = get_cell_contours(segmentation, cell_id)

                    # 使用分配的颜色绘制轮廓
                    contour_color = cell_colors[cell_id]
                    for contour in contours:
                        ax.plot(contour[:, 1], contour[:, 0], color=contour_color, linewidth=1)

                    # 标注细胞 ID (使用与轮廓相同的颜色)
                    row, col = find_cell_center(segmentation, cell_id)
                    ax.text(col, row, str(cell_id), color='white', fontsize=8)  # 设置文字为白色方便查看
                except CellNotFound as e:
                    print(f"Warning: {e}")

        # 将 Matplotlib 图转换为图像
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    # 6. 创建 GIF 动画
    imageio.mimsave(output_gif, frames, duration=duration)
    print(f"动画已保存到 {output_gif}")


class CellNotFound(Exception):
    pass


def find_cell_center(image, cell_id):
    """
    计算指定细胞ID在图像中的中心坐标。

    Args:
        image (numpy.ndarray): 分割图像.
        cell_id (int): 细胞ID.

    Returns:
        tuple: (row, col) 细胞中心的坐标。

    Raises:
        CellNotFound: 如果图像中找不到指定的细胞 ID。
    """
    rows, cols = np.where(image == cell_id)
    if len(rows) > 0 and len(cols) > 0:
        center_row = int(np.mean(rows))
        center_col = int(np.mean(cols))
        return center_row, center_col
    else:
        raise CellNotFound(f"Cell ID {cell_id} not found in image")


def get_cell_contours(image, cell_id):
    """
    获取指定细胞 ID 的轮廓。

    Args:
        image (numpy.ndarray): 分割图像。
        cell_id (int): 细胞 ID.

    Returns:
        list: 细胞轮廓坐标列表。
    """
    mask = image == cell_id
    contours = measure.find_contours(mask, 0.5)  # 使用 0.5 作为 level
    return contours


if __name__ == "__main__":
    track_file = "../datasets/Fluo-N2DH-SIM+/Test/01_RES/res_track.txt"  # res_track.txt 文件的路径
    image_dir = "../datasets/Fluo-N2DH-SIM+/Test/01"  # 原始图像所在的目录
    segmentation_dir = "../datasets/Fluo-N2DH-SIM+/Test/01_RES"  # 分割图片(处理后的)所在的目录
    output_gif = "../datasets/Fluo-N2DH-SIM+/Test/01_RES/cell_tracking.gif"  # 保存 GIF 动画的路径

    create_tracking_animation_colored(track_file, image_dir, segmentation_dir, output_gif)
