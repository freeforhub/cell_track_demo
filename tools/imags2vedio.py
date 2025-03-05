import cv2
import os
import re  # 用于排序文件名

def images_to_video(image_folder, output_video_path, fps=30):
    """
    将指定文件夹中的图像按照顺序组成视频文件。

    Args:
        image_folder (str): 包含图像的文件夹路径.
        output_video_path (str): 输出视频文件的路径.
        fps (int): 视频的帧率（每秒帧数）. Defaults to 30.
    """

    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

    # 排序图像文件，确保按照正确顺序排列
    try:
        # 尝试自然排序 (适用于文件名如 image_001.png, image_002.png)
        images = sorted(images, key=lambda f: int(re.sub('\D', '', f))) # 提取文件名中的数字部分进行排序
    except ValueError:
        # 如果文件名不符合自然排序，则使用默认的字母排序
        images = sorted(images)

    if not images:
        print(f"Error: No images found in {image_folder}")
        return

    image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not read first image {image_path}. Check if the path is valid and the image is not corrupted.")
        return

    height, width, channels = frame.shape

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用 'XVID'、'MJPG' 等
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not video.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}. Check if the codec is supported and the path is valid.")
        return

    print(f"Creating video: {output_video_path} from {len(images)} images")

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        video.write(frame)

    # 释放资源
    video.release()

    #检查文件大小

    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
      print(f"Video created successfully at {output_video_path}")
    else:
      print("Video creation failed.")

if __name__ == "__main__":
    image_folder = os.path.abspath("../datasets/OrganoID/single_images/01")  # 图片文件夹路径
    output_video_path = os.path.abspath("../datasets/OrganoID/single_images/01_VEDIO/01.mp4")  # 输出视频文件路径

    # 创建输出文件夹 (如果不存在)
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fps = 5

    images_to_video(image_folder, output_video_path, fps)