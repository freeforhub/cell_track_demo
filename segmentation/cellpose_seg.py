from cellpose import models, io
import os

# 设置输入和输出路径
input_dir = './datasets/OrganoID/single_images/01'  # 输入图像文件夹
output_dir = './datasets/OrganoID/single_images/01_SEG'  # 输出分割结果文件夹
os.makedirs(output_dir, exist_ok=True)  # 输出文件夹
pretrained_model = './models/cellpose_fine_tuned_model'
# 初始化 Cellpose 模型
model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)  # 使用 'cyto3' 模型，启用 GPU

# 遍历输入文件夹中的图像
for filename in os.listdir(input_dir):
    if filename.endswith('.tif'):  # 仅处理 TIF 格式文件
        # 加载图像
        image_path = os.path.join(input_dir, filename)
        image = io.imread(image_path)

        # 运行 Cellpose 分割
        masks, flows, styles = model.eval(image, diameter=30, channels=[0, 0])  # channels=[0, 0] 表示灰度图像,diameter指定细胞直径

        # 保存分割结果
        output_filename = f"seg_{os.path.basename(filename)}"  # 在文件名前添加 seg_
        output_path = os.path.join(output_dir, output_filename)
        io.imsave(output_path, masks)

        print(f'Saved mask for {filename} to {output_path}')