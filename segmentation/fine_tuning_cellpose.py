import os
import imageio
import numpy as np
from cellpose import models, io, train
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# 数据路径
data_dir = "./datasets/OrganoID/OriginalData"
image_dirs = [os.path.join(data_dir, "testing", "images"), os.path.join(data_dir, "training", "images"),
              os.path.join(data_dir, "validation", "images")]
mask_dirs = [os.path.join(data_dir, "testing", "segmentations"), os.path.join(data_dir, "training", "segmentations"),
              os.path.join(data_dir, "validation", "segmentations")]

# 加载图像和掩码路径
image_paths = []
mask_paths = []
for img_dir, mask_dir in zip(image_dirs, mask_dirs):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".tif", ".png", ".jpg", ".jpeg"))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith((".tif", ".png", ".jpg", ".jpeg"))])

    for img_file, mask_file in zip(img_files, mask_files):
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(mask_path) and os.path.exists(img_path):  # 确保文件存在
            image_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {mask_path}")
            elif not os.path.exists(img_path):
                print(f"Warning: Image not found for {img_path}")

print(f"Loaded {len(image_paths)} images and {len(mask_paths)} masks.")

# 划分训练集和测试集
train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

print(f"Training set: {len(train_image_paths)} images, Test set: {len(test_image_paths)} images")

# 加载训练和测试数据
train_data = [io.imread(img_path) for img_path in train_image_paths]
train_labels = [io.imread(mask_path) for mask_path in train_mask_paths]
test_data = [io.imread(img_path) for img_path in test_image_paths]
test_labels = [io.imread(mask_path) for mask_path in test_mask_paths]

# print(f"Original mask max value: {np.max(train_labels[0])}")
# print(f"Original mask min value: {np.min(train_labels[0])}")
# print(f"Random pixel value: {train_labels[0][100, 100]}")

# 转换 RGB 掩码为单通道掩码 (不再进行阈值处理)
train_labels = [mask[:, :, 0].astype(np.uint8) for mask in train_labels]  # 提取红色通道
test_labels = [mask[:, :, 0].astype(np.uint8) for mask in test_labels]    # 提取红色通道


# # 调试信息：打印数据形状和类型
# print("Debugging Information:")
# #保存第一个训练掩码查看
# output_mask_path = "./datasets/OrganoID/OriginalData/extracted_mask.png"  # 指定保存路径和文件名
# imageio.imwrite(output_mask_path, train_labels[1])  # 保存第一个掩码
# print(f"Extracted mask saved to {output_mask_path}")

# print(f"Number of training images: {len(train_data)}")
# print(f"Number of training masks: {len(train_labels)}")
# print(f"Shape of first training image: {train_data[0].shape}")
# print(f"Shape of first training mask: {train_labels[0].shape}")
# print(f"Data type of first training image: {train_data[0].dtype}")
# print(f"Data type of first training mask: {train_labels[0].dtype}")

print(type(train_data))
logger = io.logger_setup()  # 创建日志

# 创建 Cellpose 模型 (使用预训练模型)
model = models.CellposeModel(gpu=True, model_type='cyto3')  # 使用 'cyto' 预训练模型

# 初始化 TensorBoard
log_dir = "../logs/cellpose_finetuning_organoid"  # 指定 TensorBoard 日志的目录
writer = SummaryWriter(log_dir)

# 微调模型
model_dir = "../"  # 微调后的模型保存路径 会自己带models路径
n_epochs = 1000  # 训练轮数
learning_rate = 0.001  # 学习率
weight_decay = 0.0001  # 权重衰减
batch_size = 16  # 批大小
model_name = "cellpose_fine_tuned_model_organoid"  # 微调后的模型名称

# 微调
new_model_path, train_losses, test_losses = train.train_seg(
    model.net,  # 预训练模型的网络
    train_data=train_data,  # 训练图像数据
    train_labels=train_labels,  # 训练标签数据
    test_data=test_data,  # 测试图像数据
    test_labels=test_labels,  # 测试标签数据
    channels=[0, 0],  # 灰度图像
    save_path=model_dir,  # 微调后的模型保存路径
    n_epochs=n_epochs,  # 训练轮数
    learning_rate=learning_rate,  # 学习率
    weight_decay=weight_decay,  # 权重衰减
    batch_size=batch_size,  # 批大小
    SGD=True,
    model_name=model_name  # 微调后的模型名称
)

# 记录训练和测试损失到 TensorBoard
for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
    writer.add_scalar('Loss/train', train_loss, epoch)  # 记录训练损失
    writer.add_scalar('Loss/test', test_loss, epoch)  # 记录测试损失
writer.close()  # 关闭 TensorBoard 日志

print(f"Model finetuning completed and saved to {new_model_path}")
