import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.utils import to_dense_adj

from src.models.celltrack_model import CellTrack_Model


# 1. 数据准备
num_nodes = 4
handcrafted_features = torch.randn(num_nodes, 13)  # 手工设计的特征 (13 维)
learned_features = torch.randn(num_nodes, 128)     # 学习的特征 (128 维)

# 边索引 (假设细胞 0 和 1 相连，细胞 1 和 2 相连，细胞 2 和 3 相连)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                           [1, 0, 2, 1, 3, 2]], dtype=torch.long)

# 边特征 (假设每条边有一个 8 维的特征)
edge_feat = torch.randn(edge_index.size(1), 8)

# 构建图数据
data = Data(x=(handcrafted_features, learned_features), edge_index=edge_index, edge_attr=edge_feat)


# 生成标签 (二分类任务)
labels = torch.randint(0, 2, (edge_index.size(1),))  # 0 或 1

# 2. 模型初始化
hand_NodeEncoder_dic = {"input_dim": 13, "fc_dims": [64, 16], "dropout_p": 0.4}
learned_NodeEncoder_dic = {"input_dim": 128, "fc_dims": [64, 16], "dropout_p": 0.4}
intialize_EdgeEncoder_dic = {"input_dim": 239, "fc_dims": [128, 64], "dropout_p": 0.4}

# 修改 message_passing 参数
message_passing = {
    "target": "CellTrack_GNN",  # 消息传递网络的类名
    "kwargs": {  # 消息传递网络的参数
        "in_channels": 32,  # 节点消息传递的输入维度
        "hidden_channels": 32,  # 节点消息传递的隐藏维度
        "in_edge_channels": 64,  # 边特征的输入维度
        "hidden_edge_channels_conv": 16,  # 边卷积的隐藏维度
        "hidden_edge_channels_linear": [128, 64],  # 边消息传递的隐藏维度
        "dropout": 0.0,  # Dropout 概率
        "num_layers": 6,  # 消息传递层数
        "num_nodes_features": 3  # 节点特征的数量
    }
}

edge_classifier_dic = {"input_dim": 64, "fc_dims": [128, 32, 1], "dropout_p": 0.2, "use_batchnorm": False}

model = CellTrack_Model(
    hand_NodeEncoder_dic=hand_NodeEncoder_dic,
    learned_NodeEncoder_dic=learned_NodeEncoder_dic,
    intialize_EdgeEncoder_dic=intialize_EdgeEncoder_dic,
    message_passing=message_passing,  # 传入修改后的 message_passing
    edge_classifier_dic=edge_classifier_dic
)

# 3. 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    optimizer.zero_grad()
    pred = model(data.x, data.edge_index, data.edge_attr)
    loss = F.binary_cross_entropy_with_logits(pred, labels.float())
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(10):
    loss = train()
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# 4. 推理
model.eval()
with torch.no_grad():
    pred = model(data.x, data.edge_index, data.edge_attr)
    print(pred)
    pred = torch.sigmoid(pred)  # 将 logits 转换为概率

print("预测结果 (边的连接概率):")
print(pred)