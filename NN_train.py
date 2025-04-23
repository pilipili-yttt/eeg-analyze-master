import torch
from torch.optim import Adam
import numpy as np
from threading import Semaphore
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

patience = 50  # 早停耐心值
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, n_channel, n_band):
        super(Model, self).__init__()
        # 图神经网络部分
        self.gcn1 = GCNConv(n_band, n_band * 2)
        self.gcn2 = GCNConv(n_band * 2, 1)
        self.fc_g = nn.Linear(2 * n_channel, 2 * n_channel)
        
        # 二维卷积部分
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc_c = nn.Linear(16 * (2 * n_channel) * (2 * n_channel), 2 * n_channel)
        
        # 后续的全连接层
        self.fc1 = nn.Linear(4 * n_channel, 2 * n_channel)
        self.fc2 = nn.Linear(2 * n_channel, n_channel)
        self.fc3 = nn.Linear(n_channel, 1)

        self.to(torch.float32)

        self.n_channel = n_channel
        self.busy = False
    
    def create_fully_connected_edge_index(self):
        rows, cols = torch.meshgrid(torch.arange(2 * self.n_channel), torch.arange(2 * self.n_channel), indexing='ij')
        mask = rows != cols
        return torch.stack([rows[mask], cols[mask]], dim=0).to(device)

    def forward(self, x1, x2):
        # 图特征处理
        edge_index = self.create_fully_connected_edge_index()
        x1 = F.relu(self.gcn1(x1, edge_index))
        x1 = F.dropout(x1, training=self.training)
        x1 = F.relu(self.gcn2(x1, edge_index))
        x1 = x1.view(2 * self.n_channel, -1).T
        x1 = F.relu(self.fc_g(x1))
        
        # 邻接矩阵特征处理
        x2 = x2.unsqueeze(0).unsqueeze(0) # 增加一个通道维度
        x2 = F.relu(self.conv1(x2))
        x2 = F.relu(self.conv2(x2))
        x2 = x2.view(1, -1)
        x2 = F.relu(self.fc_c(x2))
        
        # 特征拼接
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class RegressionOpti:
    def __init__(self, n_channel, n_band, n_thread=4):
        self.n_channel = n_channel
        self.n_band = n_band
        self.model_pool = [Model(n_channel, n_band).to(device) for _ in range(n_thread + 1)]
        self.semaphore = Semaphore(n_thread + 1)  # 控制对模型池的访问
        self.cache = {} # 缓存训练结果，加速遗传算法

    def _train(self, model, train_data, val_data):
        model.train()  # 设置模型为训练模式
        optimizer = Adam(model.parameters())
        loss_fn = nn.MSELoss()
        
        patience_counter = 0  # 耐心计数器
        best_val_loss = float('inf')
        
        while patience_counter < patience:
            for x_conv, adj_matrix, y in train_data:
                optimizer.zero_grad()
                # 假设model.forward能够处理对应的输入
                predictions = model(x_conv, adj_matrix)
                loss = loss_fn(predictions.squeeze(), y)
                loss.backward()
                optimizer.step()
            
            val_loss = self._evaluate(model, val_data)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # 重置耐心计数器
            else:
                patience_counter += 1  # 未改善则增加耐心计数器
    def _evaluate(self, model, val_data):
        model.eval()  # 设置模型为评估模式
        loss_fn = nn.MSELoss()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():  # 在评估过程中不计算梯度
            x_conv, adj_matrix, y = val_data
            predictions = model(x_conv, adj_matrix)
            loss = loss_fn(predictions.squeeze(), y)
            total_loss += loss.item()
            total_samples += 1
        
        average_loss = total_loss / total_samples
        return average_loss



    

    def train_eval(self, data):
        # 尝试从缓存中获取结果
        data_key = self._hash_data(data)
        if data_key in self.cache:
            return self.cache[data_key]

        # 如果未缓存，则开始处理
        self.semaphore.acquire()
        model = self._get_idle_model()
        if model is None:
            raise Exception("No idle model available.")
        
        all_losses = []
        for i, (x_conv, adj_matrix, y) in enumerate(data):
            x_conv = torch.tensor(x_conv, dtype=torch.float32).to(device)
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            val_data = (x_conv, adj_matrix, y)

            train_data = [(torch.tensor(x, dtype=torch.float32).to(device),
                           torch.tensor(adj, dtype=torch.float32).to(device),
                           torch.tensor(yl, dtype=torch.float32).to(device)) for j, (x, adj, yl) in enumerate(data) if j != i]
            self._train(model, train_data, val_data)
            loss = self._evaluate(model, val_data)
            all_losses.append(loss)
        
        # 重置模型和优化器，然后标记模型为闲置
        self._reset_model(model)
        self.semaphore.release()

        # 计算结果并加入缓存
        average_loss = np.mean(all_losses)
        self.cache[data_key] = average_loss
        return average_loss

    def _hash_data(self, data):
        return hash(str(data))

    def _get_idle_model(self):
        # 返回一个空闲的模型，实际实现中需要确保线程安全
        for model in self.model_pool:
            if not model.busy:
                model.busy = True
                return model
        return None

    def _reset_model(self, model):
        # 重置模型和优化器到初始状态
        model.__init__(self.n_channel, self.n_band)  # 重置模型参数
        model.to(device)
        model.busy = False  # 标记模型为闲置

if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    from torchviz import make_dot

    m = Model(8,6).to(device)
    data = (torch.zeros([16,6]).to(device), torch.zeros([16,16]).to(device))
    with SummaryWriter("./log", comment="sample_model_visualization") as sw:
        sw.add_graph(m, data)
    torch.save(m, "./log/m.pt")
    out = m(data[0], data[1])
    g = make_dot(out)
    g.render('m.pdf', view=False)