import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=310, hidden_dim1=128,hidden_dim2=64, hidden_dim3=32, num_classes=3):
        super(BaselineMLP, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # 第一隐藏层 310->128
        self.fc2 = nn.Linear(hidden_dim1,hidden_dim2) # 第二隐藏层 128->64
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)  # 第三隐藏层 64-> 32
        self.softmax = nn.Linear(hidden_dim3, num_classes)  # 输出层 32->3
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  # 激活函数
        self.dropout = nn.Dropout(p=0.2)  # Dropout 防止过拟合

    def forward(self, x):
        # 前向传播
        x = x.reshape(x.size(0), -1)       # 转化成合适的维度

        x = self.leaky_relu(self.fc1(x))  # 第一隐藏层 + 激活
        x = self.dropout(x)  # Dropout
        x = self.leaky_relu(self.fc2(x))  # 第二隐藏层 + 激活
        x = self.dropout(x)  # Dropout
        x = self.leaky_relu(self.fc3(x))  # 第三隐藏层 + 激活
        x = self.softmax(x)  # 输出层
        return x
