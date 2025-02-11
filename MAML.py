import learn2learn as l2l
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from utils import eegDataset, transform_to_8x9
from model import DGCNN
from sklearn.model_selection import train_test_split
import os
from scipy import io as scio
from scipy.stats import zscore
from utils import set_seed

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义元学习训练过程
def meta_train(meta_model, meta_optimizer, train_tasks, num_iterations, inner_lr, inner_steps):
    """
    meta_model: 基础模型（如 DGCNN）
    meta_optimizer: 优化器（如 Adam）
    train_tasks: 元学习任务数据集
    num_iterations: 元训练迭代次数
    inner_lr: 内循环学习率
    inner_steps: 内循环优化步数
    """
    print("开始元学习训练...")
    query_acc_history = []  # 用于记录查询集的准确率
    for iteration in range(num_iterations):
        meta_optimizer.zero_grad()

        # 从任务数据集中采样一个任务
        task_idx = np.random.randint(len(train_tasks))
        support_x, support_y, query_x, query_y = train_tasks[task_idx][0]

        # 将数据移动到设备
        support_x, support_y = support_x.to(device), support_y.to(device)
        query_x, query_y = query_x.to(device), query_y.to(device)

        # 内循环优化：在支持集上训练模型
        learner = meta_model.clone()  # 克隆模型以进行任务级优化
        for step in range(inner_steps):
            support_pred = learner(support_x)
            support_loss = nn.CrossEntropyLoss()(support_pred, support_y)
            learner.adapt(support_loss, allow_nograd=False)  # 梯度下降更新模型参数

        # 外循环优化：在查询集上评估支持集训练后的模型
        query_pred = learner(query_x)
        query_loss = nn.CrossEntropyLoss()(query_pred, query_y)

        # 反向传播查询集损失，用于更新全局模型参数
        query_loss.backward()
        meta_optimizer.step()

        # 计算查询集准确率
        query_acc = (query_pred.argmax(dim=1) == query_y).float().mean().item()
        query_acc_history.append(query_acc)

        print(f"Iteration {iteration + 1}/{num_iterations}: Query Loss = {query_loss.item()}, Query Accuracy = {query_acc * 100:.2f}%")

    print("元学习训练完成！")
