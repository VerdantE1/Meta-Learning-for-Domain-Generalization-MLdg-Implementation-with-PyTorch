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


def load_DE_SEED(load_path):
    """加载SEED数据集中的差分熵(DE)特征
    返回:
        dataAll: 形状为 [样本数, 通道数, 频带数] 的DE特征
        labelAll: 对应的情感标签
    """

    filePath = load_path
    datasets = scio.loadmat(filePath)
    DE = datasets['DE']
    # DE_delta = np.squeeze(DE[:,:,0]).T
    # DE_theta = np.squeeze(DE[:,:,1]).T
    # DE_alpha = np.squeeze(DE[:,:,2]).T
    # DE_beta = np.squeeze(DE[:,:,3]).T
    # DE_gamma = np.squeeze(DE[:,:,4]).T
    # dataAll = np.concatenate([DE_delta,DE_theta,DE_alpha,DE_beta,DE_gamma], axis=1)
    dataAll = np.transpose(DE, [1,0,2])
    labelAll = datasets['labelAll'].flatten()
    labelAll = labelAll + 1

    return dataAll, labelAll

class EEGTaskDataset(Dataset):
    """用于生成支持集和查询集的数据集"""
    def __init__(self, data, labels, support_ratio=0.2):
        """
        参数:
            data: 样本数据，形状为 [样本数, 通道数, 特征数]
            labels: 标签数据，形状为 [样本数]
            support_ratio: 支持集的比例 (0.0, 1.0)
        """
        self.data = data
        self.labels = labels
        self.support_ratio = support_ratio

        # 确保支持集比例合法
        if not (0.0 < self.support_ratio < 1.0):
            raise ValueError("support_ratio must be between 0.0 and 1.0")

        # 按比例划分支持集和查询集
        self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(
            self.data, self.labels, test_size=1 - self.support_ratio, stratify=self.labels
        )
        #print("data_train shape:{}",self.data_train.shape)
        #print("data_test shape:{}", self.data_test.shape)

    def __len__(self):
        """
        返回任务的数量，这里每个 EEGTaskDataset 表示一个任务，因此返回 1。
        """
        return 1

    def __getitem__(self, idx):
        """
        返回支持集和查询集，用于元学习任务。
        """
        return (
            torch.tensor(self.data_train, dtype=torch.float32),
            torch.tensor(self.labels_train, dtype=torch.long),
            torch.tensor(self.data_test, dtype=torch.float32),
            torch.tensor(self.labels_test, dtype=torch.long),
        )
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


def evaluate_meta(meta_model, test_data, test_labels, inner_lr=0.01, inner_steps=5, support_ratio=0.2):
    """
    元测试阶段：在测试被试上进行微调并评估模型性能。

    参数:
        meta_model: 经过元学习训练后的模型
        test_data: 测试被试的 EEG 数据
        test_labels: 测试被试的标签
        inner_lr: 内循环学习率，用于支持集微调
        inner_steps: 内循环优化步数
        support_ratio: 支持集的比例 (0.0, 1.0)
    """
    print("开始元测试...")

    # 将测试数据划分为支持集和查询集
    support_x, query_x, support_y, query_y = train_test_split(
        test_data, test_labels, test_size=1 - support_ratio, stratify=test_labels
    )

    # 转换为 Tensor 并移动到设备
    support_x = torch.tensor(support_x, dtype=torch.float32).to(device)
    query_x = torch.tensor(query_x, dtype=torch.float32).to(device)
    support_y = torch.tensor(support_y, dtype=torch.long).to(device)
    query_y = torch.tensor(query_y, dtype=torch.long).to(device)

    # 克隆模型以进行任务级优化
    learner = meta_model.clone()

    # 内循环：在支持集上微调模型
    for step in range(inner_steps):
        support_pred = learner(support_x)
        support_loss = nn.CrossEntropyLoss()(support_pred, support_y)
        learner.adapt(support_loss)  # 根据支持集损失更新模型参数

    # 查询集评估：在查询集上测试微调后的模型
    query_pred = learner(query_x)
    query_loss = nn.CrossEntropyLoss()(query_pred, query_y)
    query_acc = (query_pred.argmax(dim=1) == query_y).float().mean().item()

    print(f"查询集损失: {query_loss.item():.4f}")
    print(f"查询集准确率: {query_acc * 100:.2f}%")

    return query_loss.item(), query_acc

"""
# 测试阶段
def test_model(meta_model, test_data, test_labels):

    使用元学习得到的模型参数直接测试最后一个被试的性能

    print("开始测试模型...")
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    # 直接用元学习得到的模型参数对测试数据进行预测
    meta_model.eval()
    with torch.no_grad():
        predictions = meta_model(test_data)
        accuracy = (predictions.argmax(dim=1) == test_labels).float().mean().item()

    print(f"测试集准确率: {accuracy * 100:.2f}%")
    return accuracy

"""





def main():
    # 加载数据
    data_dir = r'E:/SEED/SEED_Divide/DE/session1/'
    file_list = os.listdir(data_dir)

    set_seed(520)  #
    support_ratio = 0.1  # 支持集比例
    num_iterations = 1000  # 元训练迭代次数
    inner_lr = 0.01       # 内循环学习率
    inner_steps = 5       # 内循环优化步数

    # 定义模型
    xdim = [128, 62, 5]
    k_adj = 40
    num_out = 64
    meta_model = l2l.algorithms.MAML(DGCNN(xdim, k_adj, num_out).to(device), lr=inner_lr)

    # 优化器
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001)

    # 构建任务数据集
    train_tasks = []
    for filename in file_list[:-1]:  # 使用部分被试作为训练任务
        file_path = os.path.join(data_dir, filename)
        data, labels = load_DE_SEED(file_path)
        data = zscore(data)   # 标准化

        # 过滤样本数量不足的任务
        if len(data) > 1:  # 确保至少有一个样本用于支持集
            train_tasks.append(EEGTaskDataset(data, labels, support_ratio=support_ratio))
        else:
            print(f"跳过任务 {filename}: 样本数量不足")

    # 元训练
    meta_train(meta_model, meta_optimizer, train_tasks, num_iterations, inner_lr, inner_steps)

    # 测试阶段
    test_file = file_list[-1]  # 测试被试
    test_path = os.path.join(data_dir, test_file)
    test_data, test_labels = load_DE_SEED(test_path)
    test_data = zscore(test_data)


    # test_model(meta_model, test_data, test_labels)  # 直接用元学习后的模型参数测试 LOSO测试
    evaluate_meta(meta_model,test_data,test_labels,inner_lr=inner_lr,inner_steps=inner_steps)

if __name__ == "__main__":
    main()