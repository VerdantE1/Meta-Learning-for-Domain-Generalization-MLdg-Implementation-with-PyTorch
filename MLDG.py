import learn2learn as l2l
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset
from utils import transform_to_8x9
from model import DGCNN
from sklearn.model_selection import train_test_split
import os
from scipy import io as scio
from scipy.stats import zscore
from utils import set_seed
from torch.utils.data import DataLoader, TensorDataset


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def meta_train_mldg(meta_model, meta_optimizer, all_tasks, num_iterations,
                    inner_lr, inner_steps, beta,num_meta_val_domains,eval_interval):
    print("开始 MLDG 元学习训练...")

    query_acc_history = []  # 记录 meta-train 的准确率
    eval_results = []       # 记录评估结果 (test_loss 和 test_acc)

    ## 初步设置train_task和test_task
    train_tasks = all_tasks[:-1]
    test_task = all_tasks[-1]

    best_acc = 0
    for iteration in range(num_iterations):
        meta_optimizer.zero_grad()

        # 从所有领域中随机划分 S 和 S-V
        task_indices = np.random.permutation(len(train_tasks))
        S_indices = task_indices[:-num_meta_val_domains]  # S 部分
        S_V_indices = task_indices[-num_meta_val_domains:]  # S-V 部分


        # 初始化 meta-train 和 meta-val 的损失
        meta_train_loss = 0.0
        meta_val_loss = 0.0

        # 遍历 S 中的每个域 (meta-train)
        for idx in S_indices:
            S_data, S_labels = train_tasks[idx]

            # 转换为张量并移动到设备
            train_data = torch.tensor(S_data, dtype=torch.float32).to(device)
            train_labels = torch.tensor(S_labels, dtype=torch.long).to(device)

            # 内循环：直接在全局模型上优化
            for step in range(inner_steps):
                train_pred = meta_model(train_data)
                train_loss = nn.CrossEntropyLoss()(train_pred, train_labels)
                meta_train_loss += train_loss

        # 计算 S 的平均训练损失
        meta_train_loss /= len(S_indices)

        # 计算训练梯度并更新参数 Θ',但不优化θ
        grads = torch.autograd.grad(meta_train_loss, meta_model.parameters(), create_graph=True)


        print(1)


        # updated_params一定是tensor,故是传播中的一个结点。在这里构建updated_params到param的计算图
        # 只要 create_graph=True，updated_params 中的每个参数都能够追溯到 θ
        updated_params = [
            param - inner_lr * grad
            for param, grad in zip(meta_model.parameters(), grads)
        ]

        # 测试
        """
        original = [
            param
            for param in zip(meta_model.parameters())
        ]
        """

        # 如果不用克隆模型我就要将updated_params改到原始模型的参数或者写另外一个前向传播函数来构建计算图。所以克隆会更完事，以避免改到原始模型
        # 使用克隆模型
        cloned_model = meta_model.clone()  # l2l 提供的克隆方法

        # 使用 updated_params 更新 cloned_model 的参数
        for param, updated_param in zip(cloned_model.parameters(), updated_params):
            param.data.copy_(updated_param)  # 使用 .data 避免构建额外的计算图

        # 遍历 S-V 中的每个域 (meta-val)
        for idx in S_V_indices:
            S_V_data, S_V_labels = train_tasks[idx]

            # 转换为张量并移动到设备
            val_data = torch.tensor(S_V_data, dtype=torch.float32).to(device)
            val_labels = torch.tensor(S_V_labels, dtype=torch.long).to(device)

            # 使用克隆模型进行前向传播
            S_V_pred = cloned_model(val_data)
            val_loss = nn.CrossEntropyLoss()(S_V_pred, val_labels)
            meta_val_loss += val_loss

        # 平均验证损失
        meta_val_loss /= len(S_V_indices)
        # 结合训练损失和验证损失，计算总损失
        total_loss = meta_train_loss + beta * meta_val_loss

        ########################################  DEBUG   ######################################
        # **记录各部分损失对总梯度的贡献**
        meta_train_grads = torch.autograd.grad(meta_train_loss, meta_model.parameters(), retain_graph=True)
        meta_val_grads = torch.autograd.grad(beta * meta_val_loss, meta_model.parameters(), retain_graph=True)

        # 记录训练和验证损失的贡献
        train_grad_norm = sum(g.norm().item() for g in meta_train_grads if g is not None)
        val_grad_norm = sum(g.norm().item() for g in meta_val_grads if g is not None)
        ########################################################################################


        # 反向传播并更新全局模型参数 (更新 Θ)
        total_loss.backward()
        meta_optimizer.step()

        # 记录 meta-train 的准确率
        query_acc = (train_pred.argmax(dim=1) == train_labels).float().mean().item()
        query_acc_history.append(query_acc)

        # 显式删除 grads，并清理显存
        del grads,meta_train_grads,meta_val_grads,cloned_model
        torch.cuda.empty_cache()


        # 每隔 eval_interval 次迭代，测试模型在测试集上的性能
        if (iteration + 1) % eval_interval == 0:
            test_loss, test_acc = evaluate_meta(meta_model, test_task[0], test_task[1])
            eval_results.append((test_loss, test_acc))
            best_acc = max(best_acc,test_acc)


        # 打印详细的调试信息
        print(f"Iteration {iteration + 1}/{num_iterations}:")
        print(f"  Meta-Train Loss = {meta_train_loss.item():.4f}")
        print(f"  Meta-Val Loss = {meta_val_loss.item():.4f}")
        print(f"  Total Loss = {total_loss.item():.4f}")
        print(f"  Meta-Train Grad Norm = {train_grad_norm:.4f}")
        print(f"  Meta-Val Grad Norm (beta-scaled) = {val_grad_norm:.4f}")
        print(f"  Meta-Train Accuracy = {query_acc * 100:.2f}%")
        print(f"  Meta-Train Best_Test_Accuracy = {best_acc * 100:.2f}%")

    print("MLDG 元学习训练完成！")


def meta_train_mldg_loso(all_tasks, num_iterations, inner_lr, inner_steps, beta, num_meta_val_domains, eval_interval):
    print("开始 MLDG LOSO 元学习训练...")
    acc_all = [0] * len(all_tasks)

    for test_task_idx in range(len(all_tasks)):
        print(f"\n===== 开始进行 LOSO 测试：任务 {test_task_idx + 1}/{len(all_tasks)} =====")

        # 重置模型和优化器
        from model_baseline import BaselineMLP
        meta_model = l2l.algorithms.MAML(BaselineMLP().to(device), lr=inner_lr)
        #meta_model = l2l.algorithms.MAML(DynamicGCN_DNN().to(device), lr=inner_lr)
        meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.005, weight_decay=0.0004)

        test_task = all_tasks[test_task_idx]
        train_tasks = [task for i, task in enumerate(all_tasks) if i != test_task_idx]

        best_acc = meta_train_mldg(
            meta_model=meta_model,
            meta_optimizer=meta_optimizer,
            train_tasks=train_tasks,
            test_task=test_task,
            num_iterations=num_iterations,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            beta=beta,
            num_meta_val_domains=num_meta_val_domains,
            eval_interval=eval_interval
        )

        acc_all[test_task_idx] = best_acc
        print(f"任务 {test_task_idx + 1} 的最佳测试准确率: {best_acc * 100:.2f}%")


        trained_accs = [acc for acc in acc_all if acc > 0]
        mean_acc = sum(trained_accs) / len(trained_accs) if trained_accs else 0
        std_acc = np.std(trained_accs) if trained_accs else 0

        print("\n=== Current Best Accuracies for All Trained Subjects ===")
        for idx, acc in enumerate(acc_all):
            print(f"Subject {idx + 1}: {acc:.4f}")
        print(f"Current Mean Accuracy: {mean_acc:.4f}")
        print(f"Current Accuracy Standard Deviation: {std_acc:.4f}")
        print("================================================\n")

    print("\n===== LOSO 训练完成！ =====")
    print(f"所有任务的测试准确率: {[f'{r * 100:.2f}%' for r in acc_all]}")
    print(f"平均测试准确率: {np.mean(acc_all) * 100:.2f}%")
    print(f"测试准确率标准差: {np.std(acc_all) * 100:.2f}%")
    return acc_all



