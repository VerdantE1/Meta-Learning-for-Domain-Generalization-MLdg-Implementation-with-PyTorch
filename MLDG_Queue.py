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

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





def meta_train_mldg_with_queue(meta_model, meta_optimizer, all_tasks, num_iterations,
                               inner_steps, inner_lr, beta, eval_interval):
    print("开始 MLDG 元学习训练（基于整体交互的 inner_steps）...")


    eval_results = []       # 记录评估结果 (test_loss 和 test_acc)

    # 初始设置 S-V（队列 Q）和 V
    Q = [all_tasks[0]]  # 队列 Q，初始仅包含第一个域作为 S-V
    V = all_tasks[1:-1]   # 剩余域作为 V

    best_acc = 0
    for iteration in range(num_iterations):
        meta_optimizer.zero_grad()

        # 从 V 中选取一个域，与 S-V（队列 Q）交互训练
        if len(V) > 0:
            current_V = V.pop(0)  # 从 V 中取出一个域
        else:
            current_V = None  # 如果 V 已空，不再新增域

        # **整体交互训练，重复 inner_steps 次**
        for inner_step in range(inner_steps):
            # 初始化 meta-train 和 meta-val 的损失
            meta_train_loss = 0.0
            meta_val_loss = 0.0

            # **训练 S-V（队列 Q）**
            for S_V_data, S_V_labels in Q:
                # 转换为张量并移动到设备
                train_data = torch.tensor(S_V_data, dtype=torch.float32).to(device)
                train_labels = torch.tensor(S_V_labels, dtype=torch.long).to(device)

                # S-V 的训练
                train_pred = meta_model(train_data)
                train_loss = nn.CrossEntropyLoss()(train_pred, train_labels)
                meta_train_loss += train_loss

            # 计算训练梯度并更新参数 Θ'
            grads = torch.autograd.grad(meta_train_loss, meta_model.parameters(), create_graph=True)

            # 更新后的参数（构建计算图，以便后续 meta-val 使用）
            updated_params = [
                param - inner_lr * grad
                for param, grad in zip(meta_model.parameters(), grads)
            ]


            # **验证当前 V 域（如果存在）**
            if current_V is not None:
                V_data, V_labels = current_V
                val_data = torch.tensor(V_data, dtype=torch.float32).to(device)
                val_labels = torch.tensor(V_labels, dtype=torch.long).to(device)

                # 使用克隆模型加载更新后的参数
                cloned_model = meta_model.clone()  # l2l 提供的克隆方法
                # 使用 updated_params 更新 cloned_model 的参数
                for param, updated_param in zip(cloned_model.parameters(), updated_params):
                    param.data.copy_(updated_param)

                # V 的验证
                V_pred = cloned_model(val_data)
                val_loss = nn.CrossEntropyLoss()(V_pred, val_labels)
                meta_val_loss += val_loss

            # 计算 S-V 的平均训练损失
            meta_train_loss /= len(Q)

            # 如果当前有 V 域，计算验证损失
            if current_V is not None:
                meta_val_loss /= 1  # 这里实际上可以省略，保持一致性

            # **结合训练损失和验证损失，计算总损失**
            total_loss = meta_train_loss + beta * meta_val_loss

            # **反向传播并更新全局模型参数（在内循环中更新）**
            total_loss.backward()
            meta_optimizer.step()
            meta_optimizer.zero_grad()  # 避免梯度累计


            del grads,updated_params,cloned_model
            torch.cuda.empty_cache()

            # 记录 meta-train 的准确率
            query_acc = (train_pred.argmax(dim=1) == train_labels).float().mean().item()


        # 如果当前有 V 域，则将其加入 S-V 队列（扩展队列 Q）
        if current_V is not None:
            Q.append(current_V)



        test_task = all_tasks[-1]  # 测试任务为最后一个域
        test_loss, test_acc = evaluate_meta(meta_model, test_task[0], test_task[1])
        eval_results.append((test_loss, test_acc))
        best_acc = max(best_acc, test_acc)



        # 每隔 eval_interval 次迭代，测试模型在测试集上的性能
        # if (iteration + 1) % eval_interval == 0 or  current_V == None:


        # 打印详细的调试信息，包括 Q 和 V 的队列长度
        print(f"Iteration {iteration + 1}/{num_iterations}:")
        print(f"  Meta-Train Loss = {meta_train_loss.item():.4f}")
        print(f"  Meta-Val Loss = {meta_val_loss.item():.4f}")
        print(f"  Total Loss = {total_loss.item():.4f}")
        print(f"  Meta-Train Accuracy = {query_acc * 100:.2f}%")
        print(f"  Queue Q Size (S-V) = {len(Q)} | Remaining V Queue Size = {len(V)}")
        print(f"  Meta-Train Best_Test_Accuracy = {best_acc * 100:.2f}%")

    print("MLDG 元学习训练完成！")
