# PyTorch Implementation of Meta-Learning for Domain Generalization (MLDG)

## 项目简介

本项目实现了元学习算法 MLdg，使用 PyTorch 和 l2l 框架。当前 GitHub 上大多数 MLdg 实现采用 TensorFlow，且部分Pytorch实现存在错误。该项目旨在提供一个正确的 PyTorch 实现，以展示 MLdg 算法的有效性和应用。此外，项目中还包含了MAML算法实现，以及我的一些对MLDG的改进想法MLDG_Queue，具体内容可在 `backup` 目录中找到。
本项目仅实现自监督学习方式，强化学习暂未实现；

## 论文参考
[Learning to Generalize: Meta-Learning for Domain Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/11596)

![image](https://github.com/user-attachments/assets/df37096c-bc58-4713-b525-8c3af7bba31b)

## 背景
随着领域泛化（Domain Generalization）问题的日益重要，MLdg 作为一种元学习算法，通过在多个领域进行训练，提升模型在未见领域的适应能力。本项目为研究人员和开发者提供了一个基于 PyTorch 的实现，帮助他们更好地理解和应用该算法。
## 特点

- 使用 PyTorch 和 l2l 框架实现 MLdg 算法
- 纠正现有实现中的错误
- 简洁明了的代码结构

