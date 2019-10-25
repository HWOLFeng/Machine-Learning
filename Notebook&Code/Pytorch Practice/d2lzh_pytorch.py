import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random


# 作图
def use_svg_display():
    display.set_matplotlib_formats('svg')


# 设置图像大小
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机读取样本
    random.shuffle(indices)
    # range(start, end, step)
    for i in range(0, num_examples, batch_size):
        # 最后一次可能不足一个batch
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        # 类似return, 不同的是yield每次返回结果之后函数并没有退出，
        # 而是每次遇到yield关键字后返回相应结果，并保留函数当前的运行状态，等待下一次的调用。
        yield features.index_select(0, j), labels.index_select(0, j)


# 线性回归的矢量计算表达式
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 均方差损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 小批量梯度下降
def sgd(params, lr, batch_size):
    for param in params:
        # 批量
        param.data -= lr / batch_size * param.grad
