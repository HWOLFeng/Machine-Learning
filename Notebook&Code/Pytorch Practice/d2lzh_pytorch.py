import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

import torchvision
import torchvision.transforms as transforms
import time
import torch.utils.data as Data
import sys


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


# 将FASHION-MNIST数据集中对应的数值标签转换成文本
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress',
                   'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']
    return [text_labels[int(i)] for i in labels]


# 一行里画出多张图像和对应标签的函数
def show_fashion_mnist(images, labels):
    use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view(28, 28).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 最初的加载数据代码
def load_data_fashion_mnist(batch_size, num_workers):
    mnist_train = torchvision.datasets.FashionMNIST(
        root='~/Datasets', train=True, download=False, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(
        root='~/Datasets', train=False, download=False, transform=transforms.ToTensor())
    train_iter = Data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = Data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter
