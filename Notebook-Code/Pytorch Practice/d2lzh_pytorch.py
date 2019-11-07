import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

import numpy as np

import torchvision
import torchvision.transforms as transforms
from IPython import display
from matplotlib import pyplot as plt

import sys
import time
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
def load_data_fashion_mnist(batch_size, num_workers=0):
    mnist_train = torchvision.datasets.FashionMNIST(
        root='~/Datasets', train=True, download=False, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(
        root='~/Datasets', train=False, download=False, transform=transforms.ToTensor())
    # shuffle打乱数据
    train_iter = Data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = Data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        # 精确度计算
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# softmax 分类问题的训练函数
def softmax_train(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_loss_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' %
              (epoch + 1, train_loss_sum / n, train_acc_sum / n, test_acc))


# 输入形状转换
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# 多项式实验作图函数
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)


# 多项式实验作图
def fit_and_plot(train_features, test_features, train_labels, test_labels, num_epochs=100, loss=nn.MSELoss()):
    net = nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])
    dataset = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data, '\nbias:', net.bias.data)


def evaluate_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                # 评估模式，会关闭dropout
                net.eval()
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                # 改回训练模式
                net.train()
            else:
                # 如果有这个参数
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1)
                                == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def corr2d(X, K):
    # X输入，K卷积核，Y输出
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


def train_cuda_cpu(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            test_acc = evaluate_accuracy(test_iter, net)
            print('epoch % d, loss % .4f, train acc % .3f, test acc % .3f, time % .1f sec' % (
                epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def load_data_fashion_mnist2(batch_size, resize=None, root='~/Datasets'):
    # 用于AlexNet的手写数字辨识
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=False, transform=transform)
    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True
        #  num_workers=1
    )
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False
        # num_workers=1
    )
    return train_iter, test_iter


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])
