# ！/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Jie Xu'

"""
采用向量化计算实现全连接、激活函数为sigmoid函数、目标函数为误差平方和、优化算法为随机梯度下降优化算法
"""

import numpy as np


# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        """
        构造函数
        :param input_size: 本层输入向量维度
        :param output_size: 本层输出向量维度
        :param activator: 激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # 偏置项b
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        """
        前向计算
        :param input_array: 输入向量，维度等于input_size
        :return:
        """
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        """
        反向计算W和b的梯度
        :param delta_array:从上一层传递过来的误差项
        :return:
        """
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def updete(self, learning_rate):
        """
        使用梯度下降法更新权重
        :param learning_rate:
        :return:
        """
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad


 # sigmoid激活函数类
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


# 神经网络类
class Network(object):
    def __init__(self, layers):
        """
        构造函数
        :param layers:
        """
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i + 1], SigmoidActivator()))

    def predict(self, sample):
        """
        使用神经网络实现预测
        :param sample: 输入样本
        :return:
        """
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        """
        训练函数
        :param labels: 样本标签
        :param data_set: 输入样本
        :param rate: 学习速率
        :param epoch: 训练轮数
        :return:
        """
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)




