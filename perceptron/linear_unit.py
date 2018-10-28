# ！/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Jie Xu'

from perceptron import Perceptron

# 线性单元的激活函数与感知机有所区别，其他相同，故可由感知机类继承而来
f = lambda x: x

class LinearUnit(Perceptron):
    def __init__(self, input_num):
        """
        初始化线性单元，设置输入参数的个数
        :param input_num:
        """
        Perceptron.__init__(self, input_num, f)

# 测试
def get_training_dataset():
    """
    随意假设五个人的收入数据
    :return:
    """
    # 构建训练数据
    # 输入向量列表，每一项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，月薪
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    """
    使用数据训练线性单元
    :return:
    """
    # 创建线性单元，输入特征数为1
    lu = LinearUnit(1)
    # 训练，迭代10轮，学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    # 返回训练好的线性单元
    return lu


if __name__ == '__main__':
    """
    训练线性单元
    """
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print(linear_unit)
    # 测试
    print('work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('work 13 years, monthly salary = %.2f' % linear_unit.predict([13]))
    print('work 2 years, monthly salary = %.2f' % linear_unit.predict([2]))
    print('work 7.4 years, monthly salary = %.2f' % linear_unit.predict([7.4]))
