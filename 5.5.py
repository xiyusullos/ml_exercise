#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@LICENSE: Copyright(C) 2019, xiyusullos
@AUTHOR: xiyusullos
@BLOG: https://blog.aponder.top
@FILE: 5.5.py
@TIME: 2019-01-10 09:27:22
@DESC:
'''

# STANDARD BP-NN & ACCUMULATED BP-NN
import numpy as np


class Data(object):
    def __init__(self, data):
        self.data = np.array(data)
        self.rows = len(self.data[:, 0])
        self.cols = len(self.data[0, :])  # it include the column of labels
        self.__eta = 0.1  # initial eta=0.1
        self.__in = self.cols - 1  # number of input neurons
        self.__out = len(np.unique(self.data[:, -1]))  # number of output neurons

    def set_eta(self, n):
        self.__eta = n

    def get_eta(self):
        return self.__eta

    def get_in(self):
        return self.__in

    def get_out(self):
        return self.__out

    # 标准BP算法
    def BP_NN(self, q=10, err=0.1):
        X = self.data[:, :-1]
        # 为X矩阵左边插入列-1来计算vx-gama,在后面对b操作应该同样加一列，来计算wb-theta
        X = np.insert(X, [0], -1, axis=1)
        Y = np.array([self.data[:, -1], 1 - self.data[:, -1]]).transpose()
        d, l = self.__in, self.__out
        v = np.mat(np.random.random((d + 1, q)))  # v_0 = gama
        w = np.mat(np.random.random((q + 1, l)))  # w_0 = theta

        def f(x):  # sigmoid function
            s = 1 / (1 + np.exp(-x))
            return s

        n = self.__eta
        gap = 1
        counter = 0
        while gap > err:  # set E_k<=0.01 to quit the loop
            counter += 1
            for i in range(self.rows):
                alpha = np.mat(X[i, :]) * v  # 1*q matrix
                b_init = f(alpha)  # 1*q matrix
                # 注意把中间变量b_init增加一个b_0，且b_0 = -1,此时成为b
                b = np.insert(b_init.T, [0], -1, axis=0)  # (q+1)*1 matrix
                beta = b.T * w  # 1*l matrix
                y_cal = np.array(f(beta))  # 1*l array

                g = y_cal * (1 - y_cal) * (Y[i, :] - y_cal)  # 1*l array
                w_g = w[1:, :] * np.mat(g).T  # q*1 matrix
                e = np.array(b_init) * (1 - np.array(b_init)) * np.array(w_g.T)  # 1*q array
                d_w = n * b * np.mat(g)
                d_v = n * np.mat(X[i, :]).T * np.mat(e)

                w += d_w
                v += d_v
            gap = 0.5 * np.sum((Y[i, :] - y_cal) ** 2)
        print('BP_round:', counter)
        return v, w

    # l累积BP算法
    def ABP_NN(self, q=10, err=0.1):
        X = self.data[:, :-1]
        # 为X矩阵左边插入列-1来计算vx-gama,在后面对b操作应该同样加一列，来计算wb-theta
        X = np.insert(X, [0], -1, axis=1)
        Y = np.array([self.data[:, -1], 1 - self.data[:, -1]]).transpose()
        d, l = self.__in, self.__out
        v = np.mat(np.random.random((d + 1, q)))  # v_0 = gama
        w = np.mat(np.random.random((q + 1, l)))  # w_0 = theta

        def f(x):  # sigmoid function
            s = 1 / (1 + np.exp(-x))
            return s

        n = self.__eta
        gap = 1
        counter = 0
        while gap > err:  # set E_k<=1 to quit the loop
            d_v, d_w, gap = 0, 0, 0
            counter += 1
            for i in range(self.rows):
                alpha = np.mat(X[i, :]) * v  # 1*q matrix
                b_init = f(alpha)  # 1*q matrix
                # 注意把中间变量b_init增加一个b_0，且b_0 = -1,此时成为b
                b = np.insert(b_init.T, [0], -1, axis=0)  # (q+1)*1 matrix
                beta = b.T * w  # 1*l matrix
                y_cal = np.array(f(beta))  # 1*l array

                g = y_cal * (1 - y_cal) * (Y[i, :] - y_cal)  # 1*l array
                w_g = w[1:, :] * np.mat(g).T  # q*1 matrix
                e = np.array(b_init) * (1 - np.array(b_init)) * np.array(w_g.T)  # 1*q array
                d_w += n * b * np.mat(g)
                d_v += n * np.mat(X[i, :]).T * np.mat(e)
                gap += 0.5 * np.sum((Y[i, :] - y_cal) ** 2)
            w += d_w / self.rows
            v += d_v / self.rows
            gap = gap / self.rows
        print('ABP_round:', counter)
        return v, w


def test_NN(a, v, w):
    X = a.data[:, :-1]
    X = np.insert(X, [0], -1, axis=1)
    Y = np.array([a.data[:, -1], 1 - a.data[:, -1]]).transpose()
    y_cal = np.zeros((a.rows, 2))

    def f(x):  # sigmoid function
        s = 1 / (1 + np.exp(-x))
        return s

    for i in range(a.rows):
        alpha = np.mat(X[i, :]) * v  # 1*q matrix
        b_init = f(alpha)  # 1*q matrix
        b = np.insert(b_init.T, [0], -1, axis=0)  # (q+1)*1 matrix
        beta = b.T * w  # 1*l matrix
        y_cal[i, :] = np.array(f(beta))  # 1*l array
    print(y_cal)


# 对原始数据进行一位有效编码
D = np.array([
    [1, 1, 1, 1, 1, 1, 0.697, 0.460, 1],
    [2, 1, 2, 1, 1, 1, 0.774, 0.376, 1],
    [2, 1, 1, 1, 1, 1, 0.634, 0.264, 1],
    [1, 1, 2, 1, 1, 1, 0.608, 0.318, 1],
    [3, 1, 1, 1, 1, 1, 0.556, 0.215, 1],
    [1, 2, 1, 1, 2, 2, 0.403, 0.237, 1],
    [2, 2, 1, 2, 2, 2, 0.481, 0.149, 1],
    [2, 2, 1, 1, 2, 1, 0.437, 0.211, 1],
    [2, 2, 2, 2, 2, 1, 0.666, 0.091, 0],
    [1, 3, 3, 1, 3, 2, 0.243, 0.267, 0],
    [3, 3, 3, 3, 3, 1, 0.245, 0.057, 0],
    [3, 1, 1, 3, 3, 2, 0.343, 0.099, 0],
    [1, 2, 1, 2, 1, 1, 0.639, 0.161, 0],
    [3, 2, 2, 2, 1, 1, 0.657, 0.198, 0],
    [2, 2, 1, 1, 2, 2, 0.360, 0.370, 0],
    [3, 1, 1, 3, 3, 1, 0.593, 0.042, 0],
    [1, 1, 2, 2, 2, 1, 0.719, 0.103, 0]])
a = Data(D)  # 加载数据
v, w = a.ABP_NN(err=0.01)  # 累积BP
v1, w1 = a.BP_NN(err=0.2)  # 标准BP
# v, w = a.ABP_NN(err=0.2)#累积BP
# v1, w1 = a.BP_NN(err=0.01)#标准BP
test_NN(a, v, w)
test_NN(a, v1, w1)
