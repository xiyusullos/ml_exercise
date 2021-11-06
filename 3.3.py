#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@LICENSE: Copyright(C) 2019, xiyusullos
@AUTHOR: xiyusullos
@BLOG: https://blog.aponder.top
@FILE: 3.3.py
@TIME: 2019-01-04 15:51:29
@DESC: 使用sklearn包来实现
'''

# import matplotlib.pylab as pl
# import matplotlib.pyplot as plt
# import numpy as np  # for matrix calculation
# from sklearn import metrics
# from sklearn import model_selection
# from sklearn.linear_model import LogisticRegression
#
# SCATTER_TITLE = 'data 3.0a scatter'
# SCATTER_XLABEL = 'density'
# SCATTER_YLABEL = 'sugar ratio'
#
# # 加载对应数据
# dataset = np.genfromtxt('data/3.0a.csv', delimiter=',')
# # 将数据按照属性分离开
# X = dataset[1:, 1:3]
# y = dataset[1:, 3]
# m, n = np.shape(X)
# # 绘制原始数据散点图
# f1 = plt.figure(1)
# plt.title(SCATTER_TITLE)
# plt.xlabel(SCATTER_XLABEL)
# plt.ylabel(SCATTER_YLABEL)
#
# plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
# plt.legend(loc='upper right')
# plt.show()
#
# # 生成测试和训练数据集
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
# # 模型训练
# log_model = LogisticRegression()  # using log-regression lib model
# log_model.fit(X_train, y_train)  # fitting
# # 模型验证
# y_pred = log_model.predict(X_test)
# # 总结模型的适用性
# print(metrics.confusion_matrix(y_test, y_pred))
# print(metrics.classification_report(y_test, y_pred))
# precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
# # 显示决策边界
# f2 = plt.figure(2)
# h = 0.001
# x0_min, x0_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
# x1_min, x1_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
# x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h), np.arange(x1_min, x1_max, h))
# # 这里“模型”是你模型的预测（分类）功能
# z = log_model.predict(np.c_[x0.ravel(), x1.ravel()])
# # 将结果标记出来
# z = z.reshape(x0.shape)
# plt.contourf(x0, x1, z, cmap=pl.cm.Paired)
# plt.title(SCATTER_TITLE)
# plt.xlabel(SCATTER_XLABEL)
# plt.ylabel(SCATTER_YLABEL)
# plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
# plt.legend(loc='upper right')
# plt.show()


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def J_cost(X, y, beta):
    '''
    :param X:  sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return: the result of formula 3.27
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)

    Lbeta = -y * np.dot(X_hat, beta) + np.log(1 + np.exp(np.dot(X_hat, beta)))

    return Lbeta.sum()


def gradient(X, y, beta):
    '''
    compute the first derivative of J(i.e. formula 3.27) with respect to beta      i.e. formula 3.30
    ----------------------------------
    :param X: sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return:
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    p1 = sigmoid(np.dot(X_hat, beta))

    gra = (-X_hat * (y - p1)).sum(0)

    return gra.reshape(-1, 1)


def hessian(X, y, beta):
    '''
    compute the second derivative of J(i.e. formula 3.27) with respect to beta      i.e. formula 3.31
    ----------------------------------
    :param X: sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return:
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)

    p1 = sigmoid(np.dot(X_hat, beta))

    m, n = X.shape
    P = np.eye(m) * p1 * (1 - p1)

    assert P.shape[0] == P.shape[1]
    return np.dot(np.dot(X_hat.T, P), X_hat)


def update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost):
    '''
    update parameters with gradient descent method
    --------------------------------------------
    :param beta:
    :param grad:
    :param learning_rate:
    :return:
    '''
    for i in range(num_iterations):

        grad = gradient(X, y, beta)
        beta = beta - learning_rate * grad

        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))

    return beta


def update_parameters_newton(X, y, beta, num_iterations, print_cost):
    '''
    update parameters with Newton method
    :param beta:
    :param grad:
    :param hess:
    :return:
    '''

    for i in range(num_iterations):

        grad = gradient(X, y, beta)
        hess = hessian(X, y, beta)
        beta = beta - np.dot(np.linalg.inv(hess), grad)

        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))
    return beta


def initialize_beta(n):
    beta = np.random.randn(n + 1, 1) * 0.5 + 1
    return beta


def logistic_model(X, y, num_iterations=100, learning_rate=1.2, print_cost=False, method='gradDesc'):
    '''
    :param X:
    :param y:~
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :param method: str 'gradDesc' or 'Newton'
    :return:
    '''
    m, n = X.shape
    beta = initialize_beta(n)

    if method == 'gradDesc':
        return update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost)
    elif method == 'Newton':
        return update_parameters_newton(X, y, beta, num_iterations, print_cost)
    else:
        raise ValueError('Unknown solver %s' % method)


def predict(X, beta):
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    p1 = sigmoid(np.dot(X_hat, beta))

    p1[p1 >= 0.5] = 1
    p1[p1 < 0.5] = 0

    return p1


if __name__ == '__main__':
    data_path = 'data/4.2.csv'
    data_path = '/home/x/codes/ml_exercise/data/3.0a.csv'
    #
    data = pd.read_csv(data_path).values

    # data = np.genfromtxt(data_path)

    is_good = data[:, 3] == 1
    is_bad = data[:, 3] == 0

    X = data[:, 1:3].astype(float)
    y = data[:, 3]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)

    plt.scatter(data[:, 1][is_good], data[:, 2][is_good], c='k', marker='o')
    plt.scatter(data[:, 1][is_bad], data[:, 2][is_bad], c='r', marker='x')

    plt.xlabel('密度')
    plt.ylabel('含糖量')

    # 可视化模型结果
    beta = logistic_model(X, y, print_cost=True, method='gradDesc', learning_rate=0.3, num_iterations=1000)
    w1, w2, intercept = beta
    x1 = np.linspace(0, 1)
    y1 = -(w1 * x1 + intercept) / w2

    print(w1, w2)

    ax1, = plt.plot(x1, y1, label=r'my_logistic_gradDesc')

    lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)  # 注意sklearn的逻辑回归中，C越大表示正则化程度越低。
    lr.fit(X, y)

    lr_beta = np.c_[lr.coef_, lr.intercept_]
    print(J_cost(X, y, lr_beta))

    # 可视化sklearn LogisticRegression 模型结果
    w1_sk, w2_sk = lr.coef_[0, :]

    x2 = np.linspace(0, 1)
    y2 = -(w1_sk * x2 + lr.intercept_) / w2

    ax2, = plt.plot(x2, y2, label=r'sklearn_logistic')

    plt.legend(loc='upper right')
    plt.show()