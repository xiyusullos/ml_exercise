#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@LICENSE: Copyright(C) 2019, xiyusullos
@AUTHOR: xiyusullos
@BLOG: https://blog.xy-jit.cc
@FILE: 6.3.py
@TIME: 2019-01-10 09:44:47
@DESC:
'''

from sklearn.datasets import load_breast_cancer

data_set = load_breast_cancer()

X = data_set.data  # feature
feature_names = data_set.feature_names
y = data_set.target  # label
target_names = data_set.target_names

# 画散点图
import matplotlib.pyplot as plt

f1 = plt.figure(1)

p1 = plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r', label=target_names[0])  # feature
p2 = plt.scatter(X[y == 1, 0], X[y == 1, 1], color='g', label=target_names[1])  # feature
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend(loc='upper right')
plt.grid(True, linewidth=0.3)

plt.show()

# 数据规范化
from sklearn import preprocessing

normalized_X = preprocessing.normalize(X)

# 模型拟合和测试
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
import numpy as np

# 训练集和测试集的生成
X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.5, random_state=0)

# 模型拟合，测试，可视化
# 基于线性内核和高斯内核
for fig_num, kernel in enumerate(('linear', 'rbf')):
    accuracy = []
    c = []
    for C in range(1, 1000, 1):
        # 初始化
        clf = svm.SVC(C=C, kernel=kernel)
        # 训练
        clf.fit(X_train, y_train)
        # 测试
        y_pred = clf.predict(X_test)
        accuracy.append(metrics.accuracy_score(y_test, y_pred))
        c.append(C)

    print('max accuracy of %s kernel SVM: %.3f' % (kernel, max(accuracy)))

    # 绘制准确率
    f2 = plt.figure(2)
    plt.plot(c, accuracy)
    plt.xlabel('penalty parameter')
    plt.ylabel('accuracy')
    plt.show()

# BP net for classification on breast_cancer data set

# 加载数据
from sklearn.datasets import load_breast_cancer

data_set = load_breast_cancer()

X = data_set.data  # feature
feature_names = data_set.feature_names
y = data_set.target  # label
target_names = data_set.target_names

# data normalization
from sklearn import preprocessing

normalized_X = preprocessing.normalize(X)

# 构建数据
from pybrain.datasets import ClassificationDataSet  # 分类数据专用数据集工具包

ds = ClassificationDataSet(30, 1, nb_classes=2, class_labels=y)
for i in range(len(y)):
    ds.appendLinked(X[i], y[i])
ds.calculateStatistics()

# 分割训练和测试数据集
tstdata_temp, trndata_temp = ds.splitWithProportion(0.5)
tstdata = ClassificationDataSet(30, 1, nb_classes=2)
for n in range(0, tstdata_temp.getLength()):
    tstdata.appendLinked(tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1])
# 定义训练集，  输入维度是30，target是1，有2类，均是2d向量
trndata = ClassificationDataSet(30, 1, nb_classes=2)
for n in range(0, trndata_temp.getLength()):
    trndata.appendLinked(trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1])
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

# 建立训练
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError

n_hidden = 500
bp_nn = buildNetwork(trndata.indim, n_hidden, trndata.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(bp_nn,
                          dataset=trndata,
                          verbose=True,
                          momentum=0.5,
                          learningrate=0.0001,
                          batchlearning=True)
err_train, err_valid = trainer.trainUntilConvergence(maxEpochs=1000,
                                                     validationProportion=0.25)

# 收敛曲线用于累积BP算法的过程
import matplotlib.pyplot as plt

f1 = plt.figure(1)
plt.plot(err_train, 'b', err_valid, 'r')
plt.title('BP network classification')
plt.ylabel('error rate')
plt.xlabel('epochs')
plt.show()

# 测试
tst_result = percentError(trainer.testOnClassData(tstdata), tstdata['class'])
print("epoch: %4d" % trainer.totalepochs, " test error: %5.2f%%" % tst_result)

data_file_train = "result/btrain.csv"
data_file_valid = "result/bvalidate.csv"
data_file_test = "result/btest.csv"
data_file_datatype = "result/datatypes.csv"

# C4.5用于乳腺癌数据集的分类

from sklearn.datasets import load_breast_cancer
import pandas as pd

# 生成数据集
data_set = load_breast_cancer()
X = data_set.data  # feature
feature_names = data_set.feature_names
y = data_set.target  # label
target_names = data_set.target_names

from sklearn.model_selection import train_test_split

# 生成训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.5, random_state=0)

df_train = pd.DataFrame(X_train)
df_train.columns = feature_names
df_train['class'] = y_train
df_train.to_csv(data_file_train)

df_valid = pd.DataFrame(X_valid)
df_valid.columns = feature_names
df_valid['class'] = y_valid
df_valid.to_csv(data_file_valid)

df_test = pd.DataFrame(X_test)
df_test.columns = feature_names
df_test['class'] = y_test
df_test.to_csv(data_file_test)

# 学习和训练
import decision_tree

decision_tree.main()

# 结果
from sklearn import metrics

df_result = pd.read_csv(open('results.csv', 'r'))
y_pred = df_result['class'].values
accuracy = metrics.accuracy_score(y_test, y_pred)
print('accuracy of C4.5 tree: %.3f' % accuracy)
