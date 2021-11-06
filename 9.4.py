#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@LICENSE: Copyright(C) 2019, xiyusullos
@AUTHOR: xiyusullos
@BLOG: https://blog.aponder.top
@FILE: 9.4.py
@TIME: 2019-01-10 10:06:02
@DESC:
'''

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

if __name__ == "__main__":
    # 加载数据
    dataset = []
    fileIn = open('data/4.0.csv')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        dataset.append([float(lineArr[0]), float(lineArr[1])])
    # 设置不同的K值计算
    for k in range(2, 6):
        clf = KMeans(n_clusters=k)  # 调用KMeans算法设定k
        s = clf.fit(dataset)  # 加载数据集合
        numSamples = len(dataset)
        centroids = clf.labels_
        print(centroids, type(centroids))
        print(clf.inertia_)
        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        # 画出样例点
        for i in range(numSamples):
            plt.plot(dataset[i][0], dataset[i][1], mark[clf.labels_[i]])
        mark = ['*r', '*b', '*g', '*k', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 画出质点
        centroids = clf.cluster_centers_
        for i in range(k):
            plt.title("k:" + str(k))
            plt.xlabel('density')
            plt.ylabel('ratio_sugar')
            plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize=12)
        plt.show()
