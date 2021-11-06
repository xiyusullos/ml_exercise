#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@LICENSE: Copyright(C) 2019, xiyusullos
@AUTHOR: xiyusullos
@BLOG: https://blog.xy-jit.cc
@FILE: 4.3.py
@TIME: 2019-01-06 17:13:41
@DESC:
'''

import math

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


print(mpl.matplotlib_fname())

class TreeNode:
    '''
    树的节点类
    data:树的数组结构的一项，4值
    height:节点的高
    '''

    def __init__(self, data, height):
        self.father = data[0]  # 父节点
        self.children = []  # 子节点列表
        self.data = data[1]  # 节点标签
        self.height = height
        self.pos = 0  # 节点计算时最终位置，计算时只保存相对位置
        self.offset = 0  # 节点最终位置与初始位置的相对值
        self.data_to_father = data[2]  # 链接父节点的属性值
        # 如果有阈值，则加入阈值
        if type(data[3]) != list:
            self.data_to_father = self.data_to_father + str(data[3]);


class DtreePlot:
    def __init__(self, link, minspace, r, lh):
        '''
        树的绘制类
        link:树的数组结构
        minspace:节点间的距离
        r:节点的绘制半径
        lh:层高
        '''

        s = len(link)
        # 所有节点的列表，第一项为根节点
        treenodelist = []
        # 节点的层次结构
        treelevel = []

        # 处理树的数组结构
        for i in range(0, s):
            # 根节点的index与其父节点的index相同
            if link[i][0] == i:
                treenodelist.append(TreeNode(link[i], 0))
            else:
                treenodelist.append(TreeNode(link[i], treenodelist[link[i][0]].height + 1))
                treenodelist[link[i][0]].children.append(treenodelist[i]);
                treenodelist[i].father = treenodelist[link[i][0]];
            # 如果有新一层的节点则新建一层
            if len(treelevel) == treenodelist[i].height:
                treelevel.append([])
            treelevel[treenodelist[i].height].append(treenodelist[i])

        # 控制绘制图像的坐标轴
        self.right = 0
        self.left = 0
        # 反转层次，从底往上画
        treelevel.reverse()
        # 计算每个节点的位置
        self.__calpos(treelevel, minspace)
        # 绘制树形
        self.__drawtree(treenodelist[0], r, lh, 0)
        plt.xlim(xmin=self.left, xmax=self.right + minspace)
        plt.ylim(ymin=len(treelevel) * lh + lh / 2, ymax=lh / 2)
        plt.show()

    def __calonebyone(self, nodes, l, r, start, minspace):

        '''
        逐一绘制计算每个节点的位置
        nodes:节点集合
        l,r:左右区间
        start:当前层的初始绘制位置
        minspace:使用的最小间距
        '''

        for i in range(l, r):
            nodes[i].pos = max(nodes[i].pos, start)
            start = nodes[i].pos + minspace;
        return start;

    def __calpos(self, treelevel, minspace):
        '''
        计算每个节点的位置与相对偏移
        treelevel：树的层次结构
        minspace:使用的最小间距
        '''

        # 按层次画
        for nodes in treelevel:
            # 记录非叶节点
            noleaf = []
            num = 0;
            for node in nodes:
                if len(node.children) > 0:
                    noleaf.append(num)
                    node.pos = (node.children[0].pos + node.children[-1].pos) / 2
                num = num + 1

            start = minspace

            # 如果全是非叶节点，直接绘制
            if (len(noleaf)) == 0:
                self.__calonebyone(nodes, 0, len(nodes), 0, minspace)
            else:
                start = nodes[noleaf[0]].pos - noleaf[0] * minspace
                self.left = min(self.left, start - minspace)
                start = self.__calonebyone(nodes, 0, noleaf[0], start, minspace)
                for i in range(0, len(noleaf)):
                    nodes[noleaf[i]].offset = max(nodes[noleaf[i]].pos, start) - nodes[noleaf[i]].pos
                    nodes[noleaf[i]].pos = max(nodes[noleaf[i]].pos, start)

                    if (i < len(noleaf) - 1):
                        # 计算两个非叶节点中间的间隔，如果足够大就均匀绘制
                        dis = (nodes[noleaf[i + 1]].pos - nodes[noleaf[i]].pos) / (noleaf[i + 1] - noleaf[i])
                        start = nodes[noleaf[i]].pos + max(minspace, dis)
                        start = self.__calonebyone(nodes, noleaf[i] + 1, noleaf[i + 1], start, max(minspace, dis))
                    else:
                        start = nodes[noleaf[i]].pos + minspace
                        start = self.__calonebyone(nodes, noleaf[i] + 1, len(nodes), start, minspace)

    def __drawtree(self, treenode, r, lh, curoffset):
        '''
        采用先根遍历绘制树
        treenode:当前遍历的节点
        r:半径
        lh:层高
        curoffset:每层节点的累计偏移
        '''
        # 加上当前的累计偏差得到最终位置
        treenode.pos = treenode.pos + curoffset

        if (treenode.pos > self.right):
            self.right = treenode.pos

        # 如果是叶节点则画圈，非叶节点画方框
        if (len(treenode.children) > 0):
            drawrect(treenode.pos, (treenode.height + 1) * lh, r)
            plt.text(treenode.pos, (treenode.height + 1) * lh, treenode.data + '=?', color=(0, 0, 1), ha='center',
                     va='center')
        else:
            drawcircle(treenode.pos, (treenode.height + 1) * lh, r)
            plt.text(treenode.pos, (treenode.height + 1) * lh, treenode.data, color=(1, 0, 0), ha='center', va='center')

        num = 0;
        # 先根遍历
        for node in treenode.children:
            self.__drawtree(node, r, lh, curoffset + treenode.offset)

            # 绘制父节点到子节点的连线
            num = num + 1

            px = (treenode.pos - r) + 2 * r * num / (len(treenode.children) + 1)
            py = (treenode.height + 1) * lh - r - 0.02

            # 按到圆到方框分开画
            if (len(node.children) > 0):
                px1 = node.pos
                py1 = (node.height + 1) * lh + r
                off = np.array([px - px1, py - py1])
                off = off * r / np.linalg.norm(off)

            else:
                off = np.array([px - node.pos, -lh + 1])
                off = off * r / np.linalg.norm(off)
                px1 = node.pos + off[0]
                py1 = (node.height + 1) * lh + off[1]

            # 计算父节点与子节点连线的方向与角度
            plt.plot([px, px1], [py, py1], color=(0, 0, 0))
            pmx = (px1 + px) / 2 - (1 - 2 * (px < px1)) * 0.4
            pmy = (py1 + py) / 2 + 0.4
            arc = np.arctan(off[1] / (off[0] + 0.0000001))
            # 绘制文本以及旋转
            plt.text(pmx, pmy, node.data_to_father, color=(1, 0, 1), ha='center', va='center',
                     rotation=arc / np.pi * 180)


def drawcircle(x, y, r):
    '''
    画圆
    '''

    theta = np.arange(0, 2 * np.pi, 2 * np.pi / 1000)
    theta = np.append(theta, [2 * np.pi])
    x1 = []
    y1 = []
    for tha in theta:
        x1.append(x + r * np.cos(tha))
        y1.append(y + r * np.sin(tha))
    plt.plot(x1, y1, color=(0, 0, 0))


def drawrect(x, y, r):
    '''
    画矩形
    '''

    x1 = [x - r, x + r, x + r, x - r, x - r]
    y1 = [y - r, y - r, y + r, y + r, y - r]
    plt.plot(x1, y1, color=(0, 0, 0))


class Property:
    '''
    属性类
    '''

    def __init__(self, idnum, attribute):
        self.is_continuity = False  # 连续型属性标记
        self.attribute = attribute  # 属性标签
        self.subattributes = []  # 属性子标签
        self.id = idnum  # 属性排在输入文本的第几位
        self.index = {}  # 属性子标签的索引值


class Dtree():
    '''
    决策树生成类
    '''

    def __init__(self, filename, haveID, property_set):
        '''
        构造函数
        filename:输入文件名
        haveID:输入是否带序号
        property_set：为空则计算全部属性，否则记录set中的属性
        '''

        self.data = []
        self.data_property = []
        # 读入数据
        self.__dataread(filename, haveID)
        # 判断选择的属性集合
        if len(property_set) > 0:
            tmp_data_property = []
            for i in property_set:
                tmp_data_property.append(self.data_property[i])
            tmp_data_property.append(self.data_property[-1])
        else:
            tmp_data_property = self.data_property

        # 决策树树形数组结构
        self.treelink = []

        # 决策树主递归
        self.__TreeGenerate(range(0, len(self.data[-1])), tmp_data_property, 0, [], [])

        # 决策树绘制
        DtreePlot(self.treelink, 6, 1, -6)

    def __TreeGenerate(self, data_set, property_set, father, attribute, threshold):
        '''
        决策树主递归
        data_set:当前样本集合
        property_set：当前熟悉集合
        father:父节点索引值
        attribute:父节点连接当前节点的子属性值
        threshold:如果是连续参数就是阈值，否则为空
        '''
        # 新增一个节点
        self.treelink.append([])
        # 新节点的位置
        curnode = len(self.treelink) - 1
        # 记录新节点的父亲节点
        self.treelink[curnode].append(father)

        # 结束条件1：所有样本同一分类
        current_data_class = self.__count(data_set, property_set[-1])
        if (len(current_data_class) == 1):
            self.treelink[curnode].append(self.data[-1][data_set[0]])
            self.treelink[curnode].append(attribute)
            self.treelink[curnode].append(threshold)
            return

        # 结束条件2：所有样本相同属性，选择分类数多的一类作为分类
        if all(len(self.__count(data_set, property_set[i])) == 1 for i in range(0, len(property_set) - 1)):
            max_count = -1;
            for dataclass in property_set[-1].subattributes:
                if current_data_class[dataclass] > max_count:
                    max_attribute = dataclass
                    max_count = current_data_class[dataclass]
            self.treelink[curnode].append(max_attribute)
            self.treelink[curnode].append(attribute)
            self.treelink[curnode].append(threshold)
            return

        # 信息增益选择最优属性与阈值
        prop, threshold = self.__entropy_paraselect(data_set, property_set)

        # 记录当前节点的最优属性标签与父节点连接当前节点的子属性值
        self.treelink[curnode].append(prop.attribute)
        self.treelink[curnode].append(attribute)

        # 从属性集合中移除当前属性
        property_set.remove(prop)

        # 判断是否是连续属性
        if (prop.is_continuity):
            # 连续属性分为2子属性，大于和小于
            tmp_data_set = [[], []]
            for i in data_set:
                tmp_data_set[self.data[prop.id][i] > threshold].append(i)
            for i in [0, 1]:
                self.__TreeGenerate(tmp_data_set[i], property_set[:], curnode, prop.subattributes[i], threshold)
        else:
            # 离散属性有多子属性
            tmp_data_set = [[] for i in range(0, len(prop.subattributes))]
            for i in data_set:
                tmp_data_set[prop.index[self.data[prop.id][i]]].append(i)

            for i in range(0, len(prop.subattributes)):
                if len(tmp_data_set[i]) > 0:
                    self.__TreeGenerate(tmp_data_set[i], property_set[:], curnode, prop.subattributes[i], [])
                else:
                    # 如果某一个子属性不存没有对应的样本，则选择父节点分类更多的一项作为分类
                    self.treelink.append([])
                    max_count = -1;
                    tnode = len(self.treelink) - 1
                    for dataclass in property_set[-1].subattributes:
                        if current_data_class[dataclass] > max_count:
                            max_attribute = dataclass
                            max_count = current_data_class[dataclass]
                    self.treelink[tnode].append(curnode)
                    self.treelink[tnode].append(max_attribute)
                    self.treelink[tnode].append(prop.subattributes[i])
                    self.treelink[tnode].append(threshold)

                    # 为没有4个值得节点用空列表补齐4个值
        for i in range(len(self.treelink[curnode]), 4):
            self.treelink[curnode].append([])

    def __entropy_paraselect(self, data_set, property_set):
        '''
        信息增益算则最佳属性
        data_set:当前样本集合
        property_set:当前属性集合
        '''

        # 分离散和连续型分别计算信息增益，选择最大的一个
        max_ent = -10000
        for i in range(0, len(property_set) - 1):
            prop_id = property_set[i].id
            if (property_set[i].is_continuity):
                tmax_ent = -10000
                xlist = self.data[prop_id][:]
                xlist.sort()
                # 连续型求出相邻大小值的平局值作为待选的最佳阈值
                for j in range(0, len(xlist) - 1):
                    xlist[j] = (xlist[j + 1] + xlist[j]) / 2
                for j in range(0, len(xlist) - 1):
                    if (i > 0 and xlist[j] == xlist[j - 1]):
                        continue
                    cur_ent = 0
                    nums = [[0, 0], [0, 0]]
                    for k in data_set:
                        nums[self.data[prop_id][k] > xlist[j]][property_set[-1].index[self.data[-1][k]]] += 1
                    for k in [0, 1]:
                        subattribute_sum = nums[k][0] + nums[k][1]
                        if (subattribute_sum > 0):
                            p = nums[k][0] / subattribute_sum
                            cur_ent += (p * math.log(p + 0.00001, 2) + (1 - p) * math.log(1 - p + 0.00001,
                                                                                          2)) * subattribute_sum / len(
                                data_set)
                    if (cur_ent > tmax_ent):
                        tmax_ent = cur_ent
                        tmp_threshold = xlist[j]
                if (tmax_ent > max_ent):
                    max_ent = tmax_ent;
                    bestprop = property_set[i];
                    best_threshold = tmp_threshold;
            else:
                # 直接统计并计算
                cur_ent = 0
                nums = [[0, 0] for i in range(0, len(property_set[i].subattributes))]
                for j in data_set:
                    nums[property_set[i].index[self.data[prop_id][j]]][property_set[-1].index[self.data[-1][j]]] += 1
                for j in range(0, len(property_set[i].subattributes)):
                    subattribute_sum = nums[j][0] + nums[j][1]
                    if (subattribute_sum > 0):
                        p = nums[j][0] / subattribute_sum
                        cur_ent += (p * math.log(p + 0.00001, 2) + (1 - p) * math.log(1 - p + 0.00001,
                                                                                      2)) * subattribute_sum / len(
                            data_set)
                if (cur_ent > max_ent):
                    max_ent = cur_ent;
                    bestprop = property_set[i];
                    best_threshold = [];
        return bestprop, best_threshold

    def __count(self, data_set, prop):

        '''
        计算当前样本在某个属性下的分类情况
        '''
        out = {}

        rowdata = self.data[prop.id]
        for i in data_set:
            if rowdata[i] in out:
                out[rowdata[i]] += 1
            else:
                out[rowdata[i]] = 1;
        return out

    def __dataread(self, filename, haveID):
        '''
        输入数据处理
        '''
        file = open(filename, 'r')
        linelen = 0
        first = 1
        while 1:
            # 按行读
            line = file.readline()

            if not line:
                break

            line = line.strip('\n')
            rowdata = line.split(',')
            # 如果有编号就去掉第一列
            if haveID:
                del rowdata[0]

            if (linelen == 0):
                # 处理第一行，初始化属性类对象，记录属性的标签
                for i in range(0, len(rowdata)):
                    self.data.append([])
                    self.data_property.append(Property(i, rowdata[i]))
                    self.data_property[i].attribute = rowdata[i]
                linelen = len(rowdata)
            elif (linelen == len(rowdata)):
                if (first == 1):
                    # 处理第二行，记录属性是否是连续型和子属性
                    for i in range(0, len(rowdata)):
                        if (isnumeric(rowdata[i])):
                            self.data_property[i].is_continuity = True
                            self.data[i].append(float(rowdata[i]))
                            self.data_property[i].subattributes.append("小于")
                            self.data_property[i].index["小于"] = 0
                            self.data_property[i].subattributes.append("大于")
                            self.data_property[i].index["大于"] = 1
                        else:
                            self.data[i].append(rowdata[i])
                else:
                    # 处理后面行，记录子属性
                    for i in range(0, len(rowdata)):
                        if (self.data_property[i].is_continuity):
                            self.data[i].append(float(rowdata[i]))
                        else:
                            self.data[i].append(rowdata[i])
                            if rowdata[i] not in self.data_property[i].subattributes:
                                self.data_property[i].subattributes.append(rowdata[i])
                                self.data_property[i].index[rowdata[i]] = len(self.data_property[i].subattributes) - 1
                first = 0
            else:
                continue


def isnumeric(s):
    '''
    判断是否是数字
    '''
    return all(c in "0123456789.-" for c in s)


if __name__ == '__main__':
    link = Dtree('data/table_4.2.csv', True, range(6))
