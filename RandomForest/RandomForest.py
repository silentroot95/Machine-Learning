#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on May 12,2020
Author: silentroot95
Github: https://github.com/silentroot95/Machine-Learning
'''
import random
import numpy as np

filename = r'.\RandomForest\sonar-all-data.txt'
def LoadData(filename):
    data_set = []
    label_set = []
    with open(filename,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(',')
        label_set.append(line.pop())
        data = [float(i) for i in line]
        data_set.append(data)
    return data_set,label_set
def Bagging(data_set,label_set,ratio):
    '''
    Bagging 有放回采样
    args:
        data_set:数据集
        label_set:标签集
        ratio:采样比例
    return:
        data:采样数据
        label:采样标签
    '''
    #采样个数
    sample_len = round(len(label_set)*ratio)
    data = []
    label = []
    while len(label) < sample_len:
        #随机索引
        _index = random.randrange(len(label_set))
        data.append(data_set[_index])
        label.append(label_set[_index])
    return data,label
def Gini(label_list):
    '''
    计算基尼系数
    args:
        label_list:标签集
    return:
        gini:基尼系数
    '''
    p = label_list.count(label_list[0])/len(label_list)
    gini = 2*p*(1-p)
    return gini
def ConGini(thre,f,label_list):
    '''
    条件基尼系数
    args:
        thre:分割阈值
        f:单一特征
        label_list:标签
    return:
        con_gini:条件基尼系数
    '''
    f_len = len(f)
    #特征值小于等于阈值的标签
    below_label = [label_list[i] for i in range(f_len) if f[i] <= thre]
    #特征值大于阈值的标签
    up_label = [label_list[i] for i in range(f_len) if f[i] > thre]
    #小于等于 占比
    freq = len(below_label)/f_len
    #条件基尼系数
    con_gini = freq*Gini(below_label)+(1-freq)*Gini(up_label)
    return con_gini
def BestSplit(f,label_set):
    '''
    根据条件基尼系数找出某特征的最佳分割点
    args:
        f:单一特征
        label_set:标签集
    return:
        min_gini:此特征最佳分割点条件基尼系数
        thre:最佳分割阈值
    '''
    #特征集合
    f_set = set(f)
    #对特征值排序
    f_set_sorted = sorted(list(f_set))
    #可能的阈值
    thre_set = [(f_set_sorted[i]+f_set_sorted[i+1])/2 for i in range(len(f_set_sorted)-1)]
    con_gini = []
    for thre in thre_set:
        con_gini.append(ConGini(thre,f,label_set))
    #print(con_gini)
    min_gini = min(con_gini)
    min_index = con_gini.index(min_gini)
    return [min_gini,thre_set[min_index]]

def SplitData(thre,bf,datas,labels):
    '''
    根据阈值与特征索引分割数据集
    args:
        thre:阈值
        bf:特征索引
        datas:数据集
        labels:标签集
    return:
        le:小于等于阈值的数据
        gt:大于阈值的数据
    '''
    #第一个列表存储数据，第二个列表存储标签
    le = [[],[]]
    gt = [[],[]]
    for di,lab in zip(datas,labels):
        #这种分片是被逼无奈，直接调用pop方法删除总是出现莫名其妙的bug（有些数据删除多个值，就是莫名其妙的就少了一个值）
        dd = di[:bf]+di[bf+1:]
        #需要与阈值比较的特征值
        ft = di[bf]
        if ft <= thre:
            le[0].append(dd)
            le[1].append(lab)
        else:
            gt[0].append(dd)
            gt[1].append(lab)
    return le,gt

def CART(datas,labels,k_feature):
    '''
    递归生成CART决策树
    args:
        datas:数据
        labels:标签
        k_feature:随机选取的特征个数
    return:
        tree:CART决策树
    '''
    tree = {}
    #只有一种标签时不需要分裂
    if len(set(labels))== 1:
        tree[labels[0]] = {}
        return tree
    else:
        #数据转置 每行都是一个特征
        features = np.mat(datas).T.tolist()
        #随机选取k_feature个特征
        random_k_index = [random.randrange(len(features)) for i in range(k_feature)]
        #k_feature个特征中最佳分割点信息
        feature_splits = [BestSplit(features[k],labels) for k in random_k_index]
        gini,thre = np.mat(feature_splits).T.tolist()
        #k_feature特征中条件gini系数最小的特征
        best_index = gini.index(min(gini))
        #此特征索引
        best_feature = random_k_index[best_index]
        #最佳分割点阈值
        f_thre = thre[best_index]
        #创建子树，空字典
        tree[best_feature] = {}
        #记录阈值
        tree[best_feature]['thre'] = f_thre
        #分割数据
        le,gt = SplitData(f_thre,best_feature,datas,labels)
        #记录数据
        tree[best_feature]['le'] = le
        tree[best_feature]['gt'] = gt
        for key,data in tree[best_feature].items():
            #排除thre键，对应的值
            if isinstance(data,list):
                #递归分裂
                tree[best_feature][key] = CART(data[0],data[1],k_feature)
    return tree
def RandomForest(data_set,label_set,tree_num = 3,k_feature=3,ratio = 0.7):
    '''
    生成随机森林
    args:
        data_set:数据集
        label_set:标签集
        tree_num:决策树个数
        k_feature:随机选取特征个数
        ratio:采样比例
    return:
        forest:随机森林
    '''
    forest = []
    for i in range(tree_num):
        #采样
        datas,labels = Bagging(data_set,label_set,ratio)
        #生成决策树
        tree = CART(datas,labels,k_feature)
        forest.append(tree)
    return forest
def TestTree(new_data,tree):
    '''
    测试单一决策树
    args:
        new_data:新的数据
        tree:决策树
    return:预测结果
    '''
    for node,child_tree in tree.items():
        if child_tree and new_data[node] <= child_tree['thre']:
            #小于等于阈值的子树
            return TestTree(new_data,child_tree['le'])
        elif child_tree and new_data[node] > child_tree['thre']:
            #大于阈值的子树
            return TestTree(new_data,child_tree['gt'])
        else:
            return node
def TestForest(new_data,forest):
    '''
    测试随机森林
    '''
    ans = []
    for tree in forest:
        ans.append(TestTree(new_data,tree))
    return max(ans,key=ans.count)
def TotalTest(data_set,label_set,forest):
    '''
    测试所有数据
    '''
    error = 0
    for data,lab in zip(data_set,label_set):
        if lab != TestForest(data,forest):
            error += 1
    return error/len(data_set)

if __name__ == '__main__':
    data_set,label_set = LoadData(filename)
    forest = RandomForest(data_set,label_set,tree_num = 50,k_feature = 8,ratio = 0.5)
    error_rate = TotalTest(data_set,label_set,forest)
    print('error rate:{:.6f}'.format(error_rate))
