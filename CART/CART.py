#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on May 27,2020
Author: silentroot95
'''
import numpy as np
import matplotlib.pyplot as plt

def LoadData(filename):
    data = []
    with open (filename,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        line_float = [float(i) for i in line]
        data.append(line_float)
    return data
    data_set.append(y_set)
    return data_set
def Plot(data1,data2):
    x1,y1 = data1
    x2,y2 = data2
    plt.subplot(121)
    plt.scatter(x1,y1)
    plt.subplot(122)
    plt.scatter(x2,y2)
    plt.show()
def Error(data):
    '''
    计算data的平方和误差
    '''
    #data的均值
    ave = sum(data)/len(data)
    error2 = [(di-ave)**2 for di in data]
    return sum(error2)
def PreError(pre,data):
    '''
    计算预测误差
    Args:
        pre:预测值
        data:实际值列表
    '''
    return sum([(di-pre)**2 for di in data])
def GenThre(data):
    '''
    计算data最佳分割点
    data是已经按x排序过的
    Return:
        分割点索引
        阈值
    '''
    x,y = data
    #分割值列表
    thre_list = [(x[i]+x[i+1])/2 for i in range(len(x)-1)]
    error_list = []
    for i in range(len(thre_list)):
        l_data = y[:i+1]
        r_data = y[i+1:]
        error = Error(l_data) + Error(r_data)
        error_list.append(error)
    min_index = error_list.index(min(error_list))
    #min_index加一是因为，分割点是比x列表长度少一
    return min_index+1,thre_list[min_index]
def CART(data):
    '''
    递归生成决策树
    '''
    tree = {}
    y = data[-1]
    #y只有两个值，不必分裂
    if len(y) < 3:
        node = sum(y)/len(y)
        #阈值
        tree['thre'] = node
        #子树
        tree['childs'] = {}
        return tree
    else:
        min_index,thre = GenThre(data)
        tree['thre'] = thre
        tree['childs'] = {}
        #小于等于
        tree['childs']['le'] = CART(data[:,:min_index])
        #大于
        tree['childs']['gt'] = CART(data[:,min_index:])
    return tree
def Test(val,tree):
    '''
    预测val对应的值
    Args:
        val:即x值
        tree:决策树
    Return:
        预测值
    '''
    thre = tree['thre']
    childs = tree['childs']
    if childs:
        if val <= thre:
            return Test(val,childs['le'])
        else:
            return Test(val,childs['gt'])
    else:
        return thre
def ErrorAll(data,tree):
    '''
    计算data列表的预测误差平方和
    '''
    x,y = data
    error = 0
    for xi,yi in zip(x,y):
        pre = Test(xi,tree)
        error += (yi-pre)**2
    return error
def Mark(tree):
    '''
    递归标记树的叶节点
    '''
    #无子树，即叶节点，则标记
    if tree['childs'] == {}:
        tree['pruned'] = 1
    else:
        tree['pruned'] = 0
    if tree['childs']:
        #标记左子树
        Mark(tree['childs']['le'])
        #标记右子树
        Mark(tree['childs']['gt'])
def prun(tree,test_data):
    '''
    决策树递归剪枝
    Args:
        tree:决策树
        test_data:测试集
    '''
    #树节点，子树
    node = tree['thre']
    child_tree = tree['childs']
    #小于等于（左）子树
    le_tree = child_tree['le']
    #大于（右）子树
    gt_tree = child_tree['gt']
    #测试集分割为两部分
    le_data = test_data[:,np.where(test_data[0] <= node)[0]]
    gt_data = test_data[:,np.where(test_data[0] > node)[0]]
    #左右子树均被标记过，即均已考察过
    if le_tree['pruned'] == 1 and gt_tree['pruned'] == 1:
        #左右节点
        le_pre = le_tree['thre']
        gt_pre = gt_tree['thre']
        #左右节点的均值
        mean = (le_pre + gt_pre)/2
        #剪枝后误差
        error_prun= PreError(mean,test_data[1])
        #剪枝前误差
        error = PreError(le_pre,le_data[1]) + PreError(gt_pre,gt_data[1])
        #剪枝后误差小于剪枝前误差，则剪枝
        if error_prun <= error:
            print('Merged')
            #将均值变为叶节点
            tree['thre'] = mean
            tree['childs'] = {}
        #标记此节点
        tree['pruned'] = 1
    if le_tree['pruned'] == 0:
        #左子树剪枝，注意参数必须是tree的子树,不能是le_tree
        prun(tree['childs']['le'],le_data)
    if gt_tree['pruned'] == 0:
        #右子树剪枝
        prun(tree['childs']['gt'],gt_data)

if __name__ == '__main__':
    train_file = r'.\data3.txt'
    test_file = r'.\data3test.txt'
    data = LoadData(train_file)
    data = np.array(data).T
    #data按x排序
    data = data[:,data[0].argsort()]
    tree = CART(data)
    Mark(tree)
    test = LoadData(test_file)
    test_data = np.array(test).T
    #测试集预测误差
    error_test = ErrorAll(test_data,tree)
    #训练集预测误差
    error_train = ErrorAll(data,tree)
    print('error_test:{:.2f}'.format(error_test))
    print('error_train:{:.2f}'.format(error_train))
    #剪枝
    prun(tree,test_data)
    #剪枝后误差
    perror_test = ErrorAll(test_data,tree)
    perror_train = ErrorAll(data,tree)
    print('perror_test:{:.2f}'.format(perror_test))
    print('perror_train:{:.2f}'.format(perror_train))
    #print(tree)
