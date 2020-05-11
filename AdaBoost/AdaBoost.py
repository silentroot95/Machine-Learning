#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on May 8,2020
Author: silentroot95
Github: https://github.com/silentroot95
'''
import numpy as np
train_file = r'.\AdaBoost\horseColicTraining2.txt'
test_file = r'.\AdaBoost\horseColicTest2.txt'
def LoadData(filename):
    '''
    args:
        filename:文件名
    return:
        data_set:数据集
        label_set:标签集
    '''
    data_set=[]
    label_set = []
    with open(filename,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        data_arr = [float(data) for data in line]
        #弹出每行的最后一个值，标签
        label_set.append(data_arr.pop())
        data_set.append(data_arr)
    return data_set,label_set
def BestThre(f,label_set,weights):
    '''
    计算某一特征的最佳分割值
    分割值标签只记录<=，>相反
    args:
        f:某一维特征向量
        label_set:标签集
        weights:权值向量
    return:
        best_thre:最优分割阈值
        thre_label:fi<=此阈值对应的标签(1 or -1)
    '''
    n = len(f)
    f_set = set(f)
    f_set_sorted = sorted(list(f_set))
    #可能的阈值
    thre_list = [(f_set_sorted[i]+f_set_sorted[i+1])/2 for i in range(len(f_set_sorted)-1)]
    #每个分割值分割信息
    thre_vec=[]
    for thre in thre_list:
        #<=阈值 1 or -1对应的错误率
        error_rate = []
        for lab in [1,-1]:
            #错误分类索引
            uncorrect = []
            for i in range(n):
                if (f[i]<=thre and label_set[i] != lab) or (f[i]>thre and label_set[i] == lab ):
                    uncorrect.append(i)
            error_rate.append(sum([weights[i] for i in uncorrect]))
        #1，-1最小误差
        min_error_rate = min(error_rate)
        #最小误差索引
        min_index = error_rate.index(min_error_rate)
        #最优标签
        thre_lab = [1,-1][min_index]
        #最小误差，标签，分割值 列表
        thre_info = [min_error_rate,thre_lab,thre]
        thre_vec.append(thre_info)
    errors,labs,thres = np.mat(thre_vec).T.tolist()
    #在所有的分割点中取误差最小的分割点
    min_error = min(errors)
    best_index = errors.index(min_error)
    #最佳分割点阈值
    best_thre = thres[best_index]
    #最佳分割点标签
    thre_label = labs[best_index]
    return min_error,best_thre,thre_label
def BestFeature(data_set,label_set,weights):
    '''
    寻找误差最小特征，即最优基分类器
    args:
        data_set:样本集
        label_set:标签集
        weights:权值向量
    return:
        best_feature:最优特征字典
    '''
    m,n = np.mat(data_set).shape
    #特征矩阵
    f_mat = np.mat(data_set).T.tolist()
    #所有特征分割点信息列表
    f_thre_mat = []
    #误差最小特征
    best_feature = {}
    for i in range(n):
        #每个特征的最优分割点
        f_info = list(BestThre(f_mat[i],label_set,weights))
        f_info.append(i)
        f_thre_mat.append(f_info)
    errors,thres,labs,f_index = np.mat(f_thre_mat).T.tolist()
    err = min(errors)
    best_index = errors.index(err)
    #最优特征索引
    best_feature['index'] = f_index[best_index]
    #最优特征分割点
    best_feature['thre'] = thres[best_index]
    #最优特征分割点标签
    best_feature['lab'] = labs[best_index]
    #最优特征误差
    best_feature['err'] = err
    if err == 0 :
        best_feature['alpha'] = 0
    else:
    #最优特征alpha
        best_feature['alpha'] = np.log((1-err)/err)/2
    return best_feature
def Classify(classifier,data_set):
    '''
    args:
        classifier:基分类器
        data_set:数据集
    return:
        ans:基分类器分类结果
    '''
    features = np.mat(data_set).T.tolist()
    best_f = features[int(classifier['index'])]
    thre = classifier['thre']
    lab = classifier['lab']
    alpha = classifier['alpha']
    ans = []
    for f in best_f:
        if f <= thre:
            #-alpha*lab
            ans.append(-alpha*lab)
        else:
            #-alpha*(-lab)
            ans.append(alpha*lab)
    return ans

def AdaBoost(data_set,label_set,nums = 5):
    '''
    args:
        data_set:数据集
        label_set:标签集
        nums:基分类器个数
    return:
        classifiers:基分类器列表
    '''
    #样本数量
    m = len(data_set)
    #初始化权值向量
    weights = [1/m]*m
    #基分类器列表
    classfiers = []
    for i in range(nums):
        #基分类器
        best_feature = BestFeature(data_set,label_set,weights)
        #误差为0终止循环，误差为0很难出现
        if best_feature['err'] > 0.5 or best_feature['err'] == 0:
            break
        classfiers.append(best_feature)
        #-alpha*G_m(x) 基分类器预测结果与-alpha乘积
        ans = Classify(best_feature,data_set)
        #(-alpha*lab*G(x)) 再乘真实标签
        iyg = np.multiply(label_set,ans)
        #权值向量
        weights = np.multiply(weights,np.exp(iyg))
        #权值向量归一化
        weights = weights/sum(weights)
    return classfiers
def Test(classifiers,f):
    '''
    返回单一样本类别
    args:
        classifiers:分类器列表
        f:单一样本
    return:
        ans:类别
    '''
    res = 0
    for clf in classifiers:
        f_index =int(clf['index'])
        thre = clf['thre']
        lab = clf['lab']
        alpha = clf['alpha']
        if f[f_index] <= thre:
            res += alpha*lab
        else:
            res += alpha*(-lab)
    if res > 0:
        ans = 1
    elif res < 0:
        ans = -1
    else:
        ans = 0
    return ans
def TestSet(classifiers,data_set,label_set):
    '''
    测试数据集 预测错误率
    args:
        classifiers:分类器列表
        data_set:数据集
        label_set:标签集
    return:
        error_rate:错误率
    '''
    cnt = 0
    for data,label in zip(data_set,label_set):
        if Test(classifiers,data) != label:
            cnt += 1
    error_rate = cnt/len(data_set)
    return error_rate
if __name__ == '__main__':
    data_set,label_set = LoadData(train_file)
    nums =6
    classifiers = AdaBoost(data_set,label_set,nums)
    test_data,test_label = LoadData(test_file)
    test_err_rate = TestSet(classifiers,test_data,test_label)
    train_err_rate = TestSet(classifiers,data_set,label_set)
    print('clssifiers num:{:d}'.format(len(classifiers)))
    print('train_error_rate:{:.6f}'.format(train_err_rate))
    print('test_error_rate:{:.6f}'.format(test_err_rate))
