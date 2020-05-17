#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on May 15,2020
Author: silentroot95
Github: https://github.com/silentroot95/Machine-Learning
'''
import numpy as np
import matplotlib.pyplot as plt

filename = r'.\Regression\abalone.txt'
def LoadData(filename):
    data_mat = []
    y_vec = []
    with open(filename,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        data = [float(d) for d in line]
        y_vec.append(data.pop())
        data_mat.append(data)
    return data_mat,y_vec
def Visual(data_mat,y_vec):
    '''
    数据点的散点图，仅适用于二维点
    '''
    x_vec = np.mat(data_mat).T.tolist()[1]
    plt.scatter(x_vec,y_vec)
def Regression(data_mat,y_vec):
    '''
    线性回归
    Args:
        data_mat:数据
        y_vec:结果
    return:
        w:回归系数
    '''
    data_mat = np.mat(data_mat)
    #转化成列向量
    y_vec = np.mat(y_vec).T
    XTX = data_mat.T @ data_mat
    if np.linalg.det(XTX) == 0.0:
        print('奇异矩阵')
        return
    #回归系数
    w = XTX.I @ data_mat.T @ y_vec
    return w
def RegPlot(w,data_mat):
    '''
    画出回归图像，仅适用于二维图
    Args:
        w:回归系数
        data_mat:数据自变量
    '''
    x_vec = np.mat(data_mat).T.tolist()[1]
    x_min = min(x_vec)
    x_max = max(x_vec)
    x = [x_min,x_max]
    w = w.T.tolist()[0]
    y = [w[1]*xi+w[0] for xi in x]
    plt.plot(x,y)
def Lwlr(point,data_mat,y_vec,k=1.0):
    '''
    局部加权回归
    Args:
        point:计算点
        data_mat:自变量矩阵
        y_vec:结果，因变量
        k:高斯核参数
    return:
        predict:point对应的预测值
    '''
    data_mat = np.mat(data_mat)
    y_vec = np.mat(y_vec).T
    m = data_mat.shape[0]
    #初始化权值向量
    weights = np.mat(np.eye(m))
    for i in range(m):
        diff = data_mat[i,:] - point
        #高斯核，权值向量，离计算点越近权值越大，离计算点越远，权值越小
        weights[i,i] = np.exp(diff*diff.T/(-2*k**2))
    XTX = data_mat.T*weights*data_mat
    if np.linalg.det(XTX) == 0.0:
        print('奇异矩阵')
        return
    #回归系数
    w = XTX.I*data_mat.T*weights*y_vec
    #预测值
    predict = (point*w)[0,0]
    return predict
def LwlrTest(arr,data_mat,y_vec,k=1.0):
    '''
    计算数组arr中每个点的局部加权回归预测值
    Args:
        arr:要计算的数组
        data_mat:数据特征
        y_vec:结果
        k:高斯核参数
    return:
        arr_predict:预测结果向量
    '''
    m = len(data_mat)
    arr_predict = []
    for i in range(m):
        arr_predict.append(Lwlr(arr[i],data_mat,y_vec,k))
    return arr_predict
def LwlrPlot(data_mat,y_vec,k):
    '''
    绘制局部加权回归图像
    Args:
        data_mat:数据特征
        y_vec:结果
        k:高斯核参数
    '''
    y_predict = []
    x_mat = np.mat(data_mat)
    #对x排序便于画图
    x_mat.sort(0)
    x = x_mat[:,1]
    for data in x_mat:
        #计算每个x对应的y
        y_predict.append(Lwlr(data,data_mat,y_vec,k))
    plt.plot(x,y_predict)
def Error(y_vec,y_predict):
    '''
    计算实际值与预测值的平方和误差
    Args:
        y_vec:实际值向量
        y_predict:预测值向量
    return:
        平方和误差
    '''
    y_vec = np.array(y_vec)
    y_predict = np.array(y_predict)
    return sum((y_vec-y_predict)**2)
def AbaloneTest():
    '''
    在鲍鱼预测数据集上的测试
    '''
    data_mat,y_vec = LoadData(r'.\Regression\abalone.txt')
    #预测训练集，局部加权回归三个不同的核参数
    oy01 = LwlrTest(data_mat[0:99],data_mat[0:99],y_vec[0:99],k=0.1)
    oy1 = LwlrTest(data_mat[0:99],data_mat[0:99],y_vec[0:99],k=1)
    oy10 = LwlrTest(data_mat[0:99],data_mat[0:99],y_vec[0:99],k=10)
    print('oy01 error is:',Error(y_vec[0:99],oy01))
    print('oy1 error is:',Error(y_vec[0:99],oy1))
    print('oy10 error is:',Error(y_vec[0:99],oy10))
    #预测测试集
    ny01 = LwlrTest(data_mat[100:199],data_mat[0:99],y_vec[0:99],k=0.1)
    ny1 = LwlrTest(data_mat[100:199],data_mat[0:99],y_vec[0:99],k=1)
    ny10 = LwlrTest(data_mat[100:199],data_mat[0:99],y_vec[0:99],k=1.5)
    print('ny01 error is:',Error(y_vec[100:199],ny01))
    print('ny1 error is:',Error(y_vec[100:199],ny1))
    print('ny10 error is:',Error(y_vec[100:199],ny10))
    #经典线性回归，回归系数
    w = Regression(data_mat[0:99],y_vec[0:99])
    yw = np.mat(data_mat[100:199]) * w
    #线性回归预测值
    yw = yw.T.tolist()[0]
    print('standard regression error is:',Error(y_vec[100:199],yw))
def RigidReg(data_mat,y_vec,lam = 0.2):
    '''
    岭回归
    Args:
        data_mat:自变量
        y_vec:因变量
        lam:岭回归参数lambda
    return:
        w:回归系数
    '''
    xTx = data_mat.T*data_mat
    xTx += np.eye(len(xTx))*lam
    if np.linalg.det(xTx) == 0.0:
        print('奇异矩阵')
        return
    w = xTx.I*data_mat.T*y_vec
    return w
def RigidTest(data_mat,y_vec):
    '''
    生成不同lambda下的回归系数，用于绘制岭迹图
    Args:
        data_mat:自变量
        y_vec:因变量
    return:
        w_mat:不同lambda下的回归系数
        log_lam:lambda对数向量
    '''
    data_mat = np.mat(data_mat)
    y_vec = np.mat(y_vec).T
    #y取均值
    y_mean = np.mean(y_vec,0)
    #中心化
    y_vec -= y_mean
    #x均值
    x_mean = np.mean(data_mat,0)
    #x方差
    x_var = np.var(data_mat,0)
    #x标准化
    x_mat = (data_mat - x_mean)/x_var
    nums = 40
    w_mat = np.zeros((nums,np.shape(x_mat)[1]))
    log_lam = []
    for i in range(nums):
        #计算log_lam从-20到20的回归系数
        w = RigidReg(x_mat,y_vec,np.exp(i-20))
        w_mat[i,:] = w.T
        log_lam.append(i-20)
    return w_mat,log_lam
def RigidPlot():
    '''
    绘制岭迹图
    '''
    data_mat,y_vec = LoadData(filename)
    w_mat,log_lam = RigidTest(data_mat,y_vec)
    plt.plot(log_lam,w_mat)
    plt.show()
if __name__ == '__main__':
    #data_mat,y_vec = LoadData(filename)
   # w = Regression(data_mat,y_vec)
    #Visual(data_mat,y_vec)
    #LwlrPlot(data_mat,y_vec,k=0.02)
    #plt.show()
    RigidPlot()
