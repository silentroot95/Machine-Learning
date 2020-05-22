#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import random
filename = r'.\Logistic\TestSet.txt'
def LoadData(filename):
    data_set = []
    label_set = []
    with open(filename,'r') as f:
        lines = f.readlines()
    for line in lines:
        data = line.strip('\n').split()
        data_set.append([1.0,float(data[0]),float(data[1])])
        label_set.append(int(data[-1]))
    return data_set,label_set
def Sigmoid(x):
    return 1/(1+np.exp(-x))
def GradDecent(data_set,label_set):
    #data_set数据集，Nx3矩阵
    data_set = np.mat(data_set)
    #label_set标签集 Nx1矩阵
    label_set = np.mat(label_set).T
    m,n = data_set.shape
    #权向量初始化3x1
    weights = np.ones((n,1))
    #向量误差
    ew = 1
    alpha = 0.0001
    while(ew>0.0001):
        f = Sigmoid(data_set*weights)
        #y-t  预测标签值-实际标签值
        error = f-label_set
        #新的权向量
        tmp = alpha*np.matmul(data_set.T,error)
        new_weights = weights - alpha*np.matmul(data_set.T,error)
        #print(new_weights)
        ew =np.linalg.norm(new_weights-weights)
        weights = new_weights
    return weights
def StoGradDecent(data_set,label_set,nums=500):
    '''
    随机梯度下降
    '''
    data_set = np.mat(data_set)
    m,n =data_set.shape
    alpha = 0.001
    #初始化权值向量
    weights = np.ones((n,1))
    #循环遍数
    for j in range(nums):
        #data_index = list(range(m))
        for i in range(m):
        #    rand_index = int(random.uniform(0,len(data_index)))
            f = Sigmoid(np.dot(data_set[i],weights))
            error = float(f - label_set[i])
            weights = weights - alpha*error*data_set[i].T
            #del(data_index[rand_index])
    return weights
def Visualizition(weights,data_set,label_set):
    '''
    可视化
    '''
    class1 = []
    class2 = []
    data_set = np.array(data_set)
    weights = weights.T.tolist()[0]
    for i in range(len(label_set)):
        if label_set[i] == 1:
            class1.append(data_set[i])
        else:
            class2.append(data_set[i])
    x1,y1 = np.array(class1).T[1:,:]
    x2,y2 = np.array(class2).T[1:,:]
    plt.plot(x1,y1,'+')
    plt.plot(x2,y2,'x')
    x_min = min(data_set[:,1])
    x_max = max(data_set[:,1])
    vec_x= [x_min,x_max]
    vec_y = [-(weights[0]+weights[1]*xi)/weights[2] for xi in vec_x]
    plt.plot(vec_x,vec_y)
    plt.show()

if __name__ == '__main__':
    data_set,label_set= LoadData(filename)
    weights = StoGradDecent(data_set,label_set,nums=10)
    #print(weights)
    Visualizition(weights,data_set,label_set)
