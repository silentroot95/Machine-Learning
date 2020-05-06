# -*- coding:utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
filename = r'.\SVM\TestSet.txt'
def LoadData(filename):
    '''
    读取数据
    data:数据集
    labels:类标签
    '''
    data = []
    labels = []
    with open(filename,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        data.append([float(line[0]),float(line[1])])
        labels.append(float(line[2]))
    for i in range(len(labels)):
        if labels[i] == 0:
            labels[i] = -1
    return data,labels

def SMO(data,labels,C,tol,iter_max):
    '''
    data:数据集
    labels:类标签
    C:惩罚项
    tol:容错率（软间隔）
    iter_max:最大循环数
    return:b,alpha
    '''
    data = np.mat(data)
    m,n = data.shape
    #初始化alpha向量为0
    alpha = np.zeros(m)
    #初始化偏移b为0
    b = 0
    #初始化循环数为0
    iter_num = 0
    #alphai 与 alphai_old的误差绝对值小于epsilon认为收敛
    epsilon = 0.0001
    while(iter_num < iter_max):
        #每次改变两个alpha,改变的alpha对
        alpha_changed = 0
        for i in range(m):
            #预测值
            fxi = float(np.multiply(alpha,labels).T*(data*data[i,:].T))+b
            #预测值与实际值误差
            Ei = fxi-labels[i]
            #print(fxi,Ei,Ei*labels[i])
            '''
            KKT条件
            alpha=0，边界之外
            alpha=C，边界上
            0<alpha<C，软间隔之内
            '''
            #这里的判断条件是选择违反KKT条件的alpha值进行优化，误差的绝对值大于tol是需要优化的对象，绝对值展开为两种情况
            #fxi的值与alphai相关
            #对于yi*fxi-1 <-tol，使yi*fxi-1>= -tol，需要增大alpha，当alpha<C时才有增大的空间
            #对于yi*fxi-1 >tol，使yi*fxi-1<= tol，需要减小alpha，当alpha>0时采用减小的空间
            if((labels[i]*Ei < -tol and alpha[i] < C) or (labels[i]*Ei > tol and alpha[i] >0)):
                #随机选择一个不等于i的j
                j = RandSelect(i,m)
                #fxj
                fxj = float(np.multiply(alpha,labels).T*(data*data[j,:].T)) + b
                Ej = fxj - labels[j]
                #保存alpha_old
                alphai_old = alpha[i]
                alphaj_old = alpha[j]
                #边界约束
                if labels[i] != labels[j]:
                    L = max(0,alpha[j]-alpha[i])
                    H = min(C,C+alpha[j]-alpha[i])
                else:
                    L = max(0,alpha[j]+alpha[i]-C)
                    H = min(C,alpha[j]+alpha[i])
                #L==H没有优化空间
                if L == H:
                    continue
                eta = 2*data[i]*data[j].T-data[i]*data[i].T-data[j]*data[j].T
                if eta >= 0:
                    continue
                alpha[j] -= labels[j]*(Ei-Ej)/eta
                #根据边界约束裁剪alphaj
                if alpha[j] > H:
                    alpha[j] = H
                elif alpha[j] < L:
                    alpha[j] = L
                #alphaj认为收敛
                if(abs(alpha[j]-alphaj_old) < epsilon):
                    continue
                #更新alphai
                alpha[i] += labels[i]*labels[j]*(alphaj_old - alpha[j])
                #更新偏移值b
                b1 = b - Ei - labels[i]*(alpha[i]-alphai_old)*data[i,:]*data[i,:].T-labels[j]*(alpha[j]-alphaj_old)*data[i,:]*data[j,:].T
                b2 = b - Ej - labels[i]*(alpha[i]-alphai_old)*data[i,:]*data[j,:].T-labels[j]*(alpha[j]-alphaj_old)*data[j,:]*data[j,:].T
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2
                alpha_changed += 1
        #此时alpha的每个值都不再变化了，然后看外层循环（第一变量）发生变化后的情况
        if(alpha_changed == 0):
            iter_num += 1
        else:
            iter_num =0
    return b,alpha
def RandSelect(i,m):
    '''
    随机选择0，m的整数j且j!=i
    '''
    j = random.randint(0,m-1)
    if i != j:
        return j
    else:
        return RandSelect(i,m)
def Visualize(data,labels,alpha,b):
    '''
    结果可视化
    '''
    data = np.mat(data)
    x,y = data.T
    #x = x.tolist()[0]
    xl = [np.min(x,axis=1)[0,0],np.max(x,axis=1)[0,0]]
    #x范围
    #w向量
    w = np.multiply(alpha,labels)*data
    w = w.tolist()[0]
    #分割平面fx
    fx = [(w[0]*xi+b[0,0])/(-w[1]) for xi in xl]
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    for i in range(len(labels)):
        if labels[i] == 1:
            x1.append(x[0,i])
            y1.append(y[0,i])
        else:
            x2.append(x[0,i])
            y2.append(y[0,i])
    plt.plot(x1,y1,'x')
    plt.plot(x2,y2,'+')
    plt.plot(xl,fx)
    plt.show()
if __name__ == '__main__':
    data,labels = LoadData(filename)
    b,alpha = SMO(data,labels,1,0.2,50)
    Visualize(data,labels,alpha,b)
