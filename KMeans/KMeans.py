#-*- coding:utf-8 -*-
# use Python3
# author : 魏文彬
# time : 2020/3/29
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif']=['SimHei']      #这两行解决图例中文乱码问题
plt.rcParams['axes.unicode_minus']=False

root = r'.\聚类分析-课后作业\signal'            #信号文件路径 
test = 'E43-4-u10-123(3)_1_96_982228464.txt'    #测试文件
testfile = root + '\\' + test                   #拼接文件名
def ExtractFeature(filename):                   #提取文件数据特征
    data = []
    lmax = []
    with open(filename,'r') as fp:
        lines = fp.readlines()[12:]             #第12行开始读取数据
    for line in lines:
        num = line.split(',')
        nums = [float(num[0]),float(num[1])]
        data.append(nums)
    data = np.array(data).T
    time = data[0]
    zero_index = np.where(time == 0)[0][0]      #时间0索引，其实在这个例子中是一个定值
    volte = data[1][zero_index:]                #时间0开始 电压
    step = int(len(volte)/4)                    #电压分为四段，每段取最大值放到lmax列表中
    for i in range(4):
        vi = volte[i*step:(i+1)*step]
        vmax = max(vi)
        lmax.append(vmax)
    mean2 = (sum(lmax)-max(lmax)-min(lmax))/(2*max(lmax))   #特征值：去除最大值，最小值后的平均值 再除以最大值归一化
    if (mean2>=0.25 and mean2<0.75):                         #特征处理，使两类特征两极分化
        mean2 +=0.25                                        #对于边界处的特征，人为地使它偏向某一类
    else:
        mean2 -=0.25
    f = [mean2,1-mean2]                                     #特征向量
    return f
def WalkDir(dirname):                                       #遍历文件夹下的所有文件
    features = []
    names = os.listdir(dirname)
    for name in names:
        filename = dirname + '\\' +name
        feature = ExtractFeature(filename)
        features.append(feature)
    return features                                         #特征矩阵，实际是个列表
def Dis(la,lb):                                             #向量的2范数
    vec = np.array(la)-np.array(lb)
    dis =np.sqrt(np.dot(vec,vec))
    return dis
def KMeans(features):                                       #Kmeans聚类
    c1 = features[13]
    c2 = features[7]
    error = 4
    while(error > 0.001):
        c1s =[]
        c2s =[]
        for f in features:
            dis1 = Dis(f,c1)
            dis2 = Dis(f,c2)
            if(dis1 < dis2):
                c1s.append(f)
            else:
                c2s.append(f)
        c1n = (np.sum(np.array(c1s),axis=0)/len(c1s)).tolist()      #ndarray对象转换为列表
        c2n = (np.sum(np.array(c2s),axis=0)/len(c2s)).tolist()
        error = max(Dis(c1n,c1),Dis(c2n,c2))
        c1 = c1n
        c2 = c2n
    return  c1n,c2n,c1s,c2s                                         #返回类中心，以及最后各类的特征向量
def Test(filename,c1,c2):                                           #单一信号的测试
    test_f = ExtractFeature(filename)
    dis1 = Dis(test_f,c1)
    dis2 = Dis(test_f,c2)
    if(c1[0]>c2[0]):
        class1 = '连续'
        class2 = '突发'
    else:
        class1 = '突发'
        class2 = '连续'
    if(dis1<dis2):
        return class1
    else:
        return class2                                               #返回单一信号所属信号类别
def plot_sig(filename):                                             #单一信号数据可视化
    data=[]
    with open(filename,'r') as fp:
        lines = fp.readlines()[12:]
    for line in lines:
        num = line.split(',')
        nums = [float(num[0]),float(num[1])]
        data.append(nums)
    data = np.array(data).T
    time = data[0]
    volte = data[1]
    plt.plot(time,volte)
    plt.show()
def plot(c1,c2,c1s,c2s):                #最终分类结果可视化
    if(c1[0]>c2[0]):
        label_c1 = '连续'
        label_c2 = '突发'
    else:
        label_c1 = '突发'
        label_c2 = '连续'
    plt.plot(c1[0:1],c1[1:],'+',markersize = 13,color = 'blue',label ='center1')
    plt.plot(c2[0:1],c2[1:],'o',markersize = 13,color = 'red',label = 'center2')
    c1s = np.array(c1s).T
    c2s = np.array(c2s).T
    c1x = c1s[0]
    c1y = c1s[1]
    c2x = c2s[0]
    c2y = c2s[1]
    plt.plot(c1x,c1y,'+',color = 'blue',label = label_c1)
    plt.plot(c2x,c2y,'.',color = 'red',label = label_c2)
    plt.legend()
    plt.show()

#c1 = [-0.10067467542098962, 1.1006746754209893]    最终计算得到的类1中心
#c2 = [0.7410491951682958, 0.25895080483170424]     最终计算得到的类2中心
if __name__ == '__main__':
    features = WalkDir(root)
    c1,c2,c1s,c2s = KMeans(features)
    plot(c1,c2,c1s,c2s)
    tc = Test(testfile,c1,c2)
    print(tc)
    plot_sig(testfile)
