##### k近邻算法（KNN)

KNN 是一个简单的无显示学习过程，非泛化学习的监督学习模型。在分类和回归中均有应用。

这里的无显示学习过程，可以理解为，KNN算法不是基于学习经验判断一个新的数据，而是基于比较，看新来的数据更像哪个，就把它归为哪一类。

它的原理也比较简单，近朱者赤近墨者黑，计算新的数据与已知数据的距离，取距离最小的k个已知数据，这k个已知数据中哪一类最多，就把新的数据归为哪一类。

##### 实例

构造一个基于KNN,能够识别手写数字0到9的分类器，训练数据集在trainDigits文件夹下，测试数据在testDigits文件夹下

```python
#-*- coding:utf-8 -*-
import numpy as np
import os
train_dir = r'.\trainingDigits'
test_dir = r'.\testDigits'
def txt2vec(filename):
    '''
    文本文件转化为特征向量
    '''
    data=[]
    with open(filename,'r') as f:
        lines = f.readlines()
    for line in lines:
        #过滤换行符
        line = line.strip('\n')
        for char in line:
            data.append(int(char))
    data = np.array(data)
    return data
def knn(vec,data_set,labels,k):
    '''
    vec:测试特征向量
    data_set:已知数据集
    labels:已知数据标签
    k:KNN算法参数k
    '''
    #数据集大小
    m = len(data_set)
    #距离表
    dis=[0]*m
    for i in range(m):
        #欧氏距离，2范数
        dis[i] = np.linalg.norm(vec-data_set[i])
    dis = np.array(dis)
    #取距离最小的前k个数据的索引
    index = np.argsort(dis)[:k]
    #距离最小的k个数据类标签
    klab = [labels[i] for i in index]
    #返回类标签最多的那个类
    return max(klab,key=klab.count)
def CreatData(dirname):
    '''
    基于testDigits\目录下的文件创建数据集
    '''
    labels = []
    data_set = []
    filenames = os.listdir(dirname)
    for f in filenames:
        label = int(f.split('_')[0])
        filename = dirname+'\\'+f
        data_set.append(txt2vec(filename))
        labels.append(label)
    return labels,data_set
def Test(dirname,data_set,labels,k):
    '''
    测试testDigits\目录下的数据
    '''
    tfiles = os.listdir(dirname)
    error = 0
    m = len(tfiles)
    for t in tfiles:
        filename = dirname+'\\'+t
        tlab = int(t.split('_')[0])
        vec = txt2vec(filename)
        lab = knn(vec,data_set,labels,k)
        if(lab != tlab):
            error += 1
    print("error count:%d\n" %error)
    print("error rate: %f" %(error/m))
if __name__ == '__main__':
    k=5
    labels,data_set = CreatData(train_dir)
    Test(test_dir,data_set,labels,k)
```

