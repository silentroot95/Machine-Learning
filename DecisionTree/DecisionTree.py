#-*- coding:utf-8 -*-
import numpy as np
import math

def CARTData():
    data_set = []
    x_set = [1,2,3,4,5,6,7,8,9,10]
    y_set = [4.5,4.75,4.91,5.34,5.8,7.05,7.9,8.23,8.7,9]
    data_set.append(x_set)
    data_set.append(y_set)
    return data_set
def ID3Data():
    #年龄分为0、1、2分别代表青年、中年、老年
    #工作0、1分别代表无、有
    #房子0、1分别代表无、有
    #信贷情况0、1、2分别代表一般、好、非常好
    #最后一列表示结果，0、1分别代表否、是
    data_set = [[0,0,0,0,0],
                [0,0,0,1,0],
                [0,1,0,1,1],
                [0,1,1,0,1],
                [0,0,0,0,0],
                [1,0,0,0,0],
                [1,0,0,1,0],
                [1,1,1,1,1],
                [1,0,1,2,1],
                [1,0,1,2,1],
                [2,0,1,2,1],
                [2,0,1,1,1],
                [2,1,0,1,1],
                [2,1,0,2,1],
                [2,0,0,0,0]]
    labels = ['年龄','有工作','有房子','信贷情况']
    return data_set,labels

class CART():
    def __init__(self):
        pass
    def __MinSquareError(self,x_set,y_set):
        '''
        返回：数据集平方误差最小的切分点，为之后CART算法铺垫，数据是二分的
        x_set:自变量数据
        y_set:因变量数据
        '''
        leng = len(y_set)
        #误差列表
        error = []
        for i in range(leng):
            #左半部分数据
            l_half = y_set[:i]
            #右半部分数据
            r_half = y_set[i:]
            l_error = r_error = 0
            if(len(l_half)):
                #左半部分均值
                l_ave = sum(l_half)/len(l_half)
                for lval in l_half:
                    #平方误差累加
                    l_error +=(lval-l_ave)**2
            if(len(r_half)):
                r_ave = sum(r_half)/len(r_half)
                for rval in r_half:
                    r_error += (rval-r_ave)**2
            total_error = l_error+r_error
            error.append(total_error)
        index = error.index(min(error))
        return index
    def CART(self,data_set):
        '''
        决策树CRAT回归
        返回：递归树
        data_set：数据集
        '''
        #递归树
        tree = {}
        y_set = data_set[-1]
        x_set = data_set[0]
        #结果集小于3个，不再分裂子树
        if len(y_set) < 3:
            tree[sum(y_set)/len(y_set)] = {}
            return tree
        else:
            #最佳切分点
            index = self.__MinSquareError(x_set,y_set)
            node = x_set[index]
            tree[node] = {}
            #小于切分点数据
            l_data_set = [x_set[:index],y_set[:index]]
            #大于等于切分点
            r_data_set = [x_set[index:],y_set[index:]]
            #大于等于切分点
            tree[node][1] = self.CART(r_data_set)
            #小于切分点
            tree[node][0] = self.CART(l_data_set)
        return tree
    def TestVal(self,tree,val):
        for node,child_tree in tree.items():
            if child_tree:
                if val < node:
                    return self.TestVal(child_tree[0],val)
                else:
                    return self.TestVal(child_tree[1],val)
            else:
                return node
class ID3():
    def __init__(self):
        pass
    def __Entropy(self,data):
        '''
        返回 data的信息熵
        data:为变量的分布，ndarray对象
        '''
        entropy = 0
        num = len(data)
        #data的可能取值个数
        sdata = set(data)
        #统计data中数值个数，计算信息熵
        for sd in sdata:
            cnt = np.sum(data == sd)
            #出现频率
            prob = cnt/num
            entropy -= prob*math.log(prob,2)
        return entropy
    def __InfoGain(self,D,A):
        '''
        返回信息增益 H(D)-H(D|A)
        H(D)为经验熵
        H(D|A)为条件熵
        '''
        #经验熵
        HD = self.__Entropy(D)
        A_set = set(A)
        num = len(A)
        #A_dict字典存储(A的可能取值，此值对应的D值)键值对
        A_dict = {}
        #A_dict = {}.fromkeys(A_set,[]) fromkeys的坑append改变一个列表，其他列表也会改变
        #初始化字典
        for a in A_set:
            A_dict[a] = []
        #条件熵
        con_entropy = 0
        for i in range(num):
            A_dict[A[i]].append(D[i])
        for value in A_dict.values():
            con_entropy +=(len(value)/num)*self.__Entropy(value)
        return HD-con_entropy
    def ID3(self,data_set):
        '''
        ID3算法生成决策树
        data_set:数据集
        返回一个递归生成的树{index:{value1:{},value2:{},value3:{}}}
        其中index是树节点，对应于某个特征j的索引，valuei为该特征的可能取值，valuei对应的字典为选取该特征的取值后分裂出的子树
        '''
        #特征集，最后一行为结果
        features = np.array(data_set).T
        #与特征集对应的分类结果
        res = features[-1]
        #信息增益
        info_gain = []
        #最终返回的递归树
        tree = {}
        #当分类结果只有一种时，就到了树的叶子，不能再分裂了
        if len(set(res)) == 1:
            #键为最终结果，值为空字典
            tree[res[0]] = {}
            return tree
        else:
            #特征个数
            feature_num= len(features)-1
            for i in range(feature_num):
                #计算每个特征的信息增益
                info_gain.append(self.__InfoGain(res,features[i]))
            #最大信息增益的索引
            index = info_gain.index(max(info_gain))
            #将此索引加入树节点
            tree[index] = {}
            #初始化字典，键值对为（特征可能取值，与此值对应的数据集）
            for lab in set(features[index]):
                #初始化为空列表，存储与值对应的数据集
                tree[index][lab] = []
            for d in data_set:
                lab = d[index]
                #删除自身
                d.pop(index)
                #加入分裂后的数据集
                tree[index][lab].append(d)
            for key,data in tree[index].items():
                #对所有子树递归调用ID3算法，并用生成的树更新子树字典的值（子树字典之前的值为分裂后的数据集，现在为子树的子树）
                tree[index][key] = self.ID3(data)
        return tree
    def TestData(self,tree,data):
        '''
        测试新的数据
        tree: ID3算法生成的树
        data:新的数据
        '''
        for node,child_tree in tree.items():
            #如果子树非空
            if child_tree:
                #递归调用TestData函数，直到子树为空
                return self.TestData(child_tree[data[node]],data)
            #子树为空，返回节点即对应的测试结果
            else:
                return node
if __name__ == '__main__':
#    data_set= CARTData()
#    dt = CART()
#    tree = dt.CART(data_set)
#    print(tree)
#    print(dt.TestVal(tree,6))
    id3_data,labels= ID3Data()
    DT = ID3()
    tree = DT.ID3(id3_data)
    test_data = [1,0,0,1]
    print(tree)
    print(DT.TestData(tree,test_data))
