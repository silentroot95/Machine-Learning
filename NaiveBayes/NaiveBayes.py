#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on May 26, 2020
Author: silentroot95
'''
import os
import re
import numpy as np
import random

def CreateVocabulary(data_mat):
    '''
    创建单词表
    Args:
        data_mat:原始文件 单词矩阵
    Return:
        单词集合
    '''
    voca = set()
    for docu in data_mat:
        voca |= set(docu)
    return list(voca)
def Words2Vec(voca,words):
    '''
    根据词汇表生成文本词向量
    Args:
        voca:单词表
        words:文本
    Return:
        vec:words的词向量
    '''
    #初始化0向量
    vec = np.zeros(len(voca))
    for word in words:
        if word in voca:
            vec[voca.index(word)] = 1
    return vec
def Train(train_mat,label_vec):
    '''
    训练模型
    Args:
        train_mat:训练词向量组成的矩阵
        label_vec:类别向量
    Return:

    '''
    train_mat = np.array(train_mat)
    #邮件数
    docs_num = len(train_mat)
    #单词表长度
    words_num = len(train_mat[0])
    #垃圾邮件概率
    p_spam = sum(label_vec)/docs_num
    #初始化词向量为1，拉普拉斯平滑
    p0_num = np.ones(words_num)
    p1_num = np.ones(words_num)
    #各类初始化数目为2，拉普拉斯平滑
    num0 = 2
    num1 =2
    for i in range(docs_num):
        if label_vec[i] == 1:
            #垃圾邮件，词向量累加
            p1_num += train_mat[i]
            #统计词数
            #num1 += sum(train_mat[i])
            #统计文件数，这里我选择统计文件数
            num1 += 1
        else:
            p0_num += train_mat[i]
            #num0 += sum(train_mat[i])
            num0 += 1
    #各词在类0出现的对数概率
    p0_vec = np.log(p0_num/num0)
    #各词在类1中出现的对数概率
    p1_vec = np.log(p1_num/num1)
    return  p0_vec,p1_vec,p_spam
def Classify(test_vec,p0_vec,p1_vec,p_spam):
    '''
    新的词向量分类
    Args:
        test_vec:测试词向量
        p0_vec:非垃邮件词向量
        p1_vec:垃圾邮件词向量
        p_spam:垃圾邮件概率
    Return:
        1 or 0
    '''
    p0 = sum(test_vec*p0_vec) + np.log(1-p_spam)
    p1 = sum(test_vec*p1_vec) + np.log(p_spam)
    if p1 > p0:
        return 1
    else:
        return 0

def TextParse(big_string):
    '''
    文本解析
    Args:
        big_string:长字符串
    Return:
        非字母数字分割后字符串，保留长度大于2的单词
    '''
    #\W*匹配非字母数字
    words = re.split('\W*',big_string)
    return [word.lower() for word in words if len(word) > 2]

def WalkDir(dirname):
    '''
    读取目录下全部文件，返回文件字符列表
    '''
    files = os.listdir(dirname)
    words_list = []
    for filei in files:
        filename = dirname + '\\'+ filei
        with open (filename,'r') as f:
            big_string = f.read()
        words = TextParse(big_string)
        words_list.append(words)
    return words_list
def LoadData():
    '''
    读取spam和ham构建数据集，类别向量
    '''
    ham_dir = r'.\email\ham'
    spam_dir = r'.\email\spam'
    ham = WalkDir(ham_dir)
    spam = WalkDir(spam_dir)
    label_ham = [0]*len(ham)
    label_spam = [1]*len(spam)
    data = ham+spam
    label_vec = label_ham+label_spam
    return data,label_vec

def Test():
    '''
    随机选10个文件作为测试集，余下的做训练集
    '''
    #构建数据
    data_mat,label_vec = LoadData()
    #生成单词表
    voca = CreateVocabulary(data_mat)
    train_set = list(range(50))
    test_num = 10
    test_set = []
    #随机生成测试集
    for i in range(test_num):
        rand_index = random.randint(0,len(train_set)-1)
        test_set.append(rand_index)
        del train_set[rand_index]
    train_mat = []
    train_label = []
    #根据单词表生成测试集词向量
    for train_index in train_set:
        train_mat.append(Words2Vec(voca,data_mat[train_index]))
        train_label.append(label_vec[train_index])
    #训练模型
    p0,p1,p_spam = Train(train_mat,train_label)
    error = 0
    #统计错误率
    for test_index in test_set:
        test_vec = Words2Vec(voca,data_mat[test_index])
        if Classify(test_vec,p0,p1,p_spam) != label_vec[test_index]:
            error += 1
            print('error doc:',data_mat[test_index])
    print('error rate:{:.2f}'.format(error/test_num))

if __name__ == '__main__':
    Test()
