#!usr/bin/python
#-*- coding:utf-8 -*-

'''
Created on May 20,2020
Author: silentroot95
'''

def LoadData():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def GenC1(data,min_support):
    '''
    返回数据集的1项集
    Args:
        data:数据集
        min_support:最小支持度
    Return:
        数据集的1项集
    '''
    m = len(data)
    c1 = []
    for item in data:
        for idx in item:
            if not [idx] in c1:
                #frozenset函数变量必须为可遍历对象，所以这里的数值加方括号转化为了数组
                c1.append([idx])
    #每一个一项转化为1项集，frozenset为不可变集合，可Hash作为字典的键
    return list(map(frozenset,c1))
def GenMix(set_list):
    '''
    由n项集生成n+1项集
    Args:
        set_list:n项集
    Return:
        new_list:n+1项集
    '''
    klen = len(set_list)
    new_list = []
    for i in range(klen):
        for j in range(i+1,klen):
            #新项集为两旧项集求并
            new_set = set_list[i] | set_list[j]
            #后面的长度判断，是新的项集比上一个多一项
            if (new_set not in new_list)  and ((len(new_set) - len(set_list[i])) == 1):
                new_list.append(new_set)
    return new_list
def FilterSet(data,set_list,min_support):
    '''
    计算项集支持度
    Args:
        data:数据集
        set_list:n项集列表
        min_support:最小支持度
    Return:
        set_list:n+1项集列表
    '''
    #大于等于最小支持度的项集字典
    filter_list = []
    #样本个数
    dl = len(data)
    for se in set_list:
        se_num = 0
        for di in data:
            #如果项集se是di的子集
            if se.issubset(di):
                se_num += 1
        if se_num/dl >= min_support:
            filter_list.append(se)
    return filter_list

def FreqSet(c1):
    '''
    生成频繁项集
    Args:
        c1:1项集
    Return:
        频繁项集
    '''
    while(len(c1) > 1):
        filter_list = FilterSet(data,c1,0.5)
        c1 = GenMix(filter_list)
    return filter_list[0]
def Confidence(data,left_set,right_set):
    '''
    计算关联规则置信度Confidence(A->B) = P(B|A) = Support(AB)/Support(A)
    Args:
        left_set:左项集，即A集
        right_set:右项集，即B集
    Return:
        Support(AB)/Support(A)
    '''
    #A并B
    union_set = left_set | right_set
    union_num = 0
    left_num = 0
    for di in data:
        if left_set.issubset(di):
            #A集出现次数
            left_num += 1
        if union_set.issubset(di):
            #并集出现次数
            union_num += 1
    return union_num/left_num
def GenRule(rules,min_conf):
    '''
    根据最小置信度生成关联规则
    Args:
        rules:原始规则
        min_conf:最小置信度
    Return:
        rule_list:新的规则列表
    '''
    #新的规则列表
    rule_list = []
    r_len = len(rules)
    for i in range(r_len):
        for j in range(i+1,r_len):
            #新的左集
            left_set = rules[i]['left'] & rules[j]['left']
            #新的右集
            right_set = rules[i]['right'] | rules[j]['right']
            #关联规则置信度
            conf = Confidence(data,left_set,right_set)
            #大于等于最小置信度，则添加规则
            if conf >= min_conf:
                new_rule = {}
                new_rule['left'] = left_set
                new_rule['right'] = right_set
                new_rule['conf'] = conf
                rule_list.append(new_rule)
    return rule_list
def Rule1(freq,min_conf):
    '''
    由频繁项集，生成初始规则（右集为1项集）
    Args:
        freq:频繁项集
        min_conf:最小置信度
    Return:
        rule1:初始规则列表，强关联规则用字典存储
    '''
    rule1 = []
    for se in freq:
        #右集为1项集
        right = frozenset([se])
        #左集为频繁项集与右集的差集
        left = freq - right
        conf = Confidence(data,left,right)
        if conf >= min_conf:
            rule = {}
            rule['left'] = left
            rule['right'] = right
            rule['conf'] = conf
            rule1.append(rule)
    return rule1
def GL(rule1,min_conf):
    '''
    根据初始规则，生成强关联规则
    Args:
        rule1:初始规则列表
        min_conf:最小置信度
    Return:
        total_rules:所有的强关联规则
    '''
    rules = rule1
    total_rules = []
    #如果rules非空，最后一轮循环左项集为1项集
    while(rules and len(rules[0]['left']) > 0 ):
        #列表合并
        total_rules += rules
        new_rules = GenRule(rules,min_conf)
        rules = new_rules
    return total_rules

if __name__ == '__main__':
    data = LoadData()
    c1 = GenC1(data,0.5)
    freq_set = FreqSet(c1)
    min_conf = 0.5
    rule1 = Rule1(freq_set,min_conf)
    total_rules = GL(rule1,min_conf)
    print(total_rules)
