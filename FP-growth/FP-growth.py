#!usr/bin/python
#-*- coding:utf-8 -*-
'''
Created on May 21,2020
Author: silentroot95
'''
def LoadData():
    data = [['r','z','h','j','p'],
            ['z','y','x','w','v','u','t','s'],
            ['z'],
            ['r','x','n','o','s'],
            ['y','r','x','z','q','t','p'],
            ['y','z','x','e','q','s','t','m']]
    return data
def IniSet(data,min_sup):
    '''
    数据初始化为集合
    '''
    all_set = {}
    #对数据每一行处理
    one_set = {}
    for trans in data:
        for item in trans:
            one_set[item] = one_set.get(item,0) + 1
        #如果包含键frozenset(trans)则返回对应的值，不包含返回0 然后+1
        all_set[frozenset(trans)] = all_set.get(frozenset(trans),0) + 1
    for k in list(one_set.keys()):
        if one_set[k] < min_sup:
            del one_set[k]
    one_set_sorted = [v[0] for v in sorted(one_set.items(),key = lambda p:p[1],reverse=True)]
    return all_set,one_set_sorted
class TreeNode:
    '''
    FP 树的节点
    '''
    def __init__(self,name,occur_num,parentNode):
        self.name = name
        self.count = occur_num
        self.parent = parentNode
        self.children = {}
        #相同节点的链表
        self.nodelink = None
    def inc(self,count):
        self.count += count
    def disp(self,level = 0):
        #FP树的显示
        print(' '*level,self.name,self.count)
        for child in self.children.values():
            child.disp(level+1)
def UpdateTree(items,node,header_table,count):
    '''
    将items列表中的元素递归地插入树
    Args:
        items:按项出现次数排序后的列表
        node:在此树节点下添加子节点
        header_table:链表表头
        count:items列表在原始数据集中出现的次数
    '''
    #先插入首元素items[0]
    #如果items[0]已存在
    if items[0] in node.children:
        #增加计数
        node.children[items[0]].inc(count)
    else:
        #不存在，则新增子节点
        node.children[items[0]] = TreeNode(items[0],count,node)
        #如果此节点，在链表中未出现
        if header_table[items[0]][1] == None:
            #表头链表指向此节点
            header_table[items[0]][1] = node.children[items[0]]
        else:
            #链表已存在此节点则更新链表，在链表末尾加入此节点
            UpdateHeader(header_table[items[0]][1],node.children[items[0]])
    if len(items) > 1:
        #递归地对去除首元素之后的列表插入树中
        UpdateTree(items[1:],node.children[items[0]],header_table,count)
def UpdateHeader(tail_node,node):
    '''
    更新链表
    Args:
        tail_node:链表尾节点
        node:要插入的节点
    '''
    #循环直到tail_node是尾节点
    while(tail_node.nodelink != None):
        tail_node = tail_node.nodelink
    tail_node.nodelink = node
def CreateTree(ini_set,min_sup = 1):
    '''
    创建FP树
    Args:
        ini_set:初始集合
        min_sup:最小出现次数，对应最小支持度为min_sup/sample_num(样本数量)
    Return:
        root:FP树根节点
        header_table:节点链表
    '''
    header_table = {}
    for trans in ini_set:
        for item in trans:
            header_table[item] = header_table.get(item,0) + ini_set[trans]
    #删除小于min_sup的项集
    header_table = {k:v for k,v in header_table.items() if v >= min_sup}
    #频繁项集
    freq_item_set = set(header_table.keys())
    if len(freq_item_set) == 0:
        return None,None
    #初始化链表
    for k in header_table:
        header_table[k] = [header_table[k],None]
    #初始化根节点
    root = TreeNode('NULL',0,None)
    for tran_set,count in ini_set.items():
        local = []
        for item in tran_set:
            if item in freq_item_set:
                local.append(item)
        if len(tran_set) > 0:
            #根据项集出现次数，倒序排列，下面这个排序是非稳定的
            #ordered_items = [v[0] for v in sorted(local.items(),key = lambda p:p[1],reverse = True)]
            #下面是改进后的稳定排序
            ordered_items = StableSort(one_set,local)
            #将倒序的项集插入树
            UpdateTree(ordered_items,root,header_table,count)
    return root,header_table
def StableSort(one_set,com_set):
    '''
    基于频繁1项集的稳定排序
    Args:
        one_set:倒序排序后的频繁1项集
        com_set:未排序的数据
    Retuen:
        com_set排序的结果
    '''
    #com_set中的项在one_set中的索引
    item_index = []
    for item in com_set:
        if item in one_set:
            item_index.append(one_set.index(item))
    #对索引进行排序
    item_index.sort()
    return [one_set[i] for i in item_index]
def AscendTree(leaf,pre_path):
    '''
    递归上溯leaf节点直到根节点
    Args:
        leaf:叶节点
        pre_path:保存路径
    '''
    if leaf.parent != None:
        pre_path.append(leaf.name)
        AscendTree(leaf.parent,pre_path)
def FindPrePath(node):
    '''
    根据基模式返回条件模式
    Args:
        base_pat:基模式
        node:链表中的节点
    Return:
        cond_pats:条件模式
    '''
    cond_pats = {}
    while node != None:
        pre_path = []
        AscendTree(node,pre_path)
        if len(pre_path) > 1:
            #去除基模式
            cond_pats[frozenset(pre_path[1:])] = node.count
        node = node.nodelink
    return cond_pats
def MineTree(tree,header_table,min_sup,pre_fix,freq_list):
    '''
    根据FP树挖掘频繁项集
    Args:
        tree:树根节点
        header_table:链表
        min_sup:最小出现次数
        pre_fix:条件模式基前缀
        freq_list:保存频繁项集
    '''
    #正序排序链表
    #print(header_table.items())
    #dict_items([('x', [3, <__main__.TreeNode object at 0x0143A290>])])
    ordered_table = [v[0] for v in sorted(header_table.items(),key = lambda p:p[1][0])]
    #print(ordered_table)
    for base_pat in ordered_table:
        #拷贝条件模式前缀
        new_freq_set = pre_fix.copy()
        #添加基模式
        new_freq_set.add(base_pat)
        #添加频繁项集到列表
        freq_list.append(new_freq_set)
        #生成链表节点的条件模式基
        cond_pat_bases = FindPrePath(header_table[base_pat][1])
        #根据条件模式基生成条件FP树
        cond_tree,head = CreateTree(cond_pat_bases,min_sup)
        if head != None:
            #递归挖掘条件FP树的频繁项集
            MineTree(cond_tree,head,min_sup,new_freq_set,freq_list)
if __name__ == '__main__':
    data = LoadData()
    min_sup = 3
    ini_set,one_set= IniSet(data,min_sup)
    root,header_table = CreateTree(ini_set,min_sup)
    #root.disp()
    freq_list = []
    MineTree(root,header_table,min_sup,set(),freq_list)
    for freq in freq_list:
        print(freq)
