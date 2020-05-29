##### CART树

在DecisionTree中已经讨论过CART树的细节，此处只讨论剪枝

##### 剪枝

###### 预剪枝

在生成时根据测试集误差，若分裂后误差降低，则分裂，分裂后误差增大则不分裂。

###### 后剪枝

后剪枝的欠拟合风险较小，泛化性能往往优于预剪枝树。后剪枝是在决策树生成后进行的，后剪枝要自底向上对树中所有非叶节点进行考察，因此其训练时间开销比预剪枝树大得多。

在代码实现中，先对树中的叶节点进行标记，表示此节点已考察过。然后递归对左右子树均已考察过的节点进行考察，并把此节点标记为已考察过。