##### 线性回归

###### 一元线性模型

最简单的线性模型，如下：
$$
f(x) = wx+b
$$
通过给定训练集$(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)$,利用最小二乘法，即可得到模型参数$w,b$。

###### 多元线性模型

将一元推广到多元，自变量$\pmb x = (x_1;x_2;\cdots;x_n)$，其中$x_i$为向量的第$i$个属性，多元线性模型
$$
f(\pmb x) = w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$
写成向量形式
$$
f(\pmb x) = \pmb w^T \pmb x+b
$$
记$\bold X = (\pmb x,1)$，$\hat{\pmb{w}}= (\pmb w;b)$，则平方和误差
$$
E(\hat{\pmb{w}}) = (\pmb y-\bold X\hat{\pmb{w}})^T(\pmb y-\bold X\hat{\pmb{w}})
$$
对$\hat{\pmb{w}}$求导得
$$
\frac{\partial{E(\hat{\pmb{w}})}} {\partial{\hat{\pmb{w}}}} = 2\bold X^T(\bold X\hat{\pmb{w}}-\pmb y)
$$
矩阵求导参考https://zhuanlan.zhihu.com/p/24709748

令导数为0，得到
$$
\hat{\pmb{w}} ^* = (\bold X^T\bold X)^{-1}\bold X \pmb y
$$

###### 广义线性模型

$$
f(\pmb x) = w_1\phi_1(x_1)+w_2\phi_2(x_2)+\cdots+w_n\phi_n(x_n)+b
$$

其中$\phi_i(x_i)$为属性$x_i$的某种映射，可以是非线性函数，这样就可以以线性模型表达出非线性关系。

写成向量形式
$$
f(\pmb x) = \pmb w^T \pmb\phi(\pmb x)
$$
其中$\pmb \phi(\pmb x) = (\phi_1(x_1);\phi_2(x_2);\cdots;\phi_n(x_n))$。

##### 局部加权回归

局部加权回归对训练集的点采取不同的权，离预测点近的点权值较大，远的点权值较小。也即是，它对每一个点求出一个回归系数（相当于在局部求线性回归）,然后把这些局部的线段连接起来就是全部回归图像。

这里权值分布常用高斯核，点$x$在训练集上的权值向量$\pmb \alpha = (\alpha_1,\alpha_2,\cdots,\alpha_n)$，其中
$$
\alpha_i =\exp(-\frac{(x-x_i)^2}{2k^2})
$$
其含义为点$x$对训练集中$x_i$点的权，k为高斯核参数

此时求某点$x$的回归系数，最小化平方和误差为
$$
\begin{align}
E(\hat{\pmb{w}}) & = \sum_{i=1}^n \alpha_i(y_i-\hat{\pmb{w}}^T \pmb x_i)^2 \\
& = (\pmb y-\bold X\hat{\pmb {w}})^T A (\pmb y-\bold X\hat{\pmb {w}})
\end{align}
$$
这里矩阵$A$是对角矩阵，主对角元素为$\pmb \alpha$对应的值。

上式对$\hat{\pmb{w}}$求导得，
$$
\frac{\partial{E(\hat{\pmb{w}})}} {\partial{\hat{\pmb{w}}}} = 2\bold X^TA(\bold X\hat{\pmb{w}}-\pmb y)
$$
上面得推导中用到了$A^T=A$，因为$A$为对角矩阵。

导数为0
$$
\hat{\pmb{w}}^* = (\bold X^TA\bold X)^{-1}\bold X^TA \pmb y
$$

##### 岭回归

对于多元线性回归求得
$$
\hat{\pmb{w}} ^* = (\bold X^T\bold X)^{-1}\bold X \pmb y
$$
这里有一个问题，就是$\bold X^T \bold X$可能是奇异矩阵，其逆矩阵不存在。

对于以下情况，$\bold X^T \bold X$可能是奇异矩阵，即使其不是奇异矩阵，也很可能是病态矩阵，病态矩阵的逆矩阵是很不稳定的。

1、数据量小于特征个数，即$\bold X$不是列满秩的

2、特征存在多重共线性，即特征之间不是相互独立的，即某些列之间相关性较大

为了解决这个问题，引入岭回归。

简单地说，岭回归就是在矩阵$\bold X^T \bold X$上及一个$\lambda I$，从而使得矩阵非奇异，进而能求逆。此时
$$
\hat{\pmb{w}} ^* = (\bold X^T\bold X+\lambda I)^{-1}\bold X \pmb y
$$
岭回归最先用来处理特征数多于样本数的情况，现在也用于在估计中加入偏差，从而得到更好的估计。这里通过引入$\lambda$来限制$\pmb{w}$，通过引入惩罚项，能够减少不重要的参数，这个技术在统计学中叫做缩减(shrinkage)。

此时$\hat{\pmb w}$是$\lambda$的函数，画出$\hat{\pmb w}$随$\lambda$的变化曲线，就称为岭迹图，实际计算中，可选择非常多的$\lambda$值，做出岭迹图，看这个图取哪个值时图像稳定，由此确定$\lambda$。

岭回归是最小二乘回归的一种补充，它损失了无偏性，来换取高的数值稳定性，从而得到较高的计算精度。

##### Lasso法

Lasso法与岭回归类似，只是惩罚项（约束）不同。对于岭回归
$$
\pmb w = \mathop{\arg\min}_{\pmb w}\|\pmb y-\bold X\pmb w\|^2 \quad s.t.\sum_{i=1}^n w_i^2 \le t ,t\ge0
$$
由拉格朗日乘子法可知，上式等价于
$$
\pmb w = \mathop{\arg\min}_{\pmb w}(\|\pmb y-\bold X\pmb w\|^2 + \lambda \|\pmb w\|^2)
$$
其中$\lambda > 0$，$\lambda$与$t$一一对应。

Lasso法，带有惩罚项的误差函数
$$
\pmb w = \mathop{\arg\min}_{\pmb w}(\|\pmb y-\bold X\pmb w\|^2 + \lambda \sum_{i=1}^n|w_i|)
$$
<img src=".\pic\ys.png" style="zoom: 50%;" />

左侧为Lasso约束，右侧为岭回归约束，中心点为最小二乘法最优解