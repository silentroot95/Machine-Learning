##### 降维技术

常用数据降维技术。

###### 主成分分析(Principal Component Analysis,PCA)

###### 因子分析(Factor Analysis)

###### 独立成分分析(Indenpend Component Analysis,ICA)

###### 奇异值分解(SVD)

##### PCA

找出数据的几个最主要的特征，然后进行分析，举个例子

考察一个人的智力情况，直接看数学成绩就行了（存在数学、语文、英语成绩）

第一主成分，即数据方差最大的方向，在这个方向上数据区分度最大。

第二主成分方向，方差次大的方向，且与第一方向正交

如图，蓝色直线方向为第一主成分方向，红色直线方向为第二主成分方向

<img src=".\pic.\zcf.png" style="zoom:67%;" />

###### 方差

总体方差
$$
\sigma^2 = \frac{1}{n}\sum_{i=1}^n(x_i-\mu)^2
$$
这里的$\sigma^2$是总体方差，$\mu$是总体均值

样本方差

对于大样本，计算总体均值与总体方差代价大，往往随机抽取一定数量样本用样本均值代替总体均值，用样本方差代替总体方差。
$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n(x_i-\bar{x})^2
$$
其中$\bar{x}$表示样本均值，$S^2$表示样本方差

这里的分母为$n-1$是为了获得对$\sigma^2$的无偏估计，即
$$
E(S^2) = \sigma^2
$$

###### 协方差

考察变量X,Y的相关性
$$
Cov(X,Y) = \frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})
$$

$$
Cov(X,Y) =
\begin{cases}
>0 & 正相关 \\
<0 & 负相关 \\
=0 & 无线性相关性
\end{cases}
$$

对于多变量有协方差矩阵
$$
Cov(X,Y,Z)=
\left[
\begin{matrix}
Cov(X,X) & Cov(X,Y) & Cov(X,Z)\\
Cov(Y,X) & Cov(Y,Y) & Cov(Y,Z)\\
Cov(Z,X) & Cov(Z,Y) & Cov(Z,Z)
\end{matrix}
\right]
$$


##### 推导

这里采用最小误差的形式推导，因为在推导过程中我们用到原始数据的一种近似表示,
引入$D$维基向量的完整单位正交集合$\{\pmb {u}_i\}$，其中$i=1,\dots,D$，满足
$$
\begin{equation}
	\pmb {u}_i^T\pmb{u}_j=\delta_{ij}
\end{equation}
$$


每个数据点可以精确地表示为基向量的线性组合，即
$$
\begin{equation}
    \pmb{x}_n=\sum_{i=1}^n \alpha_{ni} \pmb{u}_i
\end{equation}
$$
将$\pmb{x}_n$与$\pmb{u}_j$做内积，利用单位正交性，可得$\alpha_{nj}=\pmb{x}_n^T\pmb{u}_j$，因此不失一般性，我们有
$$
\begin{equation}
    \pmb{x}_n=\sum_{i=1}^D(\pmb{x}_n^T\pmb{u}_i)\pmb{u}_i
\end{equation}
$$
现在我们使用$M(M<D)$维的线性子空间来近似表示$\pmb{x}_n$，不失一般性，采用$D$维子空间的前$M$个基向量，即
$$
\begin{equation}
    \tilde{\pmb{x}}_n=\sum_{i=1}^M z_{ni}\pmb{u}_i+\sum_{i=M+1}^D b_i\pmb{u}_i
\end{equation}
$$
其中$\{z_{ni}\}$依赖于特定的数据点，$\{b_i\}$是常数对所有的数据点都相同。我们的目标是选择$\{\pmb{u}_i\}$,$\{z_{ni}\}$,$\{b_i\}$最小化失真函数
$$
\begin{equation}
    J=\frac{1}{N}\sum_{i=1}^N \Vert \pmb{x}_n-\tilde{\pmb{x}}_n\Vert^2
\end{equation}
$$
首先考虑$\{z_{nj}\}$，注意这里$J,z_{nj}$是标量，$\tilde{\pmb{x}}_n$是向量

$$
\begin{equation}
\frac{\partial J}{\partial z_{nj}} = (\frac{\partial \tilde{\pmb{x}}_n}{\partial z_{nj}})^T \frac{\partial J}{\partial \tilde{\pmb{x}}_n} 
= \pmb{u}_j^T(2\tilde{\pmb{x}}_n-2\pmb{x}_n)
=2(\pmb{x}_n^T \pmb{u}_j-z_{nj})
\end{equation}
$$
另上式等于0，可得
$$
\begin{equation}
    z_{nj} = \pmb{x}_n^T\pmb{u}_j
\end{equation}
$$
考虑$b_j$
$$
\begin{equation}
\frac{\partial J}{\partial b_j} = (\frac{\partial \tilde{\pmb{x}}_n}{\partial b_j})^T \frac{\partial J}{\partial \tilde{\pmb{x}}_n} 
=\frac{1}{N} \sum_{j=1}^N \pmb{u}_j^T(2\tilde{\pmb{x}}_n-2\pmb{x}_n)
=\frac{2}{N}\sum_{j=1}^N (\pmb{x}_n^T \pmb{u}_j-b_j)
\end{equation}
$$
另上式为0，可得
$$
\begin{equation}
	b_j=\bar{\pmb{x}}^T\pmb{u}_j
\end{equation}
$$
其中$j=M+1,\cdots,D$，误差向量为
$$
\begin{equation}
	\pmb{x}_n-\tilde{\pmb{x}}_n = \sum_{i=M+1}^D\{(\pmb{x}_n-\bar{\pmb{x}}^T)\pmb{u}_i\}\pmb{u}_i
\end{equation}
$$
可以看到误差向量位于与主子空间垂直的空间中。
将上面的结果带入失真度量$J$，我们得到下式，它是一个纯粹关于$\{\pmb{u}_i\}$的函数
$$
\begin{equation}
	J=\frac{1}{N}\sum_{n=1}^N\sum_{i=M+1}^D(\pmb{x}_n^T\pmb{u}_i-\bar{\pmb{x}}^T\pmb{u}_i)^2
	=\sum_{i=M+1}^D\pmb{u}_i^TS\pmb{u}_i
\end{equation}
$$
剩下是求$\{\pmb{u}_i\}$使$J$最小化。考虑$D=2,M=1$的情况，我们限制$\pmb{u}_2^T\pmb{u}_2=1$，引入拉格朗日乘子$\lambda_2$，等价于最小化下式
$$
\begin{equation}
	\tilde{J}=\pmb{u}_2^TS\pmb{u}_2+\lambda_2(1-\pmb{u}_2^T\pmb{u}_2)
\end{equation}
$$
另上式关于$\pmb{u}_2$的导数等于0，得到$S\pmb{u}_2=\lambda_2\pmb{u}_2$，从而$\pmb{u}_2$是$S$的特征向量，特征值为$\lambda_2$。
对于任意的$D$和任意的$M<D$，最小化$J$的解可以求协方差矩阵的特征向量得到,即
$$
\begin{equation}
	S\pmb{u}_i=\lambda_i\pmb{u}_i
\end{equation}
$$
其中$i=1,\cdots,D$，这里特征向量$\{\pmb{u}_i\}$是单位正交的，失真度量为
$$
\begin{equation}
	J = \sum_{i=M+1}^D\lambda_i
\end{equation}
$$

##### PCA应用

PCA的一种应用是数据的降维压缩，另一种用途是数据预处理，我此次作业实现的是PCA对图片的压缩。
我们再看一下数据的近似过程，压缩就体现在这个近似过程中，将求得的结果带入(4)式中，可以得到数据的近似
$$
\begin{equation}
\tilde{\pmb{x}}_n = \sum_{i=1}^M(\pmb{x}_n^T \pmb{u}_i)\pmb{u}_i+\sum_{i=M+1}^D(\bar{\pmb{x}}^T\pmb{u}_i)\pmb{u}_i
	=\bar{\pmb{x}}+\sum_{i=1}^M(\pmb{x}_n^T\pmb{u}_i-\bar{\pmb{x}}^T\pmb{u}_i)\pmb{u}_i
\end{equation}
$$
由上式重构出$\tilde{\pmb{x}}_n$，我们需要

​	$\bar{\pmb{x}}$，$D$维向量，数据量$D\times 1$
​	$\{\pmb{u}_i\}$，$M$个基向量，数据量$D \times M$
​	$\pmb{x}_n^T\pmb{u}_i-\bar{\pmb{x}}^T\pmb{u}_i$，对应于基向量的$M$个系数，$N$个数据共$N \times M$个数据量。

这里我们先引入一个压缩率$R$的定义，$R$定义为重构数据需要的数据量比上原始数据数据量，即
$$
\begin{equation}
	R=\frac{D+D\times M+M\times N}{D\times N}
\end{equation}
$$

##### 计算结果

分别对灰度图与RGB彩色图进行了PCA，其中的压缩率用式(16)计算，结果如下

<img src=".\pic\L.jpeg" style="zoom:67%;" />

<img src=".\pic\rgb.jpeg" style="zoom:67%;" />

