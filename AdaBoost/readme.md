#### AdaBoost

AdaBoost是一种强化学习算法，属于Boosting家族的一员，其全称为Adaptive Boosting（自适应强化）。AdaBoost一种比较容易理解的推导方式是**前向分步算法和加法模型**。

##### 加法模型

AdaBoost给出的强分类器是一系列弱分类器的线性组合，即
$$
f(x) = \sum_{m=1}^M\beta_mb(x,\gamma_m)
$$
其中，$b(x,\gamma_m)$为弱分类器，$\gamma_m$是弱分类器的参数，$\beta_m$是系数，$f(x)$是最终的强分类器。

##### 前向分步算法

给定训练数据，及顺市函数$L(y,f(x))$的条件下，学习加法模型成为极小化损失函数问题：
$$
\mathop{\min}_{\beta_m,\gamma_m}\sum_{i=1}^NL(y_i,\sum_{m=1}^M\beta_mb(x_i,\gamma_m))
\tag1
$$
这是一个复杂的优化问题。前向分步算法求解这一问题的想法是：因为学习的是加法模型，那么从前向后，每一步只学习一个弱分类器及其系数，逐渐逼近优化目标函数式(1)，这样可以简化复杂度。具体地，每步只需优化如下损失函数：
$$
\mathop{\min}_{\beta,\gamma}\sum_{i=1}^NL(y_i,\beta b(x_i,\gamma))
$$
算法步骤

输入：训练数据集$T=\{(x_1,y_1),\cdots,(x_N,y_N)\}$，损失函数$L(y,f(x))$，基函数集$\{b(x,\gamma)\}$

输出：加法模型$f(x)$

(1)初始化$f_0(x)=0$

(1)对$m=1,\cdots,M$

​	极小化损失函数
$$
(\beta_m,\gamma_m) = \arg \mathop{\min}_{\beta,\gamma}\sum_{i=1}^NL(y_i,f_{m-1}(x_i)+\beta b(x_i,\gamma))
$$
​	得到参数$\beta_m,\gamma_m$

​	更新
$$
f_m(x) = f_{m-1}(x)+\beta_mb(x,\gamma_m)
$$
(3)得到加法模型
$$
f(x) = \sum_{m=1}^M\beta_mb(x,\gamma_m)
$$

##### AdaBoost

基学习器的线性组合
$$
f(x) = \sum_{m=1}^M\alpha_mG_m(x)
$$
输入：训练集$T=\{(x_1,y_1),\cdots,(x_N,y_N)\}$

输出：最终分类器$G(x)$

(1)初始化训练集权值分布
$$
D_1=(w_{11},\cdots,w_{1i},\cdots,w_{1N}),w_{1i} = 1/N
$$
(2)对$m=1,2,\cdots,M$

​	(a)使用权值分布$D_m$的训练集学习，得到基本分类器$G_m(x)$

​	(b)计算$G_m(x)$在训练集上的分类误差率
$$
e_m = P(G_m(x_i)\neq y_i) = \sum_{i=1}^Mw_{mi}I(G_m(x)\neq y_i)
$$
​	其中
$$
I(x)=
\begin{cases}
1 \quad x\ is\ {\rm True}\\
0\quad x\ is\ {\rm False}
\end{cases}
$$
​	应满足$e_m <0.5$。若$e_m\ge0.5$，表示此基本分类器还不如瞎猜，此时可以终止循环了。	

(c)计算$G_m(x)$系数
$$
\alpha_m = \frac{1}{2}\ln\frac{1-e_m}{e_m}
$$
​	由上面$e_m<0.5$，有$\alpha_m>0$，	

(d)更新训练集权值分布
$$
D_{m+1} = (\cdots,w_{m+1,i},\cdots)\\
w_{m+1,i} = \frac{w_{mi}}{Z_m}\exp(-\alpha_my_iG_m(x_i))
$$
​	其中，$Z_m$是规范化因子
$$
Z_m = \sum_{i=1}^Nw_{mi}\exp(\alpha_my_iG_m(x_i))
$$
​	它使得$D_{m+1}$成为一个概率分布

(3)构建基本分类器的线性组合
$$
f(x) = \sum_{i=1}^M\alpha_mG_m(x)
$$
得到最终分类器
$$
G(x) ={\rm sign}(f(x))
$$

##### AdaBoost算法推导

AdaBoost算法是前向分步算法的特例，损失函数为指数损失
$$
L(y,f(x)) = \exp(-yf(x))
$$
假设经过$m-1$轮迭代 得到$f_{m-1}(x)$
$$
f_{m-1}(x) = \sum_{i=1}^{m-1}\alpha_iG_i(x)
$$
在第$m$轮迭代中得到$\alpha_m,G_m(x)$和$f_m(x)$
$$
f_m(x) = f_{m-1}(x)+\alpha_mG_m(x)
$$
目标是求得$\alpha_m,G_m(x)$使$f_m(x)$在训练集上的指数损失最小，即
$$
(\alpha_m,G_m(x))=\mathop{\arg\min}_{\alpha,G}\sum_{i=1}^N\exp[-y_i(f_{m-1}(x_i)+\alpha G(x_i))]
$$
可表示为
$$
(\alpha_m,G_m(x))=\mathop{\arg\min}_{\alpha,G}\sum_{i=1}^N\bar{w}_{mi}\exp[-y_i\alpha G(x_i)]
$$
其中，$\bar{w}_{mi} = \exp[-y_if_{m-1}(x_i)]$，即$f_{m-1}(x)$对每一项的指数损失，亦即第$m$轮迭代未规范化的权值向量。因为$\bar{w}_{mi}$既不依赖于$\alpha$也不依赖于$G$，所以与最小化无关。但$\bar{w}_{mi}$依赖于$f_{m-1}(x)$，随着每一轮迭代改变。

首先求$\alpha_m$.
$$
\begin{align}
\sum_{i=1}^N\bar{w}_{mi}\exp[y_i-\alpha G(x_i)] &= \sum\bar{w}_{mi}e^{-\alpha}I(y_i=G(x_i))+\sum \bar{w}_{mi}e^{\alpha}I(y_i\neq  G(x_i))\\
&=(e^{\alpha}-e^{-\alpha})\sum_{i=1}^N \bar{w}_{mi}I(y_i\neq G(x_i))+e^{-\alpha}\sum_{i=1}^N\bar{w}_{mi}
\end{align}
$$
对$\alpha$求导，并使导数为0，得到
$$
\alpha_m = \frac{1}{2}\ln\frac{1-e_m}{e_m}
$$
其中，$e_m$是分类误差率
$$
e_m=\frac{\sum_{i=1}^N\bar{w}_{mi}I(y_i\neq G_m(x_i))}{\sum_{i=1}^N\bar{w}_{mi}}
=\sum_{i=1}^Nw_{mi}I(y_i\neq G_m(x_i))
$$
其中$w_{mi}$是规范化的权值向量

最后看每一轮权值更新。由
$$
f(x) = f_{m-1}(x)+\alpha_mG_m(x)
$$
及$\bar{w}_{mi}=\exp[-y_if_{m-1}(x_i)]$，可得
$$
\bar{w}_{m+1,i} = \bar{w}_{mi}\exp[-y_i\alpha_mG_m(x)]
$$
这里的权值更新，与上面只差一个规范化因子$Z_m$，是等价的。

再求$G_m(x)$，对任意$\alpha>0$
$$
\begin{align}
G_m(x) &=\mathop{\arg\min}_{G}\sum_{i=1}^N\bar{w}_{mi}\exp[-y_i\alpha G(x_i)]\\
	   &=\mathop{\arg\min}_{G}\sum_{i=1}^N\bar{w}_{mi}\exp[2I(y_i\neq G(x_i))-1]\\
	   &=\mathop{\arg\min}_{G}\sum_{i=1}^N\bar{w}_{mi}I(y_i\neq G(x_i)) 
\end{align}
$$
$G_m(x)$即是使第$m$轮加权数据分类误差最小的基本分类器。