#### 支持向量机

线性模型的二分类问题，形式如下
$$
y(x) = w^T\phi(x)+b
$$
其中$\phi(x)$表示固定的特征空间变换，$b$为偏移。训练数据集由$m$个向量$x_1,\cdots,x_m$组成，对应目标值为

$t_1,\cdots,t_m$，其中$t_i \in \{-1,1\}$，新的数据根据$y(x)$符号分类。

##### 基本型

<img src=".\pic\jc.png" alt="|" style="zoom:67%;" />

我们的目标是选择决策边界最大化间隔（margin），任意一点到决策边界的距离为
$$
\frac{|f(x_i)|}{\|w\|} = \frac{t_iy(x_i)}{\|w\|} =\frac{t_i(w^T\phi(x_i)+b)}{\|w\|}
$$
间隔由数据集里垂直距离最近的$x_n$给出，我们希望最优化参数$w$和$b$，使得这个距离最大，即
$$
\mathop{\arg\max}_{w,b}\{{\frac{1}{\|w\|}\mathop{\min}_{x_i}[t_i(w^T\phi(x_i)+b)]\}}
$$
注意，将$w$和$b$同时乘以系数$k$，任意点到决策面的距离不变。所以我们可以归一化，令
$$
t_i(w^T\phi(x_i)+b) = 1
$$
此时，所有的数据点会满足
$$
t_i(w^T\phi(x_i)+b)\ge1
$$
这是决策超平面的标准表示。

此时问题简化为最大化$\|w\|^{-1}$，等价于最小化$\|w\|^2$，即
$$
\mathop{\arg\min}_{w,b}\frac{1}{2}\|w\|^2  \\
s.t.\quad t_i(w^T\phi(x_i)+b)\ge1
$$
这就是支持向量机的基本型。

##### 拉格朗日乘子法

对于无约束的最优问题：$\min f(x)$，一般直接求导采用梯度下降法或牛顿迭代法求得最优值。

对于含有等式约束的问题，即
$$
\min f(x) \\
s.t. \quad h_i(x)=0,\quad i=1,\cdots,m
$$
由于等式约束$h_i(x)=0$的存在，无法直接求导迭代，拉格朗日乘子法是解决此类问题的常用方法，核心思想是将约束问题转化为无约束问题，即将有$d$个变量和$m$个等式约束的最优问题转化为一个有$(d+m)$变量的函数求平稳点的问题。

关键是$f(x)$的极小点必然与$h(x)=0$相切，如下左图$x_1$，为什么？看一下右图，如果$f(x)$与$h(x)=0$不相切，而是相交，由于$x$连续，则必然能找到一个新的$x_2$使得$f(x_2)$更小，且$x_2$处$f(x)$与$h(x)=0$相切。

<img src=".\pic\la.png" style="zoom: 67%;" />

由此得出两个推论：

<img src=".\pic\td.png" style="zoom: 67%;" />

（1）**对于$f(x)$的极小点$x^*$，$f(x)$在$x^*$处的梯度$\nabla f(x^*)$与$h(x)=0$的切线方向垂直**

（2）**对于$f(x)$的极小点$x^*$，$h(x)$在$x^*$处的梯度$\nabla h(x^*)$与$h(x)=0$的切线方向垂直**

根据梯度的定义第二点是显而易见的，由上面的分析第一点也是显而易见的。

由此可以得出在极小点处$\nabla f(x^*)$与$\nabla h(x^*)$平行，即存在$\lambda \neq0$使得
$$
\nabla f(x^*)+\lambda\nabla h(x^*) = 0 \tag{a}
$$
$\lambda$称为拉格朗日乘子，拉格朗日函数定义为：
$$
L(x,\lambda) = f(x)+\lambda h(x)
$$
将上式分别对$x$和$\lambda$求导置0，就得到a式和约束条件$h(x)=0$。这样就将约束问题转化为对$L(x,\lambda)$的无约束问题。然而这个方法找出的平稳点不一定都是原问题的极值点，如下左图是一个极值点，右图则不是。

<img src=".\pic\jz.png" style="zoom: 67%;" />

##### KKT条件

上面的拉格朗日乘子法解决的是等式约束，对于不等式约束也可解，只不过要加一些附加条件。

问题：
$$
\min f(x) \\
s.t. \quad g_i(x)\le=0,\quad i=1,\cdots,m \\
h_j(x)=0,\quad j=1,\cdots,n
$$
先下个定义：

对于不等式约束$g_i(x)\le0$，若在$x^*$处$g_i(x)<0$，那么称该不等式约束在$x^*$处不起作用；若在$x^*$处$g_i(x)=0$,那么称不等式约束在$x^*$处起作用。

对该定义的直观解释见下图：灰色区域为约束$g(x)\le0$的可行域，若最优点$x^*$在区域内（左图），则约束不起作用，直接通过$\nabla f(x)=0$即可获得最优解，这等价于a式中$\lambda=0$。

若最优解在区域边界上（右图），对于$f(x)$来说，在$x^*$处外部较大，内部较小，因为越靠近等值线中心处$f(x)$越小；对于$g(x)$在$x^*$处变化趋势是内部较小，外部较大。这样$\nabla f(x^*)$与$\nabla g(x^*)$方向必相反，此时$g(x)=0$，带入a式可得$\lambda>0$。

综合两种情况：
$$
f(x)=
\begin{cases}
g(x)<0,\lambda=0\\
g(x)=0,\lambda>0
\end{cases}
\Rightarrow
\lambda \ge 0,\lambda g(x)=0
$$
这称为互补松弛条件。

<img src=".\pic\KKT.png" style="zoom:67%;" />

因此推广的多个约束，拉格朗日函数：
$$
L(x,\alpha,\beta) = f(x)+\sum_{i=1}^{m}\alpha_ig_i(x)+\sum_{j=1}^{n}\beta_jh_j(x)
$$
KKT条件：
$$
h_j(x)=0\\
g_i(x)\le0\\
\alpha_i\ge 0\\
\alpha_ig_i(x)=0
$$
KKT条件是极小值的必要条件，即满足KKT条件的不一定是极小值，但极小值一点满足KKT条件。

##### 对偶问题

将原始问题转化为对偶问题，是求解带约束优化问题的一种方法，但不是唯一方法，不过转化后容易求解，因为应用广泛。

设原始问题为：
$$
\min f(x) \\
s.t. \quad g_i(x)\le0,\quad i=1,\cdots,m \\
h_j(x)=0,\quad j=1,\cdots,n
$$
拉格朗日函数为
$$
L(x,\alpha,\beta) = f(x)+\sum_{i=1}^{m}\alpha_ig_i(x)+\sum_{j=1}^{n}\beta_jh_j(x),
\quad \alpha\ge0
$$
考虑拉格朗日函数对$\alpha,\beta$取最大值，得到关于$x$的函数
$$
\theta(x) = \mathop{\max}_{\alpha,\beta}L(x,\alpha,\beta)
$$
若$x$满足约束，由KKT条件易得$\theta(x)=f(x)$

若$x$违反了一些约束（即存在$i,j$使$g_i(x)>0$或$h_j(x)\neq0$），则可令$\alpha_i\rightarrow+\infty$，令$\beta_j$使$\beta_jh_j(x) \rightarrow +\infty$，而将其余$\alpha_i,\beta_j$均取0，则$\theta(x) = +\infty$，故原始问题
$$
\mathop{\min}_{x,\alpha,\beta}L(x,\alpha,\beta)\\
s.t. \quad \alpha_i\ge0,\quad i=1,\cdots,m
$$
等价于
$$
\mathop{\min}_x\theta(x) = \mathop{\min}_x\mathop{\max}_{\alpha,\beta}L(x,\alpha,\beta)\\
s.t. \quad \alpha_i\ge0,\quad i=1,\cdots,m
$$
原始问题的对偶问题为
$$
\mathop{\max}_{\alpha,\beta}\mathop{\min}_xL(x,\alpha,\beta)\\s.t. \quad \alpha_i\ge0,\quad i=1,\cdots,m
$$
对偶问题是原始问题的下界，即
$$
\mathop{\max}_{\alpha,\beta}\mathop{\min}_xL(x,\alpha,\beta)\le
\mathop{\min}_x\mathop{\max}_{\alpha,\beta}L(x,\alpha,\beta)
\tag{b}
$$
简单的说明，因为任意值小于等于极大值，故
$$
\mathop{\min}_xL(x,\alpha,\beta)\le
\mathop{\min}_x\mathop{\max}_{\alpha,\beta}L(x,\alpha,\beta)
$$
对任意$\alpha,\beta$恒成立，那么左式对$\alpha,\beta$取极大依然成立。这就是“极小的极大$\le$极大的极小”。

b式为不等式，所以该性质被称为弱对偶性，若等式成立，则为强对偶性。

强对偶性需要满足slater条件：

原始问题为凸优化问题，即$f(x),g(x)$为凸函数，$h(x)$为仿射函数，且可行域中至少一点使不等式约束严格成立，强对偶性成立，对偶问题等价于原始问题。

注意，slater条件是充分不必要条件，即由slater条件可以导出强对偶性，强对偶性导不出slater条件。

##### 回到我们的目标问题

我们的目标是求解如下凸优化问题
$$
\mathop{\arg\min}_{w,b}\frac{1}{2}\|w\|^2  \\
s.t.\quad y_i(w^T\phi(x_i)+b)\ge1,\quad i=1,\cdots,m
$$
此处的特征空间$\phi(x)$取最简单的线性空间$x$，构造拉格朗日函数（KKT条件在下面的讨论中给出）
$$
L(w,b,\alpha) = \frac{1}{2}\|w\|^2+\sum_{i=1}^{m}\alpha_i(1-y_i(w^Tx_i+b))\\
s.t. \quad \alpha_i\ge0
$$
上式对$w$和$b$求偏导为0可得
$$
w=\sum_{i=1}^m\alpha_iy_ix_i\\
0=\sum_{i=1}^m\alpha_iy_i
$$
带入拉格朗日函数得
$$
\mathop{\min}_{w,b}L(w,b,\alpha) = \sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)
$$
由上面的讨论我们知道
$$
\mathop{\min}_{w,b}L(w,b,\alpha) = \mathop{\min}_{w,b}\mathop{\max}_\alpha L(w,b,\alpha)
$$
因为原问题为凸优化问题，由强对偶性，转换为其对偶问题
$$
\mathop{\min}_{w,b}\mathop{\max}_\alpha L(w,b,\alpha)=
\mathop{\max}_\alpha\mathop{\min}_{w,b}L(w,b,\alpha)=
\mathop{\max}_\alpha\sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)\\
s.t. \quad \sum_{i=1}^m \alpha_iy_i = 0
$$
需满足KKT条件
$$
\alpha_i\ge0\\
y_if_i(x)-1\ge0\\
\alpha_i(y_if_i(x)-1)=0
$$
求解此对偶问题，可使用SMO算法。

##### 软间隔

对于类重叠问题，并没有一个完全符合条件的可分超平面，此时引入软间隔，即允许某些样本不满足约束$y_i(w^Tx_i+b)\ge1$。

<img src=".\pic\rjg.png" style="zoom:67%;" />

此时引入惩罚项，优化目标为
$$
\mathop{\min}_{w,b}\frac{1}{2}\|w\|^2+C\sum_{i=0}^ml(y_i(w^Tx_i+b)-1)
$$
其中$C>0$为常数，$l$为损失函数
$$
l(z)=
\begin{cases}
1,\quad z<0\\
0,\quad otherwise
\end{cases}
$$

由于上述损失函数非凸，非连续，数学性质不好，故采用其他函数替代，一个自然的想法是对于正确分类和边界上的点，损失函数取0，其他的点损失函数取值是距边界的距离的线性函数，这就是常用的hinge损失函数
$$
l_{hinge}(z) = \max(0,1-z)
$$
采用hinge损失，优化目标为
$$
\mathop{\min}_{w,b}\frac{1}{2}\|w\|^2+C\sum_{i=1}^m\max(0,1-y_i(w^Tx_i+b))
$$
引入松弛变量$\xi_i\ge0$，优化目标
$$
\mathop{\min}_{w,b}\frac{1}{2}\|w\|^2+C\sum_{i=0}^m\xi_i\\
s.t. \quad y_i(w^Tx_i+b)\ge1-\xi_i\\
\xi_i\ge0, \quad i=1,\cdots,m
$$
<img src=".\pic\sc.png" style="zoom:67%;" />

拉格朗日函数
$$
L(w,b,\alpha,\xi) = \frac{1}{2}\|w\|^2+C\sum_{i=1}^m\xi_i+
\sum_{i=1}^{m}\alpha_i(1-\xi_i-y_i(w^Tx_i+b))-\sum_{i=1}^m\mu_i\xi_i\\
s.t. \quad \alpha_i\ge0,\mu_i\ge0
$$
令$L(w,b,\alpha,\xi)$对$w,b,\xi$偏导为0得
$$
w = \sum_{i=1}^m\alpha_iy_ix_i \\
0 = \sum_{i=1}^m\alpha_iy_i\\
C = \alpha_i+\mu_i
$$
对偶问题
$$
\mathop{\max}_\alpha\sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)\\
s.t. \quad \sum_{i=1}^m \alpha_iy_i = 0,0\le\alpha_i\le C,\quad i=1,\cdots,m
$$
KKT条件
$$
\alpha_i\ge0,\mu_i\ge0\\
y_if(x_i)-1+\xi_i\ge0\\
\alpha_i(y_if(x_i)-1+\xi_i))=0\\
\xi_i\ge0,\mu_i\xi_i=0
$$
关于$\alpha$与$\mu$的理解

$\alpha=0$，正确分类

$0<\alpha<C$，即$\mu>0$，即$\xi=0$，样本位于边界上，支持向量

$\alpha=C$，即$\mu=0$，此时若$\xi\le1$，样本位于最大间隔内部，若$\xi>1$，样本被错误分类

##### SMO算法

SMO算法是求解对偶问题的常见算法，对偶问题是一个包含m个参数的二次规划问题，SMO算法对m个参数进行分解，每次只求解两个参数，每次启发式地选择两个参数进行优化，直到得到最优解。
$$
\min\Psi(\alpha)=\mathop{\min}_\alpha\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_j(x_i\cdot x_j)-\sum_{i=1}^m\alpha_i\\
s.t. \quad \sum_{i=1}^m \alpha_iy_i = 0,0\le\alpha_i\le C,\quad i=1,\cdots,m
$$


对偶问题的m个参数$(\alpha_1,\cdots,\alpha_m)$，需满足约束$\sum\alpha_iy_i=0$，选择优化$\alpha_1,\alpha_2$，固定其余$\alpha_i$，目标函数变为关于$\alpha_1,\alpha_2$的二元函数。
$$
\min\Psi(\alpha_1,\alpha_2)=\min\frac{1}{2}K_{11}\alpha_1^2+\frac{1}{2}K_{22}\alpha_2^2+y_1y_2K_{12}\alpha_1\alpha_2-(\alpha_1+\alpha_2)+y_1v_1\alpha_1+y_2v_2\alpha_2+Constant
$$
其中$Constant$为常数，$v_i$为
$$
v_i=\sum_{j=3}^m\alpha_jy_jK(x_i,x_j),i=1,2
$$
由等式约束得：
$$
\alpha_1y_1+\alpha_2y_2 = -\sum_{i=3}^m\alpha_iy_i=\zeta
$$
可见$\alpha_1,\alpha_2$并不独立，上式两边乘以$y_1$得
$$
\alpha_1 = (\zeta-y_2\alpha_2)y_1
$$
代入原式并略去常数项，得
$$
\min\Psi(\alpha_2)=\frac{1}{2}K_{11}(\zeta-\alpha_2y_2)^2+\frac{1}{2}K_{22}\alpha_{2}^2+y_2K_{12}(\zeta-\alpha_2y_2)\alpha_2-(\zeta-\alpha_2y_2)y_1-\alpha_2+v_1(\zeta-\alpha_2y_2)+y_2v_2\alpha_2
$$
对上式求导为0得
$$
\frac{\partial\Psi(\alpha_2)}{\partial\alpha_2}=(K_{11}+K_{22}-2K_{12})\alpha_2-K_{11}\zeta y_2+K_{12}\zeta y_2+y_1y_2-1-v_1y_2+v_2y_2=0
$$
由
$$
v_{i,j}=f(x_i)-\sum_{j=1}^2y_j\alpha_jK_{i,j}-b,\quad i=1,2\\
E_i=f(x_i)-y_i
$$
解得
$$
\alpha^{new,unclipped} = \alpha^{old}+\frac{y_2(E_1-E_2)}{\eta}\\
\eta = K_{11}+K_{22}-2K_{12}
$$
需要对原始结果进行修剪使$\alpha$满足约束条件
$$
0\le\alpha_{1,2}\le C\\
\alpha_1y_1+\alpha_2y_2=\zeta
$$
二维平面直观表达两个约束条件

<img src=".\pic\conf.png" style="zoom:67%;" />

最优解必须在方框内且在直线上取得，因此$L\le\alpha_2^{new}\le H$

当$y_1\neq y_2$时，$L=\max(0,\alpha_2^{old}-\alpha_1^{old});H=\min(C,C+\alpha_2^{old}-\alpha_1^{old})$

当$y_1=y_2$时，$L=\max(0,\alpha_1^{old}+\alpha_2^{old}-C);H=\min(C,\alpha_2^{old}+\alpha_1^{old})$

经过上述约束修剪，最优解$\alpha_2^{new}$为
$$
\alpha_2^{new}=
\begin{cases}
H,\quad \alpha_2^{new,unclipped}>H\\
\alpha_2^{new,unclipped},\quad alpha_2^{new,unclipped}\le H\\
L,\quad \alpha_2^{new,unclipped}<L
\end{cases}
$$
解得$\alpha_1^{new}$
$$
\alpha_1^{new} = \alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_2^{new})
$$
**启发式选择变量**

SMO称第一个变量的选择为外层循环，外层循环在训练样本中选取违反KKT条件最严重的样本点，检验样本的KKT条件，该检验是在$\varepsilon$范围内进行的，外层循环首先遍历在间隔边界上的支持向量，即满足$0<\alpha_i<C$，如果这些点都满足KKT条件，那就遍历整个训练集，检验他们是否满足KKT条件。

SMO称第2个变量的选择为内层循环，假设已经找到$\alpha_1$,第二个变量的选择标准是希望能使$\alpha_2$有足够大的变化。

由前面可知$\alpha_2^{new}$依赖于$|E_1-E_2|$，为了加快计算速度，应使$|E_1-E_2|$最大。

**计算偏移$b$**

每次两个变量优化后，都要重新计算偏移$b$。当$0<\alpha_1^{new}<C$时，由KKT条件得
$$
\sum_{i=1}^m\alpha_iy_iK_{i1}+b=y_1
$$
于是
$$
b_1^{new}=y_1-\sum_{i=3}^m\alpha_iy_iK_{i1}-\alpha_1^{new}y_1K_{11}-\alpha_2^{new}y_2K_{21}
$$
又
$$
y_1-\sum_{i=3}^m\alpha_iy_iK_{i1}=-E_1+\alpha_1^{old}y_1K_{11}+\alpha_2^{old}y_2K_{21}+b^{old}
$$
代入得
$$
b_1^{new}=-E_1-y_1K_{11}(\alpha_1^{new}-\alpha_1^{old})-y_2K_{21}(\alpha_2^{new}-\alpha_2^{old})+b^{old}
$$

同理，如果$0<\alpha_2^{new}<C$，则
$$
b_2^{new}=-E_2-y_1K_{12}(\alpha_1^{new}-\alpha_1^{old})-y_2K_{22}(\alpha_2^{new}-\alpha_2^{old})+b^{old}
$$
如果$\alpha_1^{new},\alpha_2^{new}$同时满足$0<\alpha_i^{new}<C,i=1,2$那么$b_1^{new}=b_2^{new}$，如果$\alpha_1^{new},\alpha_2^{new}$是0或者$C$，那么$b_1^{new}$和$b_2^{new}$以及它们之间的数都满足KKT条件，这时取它们的中点作为$b^{new}$。

##### 参考

《Pattern Recognition And Machine Learning》.Bishop.

《机器学习》.周志华.

《统计学习方法》.李航.

https://www.cnblogs.com/massquantity/p/10807311.html

https://blog.csdn.net/luoshixian099/article/details/51227754