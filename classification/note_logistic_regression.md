<!--
 * @Author: yinzhicun
 * @Date: 2021-04-03 09:29:50
 * @LastEditTime: 2021-04-06 08:41:35
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Deep_learning/classification/note_logistic_regression.md
-->
# <center>Logistic_Regression(逻辑回归)</center>

## 简介

- 预设定相应的**函数模型**－*Model*
- 设定损失函数 －*Loss function*
- 求解最优的函数模型 －*Find the best function*
- 通过函数对未知数据的类别进行预测
- 基于概率分布的解决方法之上采用了回归模型

### 一、函数模型以及误差函数的定义

1. 以最简单的二元分类问题为例，设定有两个类别，分别为 $C_1$，$C_2$，我们需要进行分类的随机变量是 $x$，则 $x$ 属于 $C_1$ 类的概率为：
$$P(C_1|x)=\frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}$$
$$f(x)=\begin{cases} C_2\quad P(C_1|x)<0.5\\ C_1\quad P(C_1|x)>0.5  \end{cases}$$

- 这时代入classification中的计算模型，有：
$$\begin{aligned}
P(C_1|x)&=\frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}\\
&=\frac{1}{1+\frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}}\\
&=\frac{1}{1+e^{-z}}\\
&=\sigma(z)
\end{aligned}$$
$$其中\ z=ln\frac{P(C_1|x)}{P(C_2|x)}=ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}$$

- 代入高斯分布函数有：
$$\begin{aligned}
z&=ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)} \\ 
&=ln\frac{f_{\mu_1,\Sigma_1}(x)}{f_{\mu_2,\Sigma_2}(x)}+ln(\frac{P(C_1)}{P(C_2)})\\
&=-\frac{1}{2}x^T(\Sigma_1^{-1}-\Sigma_2^{-1})x+x^T(\Sigma_1^{-1}\mu_1-\Sigma_2^{-1}\mu_2)-\frac{1}{2}(\mu_1+\mu_2)\Sigma^{-1}(\mu_1-\mu_2)+\frac{1}{2}ln(\frac{|\Sigma_2|}{|\Sigma_1|})+ln(\frac{P(C_1)}{P(C_2)})
\end{aligned}$$

- 当两个分布具有相同的方差值时,在多维高斯分布中显示为两者的协方差矩阵相同，这时：
$$\Sigma_1=\Sigma_2=\Sigma$$
$$\begin{aligned}
代入上式则：z&=x^T\Sigma^{-1}(\mu_1-\mu_2)-\frac{1}{2}(\mu_1+\mu_2)\Sigma^{-1}(\mu_1-\mu_2)+ln(\frac{P(C_1)}{P(C_2)})\\
&=(\mu_1-\mu_2)^T\Sigma^{-1}x-\frac{1}{2}(\mu_1+\mu_2)\Sigma^{-1}(\mu_1-\mu_2)+ln(\frac{P(C_1)}{P(C_2)})\\
&=w^Tx+b
\end{aligned}$$
$$其中\ w=(\mu_1-\mu_2)^T\Sigma^{-1}$$
$$而\ b=\frac{1}{2}(\mu_1+\mu_2)\Sigma^{-1}(\mu_1-\mu_2)+ln(\frac{P(C_1)}{P(C_2)})$$

- 写成非向量形式：
$$z=\sum_{i=1}^n{\omega_{i}\cdot\ x_{i}}+b$$
- 所以可得：
$$\begin{aligned}
P(C_1|x)&=f(x)\\
&=\sigma(z)\\
&=\frac{1}{1+e^{-(\sum_{i=1}^n{\omega_{i}\cdot\ x_{i}}+b)}}
\end{aligned}$$
  
- 到了这里就可以 $w$ 和 $b$ 为参数进行回归运算，如果以**网络**的思想来看的话，就是在一层全联接层之后又加上了一个**sigmod的激活函数**将输入限制再在(0,1)中

2. 定义误差函数：

$$Likelihood(f)=\sum_{i=1}^{k}{f(x^i)}\cdot\sum_{i=k+1}^{n}{(1-f(x^i))}$$

- 其中 $x$ 为随机变量
- $f(x)$ 为 $P(C_1|x)$，$1-f(x)$ 为 $P(C_2|x)$
- 1-k组的训练数据是 $C_1$ 一类的，而k+1-n组得训练数据是 $C_2$ 一类的
- 实际上就是寻找使训练集分类正确可能性最大的分布

### 二、求解函数参数

1. ***cross entropy*** 交叉熵:
![avastar](./picture/2.png)
![avastar](./picture/3.png)
![avastar](./picture/4.png)
![avastar](./picture/5.png)
图片均取自知乎（https://www.zhihu.com/question/65288314/answer/244557337）

2. 设如果 $x$ 的类别属于 $C_1$ 则 $\hat{y}=1$; 如果 $x$ 的类别属于 $C_2$ 则 $\hat{y}=0$，这时对一中的误差函数两边取对数则有：
$$-ln(L(x))=-\sum_{i=1}^n{[\hat{y}^ilnf(x^i)+(1-\hat{y}^i)ln(1-f(x^i))]}$$

- 这实际上就是一个交叉熵公式,而我们要做的就是使其最小

3. 梯度下降
![avastar](./picture/6.png)
- 对比线性回归模型发现，其导数相同
![avastar](./picture/7.png)



### 三、为什么不使用二范数损失函数
1. 当结果很接近最佳概率时二范数损失函数导数接近于0，当结果原理时其导数依然接近于0；所以为了寻求更好的效果选择交叉熵作为损失函数
![avastar](./picture/8.png)
- 可视化效果如下图所示
![avastar](./picture/9.png)


### 四、Discriminative && Generative
1. 两种方法使用同一种模型，而求得参数的方法不同
![avastar](./picture/10.png)
2. 实际上数据量较小时，生成模型更加优异；而数据量大时判别模型更加优异

### 五、其他
1. 通过**Sorfmax**进行多分类问题的求解
2. 逻辑回归对于异或门一类的问题是有局限的，需要进行Transform ——> 引出了神经网络（神经网络可以看作让机器对特征做transform）


