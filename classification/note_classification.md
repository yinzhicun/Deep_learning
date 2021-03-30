<!--
 * @Author: yinzhicun
 * @Date: 2021-03-29 20:39:08
 * @LastEditTime: 2021-03-30 22:10:41
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Deep_learning/classification/note.md
-->
# <center>Classification(分类)</center>

## 简介

- 预设定相应的**函数模型**－*Model*
- 设定损失函数 －*Loss function*
- 求解最优的函数模型 －*Find the best function*
- 通过函数对未知数据的类别进行预测
- 与Regression不同的是，Classification会对函数的输出进行离散化处理，以满足其特殊的要求
- 该课程中的方法针对的是对未知对象在已知类别当中进行分类
- 课程中先提出了一种基于概率分布的解决方法

### 一、函数模型以及误差函数的定义

1. 以最简单的二元分类问题为例，设定有两个类别，分别为 $C_1$，$C_2$，我们需要进行分类的随机变量是 $x$，则 $x$ 属于 $C_1$ 类的概率为：
$$P(C_1|x)=\frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}$$
$$f(x)=\begin{cases} C_2\quad P(C_1|x)<0.5\\ C_1\quad P(C_1|x)>0.5  \end{cases}$$

-  $x$ 在为哪个类别中的概率最大，则属于哪个类别

2. 定义误差函数：
$$Likelihood(f)=\sum_{i=1}^{n}{f(x^i)}$$
- 其中 $x$ 为随机变量
- $f(x)$ 为 $x$ 的概率密度函数
- 实际上就是寻找使训练集分类正确可能性最大的分布

### 二、求解函数参数

1. 假设问题服从高斯分布，则： 
$$Likelihood(\mu,\Sigma)=\sum_{i=1}^{n}{f_{\mu,\Sigma}(x^i)}$$

2. 高斯分布：
$$f_{\mu,\Sigma}(x)=\frac{1}{(2\pi)^\frac{D}{2}}\frac{1}{|\Sigma|^\frac{1}{2}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$$
- $x$ 为随机变量的向量
- $\mu$ 为均值向量
- $\Sigma$ 为协方差矩阵


![avatar](./picture/1.png)

**注：多元正态分布的概率密度是由协方差矩阵的特征向量控制旋转(rotation)，特征值控制尺度(scale)，除了协方差矩阵，均值向量会控制概率密度的位置**

3. 最大化概率求解
$$MAX\ Likelihood(\mu,\Sigma)=\sum_{i=1}^{n}{f_{\mu,\Sigma}(x^i)}$$


- 设让Likelihood最大的 $\mu$ 和 $\Sigma$ 为$\mu^*$ 和 $\Sigma^*$ ，则：
$$\mu^*=\frac{1}{n}\sum_{i=1}^nx^i$$
$$\Sigma^*=\frac{1}{n}\sum_{i=1}^n(x^i-\mu^*)(x^i-\mu^*)^T$$

### 三、线性分类判别与二次分类判别
1. 线性分类判别（**LDA**）：一定程度上简化模型，抑制overfitting

$$\begin{aligned}
P(C_1|x)&=\frac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}\\
&=\frac{1}{1+\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}}\\
&=\frac{1}{1+e^{-z}}\\
&=\sigma(z)
\end{aligned}$$
$$z=ln\frac{P(C_1|x)}{P(C_2|x)}=ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}$$
- 当两个分布具有相同的方差值时,在多维高斯分布中显示为两者的协方差矩阵相同，这时：
$$\begin{aligned}
z=ln\frac{P(C_1|x)}{P(C_2|x)}&=ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)} \\ 
&=ln\frac{f_{\mu_1,\Sigma}(x)}{f_{\mu_2,\Sigma}(x)}+ln(\frac{P(C_1)}{P(C_2)})\\
&=x^T\Sigma^{-1}(\mu_1-\mu_2)-\frac{1}{2}(\mu_1+\mu_2)\Sigma^{-1}(\mu_1-\mu_2)+ln(\frac{P(C_1)}{P(C_2)})
\end{aligned}$$
- 可以发现上式是性的，也就是 $P(C_1|x)=P(C_2|x)$ 时边界条件是线性的，即划分两个类别区域的分界线为直线


1. 二次分类判别（**QDA**）:

- 此时两个分布方差值不同，这时有分类判别函数：
$$\begin{aligned}
ln\frac{P(C_1|x)}{P(C_2|x)}&=ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)} \\ 
&=ln\frac{f_{\mu_1,\Sigma_1}(x)}{f_{\mu_2,\Sigma_2}(x)}+ln(\frac{P(C_1)}{P(C_2)})\\
&=-\frac{1}{2}x^T(\Sigma_1^{-1}-\Sigma_2^{-1})x+x^T(\Sigma_1^{-1}\mu_1-\Sigma_2^{-1}\mu_2)-\frac{1}{2}(\mu_1+\mu_2)\Sigma^{-1}(\mu_1-\mu_2)-\frac{1}{4}ln(\frac{|\Sigma_2|}{|\Sigma_1|})+ln(\frac{P(C_1)}{P(C_2)})
\end{aligned}$$

- 可以发现上式是非线性的，也就是 $P(C_1|x)=P(C_2|x)$ 时边界条件是二次型，即划分两个类别区域的分界线为曲线

3. 通过 **LDA** 简化上述模型，即令两个分布的协方差矩阵相等

- 此时误差函数为

$$Likelihood(\mu_1,\mu_2,\Sigma)=\sum_{i=1}^{k}{f_{\mu_1,\Sigma}(x^i)}+\sum_{i=k+1}^{n}{f_{\mu_2,\Sigma}(x^i)}$$

- 最大化概率求解
$$MAX\ Likelihood(\mu_1,\mu_2,\Sigma)=\sum_{i=1}^{k}{f_{\mu_1,\Sigma}(x^i)}+\sum_{i=k+1}^{n}{f_{\mu_2,\Sigma}(x^i)}$$


- 设让Likelihood最大的 $\mu$ 和 $\Sigma$ 为$\mu_1^*$，$\mu_2^*$ 和 $\Sigma^*$ ，则：
$$\mu_1^*=\frac{1}{k}\sum_{i=1}^kx^i$$
$$\mu_2^*=\frac{1}{n-k}\sum_{i=k+1}^nx^i$$
$$\Sigma_1=\frac{1}{k}\sum_{i=1}^j(x^i-\mu_1^*)(x^i-\mu1^*)^T$$
$$\Sigma_2=\frac{1}{n-k}\sum_{i=k+1}^n(x^i-\mu_2^*)(x^i-\mu2^*)^T$$
$$\Sigma^*=\frac{k}{n}\Sigma_1+\frac{n-k}{n}\Sigma_2$$

### 四、其他
1. 当多维正太分布的特征的每一个分量相互**独立**时，可以看做多个一维的高斯分布的组合，采用**朴素贝叶斯分类方法**
