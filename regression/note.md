<!--
 * @Author: your name
 * @Date: 2021-03-27 22:45:54
 * @LastEditTime: 2021-03-28 10:20:08
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /deep_learing/regression/note.md
-->

# <center>Regression (回归)</center>

## 简介

回归的步骤实际上就是：

- 预设定相应的**函数模型**
- 通过已有的数据求解函数的参数，得到函数的关系式
- 对未知的数据进行预测。

### 一、函数模型的定义

1. 定义线性模型：
   $$y=\sum_{i=1}^n{\omega_{i}\cdot\ x_{i}}+b$$
2. 定义误差函数：
   $$L(f)=\sum_{i=1}^n(\hat{y}^i-\sum_{j=1}^k\omega_j\cdot\ x_j^i-b)^2$$

- $\hat{y}^i$ 代表用来预测函数的第 $i$ 组数据的目标值
- $x^i_j$ 表示用来预测函数的第 $i$ 组数据的第 $j$ 个参数值
- $\omega_j$ 为第 $j$ 个参数值的权重

很显而易见的是 $L(f)$ 是关于 $\omega$ 和 $b$ 的函数。我们要做的就是调整 $\omega$ 和 $b$ 让误差函数最小。

### 二、求解函数参数

1. 设置函数模型为一个参数值的简单模型：
$$y=\omega\cdot x+b$$

2. 可得误差函数为：
$$L(\omega,b)=\sum_{i=1}^n(\hat{y}^i-(\omega\cdot x^i+b))^2$$

3. 采用梯度下降法求解
$$\frac{\partial{L}}{\partial{\omega}}=2\sum_{i=1}^n(\hat{y}^i-(\omega\cdot x^i+b))\cdot (-x^i)$$
$$\frac{\partial{L}}{\partial{b}}=2\sum_{i=1}^n(\hat{y}^i-(\omega\cdot x^i+b))\cdot (-1)$$

&emsp;则有:

