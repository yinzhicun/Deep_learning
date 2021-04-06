<!--
 * @Author: yinzhicun
 * @Date: 2021-04-05 16:58:28
 * @LastEditTime: 2021-04-06 09:54:00
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Deep_Learning/deep_learning/note_preparation.md
-->
# <center>Preparation for DL</center>

## 简介

- 深度学习的步骤与前面的模型实际相同，分为以下三步
- 预设定相应的**函数模型**－*Model*
- 设定损失函数 －*Loss function*
- 求解最优的函数模型 －*Find the best function*

### 一、函数模型及误差函数的定义
#### 1. 为什么可以使用神经网络作为函数模型
> 数学原理：任何**连续**多元函数都能被一组一元函数的**有限次叠加**(注意这里的叠加可以是**非线性**的)而成，其中每一个一元函数的自变量都是一组连续单变量函数的**有限次加权叠加**。而这内层的每一个单变量函数的自变量都是一个（即一维）变量。
>> 例：$x \cdot y = e^{log(x+1)+log(y+1)}-(x+0.5)-(y+0.5)$

 具体参考：<https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem>

- 一开始叠加定理只说明了存在性，但是并未告诉人们怎么构造
- 原理在神经网络上体现则为：固定一种统一的**有限层**计算网络结构，调整每个节点的参数和每层节点之间叠加计算的权重，来一致逼近任意一个多元函数。当然最好每个节点的一元函数都具有**统一**的形式
- 后来George Cybenko在Approximation by Superpositions of a Sigmoidal Function 证明（存在性）只要**一个隐藏层**并使用**sigmoidal-type**函数就能一致逼近任意一个多元连续函数(引出问题 *why deep?* )
- 之后Kurt Hornik在Approximation Capabilities of Multilayer Feedforward Networks 中指出，如果仅考虑一致逼近，关键不在于sigmoidal-type的激活函数函数，而是多层网络的前馈结构。除了sigmoidal-type，也可以选择其他激活函数，只要他们在连续函数空间上**稠密**//TODO(稠密性概念)

摘自知乎<https://www.zhihu.com/question/24259872/answer/127219970>(有所删改)

- 全连接网络图
![avastar](./picture/1.png)
**总：** 上述材料说明了通过神经网络来设定函数模型的可行性，并且解释了激活函数的相关问题

#### 2. 误差函数的定义
实际上根据不同的问题误差函数的定义也是不同的，但是其本质都时output层的输出与训练标签之间差异的衡量
- 以分类问题为例，误差函数为交叉熵
  ![avastar](./picture/2.png)

### 二、求解函数参数
通过前向传播和反向传播的叠加进行求导来进梯度下降
![avastar](./picture/3.png)
#### 1. 前向传播 


### 三、为什么不使用二范数损失函数



### 四、Discriminative && Generative


### 五、其他


