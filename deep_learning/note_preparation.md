<!--
 * @Author: yinzhicun
 * @Date: 2021-04-05 16:58:28
 * @LastEditTime: 2021-04-06 21:51:17
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
**总**： 上述材料说明了通过神经网络来设定函数模型的可行性，并且解释了激活函数的相关问题

#### 2. 误差函数的定义
实际上根据不同的问题误差函数的定义也是不同的，但是其本质都时output层的输出与训练标签之间差异的衡量
- 以分类问题为例，误差函数为交叉熵
  ![avastar](./picture/2.png)

### 二、求解函数参数
通过**前向传播**和**反向传播**的叠加进行求导来进梯度下降
![avastar](./picture/3.png)
#### 1. 前向传播 
- 使用梯度下降法的关键就是求解误差函数的导数，即：
$$\frac{\partial{L(\theta)}}{\partial{\omega}}=\sum_{n=1}^N{\frac{\partial{C^n(\theta)}}{\partial{\omega}}}$$
![](./picture/4.png)

- 截取上图中网络的一部分，通过链式法则将 $\frac{\partial{C}}{\partial{\omega}}$ 拆成了如图所示的两项
![](./picture/5.png)

- 其中 $\frac{\partial{z}}{\partial{\omega}}$ 的结果可根据网络权重直接得出，而求解每个节点的 $\frac{\partial{z}}{\partial{\omega}}$ 过程即为前向传播
  
#### 2. 反向传播
- 如通过前向传播求得 $\frac{\partial{z}}{\partial{\omega}}$ 之后，还剩下  $\frac{\partial{C}}{\partial{z}}$ ，再次拆分可得：
$$\frac{\partial{C}}{\partial{z}}=\frac{\partial{C}}{\partial{z}}\frac{\partial{a}}{\partial{\omega}}$$ 
- 如下图，其中 $\frac{\partial{a}}{\partial{\omega}}$ 为激活函数的导数，通过前向传播中求出的 $z$ 值直接求得
![](./picture/6.png) 

- 剩余的 $\frac{\partial{C}}{\partial{z}}$ 可从output端反向推出，如下图所示
![](./picture/7.png)
![](./picture/8.png)

- 如图，要求当前结点的导数就是需要当前结点的前向传播结果和下级结点的反向传播结果，反向计算即可得到答案

**总**：通过前向传播和反向传播得到导数之后，正常按照梯度下降法求解即可

### 三、Tips
宏观上看看训练这件事
![](./picture/9.png)
- 几种调节模型的方法，如下图
![](./picture/10.png)

#### 1. 新的激活函数(举例)
- **原因**：因为一部分数据对于变化更加敏感，而另一部分权重对变化不敏感导致的**梯度消失**问题
![](./picture/11.png)
![](./picture/12.png)

- **解决方法**：替换ReLU激活函数，来改变sigmoid函数性质上的缺陷
![](./picture/13.png)

##### 1.1 ReLU是MaxOut的特例
- MaxOut的激活函数实际上就是一个选择最大值输出的开关，其特殊指出在于，激活函数也是可以**通过训练**得到的
![](./picture/14.png)
- 当MaxOut函数的输入值为线性函数时，可以转化为下图所示的样子，也就是ReLU函数
![](./picture/15.png)

#### 1.2 训练MaxOut的一些问题
- 实际上输入不同的时候，每一个MaxOut单元做出的输出也不同，也就是说随着训练的进行，连接MaxOut节点的每一条网络都可以被训练到，所以不会存在漏训练的问题
![](./picture/17.png)

### 2. 通过对学习率的优化改善梯度下降法
#### 2.1  Adagrad 
**原因**：不同方向上的下降速度不同，需要不同的学习率
- 由于一般来说 $\eta$ 会随着像最优点的靠近越来越小所以：
$$\omega^{t+1}=\omega^t-\eta^t\nabla{L}^t$$
$$其中\quad\eta^t = \frac{\eta}{\sqrt{t+1}}$$

- 用对一次微分的二范数来代替难以计算的二次微分
$$\omega^{t+1}=\omega^t-\frac{\eta}{\sqrt{\sum_{i=0}^t(\nabla{L}^i)^2}}\nabla{L}^t$$

#### 2.2 RMSProp
**原因**：同一方向上的下降速度也会不同，需要不同的学习率
![](./picture/18.png)

- 实现如下，$\alpha$ 值越大则越相信过去的值，$\alpha$ 值越小则越相信现在的值
![](./picture/19.png)

#### 2.3 Momentum
**原因**：一定程度上解决局部最优的问题，实际上可以看做引入了惯性的概念
![](./picture/20.png)
![](./picture/21.png)

#### 2.4 Adam
**原因**：RMSProp与Momentum相结合
![](./picture/22.png)

### 3. 训练到适当的时候停止
**原因**：抑制overfitting
通过有有标签的测试集实验选择更好的样本数

### ４. 正则化
**原因**：泛化特征，平滑曲线
![](./picture/23.png)
![](./picture/24.png)


### 四、Discriminative && Generative


### 五、其他


