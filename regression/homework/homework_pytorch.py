'''
Author: your name
Date: 2021-04-08 10:16:23
LastEditTime: 2021-04-08 11:28:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Deep_Learning/regression/homework/homework_pytorch.py
'''
import pandas as pd
import numpy as np
import torch
from torch import nn

#搭建网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.regression = nn.Linear(18*9 + 1, 1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.regression(x)
        return logits
    
#读取文件
data  =  pd.read_csv('/home/yinzhicun/Deep_Learning/regression/homework/train.csv')
#去除前三列的无用信息,预处理数据
data = data.iloc[ : , 3 : ]
data[data == "NR"] = 0
#处理后每天有18个观测项, 每个月取前20天, 共有12个月
#所以一共有18 * 12 * 20 = 4320 行
raw_data = data.to_numpy() #4320 * 24
print(raw_data.shape)
#将列变为18项指标, 则最后为 18 行, 24 * 20 * 12 = 5760 列

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[ : , day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), : ]
    month_data[month] = sample #18*5760

#取前9时的数据为参数数据, 10时的PM2.5为预测数据
#一个月共有480 h，所以共有480 - 9 = 471 个标签 
x = np.empty([471 * 12, 18 * 9], dtype = float)
y = np.empty([471 * 12, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
#print(x[0])
#print(y)


#归一化处理
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

            
#加上一行模拟b
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)

x_tensor = torch.from_numpy(x)
print(x_tensor)
y_tensor = torch.from_numpy(y)
print(y_tensor)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = NeuralNetwork().to(device)
for j in range(1000):
        X, y = x_tensor.to(device), y_tensor.to(device)
        prediction = model(X)
        loss = (prediction - y)*(prediction - y) / 2
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # if j % 100 == 0:
    #     loss, current = loss.item(),j * len(X)
    #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
# testdata = pd.read_csv('/home/yinzhicun/Deep_Learning/regression/homework/test.csv', header = None, encoding = 'big5')
# test_data = testdata.iloc[:, 2:]
# test_data[test_data == 'NR'] = 0
# test_data = test_data.to_numpy()
# test_x = np.empty([240, 18*9], dtype = float)
# for i in range(240):
#     test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
# for i in range(len(test_x)):
#     for j in range(len(test_x[0])):
#         if std_x[j] != 0:
#             test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
# test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
