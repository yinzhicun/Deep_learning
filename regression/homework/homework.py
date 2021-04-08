'''
Author: your name
Date: 2021-03-28 15:35:45
LastEditTime: 2021-04-08 10:45:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Deep_Learning/regression/homework/homework.py
'''
import pandas as pd
import numpy as np
#读取文件
data  =  pd.read_csv('/home/yinzhicun/Deep_Learning/regression/homework/train.csv')
#去除前三列的无用信息,预处理数据
data = data.iloc[ : , 3 : ]
data[data == "NR"] = 0
#处理后每天有18个观测项, 每个月取前20天, 共有12个月
#所以一共有18 * 12 * 20 = 4320 行
raw_data = data.to_numpy()
print(raw_data.shape)
#将列变为18项指标, 则最后为 18 行, 24 * 20 * 12 = 5760 列
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[ : , day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), : ]
    month_data[month] = sample

#取前9时的数据为参数数据, 10时的PM2.5为预测数据
#一个月共有480 h，所以共有480 - 9 = 471 个标签 
x = np.empty([471 * 12, 18 * 9], dtype = float)
y = np.empty([471 * 12, 1], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
#print(x)
#print(y)

#归一化处理
mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]



dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
print(x.shape)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    if(t%100==0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('/home/yinzhicun/Deep_Learning/regression/homework/weight.npy', w)

# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv('/home/yinzhicun/Deep_Learning/regression/homework/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('/home/yinzhicun/Deep_Learning/regression/homework/weight.npy')
ans_y = np.dot(test_x, w)
print(ans_y)
