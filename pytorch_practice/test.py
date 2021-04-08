'''
Author: yinzhicun
Date: 2021-04-07 11:10:52
LastEditTime: 2021-04-07 21:33:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Deep_Learning/pytorch_practice/test.py
'''
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data)
loss = (prediction - labels).sum()
print(prediction)

loss.backward()
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()
prediction = model(data)
print(prediction)
