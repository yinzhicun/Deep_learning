'''
Author: yinzhicun
Date: 2021-04-07 11:10:52
LastEditTime: 2021-04-07 11:28:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Deep_Learning/pytorch_practice/test.py
'''
import torch
import numpy as np

tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)