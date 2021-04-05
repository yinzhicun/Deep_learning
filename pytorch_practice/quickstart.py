'''
Author: yinzhicun
Date: 2021-04-04 21:04:55
LastEditTime: 2021-04-04 21:06:52
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Deep_Learning/pytorch_practice/quickstart.py
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

