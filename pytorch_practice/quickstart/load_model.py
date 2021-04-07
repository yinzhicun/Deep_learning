'''
Author: yinzhicun
Date: 2021-04-07 10:35:16
LastEditTime: 2021-04-07 11:18:32
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Deep_Learning/pytorch_practice/load_model.py
'''
import sys
sys.path.append('.')
from generate_model import *

test_data = datasets.FashionMNIST(
    root = "./pytorch_practice/quickstart/data",
    train = False,
    download = True,
    transform = ToTensor(),
)

model = NeuralNetwork()
model.load_state_dict(torch.load("./pytorch_practice/quickstart/model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')