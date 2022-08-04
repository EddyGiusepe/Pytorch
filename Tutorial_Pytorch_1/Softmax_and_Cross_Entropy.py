'''
Data Scientist: Dr.Eddy Giusepe Chirinos Isidro
'''
import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
'''
Em x --> é obvio: 2.0 maior prob: [0.65900114 0.24243297 0.09856589]
'''
x = np.array([2.0, 1.0, 0.1]) 
outputs = softmax(x)
print("Softmax numpy: ", outputs)

print("Podemos fazer isso com Pytorch, assim:")

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

