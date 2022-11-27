'''
Data Scientist: Dr.Eddy Giusepe Chirinos Isidro
'''
import torch
import torch.nn as nn
import numpy as np

from torchmetrics.functional import auroc


import warnings
warnings.filterwarnings("ignore")


loss = nn.CrossEntropyLoss()

y = torch.tensor([2, 0, 1]) ## 3 samples
#y = torch.tensor([0])

# nsamples x nclasses = 1x3
## nsamples x nclasses = 3x3
y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])


l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)

print(l1.item()) # Melhor prediction é com baixa Loss
print(l2.item())

# Predictions
_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)

print(predictions1)
print(predictions2)


auroc = auroc(y_pred_good, y, num_classes=3).item()
print(auroc)