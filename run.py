import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

in_data = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
inTensor = torch.Tensor(np.array(in_data))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        return x

model = torch.load('cnn.pt')
model.eval()

out = model(inTensor)

print(out)