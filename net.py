import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from random import randint

from tqdm import trange

import numpy as np


# load the mnist dataset

data = []

for i in range(100):
  value = []
  labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  for t in range(10):
    random = randint(0, 10)
    if random==1 or random==2 or random==3 or random==4 or random==5 or random==6 or random==7:
      value.append(1)
    else:
      value.append(0)
  labels = value.count(1)
  data.append((value, labels))


x = []
y = []
for i in range(len(data)):
  x.append(data[i][0])
  y.append(data[i][1])

xTensor = torch.Tensor(np.array(x))
yTensor = torch.Tensor(np.array(y))


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(10, 1)

    def forward(self, x):
        out = self.l1(x)
        return out



model = Net()

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in (t := trange(1000)):
    for i in range(len(data)):
        values = xTensor
        labels = yTensor
        output = model(values)

        optimizer.zero_grad()

        # get output from the model, given the inputs

        # get loss for the predicted output
        loss = criterion(output, labels.unsqueeze(1))
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

print('finished')

torch.save(model, "cnn.pt")
