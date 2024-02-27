import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt


# Data Generation
data_range = 15
x = data_range*(np.random.rand(data_range*100, 1))-data_range/2

frq = 2
lr = 2e-3
n_epochs = 15000
plot_every = n_epochs/5
y = np.sin(frq*x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

x_train_tensor = torch.from_numpy(x).float().to(device)
y_train_tensor = torch.from_numpy(y).float().to(device)

hidden = 200
hidden2 = 200

model = nn.Sequential(
  nn.Linear(1, hidden),
  nn.ReLU(),
  nn.Linear(hidden, hidden2),
  nn.ReLU(),
  nn.Linear(hidden2,1),
  
).to(device)

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=lr)

losses = []

# variables for ploting results
res = 10
x_axis = (np.arange(data_range*res)-data_range/2*res).reshape(data_range*res,1)/res
x_axis_torch = torch.from_numpy(x_axis).float().to(device)

# For each epoch...
for epoch in range(n_epochs):

    model.train()
    yhat = model(x_train_tensor)
    
    loss = loss_fn(y_train_tensor, yhat)
    loss.backward()    
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss)

    if (epoch % plot_every)==0:
      out = model(x_axis_torch).cpu().detach().numpy()
      out2 = ( np.sin(frq*x_axis_torch.cpu())) .detach().numpy()

      plt.plot(out)
      plt.plot(out2)
      plt.show()
