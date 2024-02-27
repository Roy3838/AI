import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import animation
from numpy.polynomial import Laguerre 

# 
def f_x(x, n = 1):
    # laguerre polynomials at x
    # N is the degree of the polynomial
    # x is a single value
    return np.polynomial.laguerre.lagval(x, np.eye(1, n+1, n)[0]) # L_n(x)

Degree = 4
data_range = 20
# x from -data_range/2 to data_range/2
x = data_range*(np.random.rand(data_range*100, 1))-data_range/2
# x but from -1 to 5

n_epochs = 1500
plot_every = n_epochs/2
y = f_x(x, Degree)

x_train_tensor = torch.from_numpy(x).float().cuda()
y_train_tensor = torch.from_numpy(y).float().cuda()

input, mid, out = 1, 200, 1

model = nn.Sequential(
  nn.Linear(input, mid),
  nn.ReLU(),
  nn.Linear(mid,out),
  
).cuda()

loss_fn = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=1e-3)

res = 10
x_axis = (np.arange(data_range*res)-data_range/2*res).reshape(data_range*res,1)/res
x_axis_torch = torch.from_numpy(x_axis).float().cuda()

outs = []
outs2 = []

running_loss = 0
graphloss = []

for epoch in range(n_epochs):

    model.train()
    yhat = model(x_train_tensor)
    
    loss = loss_fn(y_train_tensor, yhat)
    loss.backward()    
    


    optimizer.step()
    optimizer.zero_grad()


    if (epoch % plot_every)==0:
        out = model(x_axis_torch).cpu().detach().numpy()
        out2 = ( f_x(x_axis_torch.cpu(), Degree)) .detach().numpy()

plt.plot(out)
plt.plot(out2)
plt.title('Laguerre polynomials Degree = ' + str(Degree))
plt.show()






