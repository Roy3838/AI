import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Do a python code that does a Neuronal network that aproximates the sinusoide function


#uniform random inputs from -2 to 2
x_inputs = np.random.uniform(-4,4,1000)
x_inputs = torch.tensor(x_inputs)

def laguerre(x, n):
    if n == 0:
        return 1
    elif n == 1:
        return 1 - x
    else:
        return ((2*n - 1 - x)*laguerre(x, n-1) - (n - 1)*laguerre(x, n-2))/n


y_inputs = torch.sin(x_inputs)


# neural network dimensions
input_size = 1
hidden_size = 10
output_size = 1


# neural network
model = torch.nn.Sequential(
    torch.nn.Linear(input_size,hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size,output_size),
    torch.nn.Flatten(0,1)
)

# loss function
loss_func = torch.nn.MSELoss(reduction='sum')

# learning rate
learning_rate = 1e-4

# optimiser
optimiser = torch.optim.Adam(model.parameters(),lr = learning_rate)  # optim

results = []

y_history = []

# training loop
for t in range(20000):
    x = x_inputs.unsqueeze(-1)
    x = x.float()
    y_pred = model(x)
    optimiser.zero_grad()    # This is because by default, gradients are accumulated in buffers
    y_inputs = torch.sin(x_inputs)
    loss = loss_func(y_pred,y_inputs.float())
    loss.backward()
    optimiser.step()       # updates the learnable params
    results.append(loss.item())
    if (t%100 == 0):
        print(t, loss.item())
        y_pred = y_pred.detach().numpy()
        y_pred = np.sort(y_pred)
        y_history.append(y_pred)


# plot, first order data
x_inputs = x_inputs.numpy()
x_inputs = np.sort(x_inputs)
y_inputs = y_inputs.numpy()
y_inputs = np.sort(y_inputs)
y_pred = y_pred.detach().numpy()
y_pred = np.sort(y_pred)
plt.plot(x_inputs,y_inputs)
plt.plot(x_inputs,y_pred)
plt.show()

# plot loss
plt.plot(results)
plt.show()

# animation of curve fitting
fig = plt.figure()
ax = plt.axes(xlim=(-4, 4), ylim=(-1.5, 1.5))
line, = ax.plot([], [], lw=2)
def init():
    line.set_data([], [])
    return line,
def animate(i):
    x = x_inputs
    y = y_history[i]
    line.set_data(x, y)
    return line,
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=000, interval=20, blit=True)    
plt.show()
