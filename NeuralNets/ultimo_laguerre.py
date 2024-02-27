import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

n = 4
nTrainingValues = 1000
X = np.linspace(-5,10,nTrainingValues)
Y = np.zeros(nTrainingValues)

def L(l,x):
    if l == 0:
        return 1
    elif l == 1:
        return 1 - x
    else:
        return ((2*l - 1 - x)*L(l-1,x) - (l - 1)*L(l-2,x))/l


for orden_n in range(n):
    for i in range(nTrainingValues):
        Y[i] = L(orden_n,X[i])
    plt.plot(X,Y)

plt.show()






