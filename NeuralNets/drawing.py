import torch
from torchvision import datasets, transforms
import psutil
import time
import csv
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=3, shuffle=True)

# graph first image

#print(trainset[10][0].numpy().squeeze().shape)
data = trainset[9][0]

plt.imshow(data.numpy().squeeze(), cmap='gray_r')

# load torch model (model, './my_mnist_model.pt') 

model = torch.load('./my_mnist_model.pt').cpu()

# get prediction of first image

xd = data.view(1, 784)
logps = model(xd)
ps = torch.exp(logps)
probab = list(ps.detach().numpy()[0])
numero =probab.index(max(probab))

# label predicted number
stringcoso = "Predicted Number: " + str(numero)
plt.title(stringcoso)
plt.show()
