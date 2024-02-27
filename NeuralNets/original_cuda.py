# CODE TAKEN FROM https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
# MODIFIED TO WORK WITH CUDA

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import psutil

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)



input_size = 784
#hidden_sizes = [128, 64]
hidden_sizes = [784, 784] # saldaslkdaskd
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1)).cuda()


criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
print(labels.shape)


images = images.cuda()
labels = labels.cuda()

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15


g = torch.cuda.Stream()
g.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(g):
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1).cuda()        
            # Training pass
            optimizer.zero_grad()
            labels = labels.cuda()
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {} CPU: {}  GPU: {}".format(e, running_loss/len(trainloader), psutil.cpu_percent(), psutil.virtual_memory().percent))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)
torch.cuda.current_stream().wait_stream(g)

images, labels = next(iter(valloader))
images.cuda()
labels.cuda()


correct_count, all_count = 0, 0

for images,labels in valloader:
  for i in range(len(labels)):
    with torch.cuda.stream(g):
        img = images[i].view(1, 784)
        img = img.cuda()
        with torch.no_grad():
            logps = model(img)

        
        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
print(" CPU: {}  GPU: {}".format(psutil.cpu_percent(), psutil.virtual_memory().percent))


torch.save(model, './my_mnist_model.pt') 

