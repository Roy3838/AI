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


N, D_in, H, D_out = 64, 784, 128, 10

model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.Dropout(p=0.2),
                            torch.nn.Linear(H, D_out),
                            torch.nn.Dropout(p=0.1)).cuda()
print(model)

images = torch.randn(N, D_in, device='cuda')
labels = torch.randn(N, D_out, device='cuda')




criterion = nn.NLLLoss()
logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()

#warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
torch.cuda.current_stream().wait_stream(s)


#capture
g = torch.cuda.CUDAGraph()
# Sets grads to None before capture, so backward() will create
# .grad attributes with allocations from the graph's private pool
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred = model(images)
    static_loss = criterion(static_y_pred, labels)
    static_loss.backward()
    optimizer.step()


images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

images = images.cuda()
labels = labels.cuda()



for data, target in zip(images, labels):
    # Fills the graph's input memory with new data to compute on
    images.set_(data)
    labels.set_(target)
    # replay() includes forward, backward, and step.
    # You don't even need to call optimizer.zero_grad() between iterations
    # because the captured backward refills static .grad tensors in place.
    g.replay()
    # Params have been updated. static_y_pred, static_loss, and .grad
    # attributes hold values from computing on this iteration's data.











# images, labels = next(iter(valloader))
# images.cuda()
# labels.cuda()


# correct_count, all_count = 0, 0

# for images,labels in valloader:
#   for i in range(len(labels)):
#     with torch.cuda.stream(g):
#         img = images[i].view(1, 784)
#         img = img.cuda()
#         with torch.no_grad():
#             logps = model(img)

        
#         ps = torch.exp(logps)
#         probab = list(ps.cpu().numpy()[0])
#         pred_label = probab.index(max(probab))
#         true_label = labels.numpy()[i]
#         if(true_label == pred_label):
#             correct_count += 1
#         all_count += 1

# print("Number Of Images Tested =", all_count)
# print("\nModel Accuracy =", (correct_count/all_count))


# torch.save(model, './my_mnist_model.pt') 

