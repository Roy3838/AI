import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim



transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])


trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)



input_size = 784
hidden_sizes = [784, 10]
output_size = 10


model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
model = model.cuda()

loss_fn = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

#place images on GPU
images = images.cuda()
#place labels on GPU
labels = labels.cuda()

logps = model(images) #log probabilities
loss = loss_fn(logps, labels) #calculate the NLL loss
# place loss on the GPU



print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)


optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 10000


s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    cosoloss= 0
    for i in range(epochs):
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad(set_to_none=True)

        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        cosoloss += loss.item()
        if i % 1000 == 0:
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Loss: {:.4f}".format(cosoloss/1000))
            cosoloss = 0
        #print(f"Training loss: {cosoloss/len(trainloader)}")
torch.cuda.current_stream().wait_stream(s)


# g = torch.cuda.CUDAGraph()
# optimizer.zero_grad(set_to_none=True)
# with torch.cuda.graph(g):
#     output = model(images)
#     static_loss = loss_fn(output, labels)
#     static_loss.backward()
#     optimizer.step()



# for e in range(epochs):
#     running_loss = 0
#     for images, labels in trainloader:
#         # Flatten MNIST images into a 784 long vector
#         images = images.view(images.shape[0], -1)
    
#         # Training pass
#         optimizer.zero_grad()
        
#         output = model(images)
#         loss = loss_fn(output, labels)
        
#         #This is where the model learns by backpropagating
#         loss.backward()
        
#         #And optimizes its weights here
#         optimizer.step()
        
#         running_loss += loss.item()
#     else:
#         print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
# print("\nTraining Time (in minutes) =",(time()-time0)/60)


images, labels = next(iter(valloader))
images.cuda()
labels.cuda()

# with torch.no_grad():
#     logps = model(img)

# ps = torch.exp(logps)
# probab = list(ps.numpy()[0])
# print("Predicted Digit =", probab.index(max(probab)))
correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    with torch.cuda.stream(s):
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