import torch
from torchvision import datasets, transforms
import psutil
import time

"""     CARGA DE DATOS   SACADO DE https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627"""   
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=3, shuffle=True)

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1).cuda()



"""     DISEÑO DE LA RED  AUTOR: ROY MEDINA """

N, D_in, H, out, D_out = 3, 784, 784, 28, 10
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                      torch.nn.ReLU(),
                      torch.nn.Linear(H, out),
                      torch.nn.ReLU(),
                      torch.nn.Linear(out, D_out),
                      torch.nn.LogSoftmax(dim=1)).cuda()

loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

static_input = images[:N].cuda()

static_target = torch.zeros(N, D_out, device='cuda')
for i in range(N):
    static_target[i][labels[i]] = 1

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

"""             GPU CATCH               """

print("Attempting GPU Catch")

s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(static_input)
        loss = loss_fn(y_pred, static_target)
        loss.backward()
        optimizer.step()
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
# Sets grads to None before capture, so backward() will create
# .grad attributes with allocations from the graph's private pool
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred = model(static_input)
    static_loss = loss_fn(static_y_pred, static_target)
    static_loss.backward()
    optimizer.step()

"""                     TRAINING            """

N = 64*160*4

print("GPU Catched, Loading Dataset")

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=N, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=N, shuffle=True)


images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
real_inputs = images[:N].cuda()
real_targets = torch.zeros(N, D_out, device='cuda')

print("Dataset Loaded, Starting Training")

for i in range(N):
    real_targets[i][labels[i]] = 1

tiempo1 = time.time()

# MAIN TRAINING LOOP
iterations = 2
for i in range(iterations):
    for data, target in zip(real_inputs, real_targets):
        # Fills the graph's input memory with new data to compute on
        static_input.copy_(data)
        static_target.copy_(target)
        # replay() includes forward, backward, and step.
        g.replay()

    print("Iteration : {}/{} ".format(i+1,iterations))
print("Training Time: " + str(time.time() - tiempo1))





"""     EVALUAR MODELO   SACADO DE https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627"""   

images, labels = next(iter(valloader))
images.cuda()
labels.cuda()

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

torch.save(model, './my_mnist_model.pt') 


