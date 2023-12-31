import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from model import TriangleNet

# Custom Dataset class
class TriangleDataset(Dataset):
    def __init__(self, dataset_path, labels, transform=None):
        self.dataset_path = dataset_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = os.path.join(self.dataset_path, label['file_name'])
        image = Image.open(img_path).convert('L')
        vertices = np.array(label['vertices']).flatten()

        if self.transform:
            image = self.transform(image)

        return image, vertices

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("CUDA not available. Using CPU.")

# Instantiate the model and move it to the appropriate device
model = TriangleNet().to(device)


batch_size = 5
learning_rate = 0.001
num_epochs = 14
dataset_path = 'data/triangles_dataset'
labels = np.load(os.path.join(dataset_path, 'labels.npy'), allow_pickle=True)

# Remember to resize images to 640x480 if not already
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # Resize to match input size
    transforms.ToTensor(),
    # Add any additional transformations if needed
])

# Create the dataset and data loader
dataset = TriangleDataset(dataset_path, labels, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model
model = TriangleNet()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        # Move data to the appropriate device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels.float())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}')


# Save model
# Make sure to move the model back to CPU before saving
model.to('cpu')
torch.save(model.state_dict(), 'model.pth')

print("Training complete")
