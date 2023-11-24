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

# Parameters
batch_size = 4
learning_rate = 0.001
num_epochs = 3
dataset_path = 'data/triangles_dataset'
labels = np.load(os.path.join(dataset_path, 'labels.npy'), allow_pickle=True)

# Remember to resize images to 640x480 if not already
transform = transforms.Compose([
    transforms.Resize((480, 640)),  # Resize to match input size
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

def triangle_centroid_penalty(predicted_vertices, alpha=1.0):
    """
    Calculate the penalty for predicted vertices being near the centroid of the triangle.

    :param predicted_vertices: Tensor of shape (batch_size, 6) representing the predicted vertices (x1, y1, x2, y2, x3, y3).
    :param alpha: Scaling factor for the penalty.
    :return: Calculated penalty.
    """
    # Reshape vertices to (batch_size, 3, 2) for easier manipulation
    vertices = predicted_vertices.view(-1, 3, 2)

    # Calculate the centroid of the triangle
    centroid = torch.mean(vertices, dim=1)

    # Calculate the distance of each vertex from the centroid
    distances = torch.norm(vertices - centroid.unsqueeze(1), dim=2)

    # Calculate penalty (higher when vertices are closer to centroid)
    penalty = torch.mean(distances)

    return alpha * (1 / penalty)  # Inverse of average distance as penalty

# Modify your training loop to include the penalty
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        outputs = model(images)

        # Original loss
        loss_mse = criterion(outputs, labels.float())

        # Penalty term
        penalty = triangle_centroid_penalty(outputs)

        # Total loss
        loss = loss_mse + penalty

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}')
# Save model
torch.save(model.state_dict(), 'model.pth')


print("Training complete")
