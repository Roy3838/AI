import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from model import *
from draw_on_image import *
from itertools import permutations
import matplotlib.pyplot as plt

def plot_vertices(fig, ax, image, real_vertices, predicted_vertices):
    
    ax.clear()  # Clear the current axes

    # Convert tensor image to numpy and plot
    image = image.squeeze().numpy()  # Remove channel and batch dimensions
    ax.imshow(image, cmap='gray')

    # Plot real vertices
    ax.scatter(real_vertices[:, 0], real_vertices[:, 1], color='green', label='Real')

    # Plot predicted vertices
    ax.scatter(predicted_vertices[:, 0], predicted_vertices[:, 1], color='red', label='Predicted')

    ax.legend()
    plt.draw()  # Redraw the current figure
    plt.pause(0.001)  # Pause briefly to allow the plot to update
    


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

def custom_loss(predicted, target):
    """
    Custom loss function to calculate the MSE for all permutations of predicted vertices
    and return the minimum MSE.
    """
    predicted = predicted.view(-1, 3, 2)  # Assuming the shape of predicted is [batch_size, 6]
    target = target.view(-1, 3, 2)       # Reshape target similarly

    # Initialize min_loss as a tensor with a large value
    min_loss = torch.tensor(float('inf')).to(predicted.device)

    for perm in permutations([0, 1, 2]):
        # Apply the permutation to the predicted vertices
        permuted_predicted = predicted[:, list(perm)]
        # Calculate MSE loss for this permutation
        loss = torch.mean((permuted_predicted - target) ** 2)
        min_loss = torch.min(min_loss, loss)
    
    return min_loss

# Parameters
batch_size = 4
learning_rate = 0.001
num_epochs = 5
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
model = EnhancedTriangleNet()

# Loss and optimizer
criterion = custom_loss
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

def dynamic_gaussian_triangle_centroid_penalty(predicted_vertices, alpha=1.0):
    """
    Calculate the Gaussian penalty for predicted vertices being near the centroid of the triangle.
    The standard deviation of the Gaussian depends on the average distance of the vertices from the centroid.

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

    # Use the average distance as the standard deviation for the Gaussian
    sigma = torch.mean(distances, dim=1)

    # Apply Gaussian penalty
    gaussian_penalty = torch.exp(-(distances / sigma.unsqueeze(1)) ** 2)

    # Sum penalties and apply scaling factor
    total_penalty = alpha * torch.mean(gaussian_penalty)

    return total_penalty

def modified_gaussian_penalty(predicted_vertices, real_vertices, alpha=1.0):
    """
    Apply a Gaussian penalty based on the distance of the predicted vertices from the centroid of the real triangle.

    :param predicted_vertices: Tensor of shape (batch_size, 6) representing the predicted vertices.
    :param real_vertices: Tensor of shape (batch_size, 6) representing the real vertices.
    :param alpha: Scaling factor for the penalty.
    :return: Calculated penalty.
    """
    # Reshape vertices to (batch_size, 3, 2) for easier manipulation
    predicted_vertices = predicted_vertices.view(-1, 3, 2)
    real_vertices = real_vertices.view(-1, 3, 2)

    # Calculate the centroid of the real triangle
    real_centroid = torch.mean(real_vertices, dim=1)

    # Calculate the distance of each predicted vertex from the real centroid
    distances = torch.norm(predicted_vertices - real_centroid.unsqueeze(1), dim=2)

    # Use the average distance of the real vertices from their centroid as the standard deviation for the Gaussian
    real_distances = torch.norm(real_vertices - real_centroid.unsqueeze(1), dim=2)
    sigma = torch.mean(real_distances, dim=1)/4

    # Apply Gaussian penalty
    gaussian_penalty = torch.exp(-(distances / sigma.unsqueeze(1)) ** 2)

    # Sum penalties and apply scaling factor
    total_penalty = alpha * torch.mean(gaussian_penalty)

    return total_penalty


plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()




# Modify your training loop to include the penalty
for epoch in range(num_epochs):
    for i, (images, real_vertices) in enumerate(data_loader):
        # Forward pass: Compute predicted output by passing images to the model
        predicted_vertices = model(images)

        # Convert both predicted and real vertices to float for loss calculation
        predicted_vertices = predicted_vertices.float()
        real_vertices = real_vertices.float()

        # Original loss: Mean Squared Error
        loss_mse = criterion(predicted_vertices, real_vertices)

        # Penalty term
        penalty = modified_gaussian_penalty(predicted_vertices, real_vertices, 100)

        # Total loss: MSE loss + Gaussian penalty
        loss = loss_mse + penalty

        # Backward pass: Compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}, MSE Loss: {loss_mse.item()}, Penalty: {penalty.item()}')
            # Select the first image and its vertices in the batch for plotting
            
            image_to_plot = images[0].cpu().detach()
            real_vertices_to_plot = real_vertices[0].cpu().detach().view(-1, 2)
            predicted_vertices_to_plot = predicted_vertices[0].cpu().detach().view(-1, 2)

            # Call the plotting function
            plot_vertices(fig, ax, image_to_plot, real_vertices_to_plot, predicted_vertices_to_plot)

            

    

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

print("Training complete")