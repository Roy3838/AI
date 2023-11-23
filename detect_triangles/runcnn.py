import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from model import TriangleNet

# Load the trained model (make sure to provide the path to your model's state_dict)
model = TriangleNet()
print("model initialized")
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Transformation for the image (should be the same as used during training)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load an image
image_path = '/home/jay/repos/AI/data/triangles_run/photo1700692109.jpeg'
image = Image.open(image_path).convert('L')
image = transform(image)
image = image.unsqueeze(0)  # Add a batch dimension

# Predict the position of the triangle
with torch.no_grad():
    predictions = model(image)

# Convert predictions to numpy array
predicted_vertices = predictions.numpy().flatten()

# Post-process predictions if necessary
# For example, if you had normalized the positions, you would need to scale them back up

# Here you might want to scale back the predictions to the original image size
# This depends on how you preprocessed the labels during the training
# If you normalized the vertices to be between 0 and 1, you would do the opposite here

# Assuming the image is not normalized and is 64x64 as in the dataset creation example:
predicted_vertices = predicted_vertices #* image_size  # Scale the predictions if they were normalized

print(f"Predicted vertices: {predicted_vertices}")



# Convert predictions to numpy array and reshape
predicted_vertices = predictions.numpy().flatten()
predicted_vertices = predicted_vertices.reshape(-1, 2)  # Reshape to Nx2 for N vertices

# Read the original image with OpenCV
original_image = cv2.imread(image_path)

# Scale the predicted vertices if required (depends on image preprocessing)
# For demonstration, assuming the image is 64x64 and no scaling is required
scale_factor = 1  # Change this factor according to your preprocessing steps

# Draw the predicted vertices on the image
for vertex in predicted_vertices:
    x, y = (vertex * scale_factor).astype(int)
    cv2.circle(original_image, (x, y), 5, (0, 255, 0), -1)

# Show the image with predicted vertices
cv2.imshow('Predicted Triangles', original_image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

# If you want to save the result
#cv2.imwrite('predicted_image.png', original_image)