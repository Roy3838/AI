import torch
from torchvision import transforms
from PIL import Image
from model import TriangleNet
import numpy as np
import cv2
from draw_on_image import *

# Load the trained model
model_path = r'C:\Users\royme\AI\model2.pth'  # Path to the trained model
model = load_model(model_path)

# Path to the image you want to test
test_image_path = r'C:\Users\royme\AI\data\triangles_test\triangle_8.png'

# Predict the vertices
vertices = predict_vertices(model, test_image_path)
print("Predicted vertices:", vertices)

# Predict the vertices
vertices = predict_vertices(model, test_image_path)
print("Predicted vertices:", vertices)

# Draw and display the vertices on the original image
draw_vertices_on_image(test_image_path, vertices)