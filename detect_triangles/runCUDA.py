import torch
from torchvision import transforms
from PIL import Image
from model import TriangleNet
import numpy as np
import cv2

def load_model(model_path):
    # Load the trained model
    model = TriangleNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def process_image(image_path):
    # Transform the image to match the input format of the model
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict_vertices(model, image_path):
    # Process the image
    image = process_image(image_path)

    # Predict the vertices
    with torch.no_grad():
        vertices = model(image).numpy()
    
    return vertices.flatten()

# Load the trained model
model_path = r'C:\Users\royme\AI\model.pth'  # Path to the trained model
model = load_model(model_path)

# Path to the image you want to test
test_image_path = r'C:\Users\royme\AI\data\triangles_test\triangle_9.png'

# Predict the vertices
vertices = predict_vertices(model, test_image_path)
print("Predicted vertices:", vertices)

def draw_vertices_on_image(image_path, vertices, model_input_size=100):
    # Load the original image
    original_image = cv2.imread(image_path)
    original_height, original_width = original_image.shape[:2]

    # Scaling factors for width and height
    scale_x = original_width / model_input_size
    scale_y = original_height / model_input_size

    # Rescale vertices
    scaled_vertices = [(int(vertices[i] * scale_x), int(vertices[i + 1] * scale_y)) for i in range(0, len(vertices), 2)]

    # Draw circles at the rescaled vertices
    for x, y in scaled_vertices:
        cv2.circle(original_image, (x, y), 3, (0, 255, 0), -1)

    # Show the image
    cv2.imshow('Image with Predicted Vertices', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Predict the vertices
vertices = predict_vertices(model, test_image_path)
print("Predicted vertices:", vertices)

# Draw and display the vertices on the original image
draw_vertices_on_image(test_image_path, vertices)