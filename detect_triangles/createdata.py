import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Function to generate a random triangle and return its image and vertices
def generate_triangle(image_size):
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    vertices = np.array([[
        np.random.randint(0, image_size, size=2),
        np.random.randint(0, image_size, size=2),
        np.random.randint(0, image_size, size=2)
    ]], dtype=np.int32)
    cv2.fillPoly(image, vertices, 255)
    return image, vertices

def add_noise(image):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply distortion (wave-like effect)
    rows, cols = image.shape
    distortion = np.zeros_like(image, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(5.0 * np.sin(2 * np.pi * i / 60))
            offset_y = int(5.0 * np.cos(2 * np.pi * j / 60))
            if i + offset_y < rows and j + offset_x < cols:
                distortion[i, j] = blurred_image[(i + offset_y) % rows, (j + offset_x) % cols]
            else:
                distortion[i, j] = 0

    return distortion.astype(np.uint8)



# Function to create a dataset of triangles
def create_dataset(num_images, large_image_size,  dataset_path):

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    labels = []
    for i in range(num_images):
        image, vertices = generate_triangle(large_image_size, )
        noisy_image = add_noise(image)  # Add noise to the image
        file_name = f'triangle_{i}.png'
        cv2.imwrite(os.path.join(dataset_path, file_name), noisy_image)
        labels.append({'file_name': file_name, 'vertices': vertices.tolist()})

    return labels

# Parameters
num_images = 1000  # number of images to generate
large_image_size = 400  # size of the image
dataset_path = 'data/triangles_dataset'  # path to save the dataset

# Create dataset
labels = create_dataset(num_images, large_image_size,  dataset_path)

# Save labels to a file
labels_file = os.path.join(dataset_path, 'labels.npy')
np.save(labels_file, labels)

print(f"Dataset created with {num_images} images and labels saved to {labels_file}")
