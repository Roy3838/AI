import numpy as np
import cv2
import os

def generate_triangle(large_image_size, max_triangle_size):
    # Create an empty image with larger dimensions
    large_image = np.zeros((large_image_size[1], large_image_size[0]), dtype=np.uint8)
    
    # Generate random vertices for the triangle within the max_triangle_size
    vertices = np.array([[
        np.random.randint(0, max_triangle_size, size=2),
        np.random.randint(0, max_triangle_size, size=2),
        np.random.randint(0, max_triangle_size, size=2)
    ]], dtype=np.int32)

    # Offset the vertices to a random position within the larger image
    offset = np.array([np.random.randint(0, large_image_size[0] - max_triangle_size),
                       np.random.randint(0, large_image_size[1] - max_triangle_size)])
    vertices += offset

    # Draw the triangle on the larger image
    cv2.fillPoly(large_image, vertices, 255)
    return large_image, vertices

def create_dataset(num_images, large_image_size, max_triangle_size, dataset_path):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    labels = []
    for i in range(num_images):
        image, vertices = generate_triangle(large_image_size, max_triangle_size)
        file_name = f'triangle_{i}.png'
        cv2.imwrite(os.path.join(dataset_path, file_name), image)
        labels.append({'file_name': file_name, 'vertices': vertices.tolist()})

    return labels

# Parameters
num_images = 10000
large_image_size = (640, 480)  # size of the larger image (width, height)
max_triangle_size = 64  # maximum size of the triangle
dataset_path = 'data/triangles_dataset'

# Create dataset
labels = create_dataset(num_images, large_image_size, max_triangle_size, dataset_path)

# Save labels to a file
labels_file = os.path.join(dataset_path, 'labels.npy')
np.save(labels_file, labels)

print(f"Dataset created with {num_images} images and labels saved to {labels_file}")
