import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

MAX_TRIANGLE = 30
ANGLE_SIMILARITY_THRESHOLD = 20  # Degrees within which angles should be similar

# Function to generate a random triangle and return its image and vertices

def generate_triangle(image_size):
    def calculate_angle(v1, v2, v3):
        # Calculate the angle at v2 formed by the line segments v2v1 and v2v3
        a = np.linalg.norm(v1 - v2)
        b = np.linalg.norm(v3 - v2)
        c = np.linalg.norm(v3 - v1)
        # Use the cosine rule
        angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
        return np.degrees(angle)

    while True:
        image = np.zeros((image_size, image_size), dtype=np.uint8)

        # Function to generate a vertex within MAX_TRIANGLE distance
        def generate_nearby_vertex(origin, max_distance, image_size):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, max_distance)
            x = int(origin[0] + radius * np.cos(angle))
            y = int(origin[1] + radius * np.sin(angle))
            # Ensure the vertex is within image boundaries
            x = np.clip(x, 0, image_size - 1)
            y = np.clip(y, 0, image_size - 1)
            return np.array([x, y])

        # Generate the vertices
        v1 = np.random.randint(0, image_size, size=2)
        v2 = generate_nearby_vertex(v1, MAX_TRIANGLE, image_size)
        v3 = generate_nearby_vertex(v1, MAX_TRIANGLE, image_size)

        # Calculate angles
        angle1 = calculate_angle(v2, v1, v3)
        angle2 = calculate_angle(v1, v2, v3)
        angle3 = calculate_angle(v1, v3, v2)

        # Check if angles are similar
        if np.all(np.abs([angle1, angle2, angle3] - np.mean([angle1, angle2, angle3])) < ANGLE_SIMILARITY_THRESHOLD):
            vertices = np.array([[v1, v2, v3]], dtype=np.int32)
            cv2.fillPoly(image, vertices, 255)
            return image, vertices

def add_noise(image):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Shape of the image
    rows, cols = image.shape

    # Generate offset arrays
    offset_x = (5.0 * np.sin(2 * np.pi * np.arange(rows)[:, None] / 60)).astype(int)
    offset_y = (5.0 * np.cos(2 * np.pi * np.arange(cols) / 60)).astype(int)

    # Apply offsets
    distorted_indices_x = (np.arange(rows)[:, None] + offset_x) % rows
    distorted_indices_y = (np.arange(cols) + offset_y) % cols

    # Use advanced indexing to create the distorted image
    distortion = blurred_image[distorted_indices_x, distorted_indices_y]

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
num_images = 10000  # number of images to generate
large_image_size = 100  # size of the image
dataset_path = 'data/triangles_dataset'  # path to save the dataset

# Create dataset
labels = create_dataset(num_images, large_image_size,  dataset_path)

# Save labels to a file
labels_file = os.path.join(dataset_path, 'labels.npy')
np.save(labels_file, labels)

print(f"Dataset created with {num_images} images and labels saved to {labels_file}")
