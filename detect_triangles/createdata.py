import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

MAX_TRIANGLE = 60
ANGLE_SIMILARITY_THRESHOLD = 20  # Degrees within which angles should be similar

# Function to generate a random triangle and return its image and vertices
def add_square(image):
    """
    Receives an image and adds a square to it sometimes
    """
    num_squares = np.random.randint(0, 4)
    for _ in range(num_squares):
        x = np.random.randint(0, image.shape[0] - 1)
        y = np.random.randint(0, image.shape[1] - 1)
        size = np.random.randint(10, 20)
        image[x:x+size, y:y+size] = 255
        
    return image

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
            return add_square(image), vertices

# Plot vectorial field meshgrid offset_x and offset_y




def add_noise_and_distort_vertices(image, vertices):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Shape of the image
    rows, cols= image.shape  

    # Generate offset arrays
    offset_x = (5.0 * np.sin(2 * np.pi * np.arange(rows)[:, None] / 60)).astype(int)
    offset_y = (5.0 * np.cos(2 * np.pi * np.arange(cols) / 60)).astype(int)

    # Apply offsets to the image
    distorted_indices_x = (np.arange(rows)[:, None] + offset_x) % rows
    distorted_indices_y = (np.arange(cols) + offset_y) % cols
    distortion = blurred_image[distorted_indices_x, distorted_indices_y]


    # First, reshape vertices for easier handling, remove the unnecessary first dimension
    vertices = vertices.reshape((-1, 2))  # This will make vertices a (3, 2) array

    # Initialize an array to store the distorted vertices
    distorted_vertices = np.zeros_like(vertices)

    # Process each vertex individually
    for i, (y, x) in enumerate(vertices):
        # Apply the distortion offsets directly to each vertex
        # Since the offsets correspond to specific positions, we directly use them
        # Ensure that we also account for the image's dimensions to avoid out-of-bounds indices
        distorted_x = (x + offset_y[x % cols]) % cols
        distorted_y = (y + offset_x[y % rows]) % rows
        
        # Store the distorted vertex
        distorted_vertices[i] = [distorted_y, distorted_x]

    # Reshape distorted_vertices back to original shape if needed
    distorted_vertices = distorted_vertices.reshape((1, -1, 2))


    # # Adjust vertices handling based on the provided structure
    # new_vertices = np.zeros_like(vertices)
    # for i, vertex_group in enumerate(vertices):
    #     for j, vertex in enumerate(vertex_group):
    #         x, y = vertex
    #         # Ensure the vertex position is within the image bounds
    #         if 0 <= x < rows and 0 <= y < cols:
    #             distorted_x = (x + offset_x[x, 0]) % rows
    #             distorted_y = (y + offset_y[y]) % cols
    #             new_vertices[i][j] = [distorted_x, distorted_y]
    #         else:
    #             # If the vertex is out of bounds after distortion, handle as needed
    #             new_vertices[i][j] = vertex

    return blurred_image.astype(np.uint8), vertices



# Function to create a dataset of triangles
def create_dataset(num_images, large_image_size,  dataset_path):

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    labels = []
    for i in range(num_images):
        image, vertices = generate_triangle(large_image_size, )
        noisy_image, new_vertices = add_noise_and_distort_vertices(image, vertices)  # Add noise to the image
        file_name = f'triangle_{i}.png'
        cv2.imwrite(os.path.join(dataset_path, file_name), noisy_image)
        labels.append({'file_name': file_name, 'vertices': new_vertices.tolist()})
        # print Dataset size
        if i % 100 == 0:
            print(i/num_images*100, '%', end='\r')
            print(f"Generated {i} images")

    return labels

# Parameters
num_images = 40000  # number of images to generate
large_image_size = 100  # size of the image
dataset_path = 'data/triangles_dataset'  # path to save the dataset

# Delete the dataset folder if it exists using rm -rf 
os.system(f'rm -rf {dataset_path}')


# Create dataset
labels = create_dataset(num_images, large_image_size,  dataset_path)

# Save labels to a file
labels_file = os.path.join(dataset_path, 'labels.npy')
np.save(labels_file, labels)

print(f"Dataset created with {num_images} images and labels saved to {labels_file}")
