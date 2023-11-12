import os
import shutil
import numpy as np
from scipy.spatial.distance import cosine
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model

# Load pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

# Function to preprocess an image and extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()
    
# Path to the folder containing images and path to the example image
folder_path = "generated_images_materials_n_shapes_2\\"
example_image_path = "generated_images_materials_n_shapes_2\\image_cone_cardboard_73.png"  # Replace with your example image path

# Get a list of image paths from the folder
image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Extract features for the example image
example_features = extract_features(example_image_path, model)

# Extract features for each image and compute similarity to the example image
image_features = [extract_features(img_path, model) for img_path in image_paths]
similarities = [1 - cosine(example_features, img_feature) for img_feature in image_features]

# Sort image paths based on similarities
sorted_indices = np.argsort(similarities)[::-1]  # In descending order
sorted_image_paths = [image_paths[i] for i in sorted_indices]
sorted_similarities = [similarities[i] for i in sorted_indices]

# Create 8 clusters
num_clusters = 8
cluster_size = len(sorted_image_paths) // num_clusters

# Create a directory to store images for each cluster
output_directory = "output_clusters_cone_cardboard\\"
os.makedirs(output_directory, exist_ok=True)

# Copy images to corresponding cluster folders with similarity score in filename
for cluster in range(num_clusters):
    start_idx = cluster * cluster_size
    end_idx = start_idx + cluster_size if cluster != num_clusters - 1 else len(sorted_image_paths)  # All remaining images for the last cluster
    
    cluster_images = sorted_image_paths[start_idx:end_idx]
    cluster_similarities = sorted_similarities[start_idx:end_idx]
    
    cluster_folder = os.path.join(output_directory, f'cluster_{cluster}')
    os.makedirs(cluster_folder, exist_ok=True)
    
    for img_path, similarity in zip(cluster_images, cluster_similarities):
        new_img_name = f"{similarity:.4f}_{os.path.basename(img_path)}"
        shutil.copy(img_path, os.path.join(cluster_folder, new_img_name))
