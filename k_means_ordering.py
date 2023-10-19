import os
import shutil
import numpy as np
from scipy.spatial.distance import cosine
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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

# Path to the folder containing images
folder_path = "generated_images/"

# Get a list of image paths from the folder
image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Extract features for each image
image_features = [extract_features(img_path, model) for img_path in image_paths]

# Compute similarity using cosine similarity
similarities = np.zeros((len(image_paths), len(image_paths)))

for i in range(len(image_paths)):
    for j in range(len(image_paths)):
        similarities[i][j] = 1 - cosine(image_features[i], image_features[j])

# Number of clusters (you may need to tune this)
num_clusters = 5

# Perform clustering using K-means
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(similarities)

# Create a directory to store images for each cluster
output_directory = "output_clusters/"
os.makedirs(output_directory, exist_ok=True)

# Copy images to corresponding cluster folders
for cluster in range(num_clusters):
    cluster_images = [image_paths[i] for i in range(len(image_paths)) if cluster_labels[i] == cluster]
    cluster_folder = os.path.join(output_directory, f'cluster_{cluster}')
    os.makedirs(cluster_folder, exist_ok=True)
    
    # Copy images to the cluster folder
    for img_path in cluster_images:
        shutil.copy(img_path, cluster_folder)
