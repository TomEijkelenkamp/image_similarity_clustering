# Created using chatgpt-4
import os
import shutil
import numpy as np
from scipy.spatial.distance import cosine
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import torch
import tensorflow as tf

if torch.cuda.is_available():
    device = torch.device("cuda:0")

# Load pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output, device=device)

# Paths
path_element = "single_element_images\\element_1.png"  # Path to the folder containing the collection of images
path_collection = "D:\\WikiArt\\wikiart\\"  # Path to the folder containing example images
output_folder = "output_images_contain_element\\"  # Path to the output folder for similar images
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the element image
element_img = image.load_img(path_element, target_size=(299, 299))
element_img = image.img_to_array(element_img)


def load_image(img_path):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    return img


def extract_features(img, model):
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()


def match(image, element, model, similarity_threshold):
    element_features = extract_features(element, model)
    image_features = extract_features(image, model)
    similarity = 1 - cosine(element_features, image_features)
    return similarity >= similarity_threshold


def slide(image, element, kernel_size, stride):
    for i in range(0, image.shape[0] - kernel_size + 1, stride):
        for j in range(0, image.shape[1] - kernel_size + 1, stride):
            # Extract sub-image
            sub_image = image[i:i+kernel_size, j:j+kernel_size]
            
            # Check if the sub-image matches the element
            if match(sub_image, element, model, similarity_threshold):
                return True, (i, j)


# Similarity threshold
similarity_threshold = 0.85  # Define your similarity threshold here

# Extract features from all images in the collection
element_features = extract_features(element_img, model)

for root, dirs, files in os.walk(path_collection):
    for subdir in dirs:
        for file in os.listdir(os.path.join(root, subdir)):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, subdir, file)
                
                # Load the image
                img = load_image(file_path)

                # Loop scales
                for scale in [0.5, 1.0, 1.5, 2.0]:

                    # Calculate new size
                    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                
                    # Resize image
                    scaled_img = tf.image.resize(img, new_size)
                
                    # Slide match
                    match_found, (i, j) = slide(scaled_img, element_img, 50, 10)

                    if match_found:
                        # The image is similar enough; copy it to the output folder
                        src_path = file_path
                        shutil.copy(src_path, output_folder)

            