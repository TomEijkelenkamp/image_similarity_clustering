# Created using chatgpt-4
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

# Paths
path_element = "single_element_images\\element_1.png"  # Path to the folder containing the collection of images
path_collection = "D:\\WikiArt\\wikiart\\"  # Path to the folder containing example images
output_folder = "output_images_contain_element\\"  # Path to the output folder for similar images

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Similarity threshold
similarity_threshold = 0.85  # Define your similarity threshold here

# Extract features from all images in the collection
element_features = extract_features(path_element, model)

for root, dirs, files in os.walk(path_collection):
    for subdir in dirs:
        for file in os.listdir(os.path.join(root, subdir)):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, subdir, file)
                example_features = extract_features(file_path, model)

# Process each example image
for example_image_filename in os.listdir(path_collection):
    if example_image_filename.endswith(('.jpg', '.jpeg', '.png')):
        example_image_path = os.path.join(example_images_folder_path, example_image_filename)
        
        # Extract features for the example image
        example_features = extract_features(example_image_path, model)
        
        # Go through the collection and copy images that meet the similarity threshold
        for img_filename, img_features in collection_features.items():
            if img_filename not in copied_images:
                similarity = 1 - cosine(example_features, img_features)
                
                if similarity >= similarity_threshold:
                    # The image is similar enough; copy it to the output folder
                    src_path = os.path.join(collection_folder_path, img_filename)
                    shutil.copy(src_path, output_folder)
                    copied_images.add(img_filename)
