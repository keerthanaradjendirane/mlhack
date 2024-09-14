import os
import requests
from tqdm import tqdm
import pandas as pd
from urllib.parse import urlparse

# Paths
base_dir = 'D:/today'  # Base folder where everything will be saved
train_csv_path = os.path.join(base_dir, 'newtrain.csv')  # Path to train dataset
test_csv_path = os.path.join(base_dir, 'newtest.csv')  # Path to test dataset
train_image_dir = os.path.join(base_dir, 'train_images')  # Directory to save train images
test_image_dir = os.path.join(base_dir, 'test_images')  # Directory to save test images

# Create directories if they don't exist
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)

# Function to download images with error handling and progress tracking
def download_images(image_links, save_dir):
    total_images = len(image_links)
    successful_downloads = 0
    
    for idx, link in enumerate(tqdm(image_links, desc=f"Downloading images to {save_dir}")):
        try:
            # Request image data
            response = requests.get(link, stream=True, timeout=10)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Parse the image URL to extract the image filename
            parsed_url = urlparse(link)
            image_filename = os.path.basename(parsed_url.path)
            image_name = os.path.join(save_dir, image_filename)
            
            # Save the image locally
            with open(image_name, 'wb') as f:
                f.write(response.content)
            
            successful_downloads += 1
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image at index {idx}: {e}")
        
        # Display progress percentage
        percentage_completed = ((idx + 1) / total_images) * 100
        print(f"Progress: {percentage_completed:.2f}% ({idx + 1}/{total_images})")
    
    print(f"Download completed. Total successful downloads: {successful_downloads}/{total_images}")

# Load train and test datasets
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Get image links from the datasets
train_images = train_df['image_link'].tolist()
test_images = test_df['image_link'].tolist()

# Download images for train and test datasets
download_images(train_images, save_dir=train_image_dir)
download_images(test_images, save_dir=test_image_dir)

# Function to map URLs to local paths in the CSV
def map_url_to_local_path(image_url, image_dir):
    parsed_url = urlparse(image_url)
    image_filename = os.path.basename(parsed_url.path)
    
    # Construct the local image path
    local_image_path = os.path.join(image_dir, image_filename)
    
    # Check if the file exists locally
    if os.path.exists(local_image_path):
        return local_image_path
    else:
        print(f"Image not found: {local_image_path}")
        return None  # Handle missing images if needed

# Apply the mapping function to each row in the DataFrame
train_df['image_link'] = train_df['image_link'].apply(lambda url: map_url_to_local_path(url, train_image_dir))
test_df['image_link'] = test_df['image_link'].apply(lambda url: map_url_to_local_path(url, test_image_dir))

# Save the updated DataFrame with local image paths
updated_train_csv_path = os.path.join(base_dir, 'updated_amazontrain.csv')
updated_test_csv_path = os.path.join(base_dir, 'updated_amazontest.csv')

train_df.to_csv(updated_train_csv_path, index=False)
test_df.to_csv(updated_test_csv_path, index=False)

print(f"CSV updated with local image paths and saved to {updated_train_csv_path} and {updated_test_csv_path}")
