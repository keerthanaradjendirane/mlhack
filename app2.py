# Adding necessary imports
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from io import BytesIO
import requests

# Constants
IMAGE_SIZE = (224, 224)  # Define according to your model's expected input size
TRAIN_CSV = "D:/amazon/updated_amazontrain.csv"  # Path to your training labels CSV
TEST_CSV = "D:/amazon/updated_amazontest.csv"  # Path to your test labels CSV
MODEL_PATH = "D:/amazon/resnet50_amazon_model.h5"  # Path to your trained model
OUTPUT_CSV = 'D:/today/prediction_output.csv'  # Path for saving predictions

def load_data(csv_file, has_labels=True):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = row['image_link']
        try:
            # For URLs
            if img_path.startswith('http'):
                response = requests.get(img_path)
                img = load_img(BytesIO(response.content), target_size=IMAGE_SIZE)
            else:
                img = load_img(img_path, target_size=IMAGE_SIZE)
            img = img_to_array(img)
            images.append(img)
            if has_labels:
                labels.append(row['entity_value'])
        except Exception as e:
            print(f"Error loading image from {img_path}: {e}")
    
    images = np.array(images)
    if has_labels:
        labels = np.array(labels)
        return images, labels
    return images, df

def train_model(train_csv):
    X_train, y_train = load_data(train_csv)
    
    # Define your model here
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')  # Adjust according to your number of classes
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Adjust if using one-hot encoding
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=32)  # Adjust epochs and batch size
    
    model.save(MODEL_PATH)  # Save the trained model

def make_predictions(test_csv, model_path):
    X_test, test_df = load_data(test_csv, has_labels=False)
    
    model = load_model(model_path)
    y_pred = model.predict(X_test)
    
    # Assuming your model outputs probabilities and you need to get class indices
    y_pred_classes = np.argmax(y_pred, axis=1)  # Get class indices
    
    # Optionally, load LabelBinarizer if you need to map indices to class names
    # Uncomment and fit on training labels if needed
    # lb = LabelBinarizer()
    # X_train, y_train = load_data(TRAIN_CSV)
    # lb.fit(y_train)
    
    # If lb is not fitted, use `y_pred_classes` directly
    test_df['entity_value'] = y_pred_classes
    
    # Create a DataFrame with only index and entity_value
    output_df = test_df[['index', 'entity_value']]
    
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to {OUTPUT_CSV}")

# Main execution
if not os.path.exists(MODEL_PATH):
    print("Model not found. Training model...")
    train_model(TRAIN_CSV)
else:
    print("Model found. Making predictions...")
    make_predictions(TEST_CSV, MODEL_PATH)
