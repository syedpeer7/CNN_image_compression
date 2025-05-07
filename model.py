# Install dependencies (if needed)
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab import files
import os

# Upload images manually
uploaded = files.upload()

# Create a folder for images
os.makedirs("images", exist_ok=True)

# Save uploaded images to the folder
for filename in uploaded.keys():
    file_path = os.path.join("images", filename)
    with open(file_path, "wb") as f:
        f.write(uploaded[filename])

# Load and preprocess images
def load_images(image_folder="images", img_size=(64, 64)):
    images = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        img = cv2.resize(img, img_size) / 255.0  # Resize & normalize
        images.append(img)
    return np.array(images).reshape(-1, 64, 64, 1)

# Load images
X_train = load_images()

# Define the Autoencoder model
input_img = Input(shape=(64, 64, 1))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# Decoder
x = UpSampling2D((2, 2))(encoded)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Compile Model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the Autoencoder
autoencoder.fit(X_train, X_train, epochs=500, batch_size=2, shuffle=True)

# Test Image Compression & Reconstruction
encoded_imgs = autoencoder.predict(X_train)
decoded_imgs = autoencoder.predict(encoded_imgs)

# Display Results
n = len(X_train) # Number of images to show
plt.figure(figsize=(10, 4))
for i in range(n):
    # Original Image
    plt.subplot(2, n, i + 1)
    plt.imshow(X_train[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
    plt.title("Original")

    # Reconstructed Image
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
    plt.title("Reconstructed")

plt.show()
