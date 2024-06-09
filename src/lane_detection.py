import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import cv2

def build_lane_detection_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    return image

def train_lane_detection_model(model, train_images, train_labels, epochs=10):
    model.fit(train_images, train_labels, epochs=epochs, validation_split=0.2)

if __name__ == "__main__":
    train_data = pd.read_csv('../data/normalized_sensor_data.csv')
    train_images = np.array([preprocess_image(file) for file in train_data['image_path']])
    train_labels = train_data['label'].values

    model = build_lane_detection_model()
    train_lane_detection_model(model, train_images, train_labels)
    model.save('../models/lane_detection_model.h5')

