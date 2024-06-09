import cv2
import numpy as np
import tensorflow as tf
from gps_module import get_gps_data

# Load models
lane_model = tf.keras.models.load_model('../models/lane_detection_model.h5')
object_model = tf.keras.models.load_model('../models/object_detection_model.h5')

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    return np.expand_dims(image, axis=0)

def autopilot_system(image_path):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)

    lane_prediction = lane_model.predict(processed_image)
    object_prediction = object_model.predict(processed_image)

    gps_data = get_gps_data()

    if lane_prediction > 0.5:
        print("Turning Right")
    else:
        print("Turning Left")

    if object_prediction > 0.5:
        print("Object Detected - Slowing Down")

    print(f"Current GPS location: {gps_data}")

if __name__ == "__main__":
    test_image = '../data/test_image.jpg'
    autopilot_system(test_image)

