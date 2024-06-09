def normalize_image(image):
    return (image - image.min()) / (image.max() - image.min())

def resize_image(image, size=(128, 128)):
    return cv2.resize(image, size)

