import unittest
from lane_detection import build_lane_detection_model, preprocess_image

class TestLaneDetectionModel(unittest.TestCase):

    def test_model_output_shape(self):
        model = build_lane_detection_model()
        self.assertEqual(model.output_shape, (None, 1))

    def test_preprocess_image(self):
        image = preprocess_image('../data/test_image.jpg')
        self.assertEqual(image.shape, (1, 128, 128, 3))

if __name__ == "__main__":
    unittest.main()

