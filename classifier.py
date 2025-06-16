from keras.models import load_model
from keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps
import numpy as np
import warnings


# Patch the DepthwiseConv2D config before loading model
def patch_depthwise_conv2d():
    original_from_config = DepthwiseConv2D.from_config

    def patched_from_config(config):
        if 'groups' in config:
            config.pop('groups')  # Remove the problematic argument
        return original_from_config(config)

    DepthwiseConv2D.from_config = patched_from_config


class ImageClassifier:
    def __init__(self, model_path="keras_model.h5", labels_path="labels.txt"):
        # Apply the patch before loading model
        patch_depthwise_conv2d()

        # Suppress warnings during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = load_model(model_path, compile=False)

        self.class_names = open(labels_path, "r").readlines()
        self.input_shape = (224, 224)

    def predict(self, image: Image.Image) -> tuple:
        """Run prediction on a single image"""
        # Preprocess image
        img = ImageOps.fit(image, self.input_shape, Image.Resampling.LANCZOS)
        img_array = np.asarray(img)
        normalized = (img_array.astype(np.float32) / 127.5) - 1

        # Prepare batch
        data = np.expand_dims(normalized, axis=0)

        # Predict
        predictions = self.model.predict(data)
        index = np.argmax(predictions)

        return (
            self.class_names[index].strip(),  # Class name
            float(predictions[0][index]),  # Confidence score
            predictions[0].tolist()  # All probabilities
        )