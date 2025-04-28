import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D class to handle 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')  # Remove the 'groups' argument if it exists
        super().__init__(**kwargs)

# Load the pre-trained model
def load_trained_model(model_path):
    # Use custom objects to replace DepthwiseConv2D with the custom one
    custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

# Function for prediction
def predict_bread_condition(model, image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    # Predict the condition of the bread
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Interpret the predicted class
    labels = {
        0: "It's a fresh bread, use it before a week.",
        1: "It's an expired bread, throw it!",
        2: "It has mold on the right side, use the left side.",
        3: "It has mold on the left side, use the right side.",
        4: "It has mold on the top side, use the bottom side.",
        5: "It has mold on the bottom side, use the top side."
    }

    return labels[predicted_class]

if __name__ == "__main__":
    # Install Pillow if not installed
    try:
        import PIL
    except ImportError:
        raise ImportError("PIL is required. Install it by running `pip install Pillow`.")

    # Load the trained model
    model_path = 'keras_model.h5'  # Path to your saved model
    model = load_trained_model(model_path)

    # Example prediction
    image_path = r'uploads/Copy_of_a.jpg'  # Path to the test image
    result = predict_bread_condition(model, image_path)
    print(result)
