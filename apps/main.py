from flask import Flask, render_template, send_from_directory, url_for, request
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.fields.simple import SubmitField
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asldfkjlj'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Show Image')

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

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    file_url = None
    prediction = None

    # Load the trained model once to be reused for predictions
    model_path = 'keras_model.h5'  # Update this with your model path
    model = load_trained_model(model_path)

    if form.validate_on_submit():
        # When the user uploads an image
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)

        # Preprocess the uploaded image
        img_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)

        # Make prediction
        prediction = predict_bread_condition(model, img_path)

    return render_template('index.html', form=form, file_url=file_url, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
