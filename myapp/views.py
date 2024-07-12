import base64
from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
from io import BytesIO

def index(request):
    if request.method == 'POST' and request.FILES['leaf_image']:
        # Path to the saved model file (.h5)
        model_path = os.path.join(settings.BASE_DIR, 'myapp', 'static', 'file.h5')

        # Load model using TensorFlow/Keras
        model = load_model(model_path)

        # Get the uploaded image file from the form
        leaf_image = request.FILES['leaf_image']

        # Convert InMemoryUploadedFile to BytesIO
        image_stream = BytesIO(leaf_image.read())

        # Load and preprocess the image for prediction
        img = image.load_img(image_stream, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        class_names = ['early blight', 'Healthy leaves', 'late blight', 'other']  # Example class names
        prediction = class_names[predicted_class]

        # Encode image data to base64
        image_stream.seek(0)
        img_data = image_stream.read()
        img_data_b64 = base64.b64encode(img_data).decode()

        # Render the HTML template with prediction result
        return render(request, 'myapp/index.html', {
            'prediction': prediction,
            'image_url': f'data:image/jpeg;base64,{img_data_b64}'
        })

    # Render the initial form or GET request
    return render(request, 'myapp/index.html')
