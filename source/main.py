import io
import numpy as np
import tensorflow as tf
import imageio.v2 as iio
from PIL import Image
from flask import Flask, request, jsonify
import logging
logging.basicConfig(level=logging.DEBUG)


# load model 
model = tf.keras.models.load_model('./retinal_oct_api-main/model/retinal-oct-adam.h5')


# prepare images 
def prepare_image(img):
    """
    prepares the image for the api call
    """
    img = Image.open(io.BytesIO(img)).convert('RGB')
    img = img.resize((150, 150))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


# prediction
def predict_result(img):
    """predicts the result"""
    return np.argmax(model.predict(img)[0])


# initialize flask object
app = Flask(__name__)


# setting up routes and their functions
'''@app.route('/predict', methods=['POST'])
def infer_image():
    logging.info(str(request.files))
    
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The image doesn't exist"
    
    file = request.files.get('file')
    
    if not file:
        print("bob")
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return str(predict_result(img))
    '''

@app.route('/predict', methods=['POST'])
def bulk_infer_image():
    logging.info(str(request.files))
    
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The image doesn't exist"

    files = request.files.getlist('file')
    if not files:
        return
    
    batch_result = []
    for file in files:
        # Read the image
        img_bytes = file.read()

        # Prepare the image
        img = prepare_image(img_bytes)

        # Append all prediction data to dictionnary
        batch_result.append({'file': file.filename, 'prediction': int(predict_result(img))})

    # Return full dictionnary     
    return jsonify(batch_result)

@app.route('/', methods=['GET'])
def index():
    return 'Retinal OCT prediction API'

# run the API
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)




