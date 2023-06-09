from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing import image
from google.auth import credentials
from google.cloud import storage


import os
import numpy as np

CLOUD_STORAGE_BUCKET = "nst-bucket-dev-env"

app = Flask(__name__)
model = load_model('model5.h5')
target_img = os.path.join(os.getcwd(), 'static/images')


@app.route('/')
def index_view():
    return 'API is Running'


# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

# Function to load and prepare the image in right shape


def read_image(filename):
    img = load_img(filename, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.vstack([x])
    return x


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        # Create a Cloud Storage client.
        storage_client = storage.Client()

        # Get the bucket that the file will be uploaded to.
        bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)

        # Create a new blob and upload the file's content.
        blob = bucket.blob(file.filename)
        blob.upload_from_string(file.read(), content_type=file.content_type)

        if file and allowed_file(file.filename):
            # filename = file.filename
            img_url = "gs://{}/{}".format(CLOUD_STORAGE_BUCKET, blob.name)
            # file_path = os.path.join('static/images', filename)
            # file.save(file_path)
            # img = read_image(file_path)
            class_prediction = model.predict(img_url)
            if class_prediction == 0:
                result = "Gigi Sehat"
            elif class_prediction == 1:
                result = "Gigi Tidak Sehat"
            else:
                result = "Silakan Memotret Gigi Anda dengan Benar"

            data = {
                "status": True,
                "data": result,
            }
            return jsonify(data)
        else:
            return jsonify({"message": "Not support type of file"})


if __name__ == '__main__':
    app.run(debug=True, port=8080)
