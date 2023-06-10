from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing import image
from google.cloud import storage
from datetime import datetime


import os
import numpy as np

CLOUD_STORAGE_BUCKET = "nst-bucket-dev-env"

app = Flask(__name__)
model = load_model('model5.h5')
target_img = os.path.join(os.getcwd(), 'static/images')


@app.route('/')
def index_view():
    return 'API is Running'


ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


def read_image(filename):
    image_url = get_file(origin=filename)
    img = load_img(image_url, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.vstack([x])
    return x


@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        storage_client = storage.Client.from_service_account_json(
            'serviceaccountkey.json')

        bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)

        if file and allowed_file(file.filename):

            getDate = datetime.now()

            file.filename = os.path.join(getDate.strftime(
                "%d_%m_%Y_%H_%M_%S") + '.' + file.filename.rsplit('.', 1)[1])

            file_path = os.path.join('static/images', file.filename)
            file.save(file_path)

            blob = bucket.blob(file.filename)
            blob.upload_from_filename(
                os.path.join('static/images', file.filename))

            # remove image after upload to Cloud Storage
            os.remove(os.path.join('static/images', file.filename))

            img_url = "https://storage.googleapis.com/{}/{}".format(
                CLOUD_STORAGE_BUCKET, file.filename)

            print(img_url)

            remove_white_space = img_url.replace(" ", "%20")

            img = read_image(remove_white_space)

            class_prediction = model.predict(img)

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
