from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("model/model_sampah.h5")

classes = ["anorganik", "organik", "toxic"]

def preprocess(image):

    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)

    return image


@app.route("/", methods=["GET","POST"])
def index():

    prediction = None

    if request.method == "POST":

        file = request.files["file"]

        image = Image.open(file)

        img = preprocess(image)

        pred = model.predict(img)

        prediction = classes[np.argmax(pred)]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)