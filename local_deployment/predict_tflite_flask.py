# # **Using the Model as a web service**
from flask import Flask
from flask import request
from flask import jsonify

# Imports (notice NO tensorflow)
import pickle
import numpy as np
from PIL import Image
import tensorflow.lite as tflite


# Constants
LARGE_IMAGE_SIZE = 224  # Size that the base model has as default input size
IMG_SIZE = (LARGE_IMAGE_SIZE,LARGE_IMAGE_SIZE)

# preprocessing without Tensorflow
# See: https://github.com/qubvel/efficientnet/blob/master/efficientnet/model.py
#  and https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L18
#if mode == 'torch':
#        x /= 255.
#        mean = [0.485, 0.456, 0.406]
#        std = [0.229, 0.224, 0.225]
def preprocess_input(x):
    x /= 255.
    #mean = [0.485, 0.456, 0.406]
    #std = [0.229, 0.224, 0.225]
    return x

# load_img >> this replaces the load_image from the tf/keras packages
def load_img(img_path, img_size):
    #img_path: path including file name
    # img_size: tuple, e.g. (LARGE_IMAGE_SIZE,LARGE_IMAGE_SIZE)
    with Image.open(img_path) as img:
        img = img.resize(img_size, Image.NEAREST)
        x = np.array(img, dtype = 'float32')
        return x


# Load the TFLite model (converted from the keras model)
interpreter = tflite.Interpreter(model_path = "EfficientNetB3_large_07_0.927.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Load the classes
# The classes were also saved previously as pkl, so we can use them as such:
with open('encoded_classes.pkl', 'rb') as inp:
    encoded_classes = pickle.load(inp)

# serving the model
app = Flask("animal_prediction")

@app.route("/predict", methods = ["POST"])
def predict():
    print("Predicting...")
    #image_path = "./img/African Painted Dogs 0187 - Grahm S. Jones, Columbus Zoo and Aquarium.jpg"
    image_path = request.get_json()
    # Load the images using the new load_img() function already returns an np.array(dtype='float32')
    x = load_img(image_path, IMG_SIZE)
    print("\nImage: ",image_path)

    X = preprocess_input(x)
    X = np.expand_dims(X, 0) # expand dimensions (make it "batch")
    #X = np.array([X])  # expand dimensions
    
    # This was using the keras model:
    #pred = best_EfficientNetB3_model.predict(X)
    # This is actual predictions using the tflite model:
    interpreter.set_tensor(input_index,X)
    interpreter.invoke()
    pred_tflite = interpreter.get_tensor(output_index)
    # The rest is the same:
    y_pred = np.argmax(pred_tflite, axis = 1)
    predicted_class = y_pred
    predicted_proba = pred_tflite.flatten()[y_pred][0]
    predicted_label = encoded_classes[y_pred[0]]
    #print("Predicted class: ", y_pred[0])
    #print("Predicted label: ",predicted_label)
    #print("Predicted proba: ", predicted_proba)

    #print("Done.")

    result = {
        "class_label": predicted_label,
        "class_probability": float(predicted_proba)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug = True, host = "127.0.0.1", port = 9696)