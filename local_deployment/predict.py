import pickle
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import load_img  # pip install pillow, then restart the kernel

from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
LARGE_IMAGE_SIZE = 224  # Size that the base model has as default input size

print(" -- PREDICT ANIMAL CATEGORY USING EfficientNetB3 -- ")
print("Loading the model...")

best_EfficientNetB3_model = keras.models.load_model('EfficientNetB3_large_07_0.927.keras')
#best_EfficientNetB3_model.summary()

print("Loading test animals...")
test_sample_animals = [
    "./img/animals/duck/72decb40db.jpg",
    "./img/animals/crow/03c7cea31f.jpg",
    "./img/animals/possum/3f9a0f744a.jpg",
    "./img/animals/bison/4d50fcf8a7.jpg",
    "./img/animals/pelecaniformes/105d4649cc.jpg",
    "./img/animals/deer/66c58b431f.jpg"
]

# The classes were also saved previously as pkl, so we can use them as such:
with open('encoded_classes.pkl', 'rb') as inp:
    encoded_classes = pickle.load(inp)

print("Predicting...")
for i, sample_path in zip(np.arange(len(test_sample_animals)), test_sample_animals):
    test_sample_img = load_img(sample_path)
    print("\nSAMPLE image: ",sample_path)
    x = np.array(test_sample_img)
    #print(x.shape)
    x = center_crop_and_resize(x, image_size=LARGE_IMAGE_SIZE)
    #print(x.shape)
    X = preprocess_input(x)
    X = np.expand_dims(X, 0) # expand dimensions
    #X = np.array([X])  # expand dimensions
    
    #print("\t", X.shape)
    pred = best_EfficientNetB3_model.predict(X)
    y_pred = np.argmax(pred, axis = 1)
    predicted_class = y_pred
    predicted_proba = np.asarray(tf.reduce_max(pred)).flatten()[0]
    predicted_label = encoded_classes[y_pred[0]]
    print("Predicted class: ", y_pred[0])
    print("Predicted label: ",predicted_label)
    print("Predicted proba: ", predicted_proba)

print("Done.")