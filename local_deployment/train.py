# C01 - Animal Classification using CNN with Transfer Learning
# Dataset:
# https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/data  
# 
# 5400 Animal Images in 90 different categories
## IMAGES MUST BE SAVED IN THE FOLLOWING DIRECTORY:
# "./img/animals/"

# # **Training the model**


# ## Imports
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Pretrained cnn
# if you use tensorflow.keras: 
from efficientnet.tfkeras import EfficientNetB3
from efficientnet.tfkeras import preprocess_input


print(" -- ANIMAL IMAGE CLASSIFICATION USING EfficientNetB3 (TRAIN) -- ")
# **VARIABLES**

# %%
# * num_classes: Number of classes in the dataset
# * learning_rate: the learning rate
# * dropout_rate: the dropout rate
# * size_inner: the number of neurons in the inner layer
# * img_size: the image size (basically, small(150) or large(224))
# * batch_normalization: boolean that indicates whether to use the batch normalization or not
# * augmentation_layers: sequence of layers to try augmentation inside the model

SMALL_IMAGE_SIZE = 150  # 224 takes more to train, so train on smaller images
LARGE_IMAGE_SIZE = 224  # Size that the base model has as default input size
NUM_CLASSES = 90 # 90 animals in ghe dataset
# Obtained experimentally
BEST_LEARNING_RATE = 0.001
BEST_INNER_SIZE = 100
BEST_DROP_RATE = 0.25
BATCH_NORMALIZE = True
BATCH_SIZE = 15

# %%
print("Loading images...")
path = "./img/animals/"

# %%
data = {"path": [] , "label": [] }
categories = os.listdir(path)
for folder in categories:
    folder_path = os.path.join(path , folder)
    files = os.listdir(folder_path)
    for file in files:
        fpath = os.path.join(folder_path, file)
        data["path"].append(fpath)
        data["label"].append(folder)

df = pd.DataFrame(data) 

# Encode labels
print("Reading and encoding labels...")
lbl_encoder = LabelEncoder()
df['encoded_label'] = lbl_encoder.fit_transform(df['label'])

# ## Pretrained CNN model (EfficientNet)  
# Although Xception is a good Deep Learning architecture, [EfficientNet](https://pypi.org/project/efficientnet/) has become the go-to architecture for many challenging tasks, particularly in object recognition applications.  
#   
# See more: https://blog.roboflow.com/what-is-efficientnet/#:~:text=EfficientNet%20addresses%20this%20challenge%20by,of%20efficiency%20without%20compromising%20accuracy.  

# ## Train-Test splitting
print("Splitting data...")
# %%
train_df, tmp_df = train_test_split(df,  train_size= 0.60 , shuffle=True, random_state=0)
val_df , test_df = train_test_split(tmp_df ,  train_size= 0.50 , shuffle=True, random_state=0)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train ({train_df.shape[0]})")
print(train_df[["path", "label"]].head(3))

print(f"\nValidation ({val_df.shape[0]})")
print(val_df[["path", "label"]].head(3))

print(f"\nTest ({test_df.shape[0]})")
print(test_df[["path", "label"]].head(3))


# ## Transfer learning  
# ### Image generators (large image size)

# model definition function with data augmentation
# (includes batch normalization, regularization, an inner layer,
# fine tuned learning rate and data augmentation)
def extend_model_v4(num_classes         = NUM_CLASSES,          # default to NUM_CLASSES
                    img_size            = SMALL_IMAGE_SIZE,     # default to SMALL_IMAGE_SIZE
                    learning_rate       = BEST_LEARNING_RATE,   # default to BEST_LEARNING_RATE
                    size_inner          = BEST_INNER_SIZE,      # default to BEST_INNER_SIZE
                    drop_rate           = BEST_DROP_RATE,       # default to BEST_DROP_RATE
                    batch_normalization = True                  # default to true
                ):
    # pre-trained model with the same img size as the specified
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )

    # Freeze layers
    base_model.trainable = False

    #########################################
    inputs =        keras.layers.Input(shape=(img_size, img_size, 3), name = "inputLayer")
    base =          base_model(inputs, training=False)
    vectors =       keras.layers.GlobalAveragePooling2D()(base)
    # inner layer
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    # batch normalizaiton
    if(batch_normalization):
        batch_normalized = keras.layers.BatchNormalization()(inner)
        # dropout layer
        drop = keras.layers.Dropout(drop_rate)(batch_normalized)
    else:
        # dropout layer
        drop = keras.layers.Dropout(drop_rate)(inner)
        
    # outputs as raw numbers
    raw_outputs =   keras.layers.Dense(num_classes)(drop)
    # outputs as probabilities
    outputs =       keras.layers.Activation(
        # this returns probabilities
        activation="softmax", dtype=tf.float32, name='softMaxLayer' 
        )(raw_outputs)

    # final model architecture:
    model = keras.Model(inputs, outputs)

    #########################################
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)  # We will use softmax

    # compiled model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model

# %%

# ### TRAINING WITH CHECKPOINTING
# We will train the model with the best parameters on large image generators
# ## Larger image generators
print("Building image generators...")
BATCH_SIZE = 15
IMAGE_SIZE = (LARGE_IMAGE_SIZE, LARGE_IMAGE_SIZE)

gen_large = ImageDataGenerator(preprocessing_function = preprocess_input)

# data is in the dataframes
train_ds_large = gen_large.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=IMAGE_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=0,
)

val_ds_large = gen_large.flow_from_dataframe(
    dataframe=val_df,
    x_col='path',
    y_col='label',
    target_size=IMAGE_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds_large = gen_large.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=IMAGE_SIZE,
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ## Model definition, training and checkpointing
# save with checkpointing
print("Building checkpoint function...")
checkpoint_large = keras.callbacks.ModelCheckpoint(
    'EfficientNetB3_large_{epoch:02d}_{val_accuracy:.3f}_(from_train_py).keras', # .h5 is legacy
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# %%
print("Training with checkpointing...")
num_epochs = 20
model_large = extend_model_v4(
    # Though all the parameters are defaulted to the best, include them here for clarity
    num_classes         = NUM_CLASSES,
    img_size            = LARGE_IMAGE_SIZE,
    learning_rate       = BEST_LEARNING_RATE,
    drop_rate           = BEST_DROP_RATE,
    batch_normalization = BATCH_NORMALIZE
)

history_large = model_large.fit(
    train_ds_large,
    epochs=num_epochs,
    validation_data=val_ds_large,
    callbacks=[checkpoint_large]
)

print("Done. Best models were saved as .keras files.")