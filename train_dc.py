import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils

import PIL
from PIL import Image

import datetime 

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as k 
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

tf.debugging.set_log_device_placement(False)

img_width, img_height = 256, 256

train_data_dir = "/home/shihab/test"
validation_data_dir = "/home/shihab/test"

# train_data_dir = "/home/shihab/Tobacco3482"
# validation_data_dir = "/home/shihab/Tobacco3482"

# train_data_dir = "GDrive:My Drive/Dataset/rvl-cdi-zipped/rvl-cdip/dataset"
# validation_data_dir = "GDrive:My Drive/Dataset/rvl-cdi-zipped/rvl-cdip/dataset"


batch_size = 100

#Data Augmentation

datagen = ImageDataGenerator(
    rescale = 1./ 255,
    validation_split = 0.10
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    subset = "training",
    class_mode = "categorical",
    color_mode = "rgb"
    #classes=["invoice", "resume", "letter"]
)

val_datagen = ImageDataGenerator(
    rescale = 1./ 255
)

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    subset = "validation",
    class_mode = "categorical",
    color_mode = "rgb"
    #classes=["invoice", "resume", "letter"]

)

img_width, img_height = 256,256
batch_size = 128
epochs = 500

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001)
mcp_save = ModelCheckpoint('model_1.h5', save_best_only=True, monitor='acc', mode='max')

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model =keras.applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3),)
    for layer in model.layers:
        layer.trainable=True

    #Adding custom Layers 
    x = model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(16, activation="softmax")(x)

    # creating the final model 
    model_final = Model(inputs = model.input, outputs = predictions)

    # compile the model 
    model_final.compile(loss = "categorical_crossentropy", optimizer =Adam(lr=0.0001), metrics=["accuracy"])
    

model_final.fit(
train_generator,
steps_per_epoch =36005//batch_size,
epochs=500,    
validation_data=validation_generator,
validation_steps=3992//batch_size,
callbacks=[reduce_lr,mcp_save])
