import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils

import PIL
from PIL import Image

import datetime 

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

tf.debugging.set_log_device_placement(False)

img_width, img_height = 224, 224

# train_data_dir = "/home/shihab/test"
# validation_data_dir = "/home/shihab/test"

train_data_dir = "/home/shihab/Tobacco3482"
validation_data_dir = "/home/shihab/Tobacco3482"

# train_data_dir = "GDrive:My Drive/Dataset/rvl-cdi-zipped/rvl-cdip/dataset"
# validation_data_dir = "GDrive:My Drive/Dataset/rvl-cdi-zipped/rvl-cdip/dataset"


batch_size = 128

#Data Augmenttation

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

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    # Create the base model from the pre-trained model InceptionV2
    base_model = tf.keras.applications.InceptionResNetV2(input_shape=(img_width, img_height, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#ModelCheckpoint callback saves a model at some interval. 
filepath="saved_models/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5" #File name includes epoch and validation accuracy.

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

tensorboard = TensorBoard(log_dir=log_dir)

callbacks_list = [checkpoint, tensorboard]

# callbacks = [
#     tf.keras.callbacks.TensorBoard(log_dir=log_dir),
#     tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
#                                        save_weight_only=True,
#                                        monitor='val_acc',
#                                        mode='max',
#                                        save_best_only=True,
#                                        verbose=1)
# ]


history = model.fit(train_generator, 
#                     steps_per_epoch = 32003//batch_size,
#                     steps_per_epoch = 2788//batch_size,
                    steps_per_epoch = 3135//batch_size,
                    epochs=500, 
                    validation_data= validation_generator,
#                     validation_steps = 7994//batch_size,
#                     validation_steps = 694//batch_size,
                    validation_steps = 345//batch_size,
                    callbacks = [callbacks_list]
                   )
model.save('document_classifier_model_InceptionV2.h5')  # always save your weights after training or during training
