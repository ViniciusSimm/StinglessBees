import sklearn.metrics as metrics
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Nadam, Adam                  
from keras.utils import to_categorical                    
from keras.applications.vgg16 import VGG16                
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet121
from keras.layers import Dropout, Flatten, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import random
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
import cv2 as cv
import os
import glob

from data import TrainTestSplit
from utils import PrepareData
from architecture import VGG16_MODEL

#===============================================================================
# SETUP
#===============================================================================

MODEL = 'tf_model_v2'

#===============================================================================
# LOAD DATA
#===============================================================================

X_train_paths, X_test_paths, y_train_string, y_test_string = TrainTestSplit(test_size=0.1).split_train_test()
X_train = PrepareData().get_images(X_train_paths)
y_train = PrepareData().encode(y_train_string)

model_path = "./models/{}.h5".format(MODEL)
if os.path.isfile(model_path):
    model = tf.keras.models.load_model(model_path)
    print('LOADING MODEL')
else:
    model = VGG16_MODEL(freeze=False).model()
    model.compile(optimizer = Adam(0.0001) , loss = 'categorical_crossentropy', metrics=["accuracy"])
    print('CREATING NEW MODEL')

#===============================================================================
# Transfer Learning - VGG 16
#===============================================================================

# vgg16 = VGG16(
#     include_top=False,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=(224,224,3)
# )

# x = vgg16.output
# x = Flatten()(x)
# x = Dense(64,activation='relu')(x)
# x = Dropout(0.4)(x)
# out = Dense(13,activation='softmax')(x)

# tf_model = Model(inputs=vgg16.input,outputs=out)

# for layer in tf_model.layers[:20]:
#     layer.trainable=False

# tf_model = VGG16_MODEL(freeze=False).model()

#===============================================================================
# CALLBACKS
#===============================================================================

history_logger=tf.keras.callbacks.CSVLogger("./history/{}.csv".format(MODEL), separator=",", append=True)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./tmp/checkpoint',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=12, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        history_logger,
        model_checkpoint_callback]

#===============================================================================
# TRAINING
#===============================================================================

# tf_model.compile(optimizer = Adam(0.0001) , loss = 'categorical_crossentropy',
#                  metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=25, epochs = 15, validation_split=0.2, callbacks=callbacks)

#===============================================================================
# SAVE
#===============================================================================

model.load_weights('./tmp/checkpoint')
# print(model.summary())


model.save("./models/{}.h5".format(MODEL))
print('Model Saved!')
