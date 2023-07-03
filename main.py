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
from architecture import VGG16_MODEL,VGG19_MODEL,DENSENET121_MODEL

import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

#===============================================================================
# SETUP
#===============================================================================

MODEL = 'densenet_v1'

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
    model = DENSENET121_MODEL(freeze=True).model()
    model.compile(optimizer = Adam(0.0001) , loss = 'categorical_crossentropy', metrics=["accuracy"])
    print('CREATING NEW MODEL')

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

history = model.fit(X_train, y_train, batch_size=10, epochs = 15, validation_split=0.2, callbacks=callbacks)

#===============================================================================
# SAVE
#===============================================================================

model.load_weights('./tmp/checkpoint')
# print(model.summary())


model.save("./models/{}.h5".format(MODEL))
print('Model Saved!')
