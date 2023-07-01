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
from keras.optimizers import Nadam, Adam                  #I changed
from keras.utils import to_categorical                    #I changed
from keras.applications.vgg16 import VGG16                #I changed
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

#===============================================================================
#Transfer Learning - VGG 16
#===============================================================================

X_train_paths, X_test_paths, y_train_string, y_test_string = TrainTestSplit(test_size=0.1).split_train_test()
X_train = PrepareData().get_images(X_train_paths)
y_train = PrepareData().encode(y_train_string)

vgg16 = VGG16(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224,224,3)
)

x = vgg16.output
x = Flatten()(x)
x = Dense(256,activation='relu')(x)
x = Dropout(0.2)(x)
out = Dense(13,activation='softmax')(x)

tf_model = Model(inputs=vgg16.input,outputs=out)

# print(tf_model.summary())
# print(len(tf_model.layers))

for layer in tf_model.layers[:20]:
    layer.trainable=False

tf_model.compile(optimizer = Adam(0.0001) , loss = 'categorical_crossentropy',
                 metrics=["accuracy"])

history = tf_model.fit(X_train, y_train, epochs = 40)