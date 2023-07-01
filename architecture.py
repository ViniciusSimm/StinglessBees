from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Flatten, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model


class VGG16_MODEL():
    def __init__(self,freeze):
        self.freeze = freeze

    def model(self):
        vgg16 = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(224,224,3)
        )
        x = vgg16.output
        x = Flatten()(x)
        x = Dense(64,activation='relu')(x)
        x = Dropout(0.4)(x)
        out = Dense(13,activation='softmax')(x)
        tf_model = Model(inputs=vgg16.input,outputs=out)

        if self.freeze == True:
            for layer in tf_model.layers[:20]:
                layer.trainable=False

        return tf_model