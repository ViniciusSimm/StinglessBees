from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet121
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
    
class VGG19_MODEL():
    def __init__(self,freeze):
        self.freeze = freeze

    def model(self):
        vgg19 = VGG19(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(224,224,3)
        )
        x = vgg19.output
        x = Flatten()(x)
        x = Dense(64,activation='relu')(x)
        x = Dropout(0.4)(x)
        out = Dense(13,activation='softmax')(x)
        tf_model = Model(inputs=vgg19.input,outputs=out)

        if self.freeze == True:
            for layer in tf_model.layers[:20]:
                layer.trainable=False

        return tf_model

class DENSENET121_MODEL():
    def __init__(self,freeze):
        self.freeze = freeze

    def model(self):
        densenet121 = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(224,224,3)
        )
        x = densenet121.output
        x = Flatten()(x)
        x = Dense(64,activation='relu')(x)
        x = Dropout(0.4)(x)
        out = Dense(13,activation='softmax')(x)
        tf_model = Model(inputs=densenet121.input,outputs=out)

        if self.freeze == True:
            for layer in tf_model.layers[:20]:
                layer.trainable=False

        return tf_model


# if __name__ == '__main__':
    # model = VGG19(
    #         include_top=False,
    #         weights='imagenet',
    #         input_tensor=None,
    #         input_shape=(224,224,3))
    # print(model.summary())