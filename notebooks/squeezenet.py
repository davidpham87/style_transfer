import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dropout, Concatenate, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from tensorflow.keras.applications.vgg19 import decode_predictions
from tensorflow.keras.applications.vgg19 import preprocess_input

def fire_module(x, filters, padding="same", name="fire"):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Convolution2D(sq_filters, (1, 1), activation='relu', padding=padding, name=name + "/squeeze1x1")(x)
    expand1 = Convolution2D(ex1_filters, (1, 1), activation='relu', padding=padding, name=name + "/expand1x1")(squeeze)
    expand2 = Convolution2D(ex2_filters, (3, 3), activation='relu', padding=padding, name=name + "/expand3x3")(squeeze)
    x = Concatenate(axis=-1, name=name)([expand1, expand2])
    return x

def SqueezeNet(input_shape=None):

    img_input = Input(shape=input_shape)

    x = Convolution2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1', padding="same")(x)

    x = fire_module(x, (16, 64, 64), name="fire2")
    x = fire_module(x, (16, 64, 64), name="fire3")

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3', padding="same")(x)

    x = fire_module(x, (32, 128, 128), name="fire4")
    x = fire_module(x, (32, 128, 128), name="fire5")

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5', padding="same")(x)

    x = fire_module(x, (48, 192, 192), name="fire6")
    x = fire_module(x, (48, 192, 192), name="fire7")

    x = fire_module(x, (64, 256, 256), name="fire8")
    x = fire_module(x, (64, 256, 256), name="fire9")

    model = Model(img_input, x, name="squeezenet")

    weights_path = 'squeezenet_weights.h5'
    model.load_weights(weights_path, by_name=True)

    return model
