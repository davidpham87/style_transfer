from __future__ import print_function

import argparse
import numpy as np
import random
import scipy.misc
import tensorflow as tf
import time


from dotenv.main import DotEnv, find_dotenv
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications import vgg19
from tensorflow.keras.layers import UpSampling2D, Input, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

import squeezenet


class DataGenerator(Sequence):
    """Sequence Generator to speed up training"""
    def __init__(self, content_folder, style_folder, batch_size=4,
                 image_shape=(256, 256), shuffle=True):
        self.content_folder = content_folder
        self.style_folder = style_folder
        self.batch_size = batch_size
        self.input_shape = image_shape
        self.shuffle = shuffle

        self.datagenerator = ImageDataGenerator(
            horizontal_flip=True)

        self.content_generator = self.datagenerator.flow_from_directory(
            self.content_folder,
            target_size=self.input_shape,
            batch_size=self.batch_size,
            class_mode=None)

        self.style_generator = self.datagenerator.flow_from_directory(
            self.style_folder,
            target_size=self.input_shape,
            batch_size=self.batch_size,
            class_mode=None)

    def __len__(self):
        return int(np.floor(self.content_generator.n/self.batch_size))

    def __getitem__(self, idx):

        content_idx = np.random.randint(0, np.floor(self.content_generator.n/self.batch_size))
        content_images = self.content_generator[content_idx]
        content_images = vgg19.preprocess_input(content_images, 'channels_last')

        style_size = np.floor(self.style_generator.n/self.batch_size)
        style_idx = 0
        if style_size > 0:
            style_idx = np.random.randint(0, style_size)
        style_images = self.style_generator[style_idx]
        while style_images.shape[0] < self.batch_size:
            style_images = np.concatenate([style_images, self.style_generator[style_idx]])
        style_images = style_images[:self.batch_size]
        style_images = vgg19.preprocess_input(style_images, 'channels_last')

        content_vs_style_ratio = np.random.uniform(0.99, 1.0, size=(self.batch_size))
        content_vs_style_ratio_zero = np.random.uniform(0.0, 0.0, size=(self.batch_size))
        return [content_images, style_images, content_vs_style_ratio, content_vs_style_ratio_zero], None

# Image processing

def save_image(image, path):
    scipy.misc.imsave(path, image)

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")

  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x

class GenerateSampleImageOutput(keras.callbacks.Callback):
    """Custom Callback to generate pictures during training time"""

    def __init__(self, filepath, sample_input_generator, period=5, number_outputs=4):
        self.filepath = filepath # string that can be formatted with two positional arguments (epoch, idx_image)
        self.sample_input_generator = sample_input_generator
        self.period = period
        self.epochs_since_last_save = 0
        self.number_outputs = number_outputs

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            idx = np.random.randint(0, len(self.sample_input_generator))
            sample_input, _  = self.sample_input_generator[idx]
            y = self.model.predict(sample_input)
            np.random.shuffle(y) # get different inputs
            for j, y_j in enumerate(y):
                if j < self.number_outputs:
                    save_image(deprocess_img(y_j), self.filepath.format(epoch, j))

### Network Definition

def adaptive_instance_normalization(inputs):
    content_embedding, style_embedding, alpha = inputs
    style_mean, style_variance = tf.nn.moments(
        style_embedding, [1, 2], keep_dims=True)
    content_mean, content_variance = tf.nn.moments(
        content_embedding, [1, 2], keep_dims=True)
    epsilon = 1e-10
    normalized_content_features = tf.nn.batch_normalization(
        content_embedding, content_mean, content_variance, style_mean,
        tf.sqrt(style_variance), epsilon)
    alpha_broadcast = tf.expand_dims(tf.expand_dims(alpha, -1), -1)
    normalized_content_embedding = (
        alpha_broadcast * normalized_content_features +
        (1-alpha_broadcast) * content_embedding)
    return normalized_content_embedding

class AdaIN(keras.layers.Layer):

    def call(self, inputs):
        content_embedding, style_embedding, alpha = inputs
        style_mean, style_variance = tf.nn.moments(
            style_embedding, [1, 2], keep_dims=True)
        content_mean, content_variance = tf.nn.moments(
            content_embedding, [1, 2], keep_dims=True)
        epsilon = 1e-10
        normalized_content_features = tf.nn.batch_normalization(
            content_embedding, content_mean, content_variance, style_mean,
            tf.sqrt(style_variance), epsilon)
        alpha_broadcast = tf.expand_dims(tf.expand_dims(alpha, -1), -1)
        normalized_content_embedding = (
            alpha_broadcast * normalized_content_features +
            (1-alpha_broadcast) * content_embedding)

        return normalized_content_embedding

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def AdaIN():
    return keras.layers.Lambda(adaptive_instance_normalization)


class ReflectLayer(keras.layers.Layer):

    def call(self, x):
        padding = 1
        return tf.pad(
            x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
            mode='REFLECT')

    # def compute_output_shape(self, input_shape):
    #     return [input_shape[0], input_shape[1]+1, input_shape[2]+1, input_shape[3]]


def build_decoder(style_features_map, input_shape):

    # if original width, height are w, h, it stats with w/8 and h/8

    code = Input(shape=input_shape, name='decoder_input')
    ks = sorted(style_features_map.keys())
    style_features_input = [Input(shape=(None, None, style_features_map[k].shape[-1]), name=k+'_style')
                            for k in ks]
    style_features_input_dict = {k: l for k, l in zip(ks, style_features_input)}
    content_vs_style_ratio = Input(shape=(1,), name='decoder_input_content_vs_style')

    fire = squeezenet.fire_module

    x = code
    x = fire(x, (64, 256, 256), padding='same', name='fire8_decoder')
    x = fire(x, (48, 192, 192), padding='same', name='fire7_decoder')
    x = fire(x, (48, 192, 192) , padding='same', name='fire6_decoder')
    x = AdaIN()([x, style_features_input_dict['fire6'], content_vs_style_ratio])
    x = UpSampling2D()(x) # w/4 h/4
    x = fire(x, (32, 128, 128), padding='same', name='fire5_decoder')
    x = fire(x, (32, 128, 128), padding='same', name='fire4_decoder')
    x = AdaIN()([x, style_features_input_dict['fire4'], content_vs_style_ratio])
    x = UpSampling2D()(x) # w/2 h/2
    x = fire(x, (16, 64, 64), padding='same', name='fire3_decoder')
    x = fire(x, (16, 64, 64), padding='same', name='fire2_decoder')
    x = AdaIN()([x, style_features_input_dict['fire2'], content_vs_style_ratio])
    x = UpSampling2D((2,2))(x)  # w/1 h/1
    x = ReflectLayer()(x)
    outputs = Conv2D(3, 3, padding='valid', activation=None)(x)

    decoder = Model([code] + style_features_input + [content_vs_style_ratio], outputs,
                    name='decoder_model')
    print(decoder.summary())
    return decoder


def build_encoder(input_shape):
    input_tensor = keras.layers.Input(input_shape)
    squeezenet_model = squeezenet.SqueezeNet(input_shape)

    print('Model loaded.')
    # for layer in squeezenet_model.layers:
    #     layer.trainable = False # Freeze model

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in squeezenet_model.layers])
    encoder = Model(inputs=squeezenet_model.input, outputs=outputs_dict['fire8'], name='model_encoder_squeezenet')
    return encoder, outputs_dict


def compute_content_loss(content, combination):
    return tf.reduce_mean(tf.square(combination - content))


def compute_style_loss(style_fm, combination_fm, style_loss_per_layer_weights=None):
    sls = []
    if style_loss_per_layer_weights is None:
        style_loss_per_layer_weights = [1.0] * len(style_fm)

    for s_map, combination_map, style_loss_weights in zip(
            style_fm, combination_fm, style_loss_per_layer_weights):
        s_mu, s_var = tf.nn.moments(s_map, [1, 2])
        combination_mu, combination_var = tf.nn.moments(combination_map, [1, 2])
        mu_loss = tf.reduce_sum(tf.square(s_mu - combination_mu))
        var_loss = tf.reduce_sum(tf.square(tf.sqrt(s_var + 1e-6) - tf.sqrt(combination_var + 1e-6)))
        sl = (mu_loss + var_loss) / tf.cast(tf.shape(s_map)[0], tf.float32)
        sls.append(style_loss_weights*sl)
    return tf.reduce_sum(sls)


def set_initial_decoder_weights(encoder, decoder):
    encoder_layers = {l.name: l for l in encoder.layers}
    for layer in decoder.layers:
        k = layer.name
        if k in encoder_layers:
            print(k)
            layer.set_weights(encoder_layers[k].get_weights())
    return decoder

CONSTANTS  = DotEnv(find_dotenv()).dict()

data_train = CONSTANTS['data_train_folder']
data_style = CONSTANTS['style_folder']

width, height = (256, 256)# load_img(base_image_path).size

img_nrows = 256
img_ncols = int(width * img_nrows / height)

input_shape = (256, 256, 3)
content_vs_style_ratio = 1.0
losses_weights = {'content': 1, 'style': 1e-4, 'total_variation': 0.0, 'content_encoder': 1}

base_image = keras.layers.Input(input_shape, name='model_base_image')
style_reference_image = keras.layers.Input(input_shape, name='model_style_image')
content_vs_style_ratio = keras.layers.Input(shape=(1,), name='model_content_vs_style')
content_vs_style_ratio_zero = keras.layers.Input(shape=(1,), name='model_content_vs_style_zero')

## Encoding
encoder, outputs_dict = build_encoder(input_shape) # It would be smart to actually compute the encoding and store them

feature_layers = ['fire2', 'fire4', 'fire6', 'fire8']
style_loss_per_layers_weights = [21, 21, 1, 1]


style_layers = [encoder.get_layer(l).output for l in feature_layers]
style_layers_map = dict(zip(feature_layers, style_layers))
style_layer_encoder = Model(inputs=encoder.input, outputs=style_layers, name='style_layer_encoder_model')
# for layer in style_layer_encoder.layers:
#     layer.trainable = False

base_image_encoded = encoder(base_image)
style_reference_image_encoded = encoder(style_reference_image)
combination_image_latent = AdaIN()([base_image_encoded, style_reference_image_encoded, content_vs_style_ratio])

## Decoding
n_channels = combination_image_latent.get_shape()[-1].value
decoder = build_decoder(style_layers_map, (None, None, n_channels))
style_features_map = style_layer_encoder(style_reference_image)

# decoder_inputs = [combination_image_latent] + style_features_map + [content_vs_style_ratio]
decoder_inputs = [base_image_encoded] + style_features_map + [content_vs_style_ratio]
combination_image = decoder(decoder_inputs)
combination_image_encoded = encoder(combination_image)
decoder_inputs_encoder_loss  = [base_image_encoded] + style_features_map + [content_vs_style_ratio_zero]
base_image_decoded = decoder(decoder_inputs_encoder_loss)
base_image_recoded = encoder(base_image_decoded)

## Model
style_transfer_model = Model(
    inputs=[base_image, style_reference_image, content_vs_style_ratio, content_vs_style_ratio_zero],
    outputs=combination_image)

## Loss
content_loss = (
    losses_weights['content'] * compute_content_loss(
        combination_image_latent, combination_image_encoded)
    + losses_weights['content'] * compute_content_loss(
        base_image_recoded, base_image_encoded))

style_transfer_model.add_loss(content_loss)
decoder.add_loss(content_loss)

content_loss_encoder = losses_weights['content_encoder'] * compute_content_loss(
    base_image, base_image_decoded)

style_transfer_model.add_loss(content_loss_encoder)
encoder.add_loss(content_loss_encoder)

combination_features_map = style_layer_encoder(combination_image)
style_loss = losses_weights['style'] * compute_style_loss(
    style_features_map, combination_features_map, style_loss_per_layers_weights)

style_transfer_model.add_loss(style_loss)
decoder.add_loss(style_loss)

total_variation_loss = losses_weights['total_variation'] * tf.reduce_mean(tf.image.total_variation(combination_image))
style_transfer_model.add_loss(total_variation_loss)
decoder.add_loss(total_variation_loss)

train_data_generator = DataGenerator('../datasets/train_images/', '../data/style_starry_night/', batch_size=16, image_shape=input_shape[:-1])
images, _ = train_data_generator[0]
train_data_generator.datagenerator.fit(images[0])
train_data_generator.datagenerator.fit(images[1])

optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=1, clipvalue=0.5)
style_transfer_model.compile(optimizer=optimizer)

yaml_string = style_transfer_model.to_yaml()
with open('output/squeezenet_transfer_model.yaml', 'w') as f:
    f.write(yaml_string)

style_transfer_model.summary()

if True:
    style_transfer_model.load_weights('output/squeezenet_weights.h5', by_name=True)

## Callbacks

model_weights_checkpoint = keras.callbacks.ModelCheckpoint(
    'output/weights.{epoch:02d}.h5',
    monitor='loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', period=5)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'output/model.{epoch:02d}.h5',
    monitor='loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', period=20)

tensorboard_checkpoint = keras.callbacks.TensorBoard(log_dir='./logs/squeezenet_loss_weights')

reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1,
    patience=10, verbose=0, mode='auto', min_delta=0.1, cooldown=1, min_lr=1e-4)

generate_sample_image_output = GenerateSampleImageOutput('output/sample_images/squeezenet_{}_{}.png', train_data_generator, 1)

## Training loop

style_transfer_model.fit_generator(
    train_data_generator,
    callbacks=[
        model_weights_checkpoint, model_checkpoint, tensorboard_checkpoint,
        reduce_lr_on_plateau, generate_sample_image_output],
    steps_per_epoch=64, epochs=10, workers=8, use_multiprocessing=True,
    max_queue_size=20, initial_epoch=0, verbose=2)

style_transfer_model.save_weights('output/squeezenet_weights_final.h5')



