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


class DataGenerator(Sequence):
    """Sequence Generator to speed up training"""
    def __init__(self, content_folder, style_folder, batch_size=32,
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

        content_idx = np.random.randint(
            0, np.floor(self.content_generator.n/self.batch_size))
        content_images = self.content_generator[content_idx]
        content_images = vgg19.preprocess_input(content_images, 'channels_last')

        style_size = np.floor(self.style_generator.n/self.batch_size)
        if style_size == 0:
            style_idx = 0
        else:
            style_dix = np.random.randint(0, style_size)
        style_images = self.style_generator[style_idx]
        while style_images.shape[0] < self.batch_size:
            style_images = np.concatenate([style_images, self.style_generator[style_idx]])
        style_images = style_images[:self.batch_size]
        style_images = vgg19.preprocess_input(style_images, 'channels_last')

        content_vs_style_ratio = np.random.uniform(0.999, 1.0, size=(self.batch_size))

        return [content_images, style_images, content_vs_style_ratio], None

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


def pad_reflect(x, padding=1):
    return tf.pad(
        x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
        mode='REFLECT')


def reflect_layer():
    return keras.layers.Lambda(pad_reflect)


def Conv2DReflect(filters, name=None):
    return Conv2D(filters, 3, padding='valid', activation='relu', name=name)


def build_decoder(input_shape):

    # if original width, height are w, h, it stats with w/8 and h/8

    arch = [
        reflect_layer(),
        Conv2DReflect(512, name='block4_conv1_decoder'),
        UpSampling2D(),  # w/4 h/4
        reflect_layer(),
        Conv2DReflect(256, name='block3_conv4_decoder'),
        reflect_layer(),
        Conv2DReflect(256, name='block3_conv3_decoder'),
        reflect_layer(),
        Conv2DReflect(256, name='block3_conv2_decoder'),
        reflect_layer(),
        Conv2DReflect(256, name='block3_conv1_decoder'),
        UpSampling2D(), # w/2 h/2
        reflect_layer(),
        Conv2DReflect(128, name='block2_conv2_decoder'),
        reflect_layer(),
        Conv2DReflect(128, name='block2_conv1_decoder'),
        UpSampling2D(), # w h
        reflect_layer(),
        Conv2DReflect(64, name='block1_conv2_decoder'),
        reflect_layer(),
        Conv2DReflect(64, name='block1_conv1_decoder'),
        reflect_layer(),
        Conv2D(3, 3, padding='valid', activation=None)
        ]

    code = Input(shape=input_shape, name='decoder_input')
    x = code

    for layer in arch:
        x = layer(x)

    decoder = Model(code, x, name='decoder_model')
    print(decoder.summary())
    return decoder


def build_encoder(input_shape):
    input_tensor = keras.layers.Input(input_shape)
    vgg_model = vgg19.VGG19(input_tensor=input_tensor,
                          weights='imagenet', include_top=False)
    print('Model loaded.')
    for layer in vgg_model.layers:
        layer.trainable = False # Freeze model

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg_model.layers])
    encoder = Model(inputs=vgg_model.input, outputs=outputs_dict['block4_conv1'], name='model_encoder_vgg19')
    return encoder, outputs_dict


def adaptive_instance_normalization(content_embedding, style_embedding, alpha):
    style_mean, style_variance = tf.nn.moments(
        style_embedding, [1, 2], keep_dims=True)
    content_mean, content_variance = tf.nn.moments(
        content_embedding, [1, 2], keep_dims=True)
    epsilon = 1e-5
    normalized_content_features = tf.nn.batch_normalization(
        content_embedding, content_mean, content_variance, style_mean,
        tf.sqrt(style_variance), epsilon)
    alpha_broadcast = tf.expand_dims(tf.expand_dims(alpha, -1), -1)
    normalized_content_embedding = (
         alpha_broadcast * normalized_content_features +
        (1-alpha_broadcast) * content_embedding)
    return normalized_content_embedding

AdIN = keras.layers.Lambda(lambda x: adaptive_instance_normalization(*x))

def compute_content_loss(content, combination):
    return tf.reduce_mean(tf.square(combination - content))


def compute_style_loss(style_fm, combination_fm):
    sls = []
    for s_map, combination_map in zip(style_fm, combination_fm):
        s_mu, s_var = tf.nn.moments(s_map, [1, 2])
        combination_mu, combination_var = tf.nn.moments(combination_map, [1, 2])
        mu_loss = tf.reduce_sum(tf.square(s_mu - combination_mu))
        var_loss = tf.reduce_sum(tf.square(tf.sqrt(s_var) - tf.sqrt(combination_var)))
        sl = (mu_loss + var_loss) / tf.cast(tf.shape(s_map)[0], tf.float32)
        sls.append(sl)
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
losses_weights = {'content': 1, 'style': 9*1e-3, 'total_variation': 1e-10}

base_image = keras.layers.Input(input_shape)
style_reference_image = keras.layers.Input(input_shape)
content_vs_style_ratio = keras.layers.Input(shape=(1,))

## Encoding
encoder, outputs_dict = build_encoder(input_shape) # It would be smart to actually compute the encoding and store them

feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                  'block4_conv1']
style_layers = [encoder.get_layer(l).output for l in feature_layers]
style_layer_encoder = Model(inputs=encoder.input, outputs=style_layers, name='style_layer_encoder_model')
for layer in style_layer_encoder.layers:
    layer.trainable = False

base_image_encoded = encoder(base_image)
style_reference_image_encoded = encoder(style_reference_image)
combination_image_latent = AdIN([base_image_encoded, style_reference_image_encoded, content_vs_style_ratio])

## Decoding
n_channels = combination_image_latent.get_shape()[-1].value
decoder = build_decoder((None, None, n_channels))

combination_image = decoder(combination_image_latent)
combination_image_encoded = encoder(combination_image)

## Model
style_transfer_model = Model(
    inputs=[base_image, style_reference_image, content_vs_style_ratio],
    outputs=combination_image)

## Loss
content_loss = losses_weights['content'] * compute_content_loss(
    combination_image_latent, combination_image_encoded)

style_transfer_model.add_loss(content_loss)
decoder.add_loss(content_loss)

style_features_map = style_layer_encoder(style_reference_image)
combination_features_map = style_layer_encoder(combination_image)
style_loss = losses_weights['style'] * compute_style_loss(
    style_features_map, combination_features_map)

style_transfer_model.add_loss(style_loss)
decoder.add_loss(style_loss)

total_variation_loss = losses_weights['total_variation'] * tf.reduce_mean(
    tf.image.total_variation(combination_image))
style_transfer_model.add_loss(total_variation_loss)
decoder.add_loss(total_variation_loss)

if True:
    style_transfer_model.load_weights('output/weights.955.h5')

train_data_generator = DataGenerator(
    '../datasets/train_images', '../data/style_images', batch_size=8)
images, _ = train_data_generator[0]
train_data_generator.datagenerator.fit(images[0])
train_data_generator.datagenerator.fit(images[1])

optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=1, clipvalue=0.5)
style_transfer_model.compile(optimizer=optimizer)

yaml_string = style_transfer_model.to_yaml()
with open('output/style_transfer_model.yaml', 'w') as f:
    f.write(yaml_string)

## Callbacks

model_weights_checkpoint = keras.callbacks.ModelCheckpoint(
    'output/weights.{epoch:02d}.h5',
    monitor='loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', period=5)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'output/model.{epoch:02d}.h5',
    monitor='loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', period=20)

tensorboard_checkpoint = keras.callbacks.TensorBoard(log_dir='./logs/style')

reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1,
    patience=10, verbose=0, mode='auto', min_delta=0.1, cooldown=1, min_lr=1e-4)

generate_sample_image_output = GenerateSampleImageOutput(
    'output/sample_images/AdaIN_{}_{}.png', train_data_generator, 1)

## Training loop

style_transfer_model.fit_generator(
    train_data_generator,
    callbacks=[
        model_weights_checkpoint, model_checkpoint, tensorboard_checkpoint,
        reduce_lr_on_plateau, generate_sample_image_output],
    steps_per_epoch=256, epochs=2000, workers=8, use_multiprocessing=True,
    max_queue_size=20, initial_epoch=955, verbose=2)

style_transfer_model.save_weights('output/weights_final.h5')


