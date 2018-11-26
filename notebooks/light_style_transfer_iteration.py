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
            horizontal_flip=False)

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

        combination_identity_input = np.array([[1.0]])

        return [content_images, style_images, combination_identity_input], None

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

def build_encoder(input_shape):
    input_tensor = keras.layers.Input(input_shape)
    squeezenet_model = squeezenet.SqueezeNet(input_shape)

    print('Model loaded.')
    for layer in squeezenet_model.layers:
        layer.trainable = False # Freeze model

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in squeezenet_model.layers])
    encoder = Model(inputs=squeezenet_model.input, outputs=outputs_dict['fire8'], name='model_encoder_squeezenet')
    return encoder, outputs_dict


def build_encoder(input_shape):
    input_tensor = keras.layers.Input(input_shape)
    vgg_model = vgg19.VGG19(input_tensor=input_tensor,
                          weights='imagenet', include_top=False)
    print('Model loaded.')
    for layer in vgg_model.layers:
        layer.trainable = False # Freeze model

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg_model.layers])
    encoder = Model(inputs=vgg_model.input, outputs=outputs_dict['block5_conv2'], name='model_encoder_vgg19')
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

train_data_generator = DataGenerator('../data/contents_woman/', '../data/style_starry_night/', batch_size=1, image_shape=input_shape[:-1])
images, _ = train_data_generator[0]
train_data_generator.datagenerator.fit(images[0])
train_data_generator.datagenerator.fit(images[1])

losses_weights = {'content': 1, 'style': 0.5*1e-4, 'total_variation': 0.5*1e-4}

base_image = keras.layers.Input(input_shape, name='model_base_image')
style_reference_image = keras.layers.Input(input_shape, name='model_style_image')

combination_image_eye_input = keras.layers.Input((1,), name='combination_image')

combination_layer = keras.layers.Dense(np.prod(input_shape), use_bias=False)
combination_image_raw = combination_layer(combination_image_eye_input)
combination_layer.set_weights([np.reshape(images[0], (1, -1))])

combination_image = keras.layers.Reshape(input_shape)(combination_image_raw)

## Encoding
encoder, outputs_dict = build_encoder(input_shape) # It would be smart to actually compute the encoding and store them

feature_layers = ['fire2', 'fire4', 'fire6', 'fire8']
feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                  'block4_conv1', 'block5_conv1']
style_loss_per_layers_weights = [2, 2, 2, 1, 1]

style_layers = [encoder.get_layer(l).output for l in feature_layers]
style_layers_map = dict(zip(feature_layers, style_layers))
style_layer_encoder = Model(inputs=encoder.input, outputs=style_layers, name='style_layer_encoder_model')

base_image_encoded = encoder(base_image)
combination_image_encoded = encoder(combination_image)

style_features_map =  style_layer_encoder(style_reference_image)
combination_features_map = style_layer_encoder(combination_image)

## Model
style_transfer_model = Model(
    inputs=[base_image, style_reference_image, combination_image_eye_input],
    outputs=combination_image)

## Loss
content_loss = losses_weights['content'] * compute_content_loss(
    combination_image_encoded, base_image_encoded)

style_transfer_model.add_loss(content_loss)
combination_layer.add_loss(content_loss)

style_loss = losses_weights['style'] * compute_style_loss(
    style_features_map, combination_features_map, style_loss_per_layers_weights)
style_transfer_model.add_loss(style_loss)
combination_layer.add_loss(style_loss)

total_variation_loss = losses_weights['total_variation'] * tf.reduce_mean(tf.image.total_variation(combination_image))
combination_layer.add_loss(total_variation_loss)
style_transfer_model.add_loss(total_variation_loss)

optimizer = keras.optimizers.Adam(lr=100)# , clipnorm=1, clipvalue=0.5)
style_transfer_model.compile(optimizer=optimizer)

yaml_string = style_transfer_model.to_yaml()
with open('output/squeezenet_transfer_iteration_model.yaml', 'w') as f:
    f.write(yaml_string)

style_transfer_model.summary()

## Callbacks

model_weights_checkpoint = keras.callbacks.ModelCheckpoint(
    'output/weights.{epoch:02d}.h5',
    monitor='loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', period=5)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    'output/model.{epoch:02d}.h5',
    monitor='loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', period=20)

tensorboard_checkpoint = keras.callbacks.TensorBoard(log_dir='./logs/squeezenet_iteration')

reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1,
    patience=10, verbose=0, mode='auto', min_delta=0.1, cooldown=1, min_lr=1e-4)

learning_rate_scheduler = keras.callbacks.LearningRateScheduler(lambda ep, lr: max(0.5*lr, 1e-2))

generate_sample_image_output = GenerateSampleImageOutput('output/sample_images/squeezenet_iteration_{}_{}.png', train_data_generator, 1)

## Training loop

style_transfer_model.fit_generator(
    train_data_generator,
    callbacks=[
        model_weights_checkpoint, model_checkpoint, tensorboard_checkpoint,
        reduce_lr_on_plateau, generate_sample_image_output, learning_rate_scheduler],
    steps_per_epoch=128, epochs=10, workers=8, use_multiprocessing=True,
    max_queue_size=20, initial_epoch=0, verbose=1)

style_transfer_model.save_weights('output/squeezenet_weights_final.h5')



