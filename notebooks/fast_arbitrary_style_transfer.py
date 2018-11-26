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
                 image_shape=[256, 256], shuffle=True):
        self.content_folder = content_folder
        self.style_folder = style_folder
        self.batch_size = batch_size
        self.input_shape = list(image_shape)
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

    def random_crop(self, images):
        imgs = []
        for img in images:
            pos_x = np.random.randint(0, 512-self.input_shape[0])
            pos_y = np.random.randint(0, 512-self.input_shape[1])
            img_cropped = img[pos_x:(pos_x+self.input_shape[0]),
                              pos_y:(pos_y+self.input_shape[1])]
            imgs.append(img_cropped)
        return np.stack(imgs, axis=0)

    def __getitem__(self, idx):

        content_idx = np.random.randint(0, np.floor(self.content_generator.n/self.batch_size))
        content_images = self.content_generator[content_idx]
        content_images = vgg19.preprocess_input(content_images, 'channels_last')

        style_size = np.floor(self.style_generator.n/self.batch_size)
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


class ReflectLayer(keras.layers.Layer):

    def call(self, inputs):
        padding = 1
        return tf.pad(inputs, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
               mode='REFLECT')

def Conv2DReflect(filters, name=None):
    return Conv2D(filters, 3, padding='valid', activation='relu', name=name)



def build_decoder(input_shape):

    # if original width, height are w, h, it stats with w/4 and h/4
    code = Input(shape=input_shape, name='decoder_input')
    x = ReflectLayer()(code)
    x = Conv2DReflect(256 , name='block3_conv1_decoder')(x)
    x = UpSampling2D()(x) # w/2 h/2
    x = ReflectLayer()(x)
    x = Conv2DReflect(128 , name='block2_conv2_decoder')(x)
    x = ReflectLayer()(x)
    x = Conv2DReflect(128 , name='block2_conv1_decoder')(x)
    x = UpSampling2D()(x) # w h
    x = ReflectLayer()(x)
    x = Conv2DReflect(64 , name='block1_conv2_decoder')(x)
    x = ReflectLayer()(x)
    x = Conv2DReflect(64 , name='block1_conv1_decoder')(x)
    x = ReflectLayer()(x)
    outputs = Conv2D(3, 3, padding='valid', activation=None)(x)

    decoder = Model([code], outputs, name='decoder_model')
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
    encoder = Model(inputs=vgg_model.input, outputs=outputs_dict['block3_conv1'], name='model_encoder_vgg19')
    return encoder, outputs_dict


class CovarianceLayer(keras.layers.Layer):

    def call(self, x):
        batch_size, n, channels = x.shape
        x_mean, _  = tf.nn.moments(x, [2], keep_dims=True)
        x_centered = (x - x_mean)
        return tf.einsum("ijk,ljn->ijn", x_centered, x_centered)

    def compute_output_shape(self, input_shape):
        batch_size, n, channels = input_shape
        return (batch_size, channels, channels)


class BatchMultiply(keras.layers.Layer):

    def call(self, inputs):
        x, y = inputs
        return tf.einsum('aij,ajk->aik', x, y)

    def compute_output_shape(self, input_shape):
        return [input_shape[0][0], input_shape[1][1]]


class BatchDot(keras.layers.Layer):

    def call(self, inputs):
        x, y = inputs
        return tf.einsum('aji,ajk->aik', x, y)

    # def compute_output_shape(self, input_shape):
    #     return [input_shape[1][1], input_shape[1][1]]


def transformation_architecture(x, name=None):

    if name is None:
        name = "transformation"

    x = keras.layers.Conv2D(128, 3, activation='relu', padding="valid", name="{}/conv1".format(name))(x)
    x = keras.layers.Conv2D(64, 3, activation='relu', padding="valid", name="{}/conv2".format(name))(x)
    x = keras.layers.Conv2D(32, 3, activation='relu', padding="valid", name="{}/conv3".format(name))(x)
    x = keras.layers.Reshape((-1, 32))(x)
    x = CovarianceLayer()(x)
    x = keras.layers.Dense(64, 'relu', name="{}/dense1".format(name))(x)
    x = keras.layers.Dense(32, name="{}/dense2".format(name))(x)
    return x


def build_transformation_model(
        base_image_encoding, style_image_encoding, name='transformation'):
    
    base_input = keras.layers.Input(
        tensor=base_image_encoding,
        name='{}/base_image_input'.format(name))
    style_input = keras.layers.Input(
        tensor=style_image_encoding,
        name='{}/style_image_input'.format(name))

    base_image_covariance_transform = transformation_architecture(
        base_input, name+'/base_image_transform')

    style_image_covariance_transform = transformation_architecture(
        style_input, name+'/style_image_transform')

    covariance_transformation = BatchDot()(
        [base_image_covariance_transform, style_image_covariance_transform])

    outputs = [covariance_transformation, base_image_covariance_transform,
               style_image_covariance_transform]
    model = Model([base_input, style_input], outputs, name=name)
    return model

class CenterLayer(keras.layers.Layer):

    def call(self, inputs):
        x = inputs
        x_mean, _  = tf.nn.moments(x, [2], keep_dims=True)
        return x - x_mean


def build_compressor_model(input_shape, name="compressor"):

    input_layer = keras.layers.Input(shape=input_shape, name='{}/input'.format(name))
    x = keras.layers.Conv2D(128, 3, activation='relu', padding="same", name="{}/conv1".format(name))(input_layer)
    x = keras.layers.Conv2D(64, 3, activation='relu', padding="same", name="{}/conv2".format(name))(x)
    x = keras.layers.Conv2D(32, 3, activation='relu', padding="same", name="{}/conv3".format(name))(x)
    output_layer = keras.layers.Reshape((-1, 32))(x)
    model = Model(inputs=input_layer, outputs=output_layer, name=name)
    return model


def build_uncompressor_model(input_shape, image_shape, name="uncompressor"):

    input_layer = keras.layers.Input(shape=input_shape, name='{}/input'.format(name))
    reshape_size = image_shape + (32, )    
    x = keras.layers.Reshape(reshape_size)(input_layer)
    x = keras.layers.Conv2D(64, 3, activation='relu', padding="same", name="{}/conv1".format(name))(x)
    x = keras.layers.Conv2D(128, 3, activation='relu', padding="same", name="{}/conv2".format(name))(x)
    output_layer = keras.layers.Conv2D(256, 3, activation='relu', padding="same", name="{}/conv3".format(name))(x)
    model = Model(input_layer, output_layer, name=name)
    return model


class ShiftLayer(keras.layers.Layer):

    def call(self, inputs):
        x, y = inputs
        y_mean, _ = tf.nn.moments(y, [2], keep_dims=True)
        return x + y_mean

    def compute_output_shape(self, input_shape):
        return input_shape[0]


    

def compute_content_loss(content, combination):
    return tf.reduce_mean(tf.square(combination - content))


def gram_matrix(x):
    h, w = x.shape[1], x.shape[1]
    channels = int(x.shape[-1])
    features = tf.reshape(x, [-1, int(h)*int(w), channels])
    gram = K.batch_dot(features, K.permute_dimensions(features, [0, 2, 1]))
    return gram / (tf.cast(tf.shape(x)[0], tf.float32) * tf.cast(h * w, tf.float32))


def compute_style_loss(style_fm, combination_fm, style_loss_per_layer_weights=None):
    sls = []
    if style_loss_per_layer_weights is None:
        style_loss_per_layer_weights = [1.0] * len(style_fm)

    for s_map, combination_map, style_loss_weights in zip(
            style_fm, combination_fm, style_loss_per_layer_weights):
        # s_gram = gram_matrix(s_map)
        # c_gram = gram_matrix(combination_map)
        s_mu, s_var = tf.nn.moments(s_map, [1, 2])
        combination_mu, combination_var = tf.nn.moments(combination_map, [1, 2])
        mu_loss = tf.reduce_sum(tf.square(s_mu - combination_mu))
        var_loss = tf.reduce_sum(tf.square(tf.sqrt(s_var + 1e-6) - tf.sqrt(combination_var + 1e-6)))
        sl = (mu_loss + var_loss) / tf.cast(tf.shape(s_map)[0], tf.float32)
        s_dims = tf.shape(s_map)
        # sl = tf.reduce_mean(tf.square(s_gram - c_gram)) / tf.cast(s_dims[0]*s_dims[1]*s_dims[2]*s_dims[3], tf.float32)
        sls.append(style_loss_weights*sl)
    return tf.reduce_sum(sls)


CONSTANTS  = DotEnv(find_dotenv()).dict()

data_train = CONSTANTS['data_train_folder']
data_style = CONSTANTS['style_folder']

width, height = (256, 256) # load_img(base_image_path).size

img_nrows = 256
img_ncols = int(width * img_nrows / height)

input_shape = (256, 256, 3)
content_vs_style_ratio = 0.5
losses_weights = {'content': 1, 'style': 0.5*1e-1, 'total_variation': 1e-6}

base_image = keras.layers.Input(input_shape, name='model_base_image')
style_reference_image = keras.layers.Input(input_shape, name='model_style_image')

## Encoding
encoder, outputs_dict = build_encoder(input_shape) # It would be smart to actually compute the encoding and store them

feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']
style_loss_per_layers_weights = [1, 2, 2]

style_layers = [encoder.get_layer(l).output for l in feature_layers]
style_layers_map = dict(zip(feature_layers, style_layers))
style_layer_encoder = Model(inputs=encoder.input, outputs=style_layers, name='style_layer_encoder_model')
for layer in style_layer_encoder.layers:
    layer.trainable = False

base_image_encoded = encoder(base_image)
style_reference_image_encoded = encoder(style_reference_image)

transformation_model = \
    build_transformation_model(base_image_encoded, style_reference_image_encoded)

transformation_matrix, _,  style_image_covariance_transform = (
    transformation_model([base_image_encoded, style_reference_image_encoded]))


compressor_model = build_compressor_model(base_image_encoded.shape[1:])
base_image_encoded_compressed = compressor_model(base_image_encoded)
base_image_encoded_compressed_centered = CenterLayer()(base_image_encoded_compressed)


combination_image_latent_compressed = BatchMultiply()(
    [base_image_encoded_compressed_centered, transformation_matrix])

uncompressor_model = \
    build_uncompressor_model(combination_image_latent_compressed.shape[1:], (int(width/4), int(height/4)))

combination_image_latent_uncompressed = uncompressor_model(combination_image_latent_compressed)
combination_image_latent = ShiftLayer()([combination_image_latent_uncompressed, style_reference_image_encoded])

## Decoding

n_channels = combination_image_latent.get_shape()[-1].value
decoder =  build_decoder((None, None, n_channels))

decoder_inputs = [combination_image_latent]
combination_image = decoder(decoder_inputs)
combination_image_encoded = encoder(combination_image)

## Model
style_transfer_model = Model(
    inputs=[base_image, style_reference_image],
    outputs=combination_image)

## Loss
content_loss = losses_weights['content'] * compute_content_loss(
    base_image_encoded, combination_image_encoded)

style_transfer_model.add_loss(content_loss)
decoder.add_loss(content_loss)


combination_features_map = style_layer_encoder(combination_image)
style_loss = losses_weights['style'] * compute_style_loss(
    style_features_map, combination_features_map, style_loss_per_layers_weights)

style_transfer_model.add_loss(style_loss)
decoder.add_loss(style_loss)

total_variation_loss = losses_weights['total_variation'] * tf.reduce_mean(tf.image.total_variation(combination_image))
style_transfer_model.add_loss(total_variation_loss)
decoder.add_loss(total_variation_loss)

if False:
    style_transfer_model.load_weights('output/weights.2160.h5', by_name=True)

train_data_generator = DataGenerator('../datasets/train_images', '../data/style_images', batch_size=8, image_shape=input_shape[:-1])
images, _ = train_data_generator[0]
train_data_generator.datagenerator.fit(images[0])
train_data_generator.datagenerator.fit(images[1])

optimizer = keras.optimizers.Adam(lr=0.001, clipnorm=1, clipvalue=0.5)
style_transfer_model.compile(optimizer=optimizer)

yaml_string = style_transfer_model.to_yaml()
with open('output/avatar_transfer_model.yaml', 'w') as f:
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

tensorboard_checkpoint = keras.callbacks.TensorBoard(log_dir='./logs/fast_arbitrary_style_transfer_20181007')

reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1,
    patience=10, verbose=0, mode='auto', min_delta=0.1, cooldown=1, min_lr=1e-4)

generate_sample_image_output = GenerateSampleImageOutput('output/sample_images/fast_arbitrary_style_transfer_{}_{}.png', train_data_generator, 1)

## Training loop

style_transfer_model.fit_generator(
    train_data_generator,
    callbacks=[
        model_weights_checkpoint, model_checkpoint, tensorboard_checkpoint,
        reduce_lr_on_plateau, generate_sample_image_output],
    steps_per_epoch=128, epochs=1, workers=8, use_multiprocessing=True,
    max_queue_size=20, initial_epoch=0, verbose=2)

style_transfer_model.save_weights('output/fast_arbitrary_style_transfer_weights_final.h5')
