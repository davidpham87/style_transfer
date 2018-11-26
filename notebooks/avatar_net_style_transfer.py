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

### Network Definition

def adaptive_instance_normalization(content_embedding, style_embedding, alpha):

    style_mean, style_variance = tf.nn.moments(
        style_embedding, [1, 2], keep_dims=True)
    content_mean, content_variance = tf.nn.moments(
        content_embedding, [1, 2], keep_dims=True)
    epsilon = 1e-12
    normalized_content_features = tf.nn.batch_normalization(
        content_embedding, content_mean, content_variance,
        style_mean, tf.sqrt(style_variance), epsilon)

    alpha_broadcast = tf.expand_dims(tf.expand_dims(alpha, -1), -1)
    normalized_content_embedding = (
         alpha_broadcast * normalized_content_features +
        (1-alpha_broadcast) * content_embedding)

    return normalized_content_embedding

class AdaIN(keras.layers.Layer):

     def call(self, inputs):
         content_embedding, style_embedding, alpha = inputs
         return adaptive_instance_normalization(content_embedding, style_embedding, alpha)

     def compute_output_shape(self, input_shape):
         return input_shape[0]

def pad_reflect(x, padding=1):
    return tf.pad(
        x, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
        mode='REFLECT')

def reflect_layer():
    return keras.layers.Lambda(pad_reflect)


class ReflectLayer(keras.layers.Layer):

    def call(self, inputs):
        padding = 1
        return tf.pad(inputs, [[0, 0], [padding, padding], [padding, padding], [0, 0]],
               mode='REFLECT')

def Conv2DReflect(filters, name=None):
    return Conv2D(filters, 3, padding='valid', activation='relu', name=name)


def build_decoder(style_features_map, input_shape):

    # if original width, height are w, h, it stats with w/8 and h/8

    code = Input(shape=input_shape, name='decoder_input')
    ks = sorted(style_features_map.keys())
    style_features_input = [Input(shape=(None, None, style_features_map[k].shape[-1]), name=k+'_style')
                            for k in ks]
    style_features_input_dict = {k: l for k, l in zip(ks, style_features_input)}
    content_vs_style_ratio = Input(shape=(1,), name='decoder_input_content_vs_style')

    x = reflect_layer()(code)
    x = Conv2DReflect(512, name='block4_conv1_decoder')(x)
    x = UpSampling2D()(x)  # w/4 h/4
    x = reflect_layer()(x)
    x = Conv2DReflect(256 , name='block3_conv4_decoder')(x)
    x = reflect_layer()(x)
    x = Conv2DReflect(256 , name='block3_conv3_decoder')(x)
    x = reflect_layer()(x)
    x = Conv2DReflect(256 , name='block3_conv2_decoder')(x)
    x = AdaIN()([x, style_features_input_dict['block3_conv1'], content_vs_style_ratio])
    x = reflect_layer()(x)
    x = Conv2DReflect(256 , name='block3_conv1_decoder')(x)
    x = UpSampling2D()(x) # w/2 h/2
    x = reflect_layer()(x)
    x = Conv2DReflect(128 , name='block2_conv2_decoder')(x)
    x = AdaIN()([x, style_features_input_dict['block2_conv1'], content_vs_style_ratio])
    x = reflect_layer()(x)
    x = Conv2DReflect(128 , name='block2_conv1_decoder')(x)
    x = UpSampling2D()(x) # w h
    x = reflect_layer()(x)
    x = Conv2DReflect(64 , name='block1_conv2_decoder')(x)
    x = AdaIN()([x, style_features_input_dict['block1_conv1'], content_vs_style_ratio])
    x = reflect_layer()(x)
    x = Conv2DReflect(64 , name='block1_conv1_decoder')(x)
    x = reflect_layer()(x)
    outputs = Conv2D(3, 3, padding='valid', activation=None)(x)

    decoder = Model([code] + style_features_input + [content_vs_style_ratio], outputs,
                    name='decoder_model')
    print(decoder.summary())
    return decoder


def adaptive_instance_normalization(content_embedding, style_embedding, alpha=1.0):

    style_mean, style_variance = tf.nn.moments(
        style_embedding, [1, 2], keep_dims=True)
    content_mean, content_variance = tf.nn.moments(
        content_embedding, [1, 2], keep_dims=True)
    epsilon = 1e-12
    normalized_content_features = tf.nn.batch_normalization(
        content_embedding, content_mean, content_variance,
        style_mean, tf.sqrt(style_variance), epsilon)

    alpha_broadcast = tf.expand_dims(tf.expand_dims(alpha, -1), -1)
    normalized_content_embedding = (
         alpha_broadcast * normalized_content_features +
        (1-alpha_broadcast) * content_embedding)

    return normalized_content_embedding

def avatar_net_decoder(
  content_encoding, style_encoding, input_layer_vgg_name='block3_conv1', scope='vgg19_decoder', trainable=False):
    """dimension starts at height/8 width/8 from the original image for block4_conv1, dimension starts at height/4 width/4 from the original image for block3_conv1"""
    adain = adaptive_instance_normalization
    x = adain(content_encoding[input_layer_vgg_name],
              style_encoding[input_layer_vgg_name])
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if input_layer_vgg_name == 'block4_conv1':
          x = padding_layer(x)
          x = _conv2d(
            512, name='block4_conv1_decoder', trainable=trainable)(x)
          x = tf.layers.Conv2DTranspose(
            256, 3, 2, activation='relu', padding='same',
            name='block3_conv4_transpose_decoder', trainable=trainable)(x) # h/4 w/4
          x = padding_layer(x)
          x = _conv2d(
            256, name='block3_conv3_decoder', trainable=trainable)(x)
          x = padding_layer(x)
          x = _conv2d(
            256, name='block3_conv2_decoder', trainable=trainable)(x)
          x = adain(x, style_encoding['block3_conv1'])
        x = padding_layer(x)
        x = _conv2d(
            256, name='block3_conv1_decoder', trainable=trainable)(x)
        x = tf.layers.Conv2DTranspose(
            256, 3, 2, activation='relu', padding='same',
            name='block2_conv2_transpose_decoder', trainable=trainable)(x) # h/2 w/2
        x = adain(x, style_encoding['block2_conv1'])
        x = padding_layer(x)
        x = _conv2d(
            128, name='block2_conv1_decoder', trainable=trainable)(x)
        x = Conv2DTranspose(
            128, 3, 2, activation='relu', padding='same',
            name='block1_conv2_transpose_decoder', trainable=trainable)(x)  # h w
        x = adain(x, style_encoding['block1_conv1'])
        x = _conv2d(64, name='block1_conv1_decoder', padding='same',
            trainable=trainable)(x)
        x = tf.layers.Conv2D(
            3, 1, padding='valid', activation='tanh',
            name='block0_conv1', trainable=trainable)(x)
        output_image = 150*x
    return output_image


def build_decoder(style_features_map, input_shape):

    # if original width, height are w, h, it stats with w/8 and h/8

    code = Input(shape=input_shape, name='decoder_input')
    ks = sorted(style_features_map.keys())
    style_features_input = [Input(shape=(None, None, style_features_map[k].shape[-1]), name=k+'_style')
                            for k in ks]
    style_features_input_dict = {k: l for k, l in zip(ks, style_features_input)}
    content_vs_style_ratio = Input(shape=(1,), name='decoder_input_content_vs_style')

    x = ReflectLayer()(code)
    x = Conv2DReflect(512, name='block4_conv1_decoder')(x)
    x = UpSampling2D()(x)  # w/4 h/4
    x = ReflectLayer()(x)
    x = Conv2DReflect(256 , name='block3_conv4_decoder')(x)
    x = ReflectLayer()(x)
    x = Conv2DReflect(256 , name='block3_conv3_decoder')(x)
    x = ReflectLayer()(x)
    x = Conv2DReflect(256 , name='block3_conv2_decoder')(x)
    x = AdaIN(name="adain_block3_conv1")([x, style_features_input_dict['block3_conv1'], content_vs_style_ratio])
    x = ReflectLayer()(x)
    x = Conv2DReflect(256 , name='block3_conv1_decoder')(x)
    x = UpSampling2D()(x) # w/2 h/2
    x = ReflectLayer()(x)
    x = Conv2DReflect(128 , name='block2_conv2_decoder')(x)
    x = AdaIN(name="adain_block2_conv1")([x, style_features_input_dict['block2_conv1'], content_vs_style_ratio])
    x = ReflectLayer()(x)
    x = Conv2DReflect(128 , name='block2_conv1_decoder')(x)
    x = UpSampling2D()(x) # w h
    x = ReflectLayer()(x)
    x = Conv2DReflect(64 , name='block1_conv2_decoder')(x)
    x = AdaIN(name="adain_block1_conv1")([x, style_features_input_dict['block1_conv1'], content_vs_style_ratio])
    x = ReflectLayer()(x)
    x = Conv2DReflect(64 , name='block1_conv1_decoder')(x)
    x = ReflectLayer()(x)
    outputs = Conv2D(3, 3, padding='valid', activation=None)(x)

    decoder = Model([code] + style_features_input + [content_vs_style_ratio], outputs,
                    name='decoder_model')
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
        s_mu, s_var = tf.nn.moments(s_map, [1, 2])
        combination_mu, combination_var = tf.nn.moments(combination_map, [1, 2])
        mu_loss = tf.reduce_sum(tf.square(s_mu - combination_mu))
        var_loss = tf.reduce_sum(tf.square(tf.sqrt(s_var) - tf.sqrt(combination_var)))
        sl = (mu_loss + var_loss) / tf.cast(tf.shape(s_map)[0], tf.float32)
        sls.append(style_loss_weights*sl)
    return tf.reduce_sum(sls)

def compute_style_loss(style_map, combination_map):
  s_mu, s_var = tf.nn.moments(style_map, [1, 2])
  combination_mu, combination_var = tf.nn.moments(combination_map, [1, 2])
  mu_loss = tf.reduce_sum(tf.square(s_mu - combination_mu))
  var_loss = tf.reduce_sum(tf.square(tf.sqrt(s_var) - tf.sqrt(combination_var)))
  sl = (mu_loss + var_loss) / tf.cast(tf.shape(style_map)[0], tf.float32)
  return sl


CONSTANTS  = DotEnv(find_dotenv()).dict()

data_train = CONSTANTS['data_train_folder']
data_style = CONSTANTS['style_folder']

width, height = (256, 256) # load_img(base_image_path).size

img_nrows = 256
img_ncols = int(width * img_nrows / height)

input_shape = (256, 256, 3)
content_vs_style_ratio = 0.5
losses_weights = {'content': 3.0, 'style': 0.5*1e-1, 'total_variation': 2*1e-6}

base_image = keras.layers.Input(input_shape, name='model_base_image')
style_reference_image = keras.layers.Input(input_shape, name='model_style_image')
content_vs_style_ratio = keras.layers.Input(shape=(1,), name='model_content_vs_style')

## Encoding
encoder, outputs_dict = build_encoder(input_shape) # It would be smart to actually compute the encoding and store them

feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                  'block4_conv1']
style_loss_per_layers_weights = [2, 2, 1.5, 5]

style_layers = [encoder.get_layer(l).output for l in feature_layers]
style_layers_map = dict(zip(feature_layers, style_layers))
style_layer_encoder = Model(inputs=encoder.input, outputs=style_layers, name='style_layer_encoder_model')
for layer in style_layer_encoder.layers:
    layer.trainable = False

base_image_encoded = encoder(base_image)
style_reference_image_encoded = encoder(style_reference_image)
combination_image_latent = AdaIN(name="adain_block4_conv1")([base_image_encoded, style_reference_image_encoded, content_vs_style_ratio])

## Decoding
n_channels = combination_image_latent.get_shape()[-1].value
decoder = build_decoder(style_layers_map, (None, None, n_channels))
style_features_map = style_layer_encoder(style_reference_image)

decoder_inputs = [combination_image_latent] + style_features_map + [content_vs_style_ratio]
combination_image = decoder(decoder_inputs)
combination_image_encoded = encoder(combination_image)

## Model
style_transfer_model = Model(
    inputs=[base_image, style_reference_image, content_vs_style_ratio],
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

if True:
    style_transfer_model.load_weights('output/weights.3855.h5', by_name=True)

train_data_generator = DataGenerator('../datasets/train_images', '../data/style_images', batch_size=8, image_shape=input_shape[:-1])
images, _ = train_data_generator[0]
train_data_generator.datagenerator.fit(images[0])
train_data_generator.datagenerator.fit(images[1])

optimizer = keras.optimizers.Adam(lr=0.0001, clipnorm=1, clipvalue=0.5)
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

tensorboard_checkpoint = keras.callbacks.TensorBoard(log_dir='./logs/avatar_20181007')

reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1,
    patience=10, verbose=0, mode='auto', min_delta=0.1, cooldown=1, min_lr=1e-5)

generate_sample_image_output = GenerateSampleImageOutput('output/sample_images/avatar_{}_{}.png', train_data_generator, 1)

## Training loop

style_transfer_model.fit_generator(
    train_data_generator,
    callbacks=[
        model_weights_checkpoint, model_checkpoint, tensorboard_checkpoint,
        reduce_lr_on_plateau, generate_sample_image_output],
    steps_per_epoch=512, epochs=4000, workers=8, use_multiprocessing=True,
    max_queue_size=20, initial_epoch=3855, verbose=2)

style_transfer_model.save_weights('output/avatar_weights_final.h5')
