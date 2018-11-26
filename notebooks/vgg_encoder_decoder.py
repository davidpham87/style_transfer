# -*- coding: utf-8 -*-

import argparse
import glob
import itertools as it
import json
import os
import random
import time


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import tensorflow as tf

from PIL import Image

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils
from tensorflow.keras.applications import vgg19
from tensorflow.keras.layers import UpSampling2D, Input, Conv2D, Conv2DTranspose
from tensorflow.python.estimator import estimator

# Image processing
def save_image(image, path):
    scipy.misc.imsave(path, image)

import numpy as np

HEIGHT, WIDTH = (256, 256)
BATCH_SIZE = 4

standardize = lambda x: np.array(x)/np.sum(x)

loss_weights = {'content': 1, 'style': 1e3, 'total_variation': 1e-8}
style_loss_weights = {'block1_conv1': 0.5, 'block2_conv1': 0.4, 'block3_conv1': 0.3}
params_estimator = {'loss_weights': loss_weights, 'style_loss_weights': style_loss_weights}

train_steps = 2*1e5
num_shards = 8
batch_size_tpu = 8*BATCH_SIZE

# train_steps = int(1e4)
train_steps_per_eval = 4*int(1e3)
iterations_per_loop = 200

COCO_PATH = '../data/data_records/coco*'
WIKI_PATH = '../data/data_records/wiki*'

# COCO_PATH = 'drive/My Drive/style_transfer/coco/data_records/coco*'
# WIKI_PATH = 'drive/My Drive/style_transfer/wikiart/wiki*'

# COCO_PATH = 'gs://coco-tfrecords/coco*'
# WIKI_PATH = 'gs://coco-tfrecords/wiki*'

# warm_start_settings = tf.estimator.WarmStartSettings(
#      ckpt_to_initialize_from='gs://tf-model-dir/style_transfer_dir/',
#      vars_to_warm_start=linear_style_transfer_variables)
# warm_start_settings = None

# from input_dataset import content_dataset, style_dataset
# from utils import show_images, deprocess_image

def deprocess_image(x):
    x = x.copy()
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def reset_tf():
    tf.reset_default_graph()
    tf.keras.backend.clear_session()


def fixed_padding(inputs, kernel_size=3, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def padding_layer(x, name='padding'):
  return fixed_padding(x)


def vgg19_encoder(x, scope='vgg19_encoder', trainable=False):
    outputs = {}
    layers = tf.layers
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        # Block 1
        x = layers.Conv2D(
            64, (3, 3), activation='relu', padding='same',
            name='block1_conv1', trainable=trainable)(x)
        outputs['block1_conv1'] = x
        x = layers.Conv2D(
            64, (3, 3), activation='relu', padding='same',
            name='block1_conv2', trainable=trainable)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = layers.Conv2D(
            128, (3, 3), activation='relu', padding='same',
            name='block2_conv1', trainable=trainable)(x)
        outputs['block2_conv1'] = x
        x = layers.Conv2D(
            128, (3, 3), activation='relu', padding='same',
            name='block2_conv2', trainable=trainable)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(
            256, (3, 3), activation='relu', padding='same',
            name='block3_conv1', trainable=trainable)(x)
        outputs['block3_conv1'] = x

        x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
        x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
        x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3), activation='relu',
                      padding='same', name='block4_conv1')(x)
        outputs['block4_conv1'] = x
    return outputs


def _conv2d(filters, name, padding='valid', trainable=True):
  return tf.layers.Conv2D(
      filters, 3, 1, activation='relu', padding=padding, name=name, trainable=trainable)

def nearest_upsampling(data, scale):
  """Nearest neighbor upsampling implementation.
  Args:
    data: A float32 tensor of size [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].
  """
  with tf.name_scope('nearest_upsampling'):
    bs, h, w, c = data.get_shape().as_list()
    bs = -1 if bs is None else bs
    # Use reshape to quickly upsample the input.  The nearest pixel is selected
    # implicitly via broadcasting.
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, scale, 1, scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, h * scale, w * scale, c])


def vgg19_decoder(encoding, input_layer_vgg_name='block3_conv1', is_training=False, scope='vgg19_decoder', trainable=False):
    # dimension starts at height/8 width/8 from the original image for block4_conv1
    # dimension starts at height/4 width/4 from the original image for block3_conv1
    outputs = {}
    batch_norm = lambda x: tf.layers.BatchNormalization(trainable=is_training, fused=True)(x)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = encoding
        if input_layer_vgg_name == 'block4_conv1':
          x = padding_layer(x)
          x = _conv2d(
            256, name='block4_conv1_decoder', trainable=trainable)(x)
          outputs['block4_conv1'] = x
          x = nearest_upsampling(x, 2)
          x = padding_layer(x)
          x = _conv2d(256, name='block3_conv4_transpose_decoder', trainable=trainable)(x) # h/4 w/4
          x = padding_layer(x)
          x = _conv2d(
            256, name='block3_conv3_decoder', trainable=trainable)(x)
          x = padding_layer(x)
          x = _conv2d(
            256, name='block3_conv2_decoder', trainable=trainable)(x)
        x = padding_layer(x)
        x = _conv2d(
            128, name='block3_conv1_decoder', trainable=trainable)(x)
        outputs['block3_conv1'] = x
        x = nearest_upsampling(x, 2)
        x = padding_layer(x)
        x = _conv2d(
            128, name='block2_conv2_transpose_decoder', trainable=trainable)(x) # h/2 w/2
        x = padding_layer(x)
        x = _conv2d(
            64, name='block2_conv1_decoder', trainable=trainable)(x)
        outputs['block2_conv1'] = x
        x = nearest_upsampling(x, 2)
        x = padding_layer(x)
        x = _conv2d(64, name='block1_conv2_transpose_decoder',
                    trainable=trainable)(x)  # h w
        x = tf.layers.Conv2D(3, 1, activation='tanh', name='block1_conv1_decoder', padding='valid', trainable=trainable)(x)
        outputs['block1_conv1'] = x
        output_image = 150*x
    return output_image, outputs

def compute_content_loss(image, image_decoded):
  return tf.reduce_mean(tf.square(image - image_decoded))


def build_vgg19_model(input_shape, weights=None, scope='vgg19_encoder'):

    img_input = keras.layers.Input(shape=input_shape)
    outputs = vgg19_encoder(img_input, scope)
    model = keras.models.Model(
        img_input, [outputs[s] for s in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']],
        name='vgg19_encoder')
    if weights:
        WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
        weights_path = keras_utils.get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path, by_name=True)
    return model


def export_vgg19_weights(path='vgg19_encoder/weights.ckpt'):
    encoder = build_vgg19_model((None, None, 3), 'imagenet')
    sess = keras.backend.get_session()
    vgg19_encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg19_encoder')
    tf.train.Saver(vgg19_encoder_variables).save(sess, path)
    return path


def build_network_autoencoder_model_fn(content_images, params):

  content_encodings = vgg19_encoder(content_images, trainable=False)
  is_training = params.get('is_training', False)
  content_style_decodeds, decoder_outputs = vgg19_decoder(
      content_encodings['block4_conv1'], 'block4_conv1', is_training=is_training, trainable=True)

  return content_style_decodeds, content_encodings, decoder_outputs


def discriminator(image_vgg19_encoding, scope='discriminator'):
    x = image_vgg19_encoding
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = tf.layers.Flatten()(x)
        x = tf.layers.Dense(1, activation='sigmoid', name='discriminator')(image_vgg19_encoding)
        return x

def model_fn(features, labels, mode, params):
  """Constructs DCGAN from individual generator and discriminator networks."""
  del labels    # Unconditional GAN does not use labels

  use_tpu = params.get('tpu', False)
  estimator_spec_contructor = tf.estimator.EstimatorSpec
  if use_tpu:
      estimator_spec_contructor = tf.contrib.tpu.TPUEstimatorSpec

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  content_images = features['content_images']

  network_outputs = build_network_autoencoder_model_fn(content_images, {'is_training': is_training})

  content_decoded = network_outputs[0]
  content_encodings = network_outputs[1]
  real_images_encodings = content_encodings
  decoder_outputs = network_outputs[2]

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'generated_images': content_decoded,
        'content_images': content_images
    }
    return estimator_spec_contructor(mode=mode, predictions=predictions)

  content_loss = compute_content_loss(content_images, content_decoded)
  total_loss = content_loss
  # g_content_loss = 0.1*content_loss/128 # + tf.reduce_sum(content_losses)

  # Get logits from discriminator
  # d_on_data_logits = tf.squeeze(discriminator(real_images_encodings['block2_conv1']))
  # content_recoded = vgg19_encoder(content_decoded, trainable=False)

  # d_on_g_logits = tf.squeeze(discriminator(content_recoded['block2_conv1']))

  # # Calculate discriminator loss
  # d_loss_on_data = tf.nn.sigmoid_cross_entropy_with_logits(
  #     labels=tf.ones_like(d_on_data_logits),
  #     logits=d_on_data_logits)
  # d_loss_on_gen = tf.nn.sigmoid_cross_entropy_with_logits(
  #     labels=tf.zeros_like(d_on_g_logits),
  #     logits=d_on_g_logits)

  # d_loss = d_loss_on_data + d_loss_on_gen

  # # Calculate generator loss
  # g_gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(
  #     labels=tf.ones_like(d_on_g_logits),
  #     logits=d_on_g_logits)

  # d_loss = tf.reduce_mean(d_loss)
  # g_loss = tf.reduce_mean(g_gan_loss) +g_content_loss

  # if mode == tf.estimator.ModeKeys.TRAIN:
  #   d_optimizer = tf.train.AdamOptimizer(
  #       learning_rate=0.0001, beta1=0.5)
  #   g_optimizer = tf.train.AdamOptimizer(
  #       learning_rate=0.0001, beta1=0.5)
  #   if use_tpu:
  #       d_optimizer = tf.contrib.tpu.CrossShardOptimizer(d_optimizer)
  #       g_optimizer = tf.contrib.tpu.CrossShardOptimizer(g_optimizer)
  #   with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
  #     d_step = d_optimizer.minimize(
  #         d_loss,
  #         var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
  #                                    scope='discriminator'))
  #     g_step = g_optimizer.minimize(
  #         g_loss,
  #         var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
  #                                    scope='vgg19_decoder'))
  #     increment_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
  #     joint_op = tf.group([d_step, g_step, increment_step])
  #     return estimator_spec_contructor(mode=mode, loss=g_loss, train_op=joint_op)

  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = 0.0001
    global_step = tf.train.get_global_step()
    decay_steps, k = 2e3, 0.5
    learning_rate = tf.train.natural_exp_decay(learning_rate, global_step,decay_steps, k, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg19_decoder')
    gradients, variables = zip(*optimizer.compute_gradients(total_loss, var_list))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt_step = optimizer.apply_gradients(zip(gradients, variables), global_step)
    return estimator_spec_contructor(mode=mode, loss=total_loss, train_op=opt_step)

  elif mode == tf.estimator.ModeKeys.EVAL:
    def eval_metric_fn(g_content_loss, g_gan_loss, d_loss):
      # When using TPUs, this function is run on a different machine than the
      # rest of the model_fn and should not capture any Tensors defined there
      metrics = {
          'generator_content_loss': tf.metrics.mean(g_content_loss),
          'generator_gan_loss': tf.metrics.mean(g_gan_loss),
          'discriminator_loss': tf.metrics.mean(d_loss)}
          # 'style': tf.metrics.mean(style_loss),
          # 'total_variation' : tf.metrics.mean(total_variation_loss)}
      return metrics

    def eval_metric_fn(content_loss):
      # When using TPUs, this function is run on a different machine than the
      # rest of the model_fn and should not capture any Tensors defined there
      metrics = {
          'generator_content_loss': tf.metrics.mean(content_loss)
      }
      return metrics

    estimator_spec_args = dict(mode=mode, loss=total_loss)
    # eval_metrics_args = [g_content_loss, g_gan_loss, d_loss]
    eval_metrics_args = [content_loss]

    if use_tpu:
      estimator_spec_args['eval_metrics'] = (eval_metric_fn, eval_metrics_args)
    else:
      estimator_spec_args['eval_metric_ops'] = eval_metric_fn(*eval_metrics_args)

    return estimator_spec_contructor(**estimator_spec_args)

  # Should never reach here
  raise ValueError('Invalid mode provided to model_fn')


def model_fn(features, labels, mode, params):
  """Constructs DCGAN from individual generator and discriminator networks."""
  del labels    # Unconditional GAN does not use labels

  use_tpu = params.get('tpu', False)
  estimator_spec_contructor = tf.estimator.EstimatorSpec
  if use_tpu:
      estimator_spec_contructor = tf.contrib.tpu.TPUEstimatorSpec

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  content_images = features['content_images']
  network_outputs = build_network_autoencoder_model_fn(content_images, {'is_training': is_training})

  content_decoded = network_outputs[0]
  content_encodings = network_outputs[1]
  real_images_encodings = content_encodings
  decoder_outputs = network_outputs[2]

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'generated_images': content_decoded,
        'content_images': content_images
    }
    return estimator_spec_contructor(mode=mode, predictions=predictions)

  content_loss = compute_content_loss(content_images, content_decoded)
  total_loss = content_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = 0.0001
    global_step = tf.train.get_global_step()
    decay_steps, k = 2e3, 0.5
    learning_rate = tf.train.natural_exp_decay(learning_rate, global_step,decay_steps, k, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg19_decoder')
    gradients, variables = zip(*optimizer.compute_gradients(total_loss, var_list))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt_step = optimizer.apply_gradients(zip(gradients, variables), global_step)
    return estimator_spec_contructor(mode=mode, loss=total_loss, train_op=opt_step)

  elif mode == tf.estimator.ModeKeys.EVAL:
    def eval_metric_fn(content_loss):
      metrics = {
          'generator_content_loss': tf.metrics.mean(content_loss)
      }
      return metrics

    estimator_spec_args = dict(mode=mode, loss=total_loss)
    eval_metrics_args = [content_loss]

    if use_tpu:
      estimator_spec_args['eval_metrics'] = (eval_metric_fn, eval_metrics_args)
    else:
      estimator_spec_args['eval_metric_ops'] = eval_metric_fn(*eval_metrics_args)

    return estimator_spec_contructor(**estimator_spec_args)

  # Should never reach here
  raise ValueError('Invalid mode provided to model_fn')





def content_parser(serialized_example, crop_and_resize=True, image_output_shape=(HEIGHT, WIDTH)):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image/encoded": tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.image.decode_jpeg(features["image/encoded"], 3)
    image = tf.reshape(image, [features['image/height'], features['image/width'], -1])
    if crop_and_resize:
        random_size = tf.random_uniform((1,), 0.05, 1)
        random_position = tf.random_uniform((1, 2), 0, 1-random_size)
        random_box = tf.concat([random_position, random_position + random_size], axis=-1)
    else:
        random_box = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)
    image = tf.image.crop_and_resize(tf.expand_dims(image, 0), random_box, tf.constant([0]), image_output_shape)
    image = tf.squeeze(image)
    image = tf.image.random_flip_left_right(image)
    image = vgg19.preprocess_input(image, 'channels_last')
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [image_output_shape[0], image_output_shape[1], 3])
    label = tf.cast([0.0], dtype=tf.float32) # unused
    return image


def create_dataset_from_records(file_pattern, parser, batch_size=4):
    filenames = tf.data.Dataset.list_files(file_pattern)
    dataset = filenames.apply(tf.data.experimental.shuffle_and_repeat(256))
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))
    dataset = dataset.map(parser, num_parallel_calls=16).batch(batch_size, drop_remainder=True).prefetch(batch_size)
    return dataset


def input_fn(params):
    content_dataset = create_dataset_from_records(
        COCO_PATH, content_parser, BATCH_SIZE)
    style_dataset = create_dataset_from_records(
        WIKI_PATH, content_parser, BATCH_SIZE)

    content_dataset = create_dataset_from_records(
        WIKI_PATH, content_parser, BATCH_SIZE)
    style_dataset = create_dataset_from_records(
        COCO_PATH, content_parser, BATCH_SIZE)


    content_images = content_dataset.make_one_shot_iterator().get_next()
    style_images = style_dataset.make_one_shot_iterator().get_next()
    features = {'content_images': content_images,
                'style_images': style_images}
    label = None
    return features, None


def input_predict_fn(params):
    content_dataset = create_dataset_from_records(
        WIKI_PATH, lambda x: content_parser(x, False), BATCH_SIZE)
    style_dataset = create_dataset_from_records(
        COCO_PATH, lambda x: content_parser(x, False), BATCH_SIZE)

    content_images = content_dataset.take(10).make_one_shot_iterator().get_next()
    style_images = style_dataset.take(10).make_one_shot_iterator().get_next()
    features = {'content_images': content_images,
                'style_images': style_images}
    label = None
    return features, None

# model_fn = adain_style_transfer_model_fn
# Test on GPU runtime
model_dir = 'vgg19_autoencoder_block4_conv1/'

vgg19_autoencoder_variables = [
 'vgg19_decoder/block0_conv1/bias:0',
 'vgg19_decoder/block0_conv1/kernel:0',
 'vgg19_decoder/block1_conv1_decoder/bias:0',
 'vgg19_decoder/block1_conv1_decoder/kernel:0',
 'vgg19_decoder/block1_conv2_transpose_decoder/bias:0',
 'vgg19_decoder/block1_conv2_transpose_decoder/kernel:0',
 'vgg19_decoder/block2_conv1_decoder/bias:0',
 'vgg19_decoder/block2_conv1_decoder/kernel:0',
 'vgg19_decoder/block2_conv2_transpose_decoder/bias:0',
 'vgg19_decoder/block2_conv2_transpose_decoder/kernel:0',
 'vgg19_decoder/block3_conv1_decoder/bias:0',
 'vgg19_decoder/block3_conv1_decoder/kernel:0',
 'vgg19_decoder/block3_conv2_decoder/bias:0',
 'vgg19_decoder/block3_conv2_decoder/kernel:0',
 'vgg19_decoder/block4_conv1_decoder/bias:0',
 'vgg19_decoder/block4_conv1_decoder/kernel:0',
 'vgg19_encoder/block1_conv1/bias:0',
 'vgg19_encoder/block1_conv1/kernel:0',
 'vgg19_encoder/block1_conv2/bias:0',
 'vgg19_encoder/block1_conv2/kernel:0',
 'vgg19_encoder/block2_conv1/bias:0',
 'vgg19_encoder/block2_conv1/kernel:0',
 'vgg19_encoder/block2_conv2/bias:0',
 'vgg19_encoder/block2_conv2/kernel:0',
 'vgg19_encoder/block3_conv1/bias:0',
 'vgg19_encoder/block3_conv1/kernel:0',
 'vgg19_encoder/block3_conv2/bias:0',
 'vgg19_encoder/block3_conv2/kernel:0',
 'vgg19_encoder/block3_conv3/bias:0',
 'vgg19_encoder/block3_conv3/kernel:0',
 'vgg19_encoder/block3_conv4/bias:0',
 'vgg19_encoder/block3_conv4/kernel:0',
 'vgg19_encoder/block4_conv1/bias:0',
 'vgg19_encoder/block4_conv1/kernel:0']

vgg19_encoder_variables = [
 'vgg19_encoder/block1_conv1/bias:0',
 'vgg19_encoder/block1_conv1/kernel:0',
 'vgg19_encoder/block1_conv2/bias:0',
 'vgg19_encoder/block1_conv2/kernel:0',
 'vgg19_encoder/block2_conv1/bias:0',
 'vgg19_encoder/block2_conv1/kernel:0',
 'vgg19_encoder/block2_conv2/bias:0',
 'vgg19_encoder/block2_conv2/kernel:0',
 'vgg19_encoder/block3_conv1/bias:0',
 'vgg19_encoder/block3_conv1/kernel:0',
 'vgg19_encoder/block3_conv2/bias:0',
 'vgg19_encoder/block3_conv2/kernel:0',
 'vgg19_encoder/block3_conv3/bias:0',
 'vgg19_encoder/block3_conv3/kernel:0',
 'vgg19_encoder/block3_conv4/bias:0',
 'vgg19_encoder/block3_conv4/kernel:0',
 'vgg19_encoder/block4_conv1/bias:0',
 'vgg19_encoder/block4_conv1/kernel:0'
]

ws = tf.estimator.WarmStartSettings(
    ckpt_to_initialize_from="vgg19_autoencoder/",
    vars_to_warm_start=vgg19_encoder_variables)

train_steps_per_eval = int(1e3)
iterations_per_loop = 50

# CPU-based estimator used for PREDICT (generating images)
cpu_est = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=model_dir, warm_start_from=ws,
    params={'is_training': True})

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def show_images(image_batch, fig_size=24, columns=4):
    rows = (image_batch.shape[0] + 1) // (columns)
    fig = plt.figure(figsize = (fig_size, (fig_size // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        img_hwc = deprocess_image(image_batch[j])
        plt.imshow(img_hwc)
# plt.savefig('vgg19_autoencoder' + '/{}.png'.format(str(iteration).zfill(3))

metrics = cpu_est.evaluate(input_fn=input_predict_fn, steps=10)
tf.logging.info('Finished evaluating')
tf.logging.info(metrics)

current_step = estimator._load_global_step_from_checkpoint_dir(model_dir)   # pylint: disable=protected-access,line-too-long
tf.logging.info('Starting training for %d steps, current step: %d' % (1e5, current_step))

train_steps = 1e5
while current_step < train_steps:
    next_checkpoint = min(current_step + train_steps_per_eval,
                          train_steps)
    next_checkpoint = int(next_checkpoint)
    # next_checkpoint = np.cast(next_checkpoint, np.float64)
    cpu_est.train(input_fn=input_fn,
              max_steps=next_checkpoint)
    current_step = next_checkpoint
    tf.logging.info('Finished training step %d' % current_step)

    if True:
      # Evaluate loss on test set
      metrics = cpu_est.evaluate(input_fn=input_predict_fn, steps=10)
      tf.logging.info('Finished evaluating')
      tf.logging.info(metrics)

    res = cpu_est.predict(input_predict_fn)
    images = [next(res) for i in range(8)]
    images_generated = [d['generated_images'] for d in images]
    images_show = it.chain.from_iterable(list(zip(images_generated)))
    show_images(np.stack(images_show))
    plt.savefig('vgg19_autoencoder_new' + '/{}.png'.format(str(current_step/1000).zfill(5)))
    # Render some generated images
    # res = cpu_est.predict(input_predict_fn)
    # images = [next(res) for i in range(8)]
    # images = [d['generated_images'] for d in images]
    # show_images(np.stack(images))
    # generated_iter = cpu_est.predict(input_fn=input_predict_fn)
    # images = [p['generated_images'][:, :, :] for p in generated_iter]
    # assert len(images) == _NUM_VIZ_IMAGES
    # image_rows = [np.concatenate(images[i:i+10], axis=0)
    #              for i in range(0, _NUM_VIZ_IMAGES, 10)]
    # tiled_image = np.concatenate(image_rows, axis=1)



# reset_tf()
# last_ckpt = tf.train.latest_checkpoint('gs://coco-tfrecords/autoencoder_dir/')
# sess = tf.Session()
# new_saver = tf.train.import_meta_graph('gs://coco-tfrecords/autoencoder_dir/model.ckpt-11000.meta')
# new_saver.restore(sess, 'gs://coco-tfrecords/autoencoder_dir/model.ckpt-11000')
# cpu_est.latest_checkpoint()
# [cpu_est.get_variable_value(s) for s in vgg19_variables]

# sess = tf.get_default_session()

# vgg19_variables_tensor = filter(lambda v: v.name in vgg19_variables, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
# vgg19_variables_tensor = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg19*')
# vgg19_variables_tensor = list(filter(lambda v: 'Adam' not in v.name, vgg19_variables_tensor))

# [v.name for v in vgg19_variables_tensor]

# saver = tf.train.Saver(vgg19_variables_tensor)
# saver.save(sess, 'gs://coco-tfrecords/autoencoder_dir/vgg19_variables.ckpt')

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file("gs://coco-tfrecords/autoencoder_dir/model.ckpt-11520", tensor_name='', all_tensors=True)
model_vars = est.get_variable_names()
model_vars = [s for s in model_vars if 'vgg19' in s and 'Adam' not in s]

def show_vars(scope=''):
    print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))


