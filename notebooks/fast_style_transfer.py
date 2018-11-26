# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import glob
import itertools as it
import json
import os
import random
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.misc
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils
from tensorflow.keras.applications import vgg19
from tensorflow.keras.layers import UpSampling2D, Input, Conv2D, Conv2DTranspose
from tensorflow.python.estimator import estimator


################################################################################

HEIGHT, WIDTH = (480, 640)
BATCH_SIZE = 1

standardize = lambda x: np.array(x)/np.sum(x)

loss_weights = {'content': 1, 'style': 15, 'total_variation': 1e-3}

style_loss_weights = standardize([1, 1, 5, 0])
style_loss_weights = dict(
    zip(['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1'], style_loss_weights))


params_estimator = {'loss_weights': loss_weights,
                    'style_loss_weights': style_loss_weights,
                    'use_tpu': True}

train_steps = 5*1e5
num_shards = 8
batch_size_tpu = 8*BATCH_SIZE

# train_steps = int(1e4)
train_steps_per_eval = 4*int(1e3)
iterations_per_loop = int(1000/BATCH_SIZE)

COCO_PATH = 'drive/My Drive/style_transfer/coco/data_records/coco*'
WIKI_PATH = 'drive/My Drive/style_transfer/wikiart/wiki*'

COCO_PATH = 'gs://coco-tfrecords/coco*'
WIKI_PATH = 'gs://coco-tfrecords/wiki*'

COCO_PATH = '../data/cats/CAT_00/*.jpg'
WIKI_PATH = '../data/style_input/*'


################################################################################
# Image utilities

def save_image(image, path):
    scipy.misc.imsave(path, image)

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


def show_images(image_batch, fig_size=36, columns=3):
    rows = (image_batch.shape[0] + 1) // (columns)
    fig = plt.figure(figsize = (fig_size, (fig_size // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        img_hwc = deprocess_image(image_batch[j])
        plt.imshow(img_hwc)


def reset_tf():
    tf.reset_default_graph()
    tf.keras.backend.clear_session()

################################################################################



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
    return outputs


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


def vgg19_decoder(
        encoding, input_layer_vgg_name='block3_conv1', is_training=False,
        scope='vgg19_decoder', trainable=False):

    # dimension starts at height/8 width/8 from the original image for block4_conv1
    # dimension starts at height/4 width/4 from the original image for block3_conv1
    outputs = {}

    def _conv2d(filters, name, padding='valid', trainable=True):
        return tf.layers.Conv2D(
            filters, 3, 1, activation='relu', padding=padding,
            name=name, trainable=trainable)

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
        output_image = 150*x # vgg19 expect images between -150 and 150
    return output_image, outputs


def gram_matrix(feature_maps, to_reshape=True):
    """Computes the Gram matrix for a set of feature maps."""
    if to_reshape:
        batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
        denominator = tf.to_float(height * width) + 1e-16
        feature_maps = tf.reshape(
            feature_maps, tf.stack([batch_size, height * width, channels]))
    else:
        batch_size, height_width, channels = tf.unstack(tf.shape(feature_maps))
        denominator = tf.to_float(height_width) + 1e-16
    matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    return matrix / denominator


def transformation_architecture(x, name=None, is_training=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = padding_layer(x)
        x = tf.layers.Conv2D(128, 3, activation='relu', padding="valid",
                             name="{}/conv1".format(name))(x)
        x = padding_layer(x)
        x = tf.layers.Conv2D(64, 3, activation='relu', padding="valid",
                             name="{}/conv2".format(name))(x)
        x = padding_layer(x)
        x = tf.layers.Conv2D(32, 3, activation='relu', padding="valid",
                             name="{}/conv3".format(name))(x)
        x = tf.reshape(x, (tf.shape(x)[0], -1, 32))

        x_mean, x_variance  = tf.nn.moments(x, [1], keep_dims=True)
        x_centered = tf.nn.batch_normalization(
            x, x_mean, x_variance, None, None, 1e-12)
        x = gram_matrix((x-x_mean)/128, to_reshape=False)

        x = tf.layers.Dense(512, 'relu', name="{}/dense1".format(name))(x)
        x = tf.layers.BatchNormalization(trainable=is_training)(x)
        x = tf.layers.Dense(128, 'relu', name="{}/dense2".format(name))(x)
        x = tf.layers.BatchNormalization(trainable=is_training)(x)
        x = tf.layers.Dense(32, 'tanh', name="{}/dense_out".format(name))(x)
    return x


def build_transformation_model(content_encoding, style_encoding,
                               name='transformation', is_training=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        base_covariance = transformation_architecture(
            content_encoding, name+'/base_image_transform', is_training=is_training)
        style_covariance = transformation_architecture(
            style_encoding, name+'/style_image_transform', is_training=is_training)

        covariance_transformation = tf.einsum(
            'aij,ajk->aik', base_covariance, style_covariance)

        outputs = [covariance_transformation, base_covariance, style_covariance]
    return outputs


def center_tensor(x):
  x_mean, _  = tf.nn.moments(x, [1], keep_dims=True)
  return x - x_mean


def shift_tensor(x, y):
  y_mean, _ = tf.nn.moments(y, [1], keep_dims=True)
  return x + y_mean


def compressor_fn(input_tensor, name="compressor"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = input_tensor
        x = padding_layer(x)
        x = tf.layers.Conv2D(256, 3, activation='relu', padding="valid",
                             name="{}/conv1".format(name))(x)
        x = padding_layer(x)
        x = tf.layers.Conv2D(128, 3, activation='relu', padding="valid",
                             name="{}/conv2".format(name))(x)
        x = padding_layer(x)
        x = tf.layers.Conv2D(32, 3, activation='relu', padding="valid",
                             name="{}/conv3".format(name))(x)
        output = tf.reshape(x, (tf.shape(x)[0], -1, 32))
    return output


def uncompressor_fn(input_tensor, image_shape, name="uncompressor"):

  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    reshape_size = (-1, image_shape[1], image_shape[2], 32)
    x = tf.reshape(input_tensor, reshape_size)
    x = padding_layer(x)
    x = tf.layers.Conv2D(64, 3, activation='relu', padding="valid",
                         name="{}/conv1".format(name))(x)
    x = padding_layer(x)
    x = tf.layers.Conv2D(128, 3, activation='relu', padding="valid",
                         name="{}/conv2".format(name))(x)
    x = padding_layer(x)
    x = tf.layers.Conv2D(256, 3, activation='relu', padding="valid",
                         name="{}/conv3".format(name))(x)
    return x


def compute_content_loss(image, image_decoded):
  return tf.reduce_mean(tf.square(image - image_decoded))


def compute_style_loss(style_map, combination_map):
  s_gram = gram_matrix(style_map) * 1e-3
  c_gram = gram_matrix(combination_map) * 1e-3
  s_dims = tf.shape(style_map)
  denominator = tf.cast(s_dims[0]*s_dims[3], tf.float32)
  style_loss = tf.reduce_sum(tf.square(s_gram - c_gram)) / denominator
  return style_loss


def build_network_fast_linear_transfer_model_fn(content_images, style_images,
                                                is_training=False):

    content_images_dims = tf.shape(content_images)

    content_encodings = vgg19_encoder(content_images)
    style_encodings = vgg19_encoder(style_images)

    final_layer = 'block3_conv1'
    dimension_divisor = 4 if final_layer == 'block3_conv1' else 8

    with tf.variable_scope('fast_style_transfer', reuse=tf.AUTO_REUSE):

        colorization_scalar, _, _ = build_transformation_model(
          content_encodings[final_layer],
          style_encodings[final_layer],
          is_training=is_training)

        content_encoding_compressed = compressor_fn(
            content_encodings[final_layer])
        style_encoding_compress = compressor_fn(
            style_encodings[final_layer])

        content_encoding_compressed_centered = center_tensor(
            content_encoding_compressed)
        content_style_compressed = tf.matmul(
            content_encoding_compressed_centered, colorization_scalar)

        content_style_shift = shift_tensor(
          content_style_compressed, style_encoding_compress)

        content_style_encodings = uncompressor_fn(
            content_style_shift, [
              tf.shape(content_encoding_compressed)[0],
              content_images_dims[1]/4,
              content_images_dims[2]/4,
              32])

    content_style_decodeds, _ = vgg19_decoder(content_style_encodings, final_layer)

    return content_style_decodeds, content_encodings, style_encodings

def fast_linear_style_transfer_model_fn(features, labels, mode, params):
  """Constructs DCGAN from individual generator and discriminator networks."""
  del labels    # Unconditional GAN does not use labels

  content_images = features['content_images']
  style_images = features['style_images']
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  network_outputs = build_network_fast_linear_transfer_model_fn(content_images, style_images, is_training)
  content_style_decodeds = network_outputs[0]
  content_encodings = network_outputs[1]
  style_encodings = network_outputs[2]

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'generated_images': content_style_decodeds,
        'content_images': content_images,
        'style_images': style_images
    }
    export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions['generated_images'])}
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Use params['batch_size'] for the batch size inside model_fn
  # batch_size = params['batch_size']   # pylint: disable=unused-variable
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  generated_images = content_style_decodeds

  # Calculate generator loss

  # content_loss = compute_content_loss(content_images, content_style_decodeds)
  # total_loss = content_loss

  content_style_recoded = vgg19_encoder(content_style_decodeds)

  loss_weights = {'content': 1, 'style': 1000, 'total_variation': 1e-9}
  loss_weights.update(params.get('loss_weights', {}))

  style_loss_weights = {'block1_conv1': 0.65, 'block2_conv1': 0.3,
                        'block3_conv1': 0.05,  'block4_conv1': 0.05}
  style_loss_weights.update(params.get('style_loss_weights', {}))

  content_loss = loss_weights['content'] * compute_content_loss(
      content_encodings['block3_conv1'], content_style_recoded['block3_conv1'])
  style_loss = []
  for k in ['block1_conv1', 'block2_conv1', 'block3_conv1']: # , 'block4_conv1']:
    sl = compute_style_loss(style_encodings[k], content_style_recoded[k])
    style_loss.append(style_loss_weights[k]*sl)
  style_loss = loss_weights['style']*tf.reduce_sum(style_loss)
  total_variation_loss = loss_weights['total_variation'] * tf.reduce_mean(
      tf.image.total_variation(content_style_decodeds))

  total_loss = content_loss + style_loss + total_variation_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = 0.0001
    global_step = tf.train.get_global_step()
    decay_steps, k = 2*1e5, 0.5
    learning_rate = tf.train.natural_exp_decay(learning_rate, global_step, decay_steps, k, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    variables_to_train = [
        v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # if 'vgg19_encoder' not in v.name and 'vgg19_decoder' not in v.name]
        if 'vgg19_encoder' not in v.name]

    gradients, variables = zip(*optimizer.compute_gradients(total_loss, variables_to_train))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt_step = optimizer.apply_gradients(zip(gradients, variables), global_step)
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=opt_step)

  elif mode == tf.estimator.ModeKeys.EVAL:
    def _eval_metric_fn(content_loss, style_loss, total_variation_loss):
      # When using TPUs, this function is run on a different machine than the
      # rest of the model_fn and should not capture any Tensors defined there
      metrics = {
          'content': tf.metrics.mean(content_loss),
          'style': tf.metrics.mean(style_loss),
          'total_variation' : tf.metrics.mean(total_variation_loss)}
      return metrics

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        eval_metrics_ops=_eval_metric_fn(content_loss, style_loss, total_variation_loss))

  # Should never reach here
  raise ValueError('Invalid mode provided to model_fn')

################################################################################
## Input

def content_parser(serialized_example, crop_and_resize=True,
                   image_output_shape=(HEIGHT, WIDTH)):
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
        random_size = tf.random_uniform((1,), 0.99, 1)
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

def jpeg_file_parser(filename, crop_and_resize=True, image_output_shape=(HEIGHT, WIDTH)):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, 3)
    if crop_and_resize:
        random_size = tf.random_uniform((1,), 1.0, 1.0)
        random_position = tf.random_uniform((1, 2), 0, 1-random_size)
        random_box = tf.concat([random_position, random_position + random_size], axis=-1)
    else:
        random_box = tf.constant([[0, 0, 1, 1]], dtype=tf.float32)
    if image_output_shape != (None, None):
        image = tf.image.crop_and_resize(tf.expand_dims(image, 0), random_box, tf.constant([0]), image_output_shape)
    image = tf.squeeze(image)
    image = vgg19.preprocess_input(image, 'channels_last')
    image = tf.cast(image, tf.float32)
    return image

def create_dataset_from_records(file_pattern, parser, batch_size=4):
    filenames = tf.data.Dataset.list_files(file_pattern)
    dataset = filenames.apply(tf.data.experimental.shuffle_and_repeat(200))
    dataset = dataset.map(parser, num_parallel_calls=4).batch(batch_size, drop_remainder=True).prefetch(batch_size)
    return dataset

def create_dataset_from_jpeg(file_pattern, parser, batch_size=4):
    filenames = tf.data.Dataset.list_files(file_pattern)
    dataset = filenames.repeat()
    # dataset = filenames.apply(tf.data.experimental.shuffle_and_repeat(200))
    dataset = dataset.map(parser, num_parallel_calls=4).batch(batch_size, drop_remainder=True).prefetch(batch_size)
    return dataset


def input_fn(params):

    content_dataset = create_dataset_from_records(
        COCO_PATH, content_parser, BATCH_SIZE)
    style_dataset = create_dataset_from_records(
        WIKI_PATH, content_parser, BATCH_SIZE)

    content_images = content_dataset.make_one_shot_iterator().get_next()
    style_images = style_dataset.make_one_shot_iterator().get_next()
    features = {'content_images': content_images,
                'style_images': style_images}
    label = None
    return features, None


def input_predict_fn(params):
    content_dataset = create_dataset_from_jpeg(
        COCO_PATH, jpeg_file_parser, BATCH_SIZE)
    style_dataset = create_dataset_from_jpeg(
        WIKI_PATH, jpeg_file_parser, BATCH_SIZE)
    content_images = content_dataset.make_one_shot_iterator().get_next()
    style_images = style_dataset.make_one_shot_iterator().get_next()
    features = {'content_images': content_images,
                'style_images': style_images}
    label = None
    return features, None

################################################################################
### Estimators parameters

linear_style_transfer_variables = [
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

ws = tf.estimator.WarmStartSettings(
     ckpt_to_initialize_from='gs://tf-model-dir/autoencoder_dir/',
     vars_to_warm_start=linear_style_transfer_variables)
ws = None

model_dir = 'style_transfer_fast_linear_5_dir'
model_fn = fast_linear_style_transfer_model_fn

cpu_est = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=model_dir,
    params=params_estimator)

################################################################################
# cpu_est.evaluate(input_predict_fn, steps=10)

# if True:
#     res = cpu_est.predict(input_predict_fn)
#     images = [next(res) for i in range(4)]
#     images_generated = [d['generated_images'] for d in images]
#     images_content = [d['content_images'] for d in images]
#     images_style = [d['style_images'] for d in images]
#     images_show = it.chain.from_iterable(
#         list(zip(images_content, images_style, images_generated)))
#     show_images(np.stack(images_show))
# images_generated[1].shape

if True:
    res = cpu_est.predict(input_predict_fn)
    images = [next(res) for i in range(32)]
    images_generated = [d['generated_images'] for d in images]
    images_content = [d['content_images'] for d in images]
    images_style = [d['style_images'] for d in images]
    images_show = it.chain.from_iterable(
        list(zip(images_content, images_style, images_generated)))
    # show_images(np.stack(images_show))


for i, img in enumerate(images_generated):
    path = 'output' + '/david_mimi_{}.png'.format(str(int(i)).zfill(5))
    save_image(deprocess_image(img), path)


def process_image_raw(image_raw, height=512, width=512):
    x = tf.image.decode_jpeg(image_raw, 3)
    x = tf.image.resize_images(x, [height, width])
    x = tf.cast(x, tf.float32)
    x = tf.expand_dims(x, 0) # 4D tensor
    return x


def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""

  serialized_tf_example = tf.placeholder(
      dtype=tf.string, shape=(),
      name='input_example_tensor')

  receiver_tensors = {'examples': serialized_tf_example}
  feature_spec = {
      "content_images": tf.FixedLenFeature((), tf.string, ''),
      "style_images": tf.FixedLenFeature((), tf.string, ''),
  }

  features = tf.parse_single_example(serialized_tf_example, feature_spec)
  print(features)
  features = {k: process_image_raw(v) for k, v in features.items()}

  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def serving_input_reciever_raw_fn():
    """An input receiver that expects raw images."""
    columns = [('content_images', tf.float32),
               ('style_images', tf.float32),
               ('device_type', tf.string)]
    feature_placeholders = {
     name: tf.placeholder(dtype, (), name=name + "_placeholder")
     for name, dtype in columns
    }

cpu_est.export_saved_model('style_transfer_fast_linear_5_dir', serving_input_receiver_fn)
