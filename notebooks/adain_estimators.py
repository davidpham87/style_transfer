from __future__ import print_function

import argparse
import glob
import json
import numpy as np
import os
import os
import random
import scipy.misc
import time

import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils
from tensorflow.keras.applications import vgg19
from tensorflow.keras.layers import UpSampling2D, Input, Conv2D, Conv2DTranspose
from tensorflow.python.estimator import estimator


################################################################################

HEIGHT, WIDTH = (224, 224)
BATCH_SIZE = 4

standardize = lambda x: np.array(x)/np.sum(x)

loss_weights = {'content': 1, 'style': 1e5, 'total_variation': 1e-8}

style_loss_weights = standardize([0.1, 0.5, 10, 10])
style_loss_weights = dict(
    zip(['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1'], style_loss_weights))


params_estimator = {'loss_weights': loss_weights,
                    'style_loss_weights': style_loss_weights,
                    'use_tpu': True}

train_steps = 1e5
num_shards = 8
batch_size_tpu = 8*BATCH_SIZE

# train_steps = int(1e4)
train_steps_per_eval = int(1e3)
iterations_per_loop = int(1000/BATCH_SIZE)

COCO_PATH = 'drive/My Drive/style_transfer/coco/data_records/coco*'
WIKI_PATH = 'drive/My Drive/style_transfer/wikiart/wiki*'

COCO_PATH = '../data/data_records/coco*'
WIKI_PATH = '../data/data_records/wiki*'

CONTENT_TEST_PATH = '../data/data_records/content_test*'
STYLE_TEST_PATH ='../data/data_records/style_test*'

# COCO_PATH = 'gs://coco-tfrecords/coco*'
# WIKI_PATH = 'gs://coco-tfrecords/wiki*'

################################################################################


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

        x = layers.Conv2D(
            256, (3, 3), activation='relu', padding='same',
            name='block3_conv2')(x)
        x = layers.Conv2D(
            256, (3, 3), activation='relu', padding='same',
            name='block3_conv3')(x)
        x = layers.Conv2D(
            256, (3, 3), activation='relu', padding='same',
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

def instance_normalization(x):
    x_mean, x_var = tf.nn.moments(x, [1, 2], keep_dims=True)
    x_center_scaled = tf.nn.batch_normalization(x, x_mean, x_var, None, None, 1e-12)
    return x_center_scaled

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


def compute_content_loss(image, image_decoded):
  return tf.reduce_mean(tf.square(image - image_decoded))


def compute_style_loss(style_map, combination_map):
  s_mu, s_var = tf.nn.moments(style_map, [1, 2])
  combination_mu, combination_var = tf.nn.moments(combination_map, [1, 2])
  mu_loss = tf.reduce_sum(tf.square(s_mu - combination_mu))
  var_loss = tf.reduce_sum(tf.square(tf.sqrt(s_var) - tf.sqrt(combination_var)))
  style_dims = tf.shape(style_map)
  denominator = tf.cast(tf.reduce_prod(style_dims), tf.float32)
  sl = (3*mu_loss + var_loss) / denominator
  return sl, 3*mu_loss/denominator, var_loss/denominator


def build_network_adain_model_fn(content_images, style_images, is_training=False):

  content_images_dims = tf.shape(content_images)

  content_encodings = vgg19_encoder(content_images)
  style_encodings = vgg19_encoder(style_images)

  content_style_encodings = adaptive_instance_normalization(content_encodings['block4_conv1'],
      style_encodings['block4_conv1'])

  content_style_decodeds, _ = vgg19_decoder(content_style_encodings, 'block4_conv1', trainable=True)

  return content_style_decodeds, content_encodings, style_encodings


def adain_model_fn(features, labels, mode, params):
  """Constructs DCGAN from individual generator and discriminator networks."""
  del labels    # Unconditional GAN does not use labels

  content_images = features['content_images']
  style_images = features['style_images']
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  network_outputs = build_network_adain_model_fn(content_images, style_images, is_training)

  content_style_decodeds = network_outputs[0]
  content_encodings = network_outputs[1]
  style_encodings = network_outputs[2]

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'generated_images': content_style_decodeds,
        'content_images': content_images,
        'style_images': style_images,
    }

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Use params['batch_size'] for the batch size inside model_fn
  # batch_size = params['batch_size']   # pylint: disable=unused-variable
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  generated_images = content_style_decodeds

  # Calculate generator loss
  content_style_recoded = vgg19_encoder(content_style_decodeds)

  loss_weights = {'content': 1, 'style': 1000, 'total_variation': 1e-9}
  loss_weights.update(params.get('loss_weights', {}))

  style_loss_weights = {'block1_conv1': 0.65, 'block2_conv1': 0.3,
                        'block3_conv1': 0.05, 'block4_conv1': 0.05}
  style_loss_weights.update(params.get('style_loss_weights', {}))

  content_loss = loss_weights['content'] * compute_content_loss(
      content_encodings['block3_conv1'], content_style_recoded['block3_conv1'])
  style_loss = []
  style_loss_dict = {}
  for k in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
    sl, sl_mu, sl_var = compute_style_loss(style_encodings[k], content_style_recoded[k])
    style_loss.append(style_loss_weights[k]*sl)
    style_loss_dict[k + '_mu'] = sl_mu
    style_loss_dict[k + '_var'] = sl_var
  style_loss = loss_weights['style']*tf.reduce_sum(style_loss)
  total_variation_loss = loss_weights['total_variation'] * tf.reduce_mean(
      tf.image.total_variation(content_style_decodeds))

  total_loss = content_loss + style_loss + total_variation_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    global_step = tf.train.get_global_step()
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg19_decoder')
    gradients, variables = zip(*optimizer.compute_gradients(total_loss, var_list))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    opt_step = optimizer.apply_gradients(zip(gradients, variables), global_step)
    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=opt_step)

  elif mode == tf.estimator.ModeKeys.EVAL:
    def _eval_metric_fn(content_loss, style_loss, total_variation_loss, style_loss_dict):
      # When using TPUs, this function is run on a different machine than the
      # rest of the model_fn and should not capture any Tensors defined there
      metrics = {
          'content': tf.metrics.mean(content_loss),
          'style': tf.metrics.mean(style_loss),
          'total_variation' : tf.metrics.mean(total_variation_loss)}
      for k, v in style_loss_dict.items():
          metrics[k] = tf.metrics.mean(v)
      return metrics

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        eval_metric_ops=_eval_metric_fn(content_loss, style_loss, total_variation_loss, style_loss_dict))

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
        random_size = tf.random_uniform((1,), 0.85, 0.95)
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


def jpeg_file_parser(filename, crop_and_resize=True, image_output_shape=(None, None)):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, 3)
    if crop_and_resize:
        random_size = tf.random_uniform((1,), 0.5, 0.6)
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
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=1))
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

    content_test_dataset = create_dataset_from_records(
        CONTENT_TEST_PATH, lambda x: content_parser(x, False), BATCH_SIZE)

    style_test_dataset = create_dataset_from_records(
        STYLE_TEST_PATH, lambda x: content_parser(x, False), BATCH_SIZE)


    content_images = content_test_dataset.shuffle(10).make_one_shot_iterator().get_next()
    style_images = style_test_dataset.shuffle(10).make_one_shot_iterator().get_next()
    features = {'content_images': content_images,
                'style_images': style_images}
    label = None
    return features, None


model_fn = adain_model_fn

linear_style_transfer_variables = [
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
     ckpt_to_initialize_from='vgg19_autoencoder_block4_conv1',
     vars_to_warm_start=linear_style_transfer_variables)
# ws = None
# model_dir = 'gs://tf-model-dir/style_transfer_matmul_dir/'
model_dir = 'adain_dir'

cpu_est = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=model_dir,
    warm_start_from=ws,
    params=params_estimator)

# cpu_est.evaluate(input_predict_fn, steps=10)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import itertools as it

def show_images(image_batch, fig_size=24, columns=3):
    rows = (image_batch.shape[0] + 1) // (columns)
    fig = plt.figure(figsize = (fig_size, (fig_size // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        img_hwc = deprocess_image(image_batch[j])
        plt.imshow(img_hwc)

if False:
  sess = keras.backend.get_session()
  res = sess.run(dataset.make_one_shot_iterator().get_next())
  show_images(res[0][:, :, :, :, 1])
  # show_images(res[0][:, :, :, :, 0])

metrics = cpu_est.evaluate(input_fn=input_predict_fn, steps=2)
tf.logging.info('Finished evaluating')
tf.logging.info(metrics)

if True:
  tf.gfile.MakeDirs(os.path.join(model_dir, 'generated_images'))

  current_step = estimator._load_global_step_from_checkpoint_dir(model_dir)   # pylint: disable=protected-access,line-too-long
  tf.logging.info('Starting training for %d steps, current step: %d' %
                  (1e5, current_step))

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

    if True:
      res = cpu_est.predict(input_predict_fn)
      images = [next(res) for i in range(4)]
      images_generated = [d['generated_images'] for d in images]
      images_content = [d['content_images'] for d in images]
      images_style = [d['style_images'] for d in images]
      images_show = it.chain.from_iterable(list(zip(images_content, images_style, images_generated)))
      show_images(np.stack(images_show))
      plt.savefig('adain_dir' + '/{}.png'.format(str(int(current_step/1000)).zfill(5)))
      plt.close('all')

tf.logging.info('Finished generating images')

if True:
  res = cpu_est.predict(input_predict_fn)
  images = [next(res) for i in range(4)]
  images_generated = [d['generated_images'] for d in images]
  images_content = [d['content_images'] for d in images]
  images_style = [d['style_images'] for d in images]
  images_show = it.chain.from_iterable(list(zip(images_content, images_style, images_generated)))
  show_images(np.stack(images_show))

images_generated[1].shape

if True:
  images = [next(res) for i in range(16)]
  images_generated = [d['generated_images'] for d in images]
  images_content = [d['content_images'] for d in images]
  images_style = [d['style_images'] for d in images]
  images_show = it.chain.from_iterable(list(zip(images_content, images_style, images_generated)))
  show_images(np.stack(images_show), 36)
  plt.savefig('adain_dir' + '/test_{}.png'.format(str(int(20)).zfill(5)))

graph = tf.Graph()
sess = tf.Session(graph=graph)
tf.saved_model.loader.load(sess, [tag_constants.Serving], model_dir)
