from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import vgg19

COCO_PATH = '../data/data_records/coco*'
WIKI_PATH = '../data/data_records/wiki*'


# c_imgs = sess.run(content_files.make_one_shot_iterator().get_next())

# coco_path = 'gs://coco-tfrecords/coco*'
# wiki_path = 'gs://coco-tfrecords/wiki*'

def content_parser(serialized_example, crop_and_resize=True, image_output_shape=(256, 256)):
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
        random_size = tf.random_uniform((1,), 0.3, 1)
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

def create_dataset_from_records(file_pattern, parser, batch_size=8):
    filenames = tf.data.Dataset.list_files(file_pattern)
    dataset = filenames.apply(tf.data.experimental.shuffle_and_repeat(10))
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=2))
    dataset = dataset.map(parser, num_parallel_calls=2).batch(batch_size, drop_remainder=True).prefetch(batch_size)
    return dataset

content_dataset = create_dataset_from_records(
    COCO_PATH, content_parser)
style_dataset = create_dataset_from_records(
    WIKI_PATH, lambda x: content_parser(x, False))
content_style_dataset = tf.data.Dataset.zip((content_dataset, style_dataset))
# dataset = tf.data.Dataset.zip((content_dataset, zero_dataset))
# zero_dataset = tf.data.Dataset.from_tensor_slices([0.0])
# dataset = tf.data.Dataset.zip((content_style_dataset, zero_dataset))

if False:
    imgs = sess.run(content_dataset.make_one_shot_iterator().get_next())
    show_images(imgs)
    plt.show()

    imgs = sess.run(style_dataset.make_one_shot_iterator().get_next())
    show_images(imgs)
    plt.show()
