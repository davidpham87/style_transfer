from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

def deprocess_image(processed_img):
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

def show_images(image_batch, fig_size=24, columns=4):
    rows = (image_batch.shape[0] + 1) // (columns)
    fig = plt.figure(figsize = (fig_size, (fig_size // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        img_hwc = deprocess_image(image_batch[j])
        plt.imshow(img_hwc)
