# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

flags = tf.app.flags.FLAGS

class DataManager(object):
  def load(self):
    # Load dataset
    dataset_zip = np.load('data/megaman.npz')

    self.imgs       = dataset_zip['arr_0']
    self.n_samples = self.imgs.shape[0]
    # 27312 for megaman
    
  @property
  def sample_size(self):
    return self.n_samples

  def get_image(self, index):
    img = self.imgs[index]
    image_size = flags.input_width * flags.input_height * flags.input_channels
    img = img.reshape(image_size)
    return img

  def get_images(self, indices):
    images = []
    for index in indices:
      img = self.get_image(index)
      images.append(img)
    return images

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices)
