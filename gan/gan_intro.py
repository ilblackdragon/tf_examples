from functools import partial
import logging

import numpy as np

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers

import gan_model


def linear_generator(x, hidden_size):
  with tf.variable_scope('Generator'):
    h0 = tf.nn.softplus(layers.linear(x, hidden_size))
    return layers.linear(h0, 1)


def linear_discriminator(x, hidden_size, scope='Discriminator', reuse=False):
   with tf.variable_scope(scope, reuse=reuse):
     h0 = tf.tanh(layers.linear(x, hidden_size * 2))
     h1 = tf.tanh(layers.linear(h0, hidden_size * 2))
     h2 = tf.tanh(layers.linear(h1, hidden_size * 2))
     return tf.sigmoid(layers.linear(h2, 1)), None


def autoencoder_discriminator(x, hidden_size, scope='Discriminator', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    e0 = tf.nn.elu(layers.linear(x, hidden_size))
    e1 = layers.linear(e0, 1)
    d1 = tf.nn.elu(layers.linear(e1, hidden_size))
    d0 = layers.linear(d1, 1)
    return d0, e1


def main():
  mode = 'ebgan'
  params = {
    'learning_rate': 0.005,
    'z_dim': 1,
    'generator': partial(linear_generator, hidden_size=10),
  }
  if mode == 'gan':
    params.update({
      'discriminator': partial(linear_discriminator, hidden_size=10),
      'loss_builder': gan_model.make_gan_loss
    })
  elif mode == 'ebgan':
    params.update({
      'discriminator': partial(autoencoder_discriminator, hidden_size=10),
      'loss_builder': partial(gan_model.make_ebgan_loss, epsilon=0.0001)
    })
  tf.logging._logger.setLevel(logging.INFO)
  data = np.random.normal(4, 0.5, 10000).astype(np.float32)
  data.sort()
  est = learn.SKCompat(learn.Estimator(
      model_fn=gan_model.gan_model, model_dir='models/gan_intro/', params=params))
  print_monitor = tf.train.LoggingTensorHook(['loss_discr', 'loss_generator'],
      every_n_iter=100)
  est.fit(x=data, y=data, steps=10000, batch_size=32, monitors=[print_monitor])


if __name__ == "__main__":
  main()

