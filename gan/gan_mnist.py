from functools import partial
import logging

import numpy as np
import scipy.misc

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers

import gan_model


def conv_generator(x, hidden_size):
  with tf.variable_scope('Generator'):
    return layers.linear(x, 784)


def conv_discriminator(x, hidden_size, scope='Discriminator', reuse=False):
   with tf.variable_scope(scope, reuse=reuse):
     h0 = tf.tanh(layers.linear(x, hidden_size * 2))
     h1 = tf.tanh(layers.linear(h0, hidden_size * 2))
     h2 = tf.tanh(layers.linear(h1, hidden_size * 2))
     return tf.sigmoid(layers.linear(h2, 1))


def autoencoder_discriminator(x, hidden_size, scope='Discriminator', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    e0 = tf.nn.elu(layers.linear(x, hidden_size))
    e1 = layers.linear(e0, 2)
    d1 = tf.nn.elu(layers.linear(e1, hidden_size))
    d0 = layers.linear(d1, 784)
    return d0, e1


def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1]))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w] = x

    scipy.misc.imsave(save_path, img)


def main():
  mode = 'ebgan'
  params = {
    'learning_rate': 0.0001,
    'generator': partial(conv_generator, hidden_size=10),
  }
  if mode == 'gan':
    params.update({
      'discriminator': partial(conv_discriminator, hidden_size=10),
      'loss_builder': gan_model.make_gan_loss
    })
  elif mode == 'ebgan':
    params.update({
      'discriminator': partial(autoencoder_discriminator, hidden_size=10),
      'loss_builder': partial(gan_model.make_ebgan_loss, epsilon=0.05)
    })
  tf.logging._logger.setLevel(logging.INFO)
  mnist_data = learn.datasets.load_dataset('mnist')
  est = learn.SKCompat(learn.Estimator(
      model_fn=gan_model.gan_model, model_dir='gan_mnist/', params=params))
  imgs = np.array([x for idx, x in enumerate(mnist_data.train.images) if
    mnist_data.train.labels[idx] == 4])
  for i in range(100):
    print_monitor = tf.train.LoggingTensorHook(['loss_discr', 'loss_generator'],
      every_n_iter=100)
    est.fit(x=imgs, y=None, steps=2000, batch_size=32, monitors=[print_monitor])
    output = est.predict(x=np.zeros([32, 784], dtype=np.float32))
    save_visualization(np.reshape(output, [32, 28, 28]), [4, 8])


if __name__ == "__main__":
  main()

