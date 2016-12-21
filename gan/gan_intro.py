import logging

import numpy as np

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers

def generator(x, hidden_size):
  with tf.variable_scope('Generator'):
    h0 = tf.nn.softplus(layers.linear(x, hidden_size))
    return layers.linear(h0, 1)


def discriminator(x, hidden_size, scope='Discriminator', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    h0 = tf.tanh(layers.linear(x, hidden_size * 2))
    h1 = tf.tanh(layers.linear(h0, hidden_size * 2))
    h2 = tf.tanh(layers.linear(h1, hidden_size * 2))
    return tf.sigmoid(layers.linear(h2, 1))


def gan_model(feature, unused_target):
  z = tf.random_uniform(tf.shape(feature), -1, 1, dtype=feature.dtype)
  z.set_shape(feature.get_shape())
  feature_generated = generator(z, 10)
  discr_true = discriminator(feature, 10)
  discr_generated = discriminator(feature_generated, 10, reuse=True)
  loss_discr = tf.reduce_mean(-tf.log(discr_true) - tf.log(1 - discr_generated))
  loss_generator = tf.reduce_mean(-tf.log(discr_generated))

  variables = tf.trainable_variables()
  generator_params = [v for v in variables if v.name.startswith('Generator/')]
  discriminator_params = [v for v in variables if v.name.startswith('Discriminator/')]
  gc = tf.contrib.framework.get_global_step()
  learning_rate = tf.train.exponential_decay(
    0.005, gc, 150, 0.95, staircase=True)
  with tf.variable_scope('Discriminator'):
    discriminator_train_op = layers.optimize_loss(
      loss_discr, gc, variables=discriminator_params,
      learning_rate=learning_rate, optimizer='Adam', summaries=[])
  with tf.variable_scope('Generator'):
    generator_train_op = layers.optimize_loss(
      loss_generator, gc, variables=generator_params,
      learning_rate=learning_rate, optimizer='Adam', summaries=[])

  return (feature_generated, loss_discr + loss_generator,
    tf.group(discriminator_train_op, generator_train_op))


def main():
  tf.logging._logger.setLevel(logging.INFO)
  data = np.random.normal(4, 0.5, 10000).astype(np.float32)
  data.sort()
  est = learn.Estimator(model_fn=gan_model)
  est.fit(x=data, y=data, steps=10000, batch_size=32)


if __name__ == "__main__":
  main()

