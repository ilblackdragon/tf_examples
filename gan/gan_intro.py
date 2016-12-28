from functools import partial
import logging

import numpy as np

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers


def linear_generator(x, hidden_size):
  with tf.variable_scope('Generator'):
    h0 = tf.nn.softplus(layers.linear(x, hidden_size))
    return layers.linear(h0, 1)


def linear_discriminator(x, hidden_size, scope='Discriminator', reuse=False):
   with tf.variable_scope(scope, reuse=reuse):
     h0 = tf.tanh(layers.linear(x, hidden_size * 2))
     h1 = tf.tanh(layers.linear(h0, hidden_size * 2))
     h2 = tf.tanh(layers.linear(h1, hidden_size * 2))
     return tf.sigmoid(layers.linear(h2, 1))


def autoencoder_discriminator(x, hidden_size, scope='Discriminator', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    e0 = tf.nn.elu(layers.linear(x, hidden_size))
    e1 = layers.linear(e0, 1)
    d1 = tf.nn.elu(layers.linear(e1, hidden_size))
    d0 = layers.linear(d1, 1)
    return d0


def make_gan_loss(feature, feature_generated, discr_true, discr_generated):
  loss_discr = tf.reduce_mean(-tf.log(discr_true) - tf.log(1 - discr_generated))
  loss_generator = tf.reduce_mean(-tf.log(discr_generated))
  return loss_discr, loss_generator


def make_ebgan_loss(feature, feature_generated, discr_true, discr_generated,
                    epsilon):
  loss_discr = (tf.reduce_mean(tf.square(feature - discr_true)) +
    tf.reduce_mean(tf.nn.relu(epsilon -
    tf.reduce_mean(tf.square(feature_generated - discr_generated), 1))))
  loss_generator = tf.reduce_mean(tf.square(feature_generated -
    discr_generated))
  return loss_discr, loss_generator

 
def gan_model(feature, unused_target, mode, params):
  # Retrieve params
  generator = params.get('generator')
  discriminator = params.get('discriminator')
  loss_builder = params.get('loss_builder')
  initial_learning_rate = params.get('learning_rate', 0.05)
  decay_steps = params.get('decay_steps', 150)
  decay_rate = params.get('decay_rate', 0.95)

  # Create noise Z.
  z = tf.random_uniform(tf.shape(feature), -1, 1, dtype=feature.dtype)
  z.set_shape(feature.get_shape())

  # Generate fake example.
  feature_generated = generator(z)

  # Discriminate true and generated example.
  discr_true = discriminator(feature)
  discr_generated = discriminator(feature_generated, reuse=True)

  # Build GAN losses.
  loss_discr, loss_generator = loss_builder(
      feature, feature_generated, discr_true, discr_generated)
  loss_discr = tf.identity(loss_discr, name="loss_discr")
  loss_generator = tf.identity(loss_generator, name="loss_generator")
  tf.summary.scalar("loss/discr", loss_discr)
  tf.summary.scalar("loss/generator", loss_generator)
  tf.summary.scalar("loss/total", loss_discr + loss_generator)

  variables = tf.trainable_variables()
  generator_params = [v for v in variables if v.name.startswith('Generator/')]
  discriminator_params = [v for v in variables if v.name.startswith('Discriminator/')]
  gc = tf.contrib.framework.get_global_step()
  learning_rate = tf.train.exponential_decay(
    initial_learning_rate, gc, decay_steps, decay_rate, staircase=True)
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
  mode = 'ebgan'
  params = {
    'learning_rate': 0.005,
    'generator': partial(linear_generator, hidden_size=10),
  }
  if mode == 'gan':
    params.update({
      'discriminator': partial(linear_discriminator, hidden_size=10),
      'loss_builder': make_gan_loss
    })
  elif mode == 'ebgan':
    params.update({
      'discriminator': partial(autoencoder_discriminator, hidden_size=10),
      'loss_builder': partial(make_ebgan_loss, epsilon=0.0001)
    })
  tf.logging._logger.setLevel(logging.INFO)
  data = np.random.normal(4, 0.5, 10000).astype(np.float32)
  data.sort()
  est = learn.SKCompat(learn.Estimator(model_fn=gan_model,
  model_dir='gan_intro/', params=params))
  print_monitor = tf.train.LoggingTensorHook(['loss_discr', 'loss_generator'],
      every_n_iter=100)
  est.fit(x=data, y=data, steps=10000, batch_size=32, monitors=[print_monitor])


if __name__ == "__main__":
  main()

