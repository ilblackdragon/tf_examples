"""Utilities for image visualization."""

import scipy.misc
import numpy as np
import tensorflow as tf


def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1]))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w] = x

    scipy.misc.imsave(save_path, img)


class SaveVisualizationHook(tf.train.SessionRunHook):

    def __init__(self, save_path, every_n_iter=1000):
        super(SaveVisualizationHook, self).__init__()
        self._every_n_iter = every_n_iter
        self._save_path = save_path

    def begin(self):
        self._step = 0

    def before_run(self, run_context):
        if self._step % self._every_n_iter != 0:
            return None
        return tf.train.SessionRunArgs({'generated': 'generated:0'})

    def after_run(self, run_context, run_values):
        if self._step % self._every_n_iter == 0:
            output = np.reshape(run_values.results['generated'], [32, 28, 28])
            save_visualization(output, [4, 8], save_path=self._save_path)
        self._step += 1

