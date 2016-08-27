"""
This is an example of how to use TensorFlow as "interpreter" of graph
functions.
"""

import tensorflow as tf

def run_tf(func):
  def wrapper():
    with tf.Graph().as_default() as graph:
      x = func()
      with tf.Session('') as session:
        return session.run(x)
  return wrapper

@run_tf
def hello_world():
  return tf.Print([], ["Hello world!"])

@run_tf
def add_3_5():
  return tf.constant(3) + tf.constant(5)

hello_world()
print add_3_5()
