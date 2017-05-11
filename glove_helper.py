"""
GloVe embeddings in Tensorflow

Usage:
  python glove_helper.py path/to/glove.txt path/to/save/glove path/to/save/vocab

Convert GloVe embeddings from https://nlp.stanford.edu/projects/glove/
And use embeddings in your model after you created embeddings Tensor:
  tf.contrib.framework.init_from_checkpoint(path_to_saved_here, {
    'embeddings': 'embeddings'})
"""

import sys
import tensorflow as tf

args = sys.argv

f = open(args[1])
vocab = []
embeddings = []
for line in f:
    tokens = line.split()
    vocab.append(tokens[0])
    embeddings.append([float(val) for val in tokens[1:]])

with tf.Session() as sess:
    v = tf.Variable(tf.constant(embeddings, name="embeddings"))
    sess.run(tf.global_variables_initializer())
    embedding_saver = tf.train.Saver({"embeddings": v})
    embedding_saver.save(sess, args[2])

with open(args[3], 'w') as f:
  for word in vocab:
     f.write(word + '\n')

