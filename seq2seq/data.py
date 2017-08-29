import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn


def make_input_fn(mode, filename_in, filename_out, in_vocab_file, out_vocab_file, batch_size, vocab_size,
                  input_max_length, output_max_length, queue_capacity=10000, num_threads=10):
    def input_fn():
        num_epochs = None if mode == tf.estimator.ModeKeys.TRAIN else 1
        filename_in_queue = tf.train.string_input_producer(
            [filename_in], num_epochs=num_epochs)
        filename_out_queue = tf.train.string_input_producer(
            [filename_out], num_epochs=num_epochs)
        reader_in = tf.TextLineReader()
        reader_out = tf.TextLineReader()
        in_list, out_list = [], []
        for _ in range(num_threads):
            in_list.append(reader_in.read(filename_in_queue)[1])
            out_list.append(reader_out.read(filename_out_queue)[1])
        tensor_in = reader_in.read(filename_in_queue)[1]
        tensor_out = reader_out.read(filename_out_queue)[1]
        if mode == tf.estimator.ModeKeys.TRAIN:
            inputs, outputs = tf.train.shuffle_batch(
                (tensor_in, tensor_out), batch_size, capacity=queue_capacity,
                min_after_dequeue=batch_size * 3,
                enqueue_many=True
            )
        else:
            inputs, outputs = tf.train.batch(
                (tensor_in, tensor_out), batch_size, capacity=queue_capacity,
                allow_smaller_final_batch=True)

        # Preprocess inputs.
        inputs = utils.sparse_to_dense_trim(tf.string_split(inputs), output_shape=[batch_size, input_max_length], default_value='<\S>')
        outputs = utils.sparse_to_dense_trim(tf.string_split(outputs), output_shape=[batch_size, output_max_length], default_value='<\S>')
        tf.identity(inputs[0], name='inputs')
        tf.identity(outputs[0], name='outputs')
        in_vocab = tf.contrib.lookup.index_table_from_file(in_vocab_file, vocab_size=vocab_size, default_value=2)
        input_ids = in_vocab.lookup(inputs)
        out_vocab = tf.contrib.lookup.index_table_from_file(out_vocab_file, vocab_size=vocab_size, default_value=2)
        output_ids = out_vocab.lookup(outputs)
        return {'inputs': inputs_ids, 'outputs': outputs_ids}, None
    return input_fn

