import tensorflow as tf


def conv_layer(input, channels_in, channels_out, filter_size=(5,5),
               strides=[1, 1, 1, 1], name='conv', is_training=True):

    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([filter_size[0],
                                             filter_size[1],
                                             channels_in,
                                             channels_out],
                                             stddev=0.1),
                        name='W')
        B = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='B')

        conv = tf.nn.conv2d(input,
                            W,
                            strides=strides,
                            padding="SAME",
                            name='conv_layer')

        conv = tf.nn.bias_add(conv, B)
        conv = tf.layers.batch_normalization(conv, training=is_training)

        # Saving in Tensorboad
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("output", conv)

        return conv


def maxpool_layer(input, kernel_size, strides=[1, 1, 1, 1],
                  padding='VALID', name='maxpool'):

    with tf.name_scope(name):
        mp = tf.nn.max_pool(input,
                            [1,
                             kernel_size[0],
                             kernel_size[1],
                             1],
                             strides=strides,
                             padding=padding,
                             name='mp_layer')
        return mp

def fc_layer(input, size_in, size_out, name='fc'):

    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([size_in, size_out],
                                             stddev=0.1),
                                             name='W')
        B = tf.Variable(tf.constant(0.1, shape=[size_out]),
                                    name='B')

        out = tf.nn.xw_plus_b(input, W, B)

        return out
