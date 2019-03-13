import tensorflow as tf


def conv_layer(input, channels_in, channels_out, filter_size=(5,5),
               strides=[1, 1, 1, 1], name='conv'):

    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([filter_size[0],
                                             filter_size[1],
                                             channels_in,
                                             channels_out],
                                             stddev=0.1),
                        name='W')

        conv = tf.nn.conv2d(input,
                            W,
                            strides=strides,
                            padding="SAME",
                            name='conv_layer')

        # Saving in Tensorboad
        tf.summary.histogram("weights", W)
        tf.summary.histogram("output", conv)

        return conv

def maxpool_layer(input, kernel_size, strides=[1, 1, 1, 1],
                  padding='SAME', name='maxpool'):

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
