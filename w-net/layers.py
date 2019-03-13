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
        B = tf.Variable(tf.constant(0.1, shape=[channels_out]), name='B')
        conv = tf.nn.conv2d(input,
                            W,
                            strides=strides,
                            padding="SAME",
                            name='conv_layer')

        act = tf.nn.relu(conv+B)
        norm_act = tf.layers.batch_normalization(act)

        # Saving in Tensorboad
        tf.summary.histogram("weights", W)
        tf.summary.histogram("bias", B)
        tf.summary.histogram("output", conv)

        return norm_act


def maxpool_layer(input, kernel_size, strides=1, name='maxpool'):

    with tf.name_scope(name):
        mp = tf.layers.max_pooling2d(input,
                                     pool_size=kernel_size,
                                     strides=strides,
                                     name='mp_layer')

        return mp