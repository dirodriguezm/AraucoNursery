import tensorflow as tf
from tensorflow.contrib import slim

def conv_layer(input, channels_in, channels_out, filter_size=(3,3),
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
        tf.summary.histogram("output", norm_act)

        return norm_act


def maxpool_layer(input, kernel_size, strides=1, name='maxpool'):

    with tf.name_scope(name):
        mp = tf.nn.max_pool(input,
                            ksize=[1, kernel_size[0], kernel_size[1], 1],
                            strides=[1, kernel_size[0], kernel_size[1], 1],
                            padding='VALID')

        return mp

def crop_and_concat(x1,x2):
    '''
    from https://github.com/jakeret/tf_unet/blob/master/tf_unet/layers.py
    '''
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

def upconv(image, name):
    """
    upconvolute by a 2x2 kernel.
    """
    in_ch = image.get_shape()[-1].value
    out_ch = in_ch // 2
    upconv = slim.conv2d_transpose(image,out_ch,2,stride=2,
                                weights_initializer=tf.keras.initializers.he_normal(),
                                padding='SAME',activation_fn=None)
    return upconv


def t_conv_layer(input, channels_in, channels_out, filter_size=(2,2),
                 stride=2, name='deconv2d'):
    '''
    Transpose of 2d convolution
    '''
    with tf.name_scope(name):
        convt = upconv(input, 'conv_t')
        # W = tf.Variable(tf.truncated_normal([filter_size[0],
        #                                      filter_size[1],
        #                                      channels_in,
        #                                      channels_out],
        #                                      stddev=0.1),
        #                 name='W')
        #
        # input_shape = tf.shape(input)
        # output_shape = tf.stack([input_shape[0],
        #                          input_shape[1]*2,
        #                          input_shape[2]*2,
        #                          input_shape[3]//2])
        #
        # convt = tf.nn.conv2d_transpose(input,
        #                                W,
        #                                output_shape,
        #                                strides=[1, stride, stride, 1],
        #                                padding='VALID',
        #                                name="conv2d_transpose")
        #
        # # Saving in Tensorboad
        # tf.summary.histogram("weights", W)
        # tf.summary.histogram("output", convt)


        return convt
