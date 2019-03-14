import tensorflow as tf
from layers import conv_layer, t_conv_layer, maxpool_layer


def create_module_down(input, channels_in, channels_out, name,
                  f_size=(3,3), k_size=(2,2)):
    with tf.name_scope(name):
        conv_0 = conv_layer(input,
                            channels_in,
                            channels_out,
                            filter_size=f_size,
                            name='conv_0')
        conv_1 = conv_layer(conv_0,
                            channels_out,
                            channels_out,
                            filter_size=f_size,
                            name='conv_1')

        mp = maxpool_layer(conv_1, kernel_size=k_size, strides=1)

    return mp, conv_1


def create_module_up(input, channels_in, channels_out, name,
                  k_size=(2,2)):
    with tf.name_scope(name):
        conv_0 = conv_layer(input,
                            channels_in,
                            channels_out,
                            filter_size=k_size,
                            name='t_conv_0')
        conv_1 = conv_layer(conv_0,
                            channels_out,
                            channels_out,
                            filter_size=k_size,
                            name='t_conv_1')

        skip = t_conv_layer(conv_1,
                            channels_out,
                            channels_out,
                            filter_size=k_size,
                            name='t_conv_1')
    return skip

def create_module_last(input, channels_in, channels_out, name,
                  f_size=(3,3), k_size=(2,2)):
    with tf.name_scope(name):
        conv_0 = conv_layer(input,
                            channels_in,
                            channels_out,
                            filter_size=f_size,
                            name='conv_0')
        conv_1 = conv_layer(conv_0,
                            channels_out,
                            channels_out,
                            filter_size=f_size,
                            name='conv_1')

        conv_2 = conv_layer(conv_1,
                            channels_out,
                            channels_out,
                            filter_size=(1,1),
                            name='conv_1')

        softmax_image_segment = tf.nn.softmax(conv_2,
                                              name='softmax_logits')
        pred_annotation = tf.argmax(softmax_image_segment,
                                    axis=3,
                                    name="prediction")
        pred_annotation = tf.expand_dims(pred_annotation,
                                         axis=3)

        return pred_annotation, conv_2

def create_module_last_dec(input, channels_in, channels_out, name,
                  f_size=(3,3), k_size=(2,2), channels_img=3):
    with tf.name_scope(name):
        conv_0 = conv_layer(input,
                            channels_in,
                            channels_out,
                            filter_size=f_size,
                            name='conv_0')
        conv_1 = conv_layer(conv_0,
                            channels_out,
                            channels_out,
                            filter_size=f_size,
                            name='conv_1')

        conv_2 = conv_layer(conv_1,
                            channels_out,
                            channels_img,
                            filter_size=(1,1),
                            name='conv_1')

        return conv_2
