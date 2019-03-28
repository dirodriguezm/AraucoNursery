import tensorflow as tf
from layers import *

class Regressor_3:
    def __init__(self,
                 images,
                 counts,
                 lr = 1e-6):

        self.lr = lr
        self.images = images
        self.counts = counts
        self.keep_prob = tf.placeholder('float32', name='keep_prob')

        with tf.name_scope('ARCH_R1'):
            conv_0 = conv_layer(images,
                                channels_in=3,
                                channels_out=24,
                                filter_size=(5,5),
                                strides=[1,1,1,1],
                                name='conv_0')

            mp_0   = maxpool_layer(conv_0,
                                   kernel_size=[2,2],
                                   strides=[1, 2, 2, 1],
                                   name='maxpool_0')
            # ==================================================================

            conv_1 = conv_layer(mp_0,
                                channels_in=24,
                                channels_out=48,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_1')

            mp_1   = maxpool_layer(conv_1,
                                   kernel_size=[3,3],
                                   strides=[1, 2, 2, 1],
                                   name='maxpool_1')
            # ==================================================================

            conv_2 = conv_layer(mp_1,
                                channels_in=48,
                                channels_out=24,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_2')

            # ==================================================================

            conv_3 = conv_layer(conv_2,
                                channels_in=24,
                                channels_out=12,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_3')

            # ==================================================================

            conv_4 = conv_layer(conv_3,
                                channels_in=12,
                                channels_out=1,
                                filter_size=(1,1),
                                strides=[1, 1, 1, 1],
                                name='conv_4')

            flatten = tf.layers.flatten(conv_4, name='flatten')

            fc_0    = tf.layers.dense(flatten, 512)

            fc_1    = tf.layers.dense(fc_0, 256)

            fc_2    = tf.layers.dense(fc_1, 128)

            fc_3    = tf.layers.dense(fc_2, 64)

            fc_4    = tf.layers.dense(fc_3, 1)

            self.output = fc_4

    def get_logits(self):
        return self.output

class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self,
                 images,
                 counts,
                 lr = 1e-6):

        self.lr = lr
        self.images = images
        self.counts = counts
        self.keep_prob = tf.placeholder('float32', name='keep_prob')

        with tf.name_scope("AlexNet"):
            conv_0 = conv_layer(images,
                                channels_in=3,
                                channels_out=96,
                                filter_size=(11,11),
                                strides=[1,4,4,1],
                                name='conv_0')

            conv_0 = tf.nn.relu(conv_0)

            conv_0 = tf.nn.local_response_normalization(conv_0,
                                                        depth_radius=5.0,
                                                        bias=2.0,
                                                        alpha=1e-4,
                                                        beta=0.75)
            conv_0 = maxpool_layer(conv_0,
                                   kernel_size=[3,3],
                                   strides=[1, 2, 2, 1],
                                   name='maxpool_0')

            # =================================================================
            conv_1 = conv_layer(conv_0,
                                channels_in=96,
                                channels_out=256,
                                filter_size=(5,5),
                                strides=[1,1,1,1],
                                name='conv_1')

            conv_1 = tf.nn.relu(conv_1)

            conv_1 = tf.nn.local_response_normalization(conv_1,
                                                        depth_radius=5.0,
                                                        bias=2.0,
                                                        alpha=1e-4,
                                                        beta=0.75)
            conv_1 = maxpool_layer(conv_1,
                                   kernel_size=[3,3],
                                   strides=[1, 1, 1, 1],
                                   name='maxpool_1')

            # =================================================================
            conv_2 = conv_layer(conv_1,
                                channels_in=256,
                                channels_out=384,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_2')

            conv_2 = tf.nn.relu(conv_2)

            # =================================================================
            conv_3 = conv_layer(conv_2,
                                channels_in=384,
                                channels_out=384,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_3')

            conv_3 = tf.nn.relu(conv_3)

            # =================================================================
            conv_4 = conv_layer(conv_3,
                                channels_in=384,
                                channels_out=256,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_4')

            conv_4 = tf.nn.relu(conv_4)

            conv_4 = maxpool_layer(conv_4,
                                   kernel_size=[3,3],
                                   strides=[1, 2, 2, 1],
                                   name='maxpool_4')

            # =================================================================
            flatten_5 = tf.layers.Flatten()(conv_4)
            fc_5      = fc_layer(flatten_5, 4096, 4096, name='fc_5')
            fc_5      = tf.nn.relu(fc_5)
            fc_5      = tf.nn.dropout(fc_5, self.keep_prob)

            # =================================================================
            fc_6    = fc_layer(fc_5, 4096, 4096, name='fc_6')
            fc_6    = tf.nn.relu(fc_6)
            fc_6    = tf.nn.dropout(fc_6, self.keep_prob)

            # =================================================================
            fc_7    = fc_layer(fc_6, 4096, 1, name='fc_7')

            self.output = fc_7

    def get_logits(self):
        return self.output

class AlexNetMod(object):
    """Implementation of the AlexNet."""

    def __init__(self,
                 images,
                 counts,
                 lr = 1e-6):

        self.lr = lr
        self.images = images
        self.counts = counts
        self.keep_prob = tf.placeholder('float32', name='keep_prob')
        self.is_train = tf.placeholder(tf.bool, name="is_train")


        with tf.name_scope("AlexNetMod"):
            x_norm = tf.layers.batch_normalization(images, training=self.is_train)
            conv_0 = conv_layer(x_norm,
                                channels_in=3,
                                channels_out=96,
                                filter_size=(4,4),
                                strides=[1,1,1,1],
                                name='conv_0')

            conv_0 = tf.nn.relu(conv_0)

            conv_0 = maxpool_layer(conv_0,
                                   kernel_size=[3,3],
                                   strides=[1, 3, 3, 1],
                                   name='maxpool_0')
            conv_0 = tf.layers.batch_normalization(conv_0, training=self.is_train)
            

            # =================================================================
            conv_1 = conv_layer(conv_0,
                                channels_in=96,
                                channels_out=256,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_1')

            conv_1 = tf.nn.relu(conv_1)

            conv_1 = maxpool_layer(conv_1,
                                   kernel_size=[2,2],
                                   strides=[1, 2, 2, 1],
                                   name='maxpool_1')
            conv_1 = tf.layers.batch_normalization(conv_1, training=self.is_train)

            # =================================================================
            conv_2 = conv_layer(conv_1,
                                channels_in=256,
                                channels_out=384,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_2')

            conv_2 = tf.nn.relu(conv_2)

            conv_2 = tf.layers.batch_normalization(conv_2, training=self.is_train)
            # =================================================================
            conv_3 = conv_layer(conv_2,
                                channels_in=384,
                                channels_out=384,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_3')

            conv_3 = tf.nn.relu(conv_3)

            conv_3 = tf.layers.batch_normalization(conv_3, training=self.is_train)
            # =================================================================
            conv_4 = conv_layer(conv_3,
                                channels_in=384,
                                channels_out=256,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_4')

            conv_4 = tf.nn.relu(conv_4)

            conv_4 = maxpool_layer(conv_4,
                                   kernel_size=[2,2],
                                   strides=[1, 2, 2, 1],
                                   name='maxpool_4')
            conv_4 = tf.layers.batch_normalization(conv_4, training=self.is_train)

            # =================================================================
            flatten_5 = tf.layers.Flatten()(conv_4)
            fc_5      = fc_layer(flatten_5, 16384, 512, name='fc_5')
            fc_5      = tf.nn.relu(fc_5)
            fc_5      = tf.nn.dropout(fc_5, self.keep_prob)
            fc_5 = tf.layers.batch_normalization(fc_5, training=self.is_train)
            # =================================================================
            fc_6    = fc_layer(fc_5, 512, 64, name='fc_6')
            fc_6    = tf.nn.relu(fc_6)
            fc_6    = tf.nn.dropout(fc_6, self.keep_prob)
            fc_6 = tf.layers.batch_normalization(fc_6, training=self.is_train)
            # =================================================================
            fc_7    = fc_layer(fc_6, 64, 1, name='fc_7')
            self.output = fc_7

    def get_logits(self):
        return self.output

