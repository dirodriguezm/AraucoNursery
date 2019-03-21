import tensorflow as tf
from layers import conv_layer, maxpool_layer

class Regressor_1:
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
                                   strides=[1, 1, 1, 1],
                                   name='maxpool_0')

            # ==================================================================

            conv_1 = conv_layer(mp_0,
                                channels_in=24,
                                channels_out=48,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_1')

            mp_1   = maxpool_layer(conv_1,
                                   kernel_size=[2,2],
                                   strides=[1, 1, 1, 1],
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
            fc_0_d  = tf.nn.dropout(fc_0, keep_prob=self.keep_prob)

            fc_1    = tf.layers.dense(fc_0_d, 256)

            fc_2    = tf.layers.dense(fc_1, 128)
            fc_2_d  = tf.nn.dropout(fc_2, keep_prob=self.keep_prob)

            fc_3    = tf.layers.dense(fc_2_d, 64)

            fc_4    = tf.layers.dense(fc_3, 1)
            fc_4_d  = tf.nn.dropout(fc_4, keep_prob=self.keep_prob)

            self.output = fc_4_d

    def get_logits(self):
        return self.output

    def generate_loss(self, prediction):
        loss   = tf.losses.mean_squared_error(self.counts,
                                              prediction,
                                              scope='Loss')
        tf.summary.scalar("mse_loss", loss)

        return loss
