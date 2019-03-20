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

        with tf.name_scope('ARCH_R1'):
            conv_0 = conv_layer(images,
                                channels_in=3,
                                channels_out=16,
                                filter_size=(9,9),
                                strides=[1,1,1,1],
                                name='conv_0')

            mp_0   = maxpool_layer(conv_0,
                                   kernel_size=[2,2],
                                   strides=[1, 1, 1, 1],
                                   name='maxpool_0')

            # ==================================================================

            conv_1 = conv_layer(mp_0,
                                channels_in=16,
                                channels_out=32,
                                filter_size=(7,7),
                                strides=[1,1,1,1],
                                name='conv_1')

            mp_1   = maxpool_layer(conv_1,
                                   kernel_size=[2,2],
                                   strides=[1, 1, 1, 1],
                                   name='maxpool_1')

            # ==================================================================

            conv_2 = conv_layer(mp_1,
                                channels_in=32,
                                channels_out=16,
                                filter_size=(7,7),
                                strides=[1,1,1,1],
                                name='conv_2')

            mp_2   = maxpool_layer(conv_2,
                                   kernel_size=[2, 2],
                                   strides=[1, 1, 1, 1],
                                   name='maxpool_2')

            # ==================================================================

            conv_3 = conv_layer(mp_2,
                                channels_in=16,
                                channels_out=8,
                                filter_size=(7,7),
                                strides=[1,1,1,1],
                                name='conv_3')

            # ==================================================================

            conv_4 = conv_layer(conv_3,
                                channels_in=8,
                                channels_out=1,
                                filter_size=(1,1),
                                strides=[1, 1, 1, 1],
                                name='conv_4')

            self.output = conv_4


    def get_logits(self):
        return self.output

    def generate_loss(self, prediction):
        loss   = tf.losses.mean_squared_error(self.counts,
                                              prediction,
                                              scope='Loss')
        tf.summary.scalar("mse_loss", loss)

        return loss
