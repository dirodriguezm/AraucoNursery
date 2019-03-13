import tensorflow as tf
from layers import conv_layer, maxpool_layer
from conf import params
import numpy as np

class Regressor:
    def __init__(self,
                 img_dim  = (200,200),
                 channels = 1,
                 name='r1',
                 lr = 1e-6):

        self.r_name = name # Regressor type
        self.img_dim = img_dim
        self.channels = channels
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.lr = lr
        with tf.name_scope('input'):
            self.images = tf.placeholder(shape=[None,
                                                self.img_dim[0],
                                                self.img_dim[1],
                                                self.channels],
                                         dtype=tf.float32,
                                         name='input_images')

            self.counts = tf.placeholder(shape=[None, 1],
                                         dtype=tf.int32,
                                         name='output_counts')

            tf.summary.image('input', self.images, 3)

        with tf.name_scope('ARCH_'+self.r_name):
            conv_0 = conv_layer(self.images,
                                channels_in=self.channels,
                                channels_out=params[self.r_name]['output_0'],
                                filter_size=params[self.r_name]['kernel_0'],
                                strides=[1,1,1,1],
                                name='conv_0')

            mp_0   = maxpool_layer(conv_0,
                                   kernel_size=[2,2],
                                   strides=[1, 1, 1, 1],
                                   name='maxpool_0')

            # ==================================================================

            conv_1 = conv_layer(mp_0,
                                channels_in=params[self.r_name]['output_0'],
                                channels_out=params[self.r_name]['output_1'],
                                filter_size=params[self.r_name]['kernel_1'],
                                strides=[1,1,1,1],
                                name='conv_1')

            mp_1   = maxpool_layer(conv_1,
                                   kernel_size=[2,2],
                                   strides=[1, 1, 1, 1],
                                   name='maxpool_1')

            # ==================================================================

            conv_2 = conv_layer(mp_1,
                                channels_in=params[self.r_name]['output_1'],
                                channels_out=params[self.r_name]['output_2'],
                                filter_size=params[self.r_name]['kernel_2'],
                                strides=[1,1,1,1],
                                name='conv_2')

            mp_2   = maxpool_layer(conv_2,
                                   kernel_size=[2, 2],
                                   strides=[1, 1, 1, 1],
                                   name='maxpool_2')

            # ==================================================================

            conv_3 = conv_layer(mp_2,
                                channels_in=params[self.r_name]['output_2'],
                                channels_out=params[self.r_name]['output_3'],
                                filter_size=params[self.r_name]['kernel_3'],
                                strides=[1,1,1,1],
                                name='conv_3')

            mp_3   = maxpool_layer(conv_3,
                                   kernel_size=[2, 2],
                                   strides=[1, 1, 1, 1],
                                   name='maxpool_3')

            # ==================================================================

            conv_4 = conv_layer(mp_3,
                                channels_in=params[self.r_name]['output_3'],
                                channels_out=params[self.r_name]['output_4'],
                                filter_size=params[self.r_name]['kernel_4'],
                                strides=[1, 1, 1, 1],
                                name='conv_4')

            mp_4   = maxpool_layer(conv_4,
                                   kernel_size=[2, 2],
                                   strides=[1, 1, 1, 1],
                                   name='maxpool_4')


            self.output = mp_4

        # ==================================================================
        counts = tf.reduce_sum(self.output, [1, 2])
        with tf.name_scope('loss'):
            self.loss   = tf.losses.mean_squared_error(self.counts,
                                                       counts,
                                                       scope='Loss')
            tf.summary.scalar("mse_loss", self.loss)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ==================================================================
        self.summaries   = tf.summary.merge_all()
        saver            = tf.train.Saver()

        self.writer      = tf.summary.FileWriter('./logs/train')
        self.writer_test = tf.summary.FileWriter('./logs/test')

        self.sess.run(tf.global_variables_initializer())
        self.writer.add_graph(self.sess.graph)


    def train(self, X_images, y_counts, n_epochs=100):
        current_epoch  = 0
        while(current_epoch < n_epochs):
            x = X_images
            y = y_counts
            loss, _, sm = self.sess.run([self.loss, self.train_step, self.summaries],
                                    feed_dict={self.images:x,
                                               self.counts:y
                                              })
            self.writer.add_summary(sm, current_epoch)
            print('epoch: {0} - loss: {1}'.format(current_epoch, loss))
            current_epoch+=1


    def test(self, x_test, y_test, current_epoch):
        loss, sm = self.sess.run([self.loss, self.summaries],
                                    feed_dict={self.images: x_test,
                                               self.counts: y_test
                                               })
        self.writer.add_summary(sm, current_epoch)


class