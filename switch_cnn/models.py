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
        # Probabilidad de prender una neurona
        self.keep_prob = tf.placeholder('float32', name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, name="is_train");

        with tf.name_scope('ARCH_R1'):
            conv_0 = conv_layer(images,
                                channels_in=3,
                                channels_out=24,
                                filter_size=(5,5),
                                strides=[1,1,1,1],
                                name='conv_0',
                                is_training=self.is_training)

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
                                name='conv_1',
                                is_training=self.is_training)

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
                                name='conv_2',
                                is_training=self.is_training)

            # ==================================================================

            conv_3 = conv_layer(conv_2,
                                channels_in=24,
                                channels_out=12,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv_3',
                                is_training=self.is_training)

            # ==================================================================

            conv_4 = conv_layer(conv_3,
                                channels_in=12,
                                channels_out=1,
                                filter_size=(1,1),
                                strides=[1, 1, 1, 1],
                                name='conv_4',
                                is_training=self.is_training)

            flatten = tf.layers.flatten(conv_4, name='flatten')

            fc_0    = tf.layers.dense(flatten, 512)

            fc_1    = tf.layers.dense(fc_0, 256)

            fc_2    = tf.layers.dense(fc_1, 128)

            fc_3    = tf.layers.dense(fc_2, 64)

            fc_4    = tf.layers.dense(fc_3, 1)


        with tf.name_scope('Logits_Transform'):

            self.pred_counts = tf.nn.relu(fc_4,
                                          name='activated_output')


    def loss(self):
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.square(self.pred_counts -
                                  tf.cast(self.counts, 'float32')))

            return loss


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

        with tf.name_scope('Logits_Transform'):

            self.pred_counts = tf.nn.relu(fc_7,
                                          name='activated_output')
            tf.summary.tensor_summary('pred_counts', self.pred_counts)

    def loss(self):
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.square(self.pred_counts -
                                  tf.cast(self.counts, 'float32')))

            return loss


class CRNN:
    def __init__(self,
                 images,
                 counts,
                 lr = 1e-4):#1e-6):

        self.lr = lr
        # Probabilidad de prender una neurona
        self.keep_prob = tf.placeholder('float32', name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, name="is_train")
        self.images = images
        self.counts = counts

        cell   = tf.contrib.rnn.LayerNormBasicLSTMCell(128,
                          dropout_keep_prob=self.keep_prob)


        with tf.name_scope('Encoder'):

            out_0   = conv_layer(images,
                                 channels_in=3,
                                 channels_out=32,
                                 filter_size=(5,5),
                                 strides=[1,1,1,1],
                                 name='conv_0',
                                 is_training=self.is_training)

            out_1   = maxpool_layer(out_0,
                                    kernel_size=[2,2],
                                    strides=[1, 2, 2, 1],
                                    name='maxpool_1')

            out_2   = conv_layer(out_1,
                                 channels_in=32,
                                 channels_out=32,
                                 filter_size=(5,5),
                                 strides=[1,1,1,1],
                                 name='conv_2',
                                 is_training=self.is_training)

            out_3   = maxpool_layer(out_2,
                                    kernel_size=[2,2],
                                    strides=[1, 2, 2, 1],
                                    name='maxpool_3')

            out_4 = tf.layers.flatten(out_3, name='flatten_4')

            features    = tf.layers.dense(out_4, 20)

        with tf.name_scope('Decoder'):
            dimens = out_3.shape
            out_5  = tf.layers.dense(features, dimens[1]*dimens[2]*dimens[3])

            out_6  = tf.reshape(out_5, shape=(-1, dimens[1], dimens[2], dimens[3]))

            out_7  = unpooling(out_6,
                               kernel_shape=(2,2),
                               output_shape=tf.shape(out_2),
                               name='unpooling_7')

            out_8 = deconv_layer(out_7,
                                 kernel_shape=[5, 5, 32, 32],
                                 output_shape=tf.shape(out_1),
                                 name='deconv_8',
                                 is_training=self.is_training)

            out_9  = unpooling(out_8,
                               kernel_shape=(2,2),
                               output_shape=tf.shape(out_0),
                               name='unpooling_9')

            out_10 = deconv_layer(out_9,
                                 kernel_shape=[5, 5, 3, 32],
                                 output_shape=tf.shape(images),
                                 name='deconv_10',
                                 is_training=self.is_training)


        with tf.name_scope('Regressor'):
            dimens = features.shape
            input_rnn = tf.reshape(features, shape=[-1, dimens[1], 1])
            output, state = tf.nn.dynamic_rnn(cell,
                                              input_rnn,
                                              dtype='float32')

            out_11    = tf.layers.dense(output[:,-1,:], 1)


        with tf.name_scope('Logits_Transform'):

            self.pred_counts = tf.nn.relu(out_11,
                                          name='activated_output')
            self.reconstruction = tf.nn.sigmoid(out_10)
            tf.summary.image('reconstruction', self.reconstruction, 1)


    def loss(self):
        with tf.name_scope('Loss'):
            loss_regressor = tf.reduce_mean(tf.square(self.pred_counts -
                                      tf.cast(self.counts, 'float32')))

            loss_reconstruction = tf.nn.l2_loss(self.images -
                                                self.reconstruction)

            loss = loss_regressor + loss_reconstruction

            return loss
