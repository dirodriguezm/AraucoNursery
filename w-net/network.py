import tensorflow as tf
from modules import *


class W_net:
    def __init__(self, n_samples, dimen0, dimen1, channels):

        with tf.name_scope('input'):
            self.images = tf.placeholder(dtype=tf.float32,
                                         shape=[n_samples, dimen0, dimen1, channels],
                                         name='images_input')

        with tf.name_scope('U_enc'):
            h = create_module_1(self.images)



if __name__ == "__main__":

    wnet = W_net(10, 224, 224, 1)