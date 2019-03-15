import tensorflow as tf
from layers import conv_layer, maxpool_layer
from conf import params
import numpy as np
import math
import shutil
import os
import pickle


class Regressor:
    def __init__(self,
                 img_dim  = (200,200),
                 channels = 1,
                 name='r1',
                 lr = 1e-6,
                 save_path='./sessions/',
                 rotate = True):

        self.r_name = name # Regressor type
        self.img_dim = img_dim
        self.channels = channels
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.lr = lr
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(self.save_path + '/setup.txt', 'a') as self.out:
            self.out.write('Architecture: ' + str(name)+ '\n')
            self.out.write('number of channels: ' + str(self.channels) + '\n')
            self.out.write('img dimensionality: ' + str(self.img_dim) + '\n')

        with tf.name_scope('input'):
            self.img_input = tf.placeholder(shape=[None,
                                                self.img_dim[0],
                                                self.img_dim[1],
                                                self.channels],
                                         dtype=tf.float32,
                                         name='input_images')

            self.images_rot_90  = tf.contrib.image.rotate(self.img_input, angles=90)
            self.images_rot_180 = tf.contrib.image.rotate(self.img_input, angles=180)
            self.images_rot_270 = tf.contrib.image.rotate(self.img_input, angles=270)


            self.images = tf.concat([self.img_input,
                                    self.images_rot_90,
                                    self.images_rot_180,
                                    self.images_rot_270], axis=0)


            self.counts_in = tf.placeholder(shape=[None,1],
                                         dtype=tf.int32,
                                         name='output_counts')

            self.counts = tf.concat([self.counts_in,
                                     self.counts_in,
                                     self.counts_in,
                                     self.counts_in], axis=0)

            tf.summary.image('image_angle_0', self.images, 1)
            tf.summary.image('image_angle_90', self.images_rot_90, 1)
            tf.summary.image('image_angle_180', self.images_rot_180, 1)
            tf.summary.image('image_angle_270', self.images_rot_270, 1)

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
        self.counts_pred = tf.reduce_sum(self.output, [1, 2])

        with tf.name_scope('loss'):
            self.loss   = tf.losses.mean_squared_error(self.counts,
                                                       self.counts_pred,
                                                       scope='Loss')
            tf.summary.scalar("mse_loss", self.loss)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ==================================================================
        self.summaries   = tf.summary.merge_all()
        self.saver            = tf.train.Saver()

        self.writer      = tf.summary.FileWriter(self.save_path+'/logs/train')
        self.writer_test = tf.summary.FileWriter(self.save_path+'/logs/test')

        self.sess.run(tf.global_variables_initializer())
        self.writer.add_graph(self.sess.graph)


    def train(self, x_train, y_train, x_val, y_val, x_test, y_test, batch_size=32, n_epochs=1000, stop_step=20):
        n_samples  = x_train.shape[0] #Number of samples to train
        iterations = math.ceil(n_samples / batch_size)
        it              = 0         # count for iterations
        current_epoch   = 0         # count for epochs
        count           = 0         # count for early stopping
        best_model_iter = 0         # best model given the minimum loss
        val_loss        = math.inf
        self.best_loss  = math.inf  # minimum (best) loss

        # =================   Begin The Training"    =================
        while(current_epoch < n_epochs):
            down = 0         # for batch limits
            up = batch_size  # for batch limits

            epoch_loss = [] # to save the loss for each iteration and then calculate the epoch average
            for _ in range(iterations):

                x = x_train[down:up]
                y = y_train[down:up]
                loss, _, sm = self.sess.run([self.loss, self.train_step, self.summaries],
                                        feed_dict={self.img_input:x,
                                                   self.counts_in:y
                                                  })

                epoch_loss.append(loss)
                self.writer.add_summary(sm, it)

                # ========================= VALIDATION =========================
                if it % 200 == 0:
                    val_loss = self.validation(x_val, y_val, batch_size=batch_size, current_epoch=current_epoch, current_it=it)

                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        count          = 0 # reset the count
                        if os.path.exists(self.save_path+'/model/best_model'):
                            shutil.rmtree(dir)
                        best_model_iter = it
                        self.saver.save(self.sess, self.save_path+'/model/best_model')

                    else:
                        count+=1

                it+=1
                down = up
                up += batch_size

            train_loss = np.mean(epoch_loss)
            if count == stop_step:
                print('early stopping at step: {0} in epoch {1}'.format(it, current_epoch))
                with open(self.save_path + '/setup.txt', 'a') as self.out:
                    self.out.write('\nepoch_stop:' + str(current_epoch) + '\n')
                    self.out.write('n_iter:' + str(it) + '\n')
                    self.out.write('best model found in iter: ' + str(best_model_iter) + '\n')
                break


            print('epoch: {0} - iter: {1} - train loss: {2:.2f} - val loss: {3:.2f}'.format(current_epoch,
                                                                                            it,
                                                                                            math.log(train_loss),
                                                                                            math.log(val_loss)))
            current_epoch+=1

        self.predict(x_test, y_test, batch_size=batch_size)

    def validation(self, x_val, y_val, batch_size, current_epoch, current_it):

        n_samples = x_val.shape[0]
        iterations = math.ceil(n_samples / batch_size)
        epoch_loss = []
        down = 0
        up = batch_size
        for _ in range(iterations):
            x = x_val[down:up]
            y = y_val[down:up]
            loss, sm = self.sess.run([self.loss, self.summaries],
                                        feed_dict={self.img_input: x,
                                                   self.counts_in: y
                                                   })

            epoch_loss.append(loss)
            down = up
            up += batch_size

            self.writer_test.add_summary(sm, current_it)

        validation_loss = np.mean(epoch_loss)
        return validation_loss


    def predict(self, x_test, y_test, batch_size):

        n_samples = x_test.shape[0]
        iterations = math.ceil(n_samples / batch_size)
        epoch_loss = []
        down = 0
        up = batch_size
        for _ in range(iterations):
            x = x_test[down:up]
            y = y_test[down:up]
            loss, sm, counts_pred = self.sess.run([self.loss, self.summaries, self.counts_pred],
                                        feed_dict={self.img_input: x,
                                                   self.counts_in: y
                                                   })

            epoch_loss.append(loss)
            down = up
            up += batch_size

        test_loss = np.mean(epoch_loss)

        with open(self.save_path + '/setup.txt', 'a') as self.out:
            self.out.write('\ntest loss:' + str(test_loss) + '\n')

        with open(self.save_path+'test_images.pkl', 'wb') as handle:
            pickle.dump({'images':x_test, 'counts':y_test, 'count_pred':counts_pred}, handle, protocol=2)
        return test_loss
