import tensorflow as tf
from layers import conv_layer, maxpool_layer
from models import Regressor_3, AlexNet, CRNN
import numpy as np
import os
import multiprocessing
import math

class Pipeline:
    def __init__(self, save_path):
        tf.reset_default_graph()
        self.save_path = save_path
        self.sess = tf.Session()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def load_image(self, image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [self.img_dimens[0],
                                               self.img_dimens[1]])
        image = tf.image.per_image_standardization(image)
        return image, label

    def load_data(self, img_dimension=(5,5), n_channels=3):

        self.x = tf.placeholder('float32',
                                 shape=[None,
                                      img_dimension[0],
                                      img_dimension[1],
                                      3],

                                 name='input_images')

        self.y = tf.placeholder('int32',
                                 shape=[None, 1],
                                 name='counts_images')

        self.img_dimens = img_dimension
        self.n_channels = n_channels

        n_process = int(multiprocessing.cpu_count()/2)
        self.dataset_img = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        self.dataset_img = self.dataset_img.map(self.load_image,
                                                num_parallel_calls=n_process)


    def create_batches(self, batch_size=32):
        batches = self.dataset_img.batch(batch_size)
        batches = batches.prefetch(buffer_size=1)

        self.iterator = batches.make_initializable_iterator()

        self.images, self.counts = self.iterator.get_next()


    def construct_model(self, model_name='r1', lr=1e-6):
        self.model_name = model_name
        tf.summary.image('image_angle_0', self.images, 1)

        with open(self.save_path + '/setup.txt', 'a') as self.out:
            self.out.write('Architecture: ' + str(model_name)+ '\n')
            self.out.write('number of channels: ' + str(self.n_channels) + '\n')
            self.out.write('img dimensionality: ' + str(self.img_dimens) + '\n')

        if model_name == 'r3':
            self.model = Regressor_3(self.images,
                                     self.counts,
                                     lr=lr)
        if model_name == 'alexnet':
            self.model = AlexNet(self.images,
                                 self.counts,
                                 lr=0.003)
        if model_name == 'lstm':
            self.model = CRNN(self.images,
                              self.counts,
                              lr=0.003)


        self.loss = self.model.loss()

        tf.summary.scalar("loss", self.loss)

        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)

        tf.add_to_collection(name='saved', value=self.loss)
        tf.add_to_collection(name='saved', value=self.model.pred_counts)
        if self.model_name == 'lstm':
            tf.add_to_collection(name='saved', value=self.reconstruction)

        tf.add_to_collection(name='placeholder', value=self.x)
        tf.add_to_collection(name='placeholder', value=self.y)
        tf.add_to_collection(name='placeholder', value=self.images)
        tf.add_to_collection(name='placeholder', value=self.counts)
        tf.add_to_collection(name='placeholder', value=self.model.keep_prob)
        tf.add_to_collection(name='placeholder', value=self.model.is_training)
        tf.add_to_collection(name='placeholder', value=self.iterator.initializer)


        self.summaries   = tf.summary.merge_all()
        self.saver       = tf.train.Saver()

        self.writer      = tf.summary.FileWriter(self.save_path+'/logs/train')
        self.writer_test = tf.summary.FileWriter(self.save_path+'/logs/test')


    def train(self, x_train, y_train, keep_prob=0.5):

        epoch_train_loss = []

        self.sess.run(self.iterator.initializer,
                      feed_dict={self.x: x_train,
                                 self.y: y_train})

        try:
            while True:
                train_loss,_,_,_,sm = self.sess.run([self.loss,
                                                     self.images,
                                                     self.counts,
                                                     self.train_step,
                                                     self.summaries],
                        feed_dict={self.model.keep_prob: keep_prob,
                                   self.model.is_training: True})

                epoch_train_loss.append(train_loss)
                self.writer.add_summary(sm, self.it)
                self.it += 1

        except tf.errors.OutOfRangeError:
            pass

        return np.mean(epoch_train_loss)

    def validation(self, x_val, y_val):

        epoch_val_loss = []

        self.sess.run(self.iterator.initializer,
                      feed_dict={self.x: x_val,
                                 self.y: y_val})
        try:
            while True:
                #Aqui no esta reutilizando los batches
                val_loss,_,_,sm = self.sess.run([self.loss,
                                                 self.images,
                                                 self.counts,
                                                 self.summaries],
                                                 feed_dict={self.model.keep_prob: 1,
                                                            self.model.is_training: False})
                epoch_val_loss.append(val_loss)
                self.writer_test.add_summary(sm, self.it)
                self.it += 1

        except tf.errors.OutOfRangeError:
            pass

        return np.mean(epoch_val_loss)

    def test(self, x_test, y_test):

        epoch_test_loss = []

        self.sess.run(self.iterator.initializer,
                      feed_dict={self.x: x_test,
                                 self.y: y_test})
        try:
            while True:
                #Aqui no esta reutilizando los batches
                test_loss,_,_= self.sess.run([self.loss,
                                              self.images,
                                              self.counts],
                                              feed_dict={self.model.keep_prob: 1,
                                                         self.model.is_training: False})
                epoch_test_loss.append(test_loss)

        except tf.errors.OutOfRangeError:
            pass

        with open(self.save_path + '/setup.txt', 'a') as self.out:
            self.out.write('best model found in iter: ' + str(self.best_model_epoch) + '\n')
        return np.mean(epoch_test_loss)



    def fit(self, x_train, y_train, x_val, y_val, n_epochs=10,
            stop_step=20, keep_prob=0.5):

        # init variables
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        self.writer.add_graph(self.sess.graph)
        # Variable for early stopping
        best_loss = math.inf
        nochanges = 0 # count to break the train
        # GLobal train iterations
        self.it = 0
        self.best_model_epoch  = n_epochs

        for epoch in range(n_epochs):
            train_loss = self.train(x_train, y_train, keep_prob)

            if epoch % 2 == 0:
                val_loss   = self.validation(x_val, y_val)
                print('Epoch: {0} Train Loss: {1} Val Loss: {2}'.format(epoch,
                                                                        train_loss,
                                                                        val_loss))
                if val_loss < best_loss:
                    print('saving best model on epoch {0}'.format(epoch))
                    best_loss = val_loss
                    nochanges = 0

                    if os.path.exists(self.save_path+'/model/best_model'):
                        shutil.rmtree(dir)

                    self.best_model_epoch = epoch
                    self.saver.save(self.sess, self.save_path+'/model/best_model')
                else:
                    nochanges += 1

            if nochanges == stop_step:
                print('Early stopping at epoch: {}'.format(self.best_model_epoch))
                break
