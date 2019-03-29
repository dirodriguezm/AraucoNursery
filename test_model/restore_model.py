import tensorflow as tf
from tensorflow.contrib import image
import multiprocessing
import numpy as np

class RestoreModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.name = 'best_model'
        self.path = self.model_path +'/model/'

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.path + self.name + '.meta')
            saver.restore(sess , save_path=self.path + self.name)

            t_v = tf.trainable_variables() # get all trained variables

            self.t_v = t_v

        self.load_data(img_dimension=(100,100), n_channels=3)
        self.make_graph()

    def load_image(self, image, label):
        #image_string = tf.read_file(path)
        # Don't use tf.image.decode_image, or the output shape will be undefined
        #image = tf.image.decode_png(image_string, channels=3)

        # This will convert to float values in [0, 1]
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

        batches = self.dataset_img.batch(10)
        batches = batches.prefetch(buffer_size=1)

        self.iterator = batches.make_initializable_iterator()

        self.images, self.counts = self.iterator.get_next()


    def make_graph(self):

        conv_0 = tf.nn.conv2d(self.images,
                              self.t_v[0],
                              strides=[1,1,1,1],
                              padding="SAME",
                              name='conv_layer')

        conv_0 = tf.nn.bias_add(conv_0, self.t_v[1])

        mp_0 = tf.nn.max_pool(conv_0,
                            [1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='mp_layer')
        # ==================================================================
        conv_1 = tf.nn.conv2d(mp_0,
                              self.t_v[2],
                              strides=[1,1,1,1],
                              padding="SAME",
                              name='conv_layer')

        conv_1 = tf.nn.bias_add(conv_1, self.t_v[3])

        mp_1 = tf.nn.max_pool(conv_1,
                            [1, 3, 3, 1],
                             strides=[1, 2, 2, 1],
                             padding='VALID',
                             name='mp_layer')
        # ==================================================================

        conv_2 = tf.nn.conv2d(mp_1,
                              self.t_v[4],
                              strides=[1,1,1,1],
                              padding="SAME",
                              name='conv_layer')

        conv_2 = tf.nn.bias_add(conv_2, self.t_v[5])

        # ==================================================================
        conv_3 = tf.nn.conv2d(conv_2,
                              self.t_v[6],
                              strides=[1,1,1,1],
                              padding="SAME",
                              name='conv_layer')

        conv_3 = tf.nn.bias_add(conv_3, self.t_v[7])
        # ==================================================================
        conv_4 = tf.nn.conv2d(conv_3,
                              self.t_v[8],
                              strides=[1,1,1,1],
                              padding="SAME",
                              name='conv_layer')

        conv_4 = tf.nn.bias_add(conv_4, self.t_v[9])

        flatten = tf.layers.flatten(conv_4, name='flatten')

        fc_0    = tf.matmul(flatten, self.t_v[10]) + self.t_v[11]

        fc_1    = tf.matmul(fc_0, self.t_v[12]) + self.t_v[13]

        fc_2    = tf.matmul(fc_1, self.t_v[14]) + self.t_v[15]

        fc_3    = tf.matmul(fc_2, self.t_v[16]) + self.t_v[17]

        fc_4    = tf.matmul(fc_3, self.t_v[18]) + self.t_v[19]

        self.output = fc_4


        self.pred_counts = tf.nn.relu(self.output, name='activated_output')

        self.loss = tf.reduce_mean(tf.square(self.pred_counts -
                                   tf.cast(self.counts, 'float32')))

        self.writer      = tf.summary.FileWriter('.')

    def test(self, images, counts):

        total_loss = []
        total_pred = []

        with tf.Session() as sess:
            # init variables
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            #self.writer.add_graph(sess.graph)

            sess.run(self.iterator.initializer,
                     feed_dict={self.x: images,
                                self.y: counts})

            try:
                while True:
                    loss, y_pred, _, _ = sess.run([self.loss, self.pred_counts,
                                                     self.images, self.counts])
                    total_loss.append(loss)
                    total_pred.append(y_pred)
            except:
                pass
            print('test loss: ',np.mean(loss))
            total_pred = np.concatenate(total_pred, axis=0)

            return total_pred.flatten()
