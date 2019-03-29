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

            saved_collection = tf.get_collection('saved')

            placeholder_collection = tf.get_collection('placeholder')

            self.x           = placeholder_collection[0]
            self.y           = placeholder_collection[1]
            self.images      = placeholder_collection[2]
            self.counts      = placeholder_collection[3]
            self.keep_prob   = placeholder_collection[4]
            self.is_training = placeholder_collection[5]
            self.iter_init   = placeholder_collection[6]

            self.loss        = saved_collection[0]
            self.pred_counts = saved_collection[1]


    def test(self, images, counts):

        total_loss = []
        total_pred = []

        with tf.Session() as sess:
            # init variables
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            #self.writer.add_graph(sess.graph)

            sess.run(self.iter_init,
                     feed_dict={self.x: images,
                                self.y: counts})

            try:
                while True:
                    loss, y_pred, _, _ = sess.run([self.loss, self.pred_counts,
                                                     self.images, self.counts],
                                                feed_dict={self.is_training:False,
                                                           self.keep_prob:1}
                                                )
                    total_loss.append(loss)
                    total_pred.append(y_pred)
            except:
                pass
            print('test loss: ',np.mean(loss))
            total_pred = np.concatenate(total_pred, axis=0)

            return total_pred.flatten()
