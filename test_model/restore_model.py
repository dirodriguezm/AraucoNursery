import tensorflow as tf
from tensorflow.contrib import image

class RestoreModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.name = 'best_model'
        self.path = self.model_path +'/model/'

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.path + self.name + '.meta')
            saver.restore(sess , save_path=self.path + self.name)
            t_v = tf.trainable_variables() # get all trained variables

            print(t_v)
