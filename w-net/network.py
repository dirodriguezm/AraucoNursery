import tensorflow as tf
from modules import create_module_down, create_module_up, \
                    create_module_last, create_module_last_dec
from layers import crop_and_concat
from utils import brightness_weight, convert_to_batchTensor,\
                  soft_ncut, batch_colorize, gaussian_neighbor


class W_net:
    def __init__(self,
                 n_samples,
                 dimen0,
                 dimen1,
                 channels,
                 K = 10):
        self.channels  = channels
        self.dimen0    = dimen0
        self.dimen1    = dimen1
        self.n_samples = n_samples
        self.K         = K

        self.sess      = tf.Session()

        with tf.name_scope('input'):
            self.images = tf.placeholder(dtype=tf.float32,
                                         shape=[n_samples,
                                                dimen0,
                                                dimen1,
                                                channels],
                                         name='images_input')
            self.neighbor_indeces = tf.placeholder(tf.int64, name="neighbor_indeces")
            self.neighbor_vals = tf.placeholder(tf.float32, name="neighbor_vals")
            self.neighbor_shape = tf.placeholder(tf.int64, name="neighbor_shape")
            tf.summary.image("input_images", self.images, max_outputs=2)

        with tf.name_scope('U_enc'):

            mp_1, conv_1 = create_module_down(self.images, 3, 64, 'module_1')
            mp_2, conv_2 = create_module_down(mp_1, 64,  128, 'module_2')
            mp_3, conv_3 = create_module_down(mp_2, 128, 256, 'module_3')
            mp_4, conv_4 = create_module_down(mp_3, 256, 512, 'module_4')
            t_conv_5     = create_module_up(mp_4, 512, 1024, 'module_5')
            skip_6       = crop_and_concat(conv_4, t_conv_5)
            t_conv_6     = create_module_up(skip_6, 1024, 512, 'module_6')
            skip_7       = crop_and_concat(conv_3, t_conv_6)
            t_conv_7     = create_module_up(skip_7, 512, 256, 'module_7')
            skip_8       = crop_and_concat(conv_2, t_conv_7)
            t_conv_8     = create_module_up(skip_8, 256, 128, 'module_8')
            skip_9       = crop_and_concat(conv_1, t_conv_8)
            self.pred_annotations, logits  = create_module_last(skip_9,
                                                           128,
                                                           64,
                                                           'module_9')

        with tf.name_scope('U_dec'):
            softmax_logits = tf.nn.softmax(logits, name='softmax_logits')
            mp_10, conv_10 = create_module_down(softmax_logits, 64, 64, 'module_10')
            mp_11, conv_11 = create_module_down(mp_10, 64,  128, 'module_11')
            mp_12, conv_12 = create_module_down(mp_11, 128, 256, 'module_12')
            mp_13, conv_13 = create_module_down(mp_12, 256, 512, 'module_14')
            t_conv_14     = create_module_up(mp_13, 512, 1024, 'module_14')
            skip_15       = crop_and_concat(conv_13, t_conv_14)
            print(skip_15)
            t_conv_15     = create_module_up(skip_15, 1024, 512, 'module_15')
            skip_16       = crop_and_concat(conv_12, t_conv_15)

            t_conv_16     = create_module_up(skip_16, 512, 256, 'module_16')
            skip_17       = crop_and_concat(conv_11, t_conv_16)

            t_conv_17     = create_module_up(skip_17, 256, 128, 'module_17')
            skip_18       = crop_and_concat(conv_10, t_conv_17)

            reconstruct_image = create_module_last_dec(skip_18,
                                                       128,
                                                       64,
                                                       'module_18',
                                                       channels_img=self.channels)
            print(reconstruct_image)
        with tf.name_scope('Loss'):
            self.reconstruct_images = reconstruct_image
            error = (self.images - self.reconstruct_images)**2
            self.reconstruct_loss = tf.reduce_mean(tf.reshape(error, shape=[-1]))
            neighbor_filter = (self.neighbor_indeces, self.neighbor_vals, self.neighbor_shape)
            _image_weights = brightness_weight(self.images, neighbor_filter, sigma_I = 0.05)
            image_weights = convert_to_batchTensor(*_image_weights)
            batch_soft_ncut = soft_ncut(self.images, softmax_logits, image_weights)
            self.soft_ncut = tf.reduce_mean(batch_soft_ncut)
            self.loss = self.reconstruct_loss + self.soft_ncut

            trainable_var = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(5e-5)
            grads = optimizer.compute_gradients(self.loss, var_list=trainable_var)
            self.train_op = optimizer.apply_gradients(grads)

            self.colorized_pred_annotation = batch_colorize(self.pred_annotations,
                                                            0,
                                                            self.K,
                                                            'viridis')

            tf.summary.image("reconstruct_images", self.reconstruct_images, max_outputs=2)


        self.summaries   = tf.summary.merge_all()
        saver            = tf.train.Saver()

        self.writer      = tf.summary.FileWriter('./logs/train')
        self.writer_test = tf.summary.FileWriter('./logs/test')

        self.sess.run(tf.global_variables_initializer())
        self.writer.add_graph(self.sess.graph)

    def train(self, X_images, n_epochs=100):
        current_epoch  = 0
        image_shape = self.images.get_shape().as_list()[1:3]
        gauss_indeces, gauss_vals = gaussian_neighbor(image_shape, sigma_X = 4, r = 5)
        weight_shapes = np.prod(image_shape).astype(np.int64)

        while(current_epoch < n_epochs):
            x = X_images
            loss, _, sm = self.sess.run([self.loss, self.train_op, self.summaries],
                                    feed_dict={self.images:x,
                                               self.neighbor_indeces: gauss_indeces,
                                               self.neighbor_vals: gauss_vals,
                                               self.neighbor_shape: [weight_shapes, weight_shapes]
                                              })
            self.writer.add_summary(sm, current_epoch)
            print('epoch: {0} - loss: {1}'.format(current_epoch, loss))
            current_epoch+=1

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    wnet = W_net(1, 300, 600, 3)
    im = Image.open("img/img.jpg")
    np_im = np.array(im)[None, :]
    print(np_im.shape)
    wnet.train(np_im)
