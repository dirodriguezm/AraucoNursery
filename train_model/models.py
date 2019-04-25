import tensorflow as tf
from layers import *
from cost import *


def image_rotation(images, density,counts, go,semilla=42):
    if go:
        print("rotando los datos")
        images_180    = tf.image.flip_up_down(images)
        images_fliped = tf.image.flip_left_right(images)
        images_fliped2 = tf.image.flip_left_right(images_180)

        density_180    = tf.image.flip_up_down(density)
        density_fliped = tf.image.flip_left_right(density)
        density_fliped2 = tf.image.flip_left_right(density_180)

        new_images = tf.concat([images_180,
                                images_fliped,
                                images_fliped2,
                                images], axis=0)

        new_density = tf.concat([density_180, density_fliped, density_fliped2, density], axis=0)
        new_counts = tf.concat([counts, counts, counts, counts], axis=0)

        #revovemos los datos
        new_images = tf.random_shuffle(new_images, seed=semilla)
        new_density = tf.random_shuffle(new_density, seed=semilla)
        new_counts = tf.random_shuffle(new_counts, seed=semilla)


        return new_images, new_density,new_counts
    else:
        images = tf.random_shuffle(images, seed=semilla)
        density = tf.random_shuffle(density, seed=semilla)
        counts = tf.random_shuffle(counts, seed=semilla)
        return images, density,counts
    

class Regressor_3:
    def __init__(self,
                 images,
                 density,
                 counts,
                 lr = 1e-8):
        self.lr = 1e-8
        self.is_training = tf.placeholder(tf.bool,name="is_train")
        self.keep_prob = tf.placeholder('float32', name='keep_prob')
        

        images, density,counts = tf.cond(self.is_training,
                                lambda: image_rotation(images,density,counts,True),
                                lambda: image_rotation(images,density,counts,False))

        self.lr = lr
        self.images = images
        self.counts = counts
        self.density = density
        

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

            self.prediction = tf.nn.relu(fc_4,
                                          name='activated_output')


    def loss(self): 
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.square(self.prediction -
                                  tf.cast(self.counts, 'float32')))

            return loss


class MCNN:
    def __init__(self,
                 images,
                 density,
                 counts,
                 lr = 1e-6):
        self.lr = lr
        self.is_training = tf.placeholder(tf.bool,name="is_train")
        self.keep_prob = tf.placeholder('float32', name='keep_prob')

        self.counting = tf.placeholder(tf.bool,name="counting")

        images, density,counts = tf.cond(self.is_training,
                                lambda: image_rotation(images,density,counts,True),
                                lambda: image_rotation(images,density,counts,False))
                                
        self.images = images
        self.density = density
        self.counts = counts




        with tf.name_scope("COLUMN1"):

            #9x9 layers
            # =================================================================
            conv1_0 = conv_layer(self.images,
                                channels_in=3,
                                channels_out=16,
                                filter_size=(9,9),
                                strides=[1,1,1,1],
                                name='conv1_0',
                                is_training=self.is_training)
            conv1_0 = tf.nn.relu(conv1_0)
            conv1_0 = maxpool_layer(conv1_0,
                                    kernel_size=[2,2],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='maxpool_1')



            conv2_0 = conv_layer(conv1_0,
                                 channels_in=16,
                                 channels_out=32,
                                 filter_size=(7,7),
                                 strides=[1,1,1,1],
                                 name='conv2_0',
                                 is_training=self.is_training)
            conv2_0 = tf.nn.relu(conv2_0)
            conv2_0 = maxpool_layer(conv2_0,
                                    kernel_size=[2,2],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='maxpool_1')


            conv3_0 = conv_layer(conv2_0,
                                channels_in=32,
                                channels_out=16,
                                filter_size=(7,7),
                                strides=[1,1,1,1],
                                name='conv3_0',
                                is_training=self.is_training)
            conv3_0 = tf.nn.relu(conv3_0)
            conv4_0 = conv_layer(conv3_0,
                                channels_in=16,
                                channels_out=8,
                                filter_size=(7,7),
                                strides=[1,1,1,1],
                                name='conv4_0',
                                is_training=self.is_training)
            conv4_0 = tf.nn.relu(conv4_0)
            # tf.summary.image('pred 9x9', conv4_0, 1)

        with tf.name_scope("COLUMN2"):
            # 7x7 layer
            # =================================================================
            conv1_1 = conv_layer(self.images,
                                channels_in=3,
                                channels_out=20,
                                filter_size=(7,7),
                                strides=[1,1,1,1],
                                name='conv1_0',
                                is_training=self.is_training)
            conv1_1 = tf.nn.relu(conv1_1)
            conv1_1 = maxpool_layer(conv1_1,
                                    kernel_size=[2,2],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='maxpool_1')



            conv2_1 = conv_layer(conv1_1,
                                channels_in=20,
                                channels_out=40,
                                filter_size=(5,5),
                                strides=[1,1,1,1],
                                name='conv2_0',
                                is_training=self.is_training)
            conv2_1 = tf.nn.relu(conv2_1)
            conv2_1 = maxpool_layer(conv2_1,
                                    kernel_size=[2,2],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='maxpool_1')


            conv3_1 = conv_layer(conv2_1,
                                channels_in=40,
                                channels_out=20,
                                filter_size=(5,5),
                                strides=[1,1,1,1],
                                name='conv3_0',
                                is_training=self.is_training)
            conv3_1 = tf.nn.relu(conv3_1)
            conv4_1 = conv_layer(conv3_1,
                                channels_in=20,
                                channels_out=10,
                                filter_size=(5,5),
                                strides=[1,1,1,1],
                                name='conv4_0',
                                is_training=self.is_training)
            conv4_1 = tf.nn.relu(conv4_1)
            # tf.summary.image('pred 7x7', conv4_1, 1)
        
        with tf.name_scope("COLUMN3"):
            # 5x5 layer
            # =================================================================
            conv1_2 = conv_layer(self.images,
                                channels_in=3,
                                channels_out=24,
                                filter_size=(5,5),
                                strides=[1,1,1,1],
                                name='conv1_0',
                                is_training=self.is_training)
            conv1_2 = tf.nn.relu(conv1_2)
            conv1_2 = maxpool_layer(conv1_2,
                                    kernel_size=[2,2],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='maxpool_1')



            conv2_2 = conv_layer(conv1_2,
                                channels_in=24,
                                channels_out=48,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv2_0',
                                is_training=self.is_training)
            conv2_2 = tf.nn.relu(conv2_2)
            conv2_2 = maxpool_layer(conv2_2,
                                    kernel_size=[2,2],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='maxpool_1')


            conv3_2 = conv_layer(conv2_2,
                                channels_in=48,
                                channels_out=24,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv3_0',
                                is_training=self.is_training)
            conv3_2 = tf.nn.relu(conv3_2)
            conv4_2 = conv_layer(conv3_2,
                                channels_in=24,
                                channels_out=12,
                                filter_size=(3,3),
                                strides=[1,1,1,1],
                                name='conv4_0',
                                is_training=self.is_training)
            conv4_2 = tf.nn.relu(conv4_2)

                     
        with tf.name_scope("FuseLayer"):
            # fuse layer
            # =================================================================
            suma = tf.concat([conv4_0,conv4_1,conv4_2],axis = 3)
            # print(suma)

            #spatialdropout
            suma =spatial_dropout(suma, self.keep_prob, seed=42)

            conv_final = conv_layer(suma,
                                    channels_in=30,
                                    channels_out=1,
                                    filter_size=(1,1),
                                    strides=[1,1,1,1],
                                    name='conv_final',
                                    is_training=self.is_training)



        with tf.name_scope('prediction'):
            self.prediction = tf.nn.relu(conv_final)
            tf.summary.image('density_pred', self.prediction, 1)
        with tf.name_scope("INPUT"):
            tf.summary.image("image", self.images,1)
        with tf.name_scope("density_gt"):
            tf.summary.image("density", self.density,1)





    def get_count(self):
        suma_p = tf.reduce_sum(self.prediction) 
        suma_d = tf.reduce_sum(self.density)
        return suma_p,suma_d

    def loss(self):
        # L2 Loss
        loss = tf.reduce_sum((self.prediction - self.density) * (self.prediction - self.density))
        # loss = tf.losses.mean_squared_error(tf.reduce_sum(self.prediction),tf.reduce_sum(self.density)) 

        
        #mse a mano
        # loss = tf.sqrt(tf.reduce_mean(tf.square(self.prediction - self.density)))
        # act_sum = tf.reduce_sum(self.prediction)
        # act_sum = tf.reduce_sum(self.density)
        # MAE = tf.abs(self.act_sum - self.pre_sum)
        
        #loss normal
        # loss = tf.losses.mean_squared_error(self.prediction,self.density)
        
        #loss con suma
        # suma = tf.math.reduce_sum(self.prediction)
        # suma =tf.reshape(suma,[-1,1])
        # loss = tf.losses.mean_squared_error(self.prediction,
        #                                     self.density) + tf.losses.mean_squared_error(suma,self.counts)
    
        #loss ponderado
        # cte = 100
        # loss = tf.losses.mean_squared_error(cte*self.prediction,
        #                                     cte*self.density) 
        return loss


class CCNN:
    def __init__(self,
                 images,
                 density,
                 counts,
                 lr = 1e-8):
        self.lr = lr
        self.is_training = tf.placeholder(tf.bool,name="is_train")
        self.keep_prob = tf.placeholder('float32', name='keep_prob')

        self.counting = tf.placeholder(tf.bool,name="counting")

        # images, density,counts = tf.cond(self.is_training,
        #                         lambda: image_rotation(images,density,counts,True),
        #                         lambda: image_rotation(images,density,counts,False))
                                
        self.images = images
        self.density = density
        self.counts = counts


        with tf.name_scope("CCNN"):


            conv1 = conv_layer(self.images,
                                channels_in=3,
                                channels_out=32,
                                filter_size=(7,7),
                                strides=[1,1,1,1],
                                name='conv1',
                                is_training=self.is_training)
            conv1 = tf.nn.relu(conv1)


            conv2 = conv_layer(conv1,
                                 channels_in=32,
                                 channels_out=32,
                                 filter_size=(7,7),
                                 strides=[1,1,1,1],
                                 name='conv2',
                                 is_training=self.is_training)
            conv2 = tf.nn.relu(conv2)



            conv2 = maxpool_layer(conv2,
                                    kernel_size=[2,2],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='maxpool_1')

            conv3 = conv_layer(conv2,
                                channels_in=32,
                                channels_out=64,
                                filter_size=(5,5),
                                strides=[1,1,1,1],
                                name='conv3',
                                is_training=self.is_training)
            conv3 = tf.nn.relu(conv3)

            conv3 = maxpool_layer(conv3,
                        kernel_size=[2,2],
                        strides=[1, 2, 2, 1],
                        padding='SAME',
                        name='maxpool_1')


            conv4 = conv_layer(conv3,
                                channels_in=64,
                                channels_out=1000,
                                filter_size=(1,1),
                                strides=[1,1,1,1],
                                name='conv4',
                                is_training=self.is_training)

            conv4 = tf.nn.relu(conv4)

            conv5 = conv_layer(conv4,
                    channels_in=1000,
                    channels_out=400,
                    filter_size=(1,1),
                    strides=[1,1,1,1],
                    name='conv5',
                    is_training=self.is_training)

            conv5 = tf.nn.relu(conv5)

            conv6 = conv_layer(conv5,
                    channels_in=400,
                    channels_out=1,
                    filter_size=(1,1),
                    strides=[1,1,1,1],
                    name='conv6',
                    is_training=self.is_training)

            conv6 = tf.nn.relu(conv6)

        with tf.name_scope('prediction'):
            self.prediction = conv6
            tf.summary.image('density_pred', self.prediction, 1)
            # tf.summary.image("image", self.images,1)
            # tf.summary.image("density", self.density,1)

        with tf.name_scope("INPUT"):
            tf.summary.image("image", self.images,1)
        with tf.name_scope("density_gt"):
            tf.summary.image("density", self.density,1)

    def get_count(self):
        suma_p = tf.reduce_sum(self.prediction) 
        suma_d = tf.reduce_sum(self.density)
        return suma_p,suma_d

    def loss(self):
        # L2 Loss
        loss = tf.reduce_sum((self.prediction - self.density) * (self.prediction - self.density))
        return loss
        

class HYDRACNN:
    def __init__(self,
                 images,
                 density,
                 counts,
                 lr = 1e-8):
        self.lr = lr
        self.is_training = tf.placeholder(tf.bool,name="is_train")
        self.keep_prob = tf.placeholder('float32', name='keep_prob')

        self.counting = tf.placeholder(tf.bool,name="counting")

        images, density,counts = tf.cond(self.is_training,
                                lambda: image_rotation(images,density,counts,True),
                                lambda: image_rotation(images,density,counts,False))
                                
        self.images = images
        self.density = density
        self.counts = counts


        with tf.name_scope("HYDRA"):
            pass

           
        with tf.name_scope('prediction'):
            #salida
            # self.prediction = 
            tf.summary.image('density_pred', self.prediction, 1)
            

        with tf.name_scope("INPUT"):
            tf.summary.image("image", self.images,1)
        with tf.name_scope("density_gt"):
            tf.summary.image("density", self.density,1)

    def get_count(self):
        suma_p = tf.reduce_sum(self.prediction) 
        suma_d = tf.reduce_sum(self.density)
        return suma_p,suma_d

    def loss(self):
        # L2 Loss
        loss = tf.reduce_sum((self.prediction - self.density) * (self.prediction - self.density))
        return loss
        