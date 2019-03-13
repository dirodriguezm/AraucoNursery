import tensorflow as tf
from layers import *


def create_module_1(input):

    with tf.name_scope('module_1'):
        conv_0 = conv_layer(input, 1, 64, filter_size=(3,3), name='conv_0')
        conv_1 = conv_layer(conv_0, 64, 64, filter_size=(3,3), name='conv_1')

        mp = maxpool_layer(conv_1, kernel_size=(2,2), strides=1)
        skip = tf.concat([conv_0, conv_1], axis=-1)
        print(mp)
        print(skip)
    return mp, skip

def create_module_2():
    pass

def create_module_3():
    pass

def create_module_4():
    pass

def create_module_5():
    pass

def create_module_6():
    pass

def create_module_7():
    pass

def create_module_8():
    pass

def create_module_9():
    pass

def create_module_10():
    pass

def create_module_11():
    pass

def create_module_12():
    pass

def create_module_13():
    pass

def create_module_14():
    pass

def create_module_15():
    pass

def create_module_16():
    pass

def create_module_17():
    pass

def create_module_18():
    pass

def create_module_19():
    pass

def create_module_20():
    pass
