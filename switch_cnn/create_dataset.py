import tensorflow as tf
import sys
from PIL import Image
import numpy as np
import os

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(dir_img):
    im_frame = Image.open('./images/'+dir_img)
    np_frame = np.array(im_frame.getdata())
    return np_frame.reshape((101, 101, 4))

def createDataRecord(out_filename, addrs, labels):
    writer = tf.python_io.TFRecordWriter(out_filename)

    for i in range(len(addrs)):
        #if not i % 1000:
        print('Train data: {}/{}'.format(i, len(addrs)))
        sys.stdout.flush()

        img   = load_image(addrs[i])
        label = labels[i]

        if img is None:
            continue

        data = {
            'image': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)

        }

        feature = tf.train.Features(feature=data)
        example = tf.train.Example(features = feature)
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()


def _parse_(serialized_example):
    feature = {'image' : tf.FixedLenFeature([],tf.string),
                'label': tf.FixedLenFeature([],tf.int64)}

    example = tf.parse_single_example(serialized_example, feature)
    image = tf.decode_raw(example['image_raw'],tf.uint8)
    image = tf.cast(image, tf.float32)

    label = tf.cast(example['label'],tf.int32)
    return {'image':image, 'label':label}
