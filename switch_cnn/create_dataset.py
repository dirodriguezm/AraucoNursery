import tensorflow as tf
import sys


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image():
    pass

def createDataRecord(out_filename, addrs, labels):
    writer = tf.python_io.TFRecordWriter(out_filename)

    for i in range(len(addrs)):
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()

        img   = load_image(addrs[i])
        label = labels[i]

        if img is None:
            continue

        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)

        }

        example = tf.train.Example(features = tf.train.Feature(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()