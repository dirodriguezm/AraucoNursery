import os
import tensorflow as tf
import matplotlib.image as mpimg

def save_images_from_event(n_ruta, tag, output_dir='./'):
    """
    extrae las imagenes que se encuentran guardadas en el event que es generado durante el entrenamiento
    :n_ruta: ruta del evento
    :tag: nombre de lo que se guarda en el tf
    :output_dir: directorio donde se guardaran las imagenes
    """
    ruta = n_ruta.split('/')
    ruta = ruta[:len(ruta)-1]

    string = ""
    for element in ruta:
        string += element +'/'
    
    string +=  tag +'/'

    if not os.path.exists(string):
        os.makedirs(string)
    
    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(n_ruta):
            for v in e.summary.value:
                if tag in v.tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_n_ruta = os.path.realpath('{}/image_{:05d}.png'.format(string, count))
                    print("Saving '{}'".format(output_n_ruta))
                    if(im.shape[2] ==1):
                        im=im[:,:,0]

                    mpimg.imsave(output_n_ruta, im)

                    count += 1  

test= "/media/daniel/Respaldo/Memoria/Codigo/AraucoNursery/train_model/sessions/mcnn_color_1/logs/test/events.out.tfevents.1554203230.daniel-tarro"
train = "/media/daniel/Respaldo/Memoria/Codigo/AraucoNursery/train_model/sessions/mcnn_color_1/logs/train/events.out.tfevents.1554203220.daniel-tarro"

save_images_from_event(train, 'density_pred')
# image
# density_gt
# density_pred

