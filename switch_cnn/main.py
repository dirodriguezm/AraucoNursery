import numpy as np
from sklearn.model_selection import train_test_split
from models import Regressor
from create_dataset import createDataRecord, load_image
import os
import pandas as pd
import datetime
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

if __name__ == "__main__":

    directories = [file for here, dir, files in os.walk('./images/train') for file in files if file.endswith('.Png')]

    df = pd.read_csv('./images/cuentas.csv')

    x_images = []
    y_counts = []
    for dir in directories[0:10]:
        im = load_image('train/'+dir)
        count = df.loc[df['imagen'] == dir]['cant_arboles'].values[0]
        y_counts.append(count)
        x_images.append(im)

    x_images = np.array(x_images, dtype=np.float32)
    y_counts = np.array(y_counts, dtype=np.int32).reshape((len(y_counts),1))


    train_images, t_images, train_counts, t_counts = train_test_split(x_images,
                                                                      y_counts,
                                                                      test_size=0.3,
                                                                      shuffle=True,
                                                                      random_state=42)


    val_images, test_images, val_counts, test_counts = train_test_split(t_images,
                                                                        t_counts,
                                                                        test_size=0.4,
                                                                        shuffle=True,
                                                                        random_state=42)

    print('Train:{0}\nValidation:{1}\nTest:{2}'.format(train_images.shape, val_images.shape, test_images.shape))


    dt= datetime.datetime.now()
    proyect_name = str(dt.month)+'_'+str(dt.day)+'_'+str(dt.year)+'.'+str(dt.hour)+'_'+str(dt.minute)+'_'+str(dt.second)
    bs = 2
    rotate = True
    r1 = Regressor(img_dim=(101,101),
                   channels=4,
                   name='r1',
                   lr=1e-6,
                   save_path='./sessions/'+proyect_name,
                   rotate=rotate)

    with open(r1.save_path + '/setup.txt', 'a') as r1.out:
        r1.out.write('Rotation: '+str(rotate) + '\n')
        r1.out.write('TRAIN: ' + str(len(train_counts)) + '\n')
        r1.out.write('VALIDATION: ' + str(len(val_counts)) + '\n')
        r1.out.write('TEST: ' + str(len(test_counts)) + '\n')


    r1.train(train_images, train_counts,
             val_images, val_counts,
             test_images, test_counts,
             batch_size=bs,
             n_epochs=1000,
             stop_step=20)

