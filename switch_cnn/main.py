import numpy as np
from sklearn.model_selection import train_test_split
from models import Regressor
from create_dataset import createDataRecord, load_image
import os
import pandas as pd


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


    bs = 2

    r1 = Regressor(img_dim=(101,101),
               channels=4,
               name='r1',
               lr=1e-6)

    r1.train(train_images, train_counts,val_images, val_counts, batch_size=bs)


