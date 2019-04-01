import os
import datetime
import sys
import h5py
from pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


if __name__ == "__main__":

    # ===================================================
    experiment_name = sys.argv[2]
    model_used      = sys.argv[3]
    n_epochs        = 1000
    batch_size      = 64
    keep_prob       = 0.8
    # ===================================================

    with h5py.File('../images/data_mapa.h5', 'r') as hf:
        images = hf['images'].value[0:100]
        target = hf['density'].value[0:100]

    densidades = []
    for element in target:
        den = element
        den_quarter = np.zeros((int(den.shape[0] / 4), int(den.shape[1] / 4)))

        for i in range(den_quarter.shape[0]):
            for j in range(den_quarter.shape[1]):
                for p in range(4):
                    for q in range(4):
                        den_quarter[i][j] += den[i * 4 + p][j * 4 + q]
        den_quarter = den_quarter[:,:,None]
        densidades.append(den_quarter)

    target = densidades
    dimensions = images.shape
    x_train, x_rest, y_train, y_rest = train_test_split(
    images, target, test_size=0.4, random_state=42, shuffle=True)

    x_val, x_test, y_val, y_test = train_test_split(
    x_rest, y_rest, test_size=0.4, random_state=42, shuffle=True)

    # ====================================================

    pip = Pipeline(save_path='./sessions/'+experiment_name+'/')

    pip.load_data(img_dimension=(dimensions[1],dimensions[2]),
                  n_channels=dimensions[3], target_dim = [None, 25, 25, 1],
                  type_target='float32')

    pip.create_batches(batch_size)

    pip.construct_model(model_name=model_used)

    # pip.fit(x_train, y_train[:, None], x_val, y_val[:, None],
    #         n_epochs=n_epochs, stop_step=100000, keep_prob=keep_prob)
    pip.fit(x_train, y_train, x_val, y_val,
            n_epochs=n_epochs, stop_step=100000, keep_prob=keep_prob)
    pip.test(x_test, y_test[:,None])

    # =====================================================

    with h5py.File('./sessions/'+experiment_name+'/'+'train_set.h5', 'w') as hf:
         hf.create_dataset("images",  data=x_train)
         hf.create_dataset("counts",  data=y_train)

    with h5py.File('./sessions/'+experiment_name+'/'+'val_set.h5', 'w') as hf:
         hf.create_dataset("images",  data=x_val)
         hf.create_dataset("counts",  data=y_val)

    with h5py.File('./sessions/'+experiment_name+'/'+'test_set.h5', 'w') as hf:
         hf.create_dataset("images",  data=x_test)
         hf.create_dataset("counts",  data=y_test)
