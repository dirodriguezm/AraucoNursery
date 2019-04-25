import os
import datetime
import sys
import h5py
from pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


# python main.py 0 mcnn_dron multicolumn 1
# python main.py 0 ccnn_1 ccnn 1

if __name__ == "__main__":

    # ===================================================
    experiment_name = sys.argv[2]
    model_used      = sys.argv[3]
    count           = sys.argv[4]
    n_epochs        = 4000
    batch_size      = 1
    keep_prob       = 0.8

    # ===================================================

    with h5py.File('./images/data_mapa.h5', 'r') as hf:
        print(hf.keys())
        images = hf['images'].value
        target = hf['reduced_density'].value
        if(count):
             counts = hf['counts'].value


    target = target[:,:,:,None]
    counts = counts[:, None]
    dimensions = images.shape


    x_train, x_rest, y_train, y_rest = train_test_split(
    images, target, test_size=0.4, random_state=42, shuffle=True)

    x_val, x_test, y_val, y_test = train_test_split(
    x_rest, y_rest, test_size=0.4, random_state=42, shuffle=True)


    if(count):
          x_train, x_rest, y_train, y_rest,z_train,z_rest = train_test_split(images, target,counts, 
          test_size=0.4, random_state=42, shuffle=True)

          x_val, x_test, y_val, y_test,z_val,z_test = train_test_split(
    x_rest, y_rest,z_rest, test_size=0.4, random_state=42, shuffle=True)

    # ====================================================
    print("entrando a pipeline")
    pip = Pipeline(save_path='./sessions/'+experiment_name+'/')

    with h5py.File('./sessions/'+experiment_name+'/'+'train_set.h5', 'w') as hf:
         hf.create_dataset("images",  data=x_train)
         hf.create_dataset("counts",  data=y_train)

    with h5py.File('./sessions/'+experiment_name+'/'+'val_set.h5', 'w') as hf:
         hf.create_dataset("images",  data=x_val)
         hf.create_dataset("counts",  data=y_val)

    with h5py.File('./sessions/'+experiment_name+'/'+'test_set.h5', 'w') as hf:
         hf.create_dataset("images",  data=x_test)
         hf.create_dataset("counts",  data=y_test)

    # =====================================================
    pip.load_data(img_dimension=(dimensions[1],dimensions[2]),
                  n_channels=dimensions[3], target_dim = [None, target[0].shape[0], target[0].shape[1], 1])

    pip.create_batches(batch_size)

    pip.construct_model(model_name=model_used)

    pip.fit(x_train, y_train,z_train, x_val, y_val,z_val,
            n_epochs=n_epochs, stop_step=100000, keep_prob=keep_prob)
    pip.test(x_test, y_test,z_test)

    