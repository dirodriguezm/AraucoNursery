import numpy as np
from restore_model import RestoreModel
import h5py


if __name__ == "__main__":
    model_path = './saved_model/r3_newimages'

    model_path = './saved_model/final2/'
    dir_whole = '../images/data_cuentas.h5'
    dir_model = model_path+'test_set.h5'
    with h5py.File(dir_whole, 'r') as hf:
        images = hf['images'].value
        counts = hf['counts'].value


    model = RestoreModel(model_path)
    y_pred = model.test(images, counts[:,None])

    print(y_pred)
    print('-'*100)
    print(counts)
