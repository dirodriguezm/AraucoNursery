import numpy as np
from restore_model import RestoreModel
import h5py


if __name__ == "__main__":
    model_path = './saved_model/popo/'
    dir_model = model_path+'train_set.h5'

    with h5py.File(dir_model, 'r') as hf:
        images = hf['images'].value
        counts = hf['counts'].value


    model = RestoreModel(model_path)
    y_pred = model.test(images, counts[:,None])
