import numpy as np
from restore_model import RestoreModel
import h5py


if __name__ == "__main__":
    model_path = './saved_model/r3final'

    with h5py.File('./saved_model/r3final/test_set.h5', 'r') as hf:
        images = hf['images'].value[0:100]
        counts = hf['counts'].value[0:100]


    model = RestoreModel(model_path)
    y_pred = model.test(images, counts[:,None])

    print(y_pred)
    print('-'*100)
    print(counts)
