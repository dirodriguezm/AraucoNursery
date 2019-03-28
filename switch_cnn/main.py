import os
import datetime
import sys
import h5py
from pipeline import Pipeline
from sklearn.model_selection import train_test_split


os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]


if __name__ == "__main__":

    # ===================================================
    experiment_name = sys.argv[2]
    model_used      = sys.argv[3]
    n_epochs        = 1000
    batch_size      = 128
    keep_prob       = 0.8
    # ===================================================

    with h5py.File('./images/data_cuentas.h5', 'r') as hf:
        images = hf['images'].value
        counts = hf['counts'].value

    x_train, x_rest, y_train, y_rest = train_test_split(
    images, counts, test_size=0.4, random_state=42, shuffle=True)

    x_val, x_test, y_val, y_test = train_test_split(
    x_rest, y_rest, test_size=0.4, random_state=42, shuffle=True)

    # ====================================================

    pip = Pipeline(save_path='./sessions/'+experiment_name+'/')

    pip.load_data(img_dimension=(100,100), n_channels=3)

    pip.create_batches(batch_size)

    pip.construct_model(model_name=model_used)

    pip.fit(x_train, y_train[:, None], x_val, y_val[:, None],
            n_epochs=n_epochs, stop_step=100000, keep_prob=keep_prob)

    pip.test(x_test, y_test[:,None])

    # =====================================================
    with h5py.File('./sessions/'+experiment_name+'/'+'test_set.h5', 'w') as hf:
        hf.create_dataset("images",  data=x_test)
        hf.create_dataset("counts",  data=y_test)
