import numpy as np
from sklearn.model_selection import train_test_split
from models import Regressor
from create_dataset import *


if __name__ == "__main__":
    n_samples = 100
    synthetic_images = np.random.normal(0, 1, size=[n_samples, 100, 100, 1])
    synthetic_labels = np.random.randint(100, size=(n_samples,1))

    train_images, t_images, train_counts, t_counts = train_test_split(synthetic_images,
                                                                      synthetic_labels,
                                                                      test_size=0.3,
                                                                      shuffle=True,
                                                                      random_state=42)


    val_images, test_images, val_counts, test_counts = train_test_split(t_images,
                                                                        t_counts,
                                                                        test_size=0.4,
                                                                        shuffle=True,
                                                                        random_state=42)

    print('Train:{0}\nValidation:{1}\nTest:{2}'.format(train_images.shape, val_images.shape, test_images.shape))

    r1 = Regressor(img_dim=(100,100),
                   channels=1,
                   name='r1',
                   lr=1e-6)

    r1.train(train_images, t_counts)
