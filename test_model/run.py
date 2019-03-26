import numpy as np
from restore_model import RestoreModel

if __name__ == "__main__":
    model_path = '../saved_model/alexnet'
    model = RestoreModel(model_path)
