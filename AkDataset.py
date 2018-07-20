import numpy as np
import h5py
from tflearn.data_utils import image_preloader
from tflearn.data_utils import build_hdf5_image_dataset


# Cross subject dataset
dataset_train = 'D:\PythonProj\DataSet\Hand4/train'
dataset_test = 'D:\PythonProj\DataSet\Hand4/test'

# build_hdf5_image_dataset(dataset_train, image_shape=(32, 32), mode='folder', output_path='hand_train_32.h5', categorical_labels=True, normalize=True)
# build_hdf5_image_dataset(dataset_test, image_shape=(32, 32), mode='folder', output_path='hand_test_32.h5', categorical_labels=True, normalize=True)

# testing
# X, Y = image_preloader(dataset_file, image_shape=(32, 32), mode='folder', categorical_labels=True, normalize=True)
# # Load HDF5 dataset
# h5f = h5py.File('hand_train.h5', 'r')

# CrossValidation dataset
dataset_train = 'D:\PythonProj\DataSet\Hand4\CrossValidation/4/train'
dataset_test = 'D:\PythonProj\DataSet\Hand4\CrossValidation/4/test'

build_hdf5_image_dataset(dataset_train, image_shape=(32, 32), mode='folder', output_path='hand_train_32_S4.h5', categorical_labels=True, normalize=True)
build_hdf5_image_dataset(dataset_test, image_shape=(32, 32), mode='folder', output_path='hand_test_32_S4.h5', categorical_labels=True, normalize=True)