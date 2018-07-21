import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import h5py
from test_keras.keras_hand_train import ak_train
from test_keras.keras_hand_test import ak_test


p = ak_test()


