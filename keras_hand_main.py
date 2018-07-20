import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import h5py
from test_keras.keras_hand_train import ak_train
from test_keras.keras_hand_test import ak_test

#ak_train(300)  # 38

p = ak_test()



# min=100
# selected_index=0
# for i in range(25,55):
#     ak_train(i)
#     p=ak_test()
#     if p<min:
#         min=p
#         selected_index=i
#         print (min," with epoch ",selected_index)
#
# print("The minimum error",min ,"is appeared with: ", selected_index," epochs")
