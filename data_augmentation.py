import cv2
import numpy as np
import os
import pickle
from random import shuffle
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#from data import *

train_X = pickle.load(open('processed_data/X_train_new_3.pickle', 'rb'))
train_Y = pickle.load(open('processed_data/Y_train_new_3.pickle', 'rb'))

datagen = ImageDataGenerator(
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=None,
    zoom_range=0.3,
    horizontal_flip=True,
)

X_shape = train_X.shape

times = 3
new_data = np.zeros(((times + 1) * X_shape[0], X_shape[1], X_shape[2], X_shape[3]))
new_data[:X_shape[0]] = train_X

new_y = np.zeros(((times + 1)*X_shape[0], ))
new_y[:X_shape[0]] = train_Y

prev = X_shape[0]

print('loaded train_X, starting data augmentation')
for i in range(times):
    print(str(i) + '/' + str(times))
    new_data[prev: prev + X_shape[0]] = datagen.flow(train_X, batch_size=X_shape[0])[0]
    new_y[prev: prev + X_shape[0]] = train_Y
    prev += X_shape[0]


print('done! saving results')
pickle.dump(new_data, open('processed_data/X_train_new_3_augmented.pickle', 'wb'))
pickle.dump(new_y, open('processed_data/Y_train_new_3_augmented.pickle', 'wb'))