import cv2
import numpy as np
import os
import pickle
from random import shuffle
#from metadata import set_data

CLASSES = ['happy', 'sad', 'surprise']#['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'] #, 'disgust']
NUM_CLASSES = len(CLASSES)

TRAIN_DATA_DIR = 'dataset/images/train/'
TEST_DATA_DIR = 'dataset/images/test/'

def load_train_set(cvs_file):
    train_X = 0
    train_Y = 0
    
    return train_X, train_Y

def shuffle_data(X, Y):
    X_Y = []
    n = len(Y)
    for i in range(n):
        X_Y.append([X[i], Y[i]])
    shuffle(X_Y)
    
    for i in range(n):
        X[i] = X_Y[i][0]
        Y[i] = X_Y[i][1]
    return X, Y

def img_to_pickle(directory, classes, samples, filename_X, filename_Y):
    num_class = len(classes)
    X = []
    Y = []
    i = 0
    for emotion in classes:
        files = os.listdir(directory + emotion)[:samples]
        try:
            files = files[:samples]
        except Exception:
            print('class ' + emotion + ' does not have enough samples')
        
        for file in files:
            X.append(cv2.imread(directory + emotion + '/' + file, 0) / 255)
            Y.append(i)
        i += 1
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    Y = np.array(Y)
    X, Y = shuffle_data(X, Y)
    file = open(filename_X, 'wb')
    pickle.dump(X, file)
    file.close()
    file = open(filename_Y, 'wb')
    pickle.dump(Y, file)
    file.close()
    
    print('shape of X: ', X.shape)
    print('shape of Y: ', Y.shape)
        




if __name__ == "__main__":

    # downlaod data and save as pickle
    img_to_pickle(TRAIN_DATA_DIR, CLASSES, 3205, 'train_X_3.pickle', 'train_Y_3.pickle')
    img_to_pickle(TEST_DATA_DIR, CLASSES, 797, 'test_X_3.pickle', 'test_Y_3.pickle')
    
    
#%%
#pretrain dataset
img_to_pickle(TRAIN_DATA_DIR, 
              ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'disgust'],
              10000, 'pretrain_X.pickle', 'prerain_Y.pickle')
train_X = pickle.load(open('train_X.pickle', 'rb'))
train_Y = pickle.load(open('train_Y.pickle', 'rb'))

for i in range(len(train_Y)):
    if train_Y[i] in [0, 1, 5, 4, 6]:
        train_Y[i] = 0
    else:
        train_Y[i] = 1
file = open('train_Y_positivity_big.pickle', 'wb')
pickle.dump(train_Y, file)
file.close()
    



#%%

Xs = [[], [], []]
Ys = [[], [], []]
i = 0

samples = 3205 + 797
split = [3002, 500, 500]

directories = [TRAIN_DATA_DIR, TEST_DATA_DIR]
classes = ['happy', 'sad', 'surprise'] #['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
for emotion in classes:
    X = []
    Y = []
    for directory in directories:
        files = os.listdir(directory + emotion) 
        for file in files:
            X.append(cv2.imread(directory + emotion + '/' + file, 0) / 255)
    Xs[0] += X[:split[0]]
    Xs[1] += X[split[0]:split[0] + split[1]]
    Xs[2] += X[split[0] + split[1]:split[0] + split[1] + split[2]]
    Ys[0] += list(np.ones(split[0]) * i)
    Ys[1] += list(np.ones(split[1]) * i)
    Ys[2] += list(np.ones(split[2]) * i)
    
    i+= 1

X_filenames = ['X_train_new_3.pickle', 'X_val_new_3.pickle', 'X_test_new_3.pickle']
Y_filenames = ['Y_train_new_3.pickle', 'Y_val_new_3.pickle', 'Y_test_new_3.pickle']

for i in range(3):
    X = np.array(Xs[i])
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    Y = np.array(Ys[i])
    
    print(X.shape, Y.shape)
    
    X, Y = shuffle_data(X, Y)
    file = open(X_filenames[i], 'wb')
    pickle.dump(X, file)
    file.close()
    file = open(Y_filenames[i], 'wb')
    pickle.dump(Y, file)
    file.close()

print('shape of X: ', X.shape)
print('shape of Y: ', Y.shape)
















    
    
            
