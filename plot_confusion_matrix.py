"""

plot confusion matrices

"""

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import json
import pickle



CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']


DIR = 'confusion_matrix/'


def pred_to_num(pred, indices):
    num = []
    for p_raw in pred:
        
        p = []
        for i in indices:
            p.append(p_raw[i])
        
        #print(p)
        num.append(list(p).index(max(p)))
    
    return num

def num_to_classes(num, classes):
        
    pred_classes = []
    
    for n in num:
        pred_classes.append(classes[int(n)]) #not sure why it is not already an int
    return pred_classes

def filter_classes(y_pred, y_true, classes):
    y_true_filtered = []
    y_pred_filtered = []
    i = 0
    for y in y_true:
        y = int(y) #not sure why it is not already an int
        if y in classes:
            y_true_filtered.append(y)
            y_pred_filtered.append(y_pred[i])
        i += 1
    return y_pred_filtered, y_true_filtered

def plot_confusion_matrix(model, name, classes, X, Y, indices = None, show = True):
    
    if indices == None:
        indices = range(len(classes))
    
    y_pred = model.predict(X)
    
    y_pred_num = pred_to_num(y_pred, indices)
    
    y_pred_num, y_true_num = filter_classes(y_pred_num, Y, indices)
    
    y_pred_classes = num_to_classes(y_pred_num, classes)
    y_true_classes = num_to_classes(y_true_num, classes)
    
    mat = metrics.confusion_matrix(y_pred_classes, y_true_classes, labels = classes)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(mat)
    plt.title(name)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    if show:
        plt.show()
    return mat



# load data
test_X = pickle.load(open('processed_data/X_test_new.pickle', 'rb'))
test_Y = pickle.load(open('processed_data/Y_test_new.pickle', 'rb'))


# conv_4
#conv_4 = keras.models.load_model('')

# conv_4 regularized
model = keras.models.load_model('models/conv_4 regularized_lr0.001_highest_val')

mat = plot_confusion_matrix(model, 'CNN-1 with regularization, 6 classes', 
                           ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'], 
                           test_X, test_Y)


# conv_4 data aug
model = keras.models.load_model('models/conv_4 ----data aug_highest_val')

mat = plot_confusion_matrix(model, 'CNN-1 with data augmentation, 6 classes', 
                           ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'], 
                           test_X, test_Y)

# conv_5

# conv_5 data aug
model = keras.models.load_model('models/conv_5_lr0.0001_data_aug_highest_val')

mat = plot_confusion_matrix(model, 'CNN-2 with data augmentation, 6 classes', 
                           ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'], 
                           test_X, test_Y)


# conv_5 data aug 2 classes
mat = plot_confusion_matrix(model, 'CNN-2 with data augmentation, 2 classes', 
                           ['sad', 'surprise'], 
                           test_X, test_Y, indices = [4, 5])

# conv_5 data aug 3 classes
mat = plot_confusion_matrix(model, 'CNN-2 with data augmentation, 3 classes', 
                           ['happy', 'sad', 'surprise'], 
                           test_X, test_Y , indices = [2, 4, 5])
















