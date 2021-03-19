from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import json
import pickle


#%%
# Plot non-normalized confusion matrix
model = keras.models.load_model('conv_all_classes')
test_X = pickle.load(open('test_X.pickle', 'rb'))
test_Y = pickle.load(open('test_Y.pickle', 'rb'))
test_Y = keras.utils.to_categorical(test_Y, 6)
pred_Y = model.predict(test_X)

matrix = confusion_matrix(test_Y.argmax(axis=1), pred_Y.argmax(axis=1))

CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'] #'disgust', 

#%%

def evaluate_model(model, test_X, test_Y):
    score = model.evaluate(test_X, test_Y)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

#evaluate_model(mlp_1, test_X, test_Y)
#evaluate_model(conv_1, test_X, test_Y)

#%%


def plot_confusion_matrix(model, X, y):
    y_pred = model.predict(X)
    y_pc = []
    for pred in y_pred:
        y_pc.append(list(pred).index(max(pred)))
    y_pred = np.array(y_pc)
    
    mat = metrics.confusion_matrix(y, y_pred)
    plt.imshow(mat)
    plt.show()
    return mat


def plot_cf_nclasses(model, X, y):
    y_pred = model.predict(X)
    y_pc = []
    
    y_new = []
    y_pred_new = []
    
    for i in range(len(y)):
        if y[i] in [4, 5]: #['happy', 'sad', 'surprise']:
            y_new.append(y[i])
            p = [y_pred[i][4], y_pred[i][5]] #y_pred[i][2], 
            y_pred_new.append(p.index(max(p)))

    
    mat = metrics.confusion_matrix(y_new, y_pred_new)
    plt.imshow(mat)
    plt.show()
    return mat
















