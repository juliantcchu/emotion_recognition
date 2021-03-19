#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:35:22 2021

@author: julianchu
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import json
import pickle
import os

#%%
 

callbacks = []

def train_model(model, name, train_X, train_Y, val = 0.1, 
                epochs = 32, batch_size = 128, plot_loss = True):
    
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath='/tmp/checkpoint',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    callbacks.append(model_checkpoint_callback)
    
    
    if type(val) == float:
        hist = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, 
                         validation_split = val, callbacks=[model_checkpoint_callback])
    else:
        hist = model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, 
                         validation_data=(val[0], val[1]), callbacks=[model_checkpoint_callback])
    
    model.save('models/' + name)
    with open('plots/' + name + '.json', 'w') as f:
        json.dump(hist.history, f)
        
    model.load_weights('/tmp/checkpoint')
    model.save('models/' + name + '_highest_val')
    
    return model, hist

def plot_acc_loss(hist, name):
    #accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title(name + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    #plt.savefig(name + '.png')
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title(name + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    #plt.savefig(name + '.png')