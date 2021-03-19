#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 17:10:13 2021

@author: julianchu
"""


def set_data():
    global CLASSES, NUM_CLASSES, TRAIN_DATA_DIR, TEST_DATA_DIR
    
    
    CLASSES = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise'] #'disgust', 
    NUM_CLASSES = len(CLASSES)
    
    TRAIN_DATA_DIR = 'dataset/images/train/'
    TEST_DATA_DIR = 'dataset/images/test/'

