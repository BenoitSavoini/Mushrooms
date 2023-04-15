# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:19:17 2023

@author: Alkios
"""

import tensorflow as tf
print(tf.__version__)


tf.test.gpu_device_name()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, LSTM
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder


path = 'C:/Users/Alkios/Downloads/mushrooms/mushrooms.csv'


def load_training_data(data_directory):
    df = pd.read_csv(data_directory, encoding = 'latin-1')
    return df



df = load_training_data(path)
df.columns

features = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']

X = df[features]
y = df['class']
# Textes de tweets d'entraînement et labels correspondants

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2, random_state = 42)

clf = DecisionTreeClassifier()


enc = OrdinalEncoder()
enc.fit(Xtrain)
Xtrain_encoded = enc.fit_transform(Xtrain)
Xtest_encoded = enc.fit_transform(Xtest)

# Train Decision Tree Classifer
clf = clf.fit(Xtrain_encoded, ytrain)

#Predict the response for test dataset
ypred = clf.predict(Xtest_encoded)

print("Accuracy:",metrics.accuracy_score(ytest, ypred))

