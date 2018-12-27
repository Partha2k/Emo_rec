#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 16:27:02 2017
@author: prudhvi
"""

"""
The program trains the CNN
on the Jaffe training set.
The number of iterations can be increased, 
but as default, this runs for 5 iterations on a 
20% divided validation set. 
We trained this for around 400 iterations, 
20 epochs at a time. Each epoch takes 75 seconds
Arguments:
first : Number of epochs
"""

import numpy as np
import os
import math
import sys
from PIL import Image
np.random.rand(2)
#from keras.layers import Conv1D
files = os.listdir('/home/parthasarathidas/Documents/train_set/')

n_iter = int(sys.argv[1])
tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE']

def targets(filename):
    targets = []
    for f in filename:
        if tag_list[0] in f:
            targets.append(0)
        if tag_list[1] in f:
            targets.append(1)
        if tag_list[2] in f:
            targets.append(2)
        if tag_list[3] in f:
            targets.append(3)
        if tag_list[4] in f:
            targets.append(4)
        if tag_list[5] in f:
            targets.append(5)
        if tag_list[6] in f:
            targets.append(6)
    return np.array(targets)


def data(filename):
    train_images = []
    for f in filename:
        current = f
        train_images.append(np.array(Image.open('/home/parthasarathidas/Documents/train_set/'+current).getdata()))        
    return np.array(train_images)

y = targets(files)
print "Fetching Data. Please wait......"

x = data(files)

print "Fetching Complete."


x = np.reshape(x, (np.size(files), 100, 100, 3)) #for image of pixels 64x64 only

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =2)
#print x_train.shape , y_train.shape, math.floor(int(np.size(files))*0.75)
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, num_classes = 7)
y_test = np_utils.to_categorical(y_test, num_classes = 7)
y = np_utils.to_categorical(y)



#x_train = np.reshape(x_train, (2556, 48, 48, 1))
#x_test = np.reshape(x_test, (852, 48, 48, 1))
#print x.shape
#print y.shape
#x_train = x[0:2556, :, :, :]
#x_test = x[2556:, :, :, :]
#y_train = y[0:2556, :]
#y_test = y[2556:, :]

#from sklearn.metrics import confusion_matrix
#print "\nConfusion Matrix\n"
#print confusion_matrix(y_test, predictions)
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD


model = Sequential()  
model.add(Conv2D(10, (5, 5), activation = 'relu', input_shape = x.shape[1:]))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(10, (5, 5), activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))

#model.add(Conv2D(10, 3, 3, activation = 'relu'))
#model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(10, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax'))


#ada = optimizers.adam(lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay= 0)
#ada = optimizers.adam(lr = 0.005)
model.compile(optimizer= 'adam' , loss = 'categorical_crossentropy',
              metrics= ['accuracy'])

history = model.fit(x_train, y_train, batch_size= 100, epochs= n_iter, validation_split=0.2)





'''
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''


#predictions = model.predict(x_test, batch_size = None, verbose = 0, steps = None)

score = model.evaluate(x_test,y_test, batch_size = 100)
print score 

model.save("/home/parthasarathidas/Documents/CNN_retrain/" + str(n_iter) + "_epoch.h5")


