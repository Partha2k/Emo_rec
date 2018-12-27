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
import matplotlib.pyplot as plt
np.random.rand(2)
#from keras.layers import Conv1D
files = os.listdir('/home/parthasarathidas/Documents/emotion_rec/Custom_Model/')

n_iter = int(sys.argv[1])
tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE', 'CT']

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
        if tag_list[7] in f:
            targets.append(7)    
    return np.array(targets)


def data(filename):
    train_images = []
    for f in filename:
    	if f.endswith(".PNG"):
           current = f
           org_img =Image.open('/home/parthasarathidas/Documents/emotion_rec/Custom_Model/'+current)
           train_images.append(np.array(org_img.getdata()))        
    return np.array(train_images)

y = targets(files)
print "Fetching Data. Please wait......"

x = data(files)

print "Fetching Complete."

'''
x = np.reshape(x, (np.size(files), 100, 100, 3)) #for image of pixels 100x100 only

from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =2)
#print x_train.shape , y_train.shape, math.floor(int(np.size(files))*0.75)
from keras.utils import np_utils
#y_train = np_utils.to_categorical(y_train, num_classes = 7)
y = np_utils.to_categorical(y, num_classes = 7)
#y = np_utils.to_categorical(y)



#x_train = np.reshape(x_train, (2556, 48, 48, 1))
#x_test = np.reshape(x_test, (852, 48, 48, 1))
#print x.shape
#print y.shape
#x_train = x[0:2556, :, :, :]
#x_test = x[2556:, :, :, :]
#y_train = y[0:2556, :]
#y_test = y[2556:, :]
'''

'''
#from sklearn.metrics import confusion_matrix
#print "\nConfusion Matrix\n"
#print confusion_matrix(y_test, predictions)
from keras.models import Sequential, Model
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing import image
from keras import optimizers


model = Sequential()  
model.add(Conv2D(16, (5, 5), activation = 'relu', input_shape = x.shape[1:]))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (5, 5), activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size= (2, 2)))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax'))

	

#ada = optimizers.adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, decay= 0)
ada = optimizers.adam(lr = 0.0009)
model.compile(optimizer= ada, loss = 'binary_crossentropy',
              metrics= ['accuracy'])




from keras.utils import plot_model
#plot_model(model, to_file='model.png')


#history = model.fit(x, y, batch_size= 16, epochs= n_iter, validation_split=0.2)
#model.save("/home/parthasarathidas/Documents/emotion_rec/Custom_Model/" + str(n_iter) + "_epoch.h5")

#print 10*"#"
#print y_test
#print confusion_matrix(y_test.values.argmax(axis = 1), predictions.argmax(axis = 1))
model_layer = [layer.name for layer in model.layers]
print model_layer

img_path_1 = '/home/parthasarathidas/Documents/train_set2/KA.AN1.39.jpg'
img_path_2 = '/home/parthasarathidas/Documents/train_set2/KL.AN1.16.jpg'
img_path_3 = '/home/parthasarathidas/Documents/train_set2/KM.AN1.17.jpg'
img_path_4 = '/home/parthasarathidas/Documents/train_set2/KR.AN3.85.jpg'
img_path_5 = '/home/parthasarathidas/Documents/train_set2/MK.AN1.12.jpg'
img_path_6 = '/home/parthasarathidas/Documents/train_set2/NA.AN2.21.jpg'
img_path_7 = '/home/parthasarathidas/Documents/train_set2/NM.AN1.10.jpg'
img_path_8 = '/home/parthasarathidas/Documents/train_set2/TM.AN1.19.jpg'
img_path_9 = '/home/parthasarathidas/Documents/train_set2/UY.AN3.14.jpg'

img_1 = image.load_img(img_path_1, target_size=(100, 100))
img1 = image.img_to_array(img_1)
img1_ = np.expand_dims(img1, 0)
img_2 = image.load_img(img_path_2, target_size=(100, 100))
img2 = image.img_to_array(img_2)
img2_ = np.expand_dims(img2, 0)
img_3 = image.load_img(img_path_3, target_size=(100, 100))
img3 = image.img_to_array(img_3)
img3_ = np.expand_dims(img3, 0)
img_4 = image.load_img(img_path_4, target_size=(100, 100))
img4 = image.img_to_array(img_4)
img4_ = np.expand_dims(img4, 0)
img_5 = image.load_img(img_path_5, target_size=(100, 100))
img5 = image.img_to_array(img_5)
img5_ = np.expand_dims(img5, 0)
img_6 = image.load_img(img_path_6, target_size=(100, 100))
img6 = image.img_to_array(img_6)
img6_ = np.expand_dims(img6, 0)
img_7 = image.load_img(img_path_7, target_size=(100, 100))
img7 = image.img_to_array(img_7)
img7_ = np.expand_dims(img7, 0)
img_8 = image.load_img(img_path_8, target_size=(100, 100))
img8 = image.img_to_array(img_8)
img8_ = np.expand_dims(img8, 0)
img_9 = image.load_img(img_path_9, target_size=(100, 100))
img9 = image.img_to_array(img_9)
img9_ = np.expand_dims(img9, 0)


max_pooling2d_2_extract = Model(inputs = model.input, outputs = model.get_layer('max_pooling2d_2').output)
outcome_1 = max_pooling2d_2_extract.predict(img1_)
outcome_2 = max_pooling2d_2_extract.predict(img2_)
outcome_3 = max_pooling2d_2_extract.predict(img3_)
outcome_4 = max_pooling2d_2_extract.predict(img4_)
outcome_5 = max_pooling2d_2_extract.predict(img5_)
outcome_6 = max_pooling2d_2_extract.predict(img6_)
outcome_7 = max_pooling2d_2_extract.predict(img7_)
outcome_8 = max_pooling2d_2_extract.predict(img8_)
outcome_9 = max_pooling2d_2_extract.predict(img9_)

plt.title('Model Output')
plt.subplot(331)
plt.imshow(outcome_1[0,:,:,0])
plt.subplot(332)
plt.imshow(outcome_2[0,:,:,0])
plt.subplot(333)
plt.imshow(outcome_3[0,:,:,0])
plt.subplot(334)
plt.imshow(outcome_4[0,:,:,0])
plt.subplot(335)
plt.imshow(outcome_5[0,:,:,0])
plt.subplot(336)
plt.imshow(outcome_6[0,:,:,0])
plt.subplot(337)
plt.imshow(outcome_7[0,:,:,0])
plt.subplot(338)
plt.imshow(outcome_8[0,:,:,0])
plt.subplot(339)
plt.imshow(outcome_9[0,:,:,0])
plt.show()



	
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


getFiles = "/home/parthasarathidas/Documents/lfw_test/"
filesList = os.listdir(getFiles)
for files in filesList:
    current = Image.open(os.path.join(getFiles,files))

#current
    #current = current.convert('L')
    #current = current.resize((48, 48))
    current = current.resize((100,100))
    data = np.array(current.getdata())
#10 on top
#11 on top
#13 okay
#15 okay
    #data = np.reshape(data, (1, 48, 48, 1))
    data = np.reshape(data, (1, 100, 100, 3))

    prediction = model.predict(data)
    print prediction
    #order = np.argsort(prediction)[0,:]
#predictions = model.predict(x_test, batch_size = None, verbose = 0, steps = None)

#score = model.evaluate(x_test,y_test, batch_size = 100)
#print score 

#model.save("/home/parthasarathidas/Documents/CNN_retrain/" + str(n_iter) + "_epoch.h5")

'''