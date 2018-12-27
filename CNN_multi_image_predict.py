import numpy as np
from PIL import Image
import sys
import os
from keras.models import load_model
#360 824
#model = load_model('my_model_360_iter_batch_100.h5')
model = load_model('8_epoch.h5') #custom model created by CNN_trainTest.py for 100 x 100 images

"""
This program takes a single image as its input
and returns the top three predicted emotions
"""
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
    order = np.argsort(prediction)[0,:]
#print order
#print prediction
#first = np.argmax(prediction)
#prediction[0, first] = 0
#second = np.argmax(prediction)

    tag_dict = {0: 'Angry', 1: 'Sadness', 2: 'Surprise', 3: 'Happiness', 4: 'Disgust', 5: 'Fear', 6: 'Neutral'}

#print tag_dict
    print(5*"#")
    print(files)
    print ("First: " + tag_dict[order[-1]])
    print ("Second: " +tag_dict[order[-2]])
    print ("Third: " +tag_dict[order[-3]])
