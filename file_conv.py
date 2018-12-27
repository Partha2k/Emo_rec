import numpy as np
from PIL import Image
import os
from scipy import misc, ndimage
import sys

dumpPath = sys.argv[1]
filenames = os.listdir(dumpPath)
for files in filenames:
	image = ndimage.imread(dumpPath + files, mode = 'RGB')
	image_resized = misc.imresize(image, (100,100))
	misc.imsave('/home/parthasarathidas/Documents/train_set2/'+files[0:9]+".jpg", image_resized)


