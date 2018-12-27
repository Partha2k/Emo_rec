import os
import getpass
from shutil import copy, copy2
import math
import sys

username = getpass.getuser()

labelled_data = '/home/'+username+'/Documents/train_set2/'

path = '/home/'+username+'/Documents'

dirname = 'ImnetFormat'

command = 'cd && cd Documents && mkdir ImnetFormat && cd ImnetFormat && mkdir train && mkdir test && mkdir valid'

command2 = 'cd && cd Documents/ImnetFormat/train/ && mkdir Angry && mkdir Contempt && mkdir Fear && mkdir Neutral && mkdir Sad && mkdir Surpised && mkdir Disgusted && mkdir Happy'

command3 = 'cd && cd Documents/ImnetFormat/valid/ && mkdir Angry && mkdir Contempt && mkdir Fear && mkdir Neutral && mkdir Sad && mkdir Surpised && mkdir Disgusted && mkdir Happy'

command4 = 'cd && cd Documents/ImnetFormat/test/ && mkdir Angry && mkdir Contempt && mkdir Fear && mkdir Neutral && mkdir Sad && mkdir Surpised && mkdir Disgusted && mkdir Happy'

an_train = path + '/ImnetFormat/train/Angry/'
ct_train = path + '/ImnetFormat/train/Contempt/'
di_train = path + '/ImnetFormat/train/Disgusted/'
fe_train = path + '/ImnetFormat/train/Fear/'
ha_train = path + '/ImnetFormat/train/Happy/'
ne_train = path + '/ImnetFormat/train/Neutral/'
sa_train = path + '/ImnetFormat/train/Sad/'
su_train = path + '/ImnetFormat/train/Surpised/'

an_valid = path + '/ImnetFormat/valid/Angry/'
ct_valid = path + '/ImnetFormat/valid/Contempt/'
di_valid = path + '/ImnetFormat/valid/Disgusted/'
fe_valid = path + '/ImnetFormat/valid/Fear/'
ha_valid = path + '/ImnetFormat/valid/Happy/'
ne_valid = path + '/ImnetFormat/valid/Neutral/'
sa_valid = path + '/ImnetFormat/valid/Sad/'
su_valid = path + '/ImnetFormat/valid/Surpised/'

an_test = path + '/ImnetFormat/test/Angry/'
ct_test = path + '/ImnetFormat/test/Contempt/'
di_test = path + '/ImnetFormat/test/Disgusted/'
fe_test = path + '/ImnetFormat/test/Fear/'
ha_test = path + '/ImnetFormat/test/Happy/'
ne_test = path + '/ImnetFormat/test/Neutral/'
sa_test = path + '/ImnetFormat/test/Sad/'
su_test = path + '/ImnetFormat/test/Surpised/'


tag_list = ['AN', 'SA', 'SU', 'HA', 'DI', 'FE', 'NE', 'CT']

def split(src):
	fileList = os.listdir(src)
	train_split = round(len(fileList)*0.6)
	test_split = round(len(fileList)*0.2)
	valid_aplit = len(fileList)-(train_split+test_split)
	for idx, file in enumerate(fileList):
		if idx <= train_split:
			if file.endswith(".jpg"):
				if tag_list[0] in file:
					copy(src+file, an_train)
					print idx, file
				if tag_list[1] in file:
					copy(src+file, sa_train)
					print idx, file
				if tag_list[2] in file:
					copy(src+file, su_train)
					print idx, file
				if tag_list[3] in file:
					copy(src+file, ha_train)
					print idx, file
				if tag_list[4] in file:
					copy(src+file, di_train)
					print idx, file
				if tag_list[5] in file:
					copy(src+file, fe_train)
					print idx, file
				if tag_list[6] in file:
					copy(src+file, ne_train)
					print idx, file
				if tag_list[7] in file:
					copy(src+file, ct_train)
					print idx, file   

		if idx > train_split and idx <= train_split+test_split:
			if file.endswith(".jpg"):
				if tag_list[0] in file:
					copy(src+file, an_test)
				if tag_list[1] in file:
					copy(src+file, sa_test)
				if tag_list[2] in file:
					copy(src+file, su_test)
				if tag_list[3] in file:
					copy(src+file, ha_test)
				if tag_list[4] in file:
					copy(src+file, di_test)
				if tag_list[5] in file:
					copy(src+file, fe_test)
				if tag_list[6] in file:
					copy(src+file, ne_test)
				if tag_list[7] in file:
					copy(src+file, ct_test)

		if idx > train_split+test_split and idx <= train_split+test_split+valid_aplit:
			if file.endswith(".jpg"):
				if tag_list[0] in file:
					copy(src+file, an_valid)
				if tag_list[1] in file:
					copy(src+file, sa_valid)
				if tag_list[2] in file:
					copy(src+file, su_valid)
				if tag_list[3] in file:
					copy(src+file, ha_valid)
				if tag_list[4] in file:
					copy(src+file, di_valid)
				if tag_list[5] in file:
					copy(src+file, fe_valid)
				if tag_list[6] in file:
					copy(src+file, ne_valid)
				if tag_list[7] in file:
					copy(src+file, ct_valid) 

def keras_to_imnet():
	if os.path.exists(os.path.join(path,dirname)) == True:
		split(labelled_data)
	else:
		os.system(command)
		os.system(command2)
		os.system(command3)
		os.system(command4)
		split(labelled_data)
		print "Converted to ImageNet Format Dataset"

def Imnet_to_keras():
	dest = "/home/"+username+"/Documents/Custom_Keras_Dataset/"
	if os.path.exists(dest) == True:
		pass
	else:
		os.mkdir(dest)

	path1 =  "/home/"+username+"/Documents/ImnetFormat/"
	if os.path.exists(path1) == True:
		for root, dir, files in os.walk(path1):
			for file in files:
				file_loc = os.path.join(root, file)
				if os.path.isfile(file_loc):
					copy2(file_loc, dest)
		print "Converted to Custom Keras Dataset"
	else:
		print "ImageNet Format folder does not exist."




try:
	choice = sys.argv[1]
	if choice == '1':
		keras_to_imnet()
	elif choice == '2':
		Imnet_to_keras()
	else:
		print "Invalid Selection"

except IndexError:
	print "Please enter a valid Selection"