import os
from shutil import copy2
import getpass
username = getpass.getuser()

dest = "/home/"+username+"/Documents/CK_exCK/"
if os.path.exists(dest) == True:
	pass
else:
	os.mkdir(dest)

path1 =  "/home/"+username+"/Documents/cohn_kanade/"
for root, dir, files in os.walk(path1):
    for file in files:
        file_loc = os.path.join(root, file)
        if os.path.isfile(file_loc):
        	copy2(file_loc, dest)
print 10*"#"
'''
path2 = "/home/parthasarathidas/Documents/cohn_kanade_images/"
for root, dir, files in os.walk(path2):
    for file in files:
        file_loc_2 = os.path.join(root, file)
        if os.path.isfile(file_loc_2):
        	copy2(file_loc_2, dest)
print 10*"#"
'''