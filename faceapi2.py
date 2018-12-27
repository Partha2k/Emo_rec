subscription_key = 'ddfcea46e29d49b2829855a207783c49'
assert subscription_key

import urllib
params = urllib.urlencode({
    # Request parameters
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
})

face_detection_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect?%s' %params
#emotion_recognition_url = "https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize"


header = {'Ocp-Apim-Subscription-Key': subscription_key }

import os
import time
import shutil

image_path = "/home/parthasarathidas/Documents/CK+/"
files = os.listdir(image_path)
num_iter = len(os.listdir("/home/parthasarathidas/Documents/emotion_rec/Custom_Model/"))-4
for idx, file in enumerate(files):
    image_data = open(os.path.join(image_path,file), "rb").read()

    import requests
    import json
    headers  = {'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream" }
    response = requests.post(face_detection_url, headers=headers, data=image_data)
    response.raise_for_status()
    analysis = response.json()
    temp = json.dumps(analysis)
    attribs = json.loads(temp)
    temp_ = attribs[0]['faceAttributes']['emotion']
    dominantAttrib = max(temp_.values())
    for emot, prob in temp_.iteritems():
        if prob == dominantAttrib:
        	#print idx , file, emot
	       if emot == 'anger':
	       	  shutil.move(os.path.join(image_path,file),"IMG."+str(000000)+str(idx+num_iter)+".AN.anger.PNG")
	       	  print "Renamed with suffix AN"
	       if emot == 'contempt':
	       	  shutil.move(os.path.join(image_path,file),"IMG."+str(000000)+str(idx+num_iter)+".CT.contempt.PNG")
	       	  print "Renamed with suffix CT"
	       if emot == 'disgust':
	       	  shutil.move(os.path.join(image_path,file),"IMG."+str(000000)+str(idx+num_iter)+".DI.disgust.PNG")
	       	  print "Renamed with suffix DI"
	       if emot == 'fear':
	       	  shutil.move(os.path.join(image_path,file),"IMG."+str(000000)+str(idx+num_iter)+".FE.fear.PNG")
	       	  print "Renamed with suffix FE"
	       if emot == 'happiness':
	       	  shutil.move(os.path.join(image_path,file),"IMG."+str(000000)+str(idx+num_iter)+".HA.happiness.PNG")
	       	  print "Renamed with suffix HA"
	       if emot == 'neutral':
	       	  shutil.move(os.path.join(image_path,file),"IMG."+str(000000)+str(idx+num_iter)+".NE.neutral.PNG")
	       	  print "Renamed with suffix NE"
	       if emot == 'sadness':
	       	  shutil.move(os.path.join(image_path,file),"IMG."+str(000000)+str(idx+num_iter)+".SA.sadness.PNG")
	       	  print "Renamed with suffix SA"
	       if emot == 'surprise':
	       	  shutil.move(os.path.join(image_path,file),"IMG."+str(000000)+str(idx+num_iter)+".SU.surprise.PNG")
	       	  print "Renamed with suffix SU"
    time.sleep(8)

#analysis = json.dumps(analysis)
#open("face.json","w").write(analysis)
