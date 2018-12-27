subscription_key = 'b165685d5fdd43119c2c3e086e36f87c'
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

image_path = "/home/parthasarathidas/Documents/CK_exCK/"
files = os.listdir(image_path)
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
	       	  os.rename(os.path.join(image_path,file),str(000000)+str(idx)+".AN.anger.jpg")
	       	  print "Renamed with suffix AN"
	       if emot == 'contempt':
	       	  os.rename(os.path.join(image_path,file),str(000000)+str(idx)+".CT.contempt.jpg")
	       	  print "Renamed with suffix CT"
	       if emot == 'disgust':
	       	  os.rename(os.path.join(image_path,file),str(000000)+str(idx)+".DI.disgust.jpg")
	       	  print "Renamed with suffix DI"
	       if emot == 'fear':
	       	  os.rename(os.path.join(image_path,file),str(000000)+str(idx)+".FE.fear.jpg")
	       	  print "Renamed with suffix FE"
	       if emot == 'happiness':
	       	  os.rename(os.path.join(image_path,file),str(000000)+str(idx)+".HA.happiness.jpg")
	       	  print "Renamed with suffix HA"
	       if emot == 'neutral':
	       	  os.rename(os.path.join(image_path,file),str(000000)+str(idx)+".NE.neutral.jpg")
	       	  print "Renamed with suffix NE"
	       if emot == 'sadness':
	       	  os.rename(os.path.join(image_path,file),str(000000)+str(idx)+".SA.sadness.jpg")
	       	  print "Renamed with suffix SA"
	       if emot == 'surprise':
	       	  os.rename(os.path.join(image_path,file),str(000000)+str(idx)+".SU.surprise.jpg")
	       	  print "Renamed with suffix SU"
    time.sleep(10)

#analysis = json.dumps(analysis)
#open("face.json","w").write(analysis)
