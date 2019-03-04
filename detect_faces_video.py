#import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

#Usage:
#python detect_faces_video.py --prototxt depley.prototxt.txt \ --model res10_300x300x_ssd_iter_140000.caffemodel

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ad.add_argument("-p", "--prototxt", required=True, help = "Path to Caffe 'deploy' prototxt file")
ad.add_argument("-m", "--model", required=True, help="Path to pre-trained Caffe model")
ad.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum propability to filter waeak detections")
args=vars(ap.parse_args())

#Load the serialized data from disk
print("[INFO] loading model...")
net = cv2.dnn.radNetFromCaffe(args["prototxt"], args["model"])

#Initialize the video stream and allow the camera cursor to warm up
print("[INFO] starting video stream...")
#We initialize a Video Stream object specifying camera with index zero as the
#source (in general this would be your laptop's build in camera or your
#desktop's first camera detected)
vs=VideoStream(src=0).start()
time.sleep(2.0) #2 seconds

#From there we loop over the frames from the video stream
while True:
    #grab the frame from the released video stream and resize to have
    #a maxmimum width of 400 pixels
    frame=vs.read()
    frame=imutils.resize(frame,width=400)

    #grab the frame dimension and convert it into a blob
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300,300), (104.0, 177.0,123.0))

    net.setInput(blob)
    detections = net.forward()
    #We can now loop over the detections, compare to the confidence treshold,
    #and draw face boxes + confidence values on the screen
    for i in range(0, detections.shape[2]):
        #extract the confidece (i.e. the propability) associated with the prediction
        confidence=detections[0,0,i,2]

        #filter out weak detections by ensuring the 'confidence' is greater than
        #the minimum confidence
        if confidence<args["confidence"]:
            continue

        #Compute the (x,y)-coordinates of the bounding boy for the object
        box = detections[0,0,i, 3:7]*np.array([w,h,w,h])
        (startX, startY, endX,endY) = box.astype("int")

        #Draw the bounding box of the face along with the associated propablity
        test ="{:.2f}%".format(confidence*100)
        y= startY - 10 if startY - 10 >10 else startY+10
        cv2.rectangle(frame,(startX, startY), (endX,endY),(0,0,255),2)
        cv2.putText(frame,text,(startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,255),2)

        #As that our openCV face detection has been drawn, let's display the
        #frame on the screen and wait for a keypress
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        #if the 'q' key was pressed, break from the loop
        if key ==ord("q"):
            break

#Just a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
