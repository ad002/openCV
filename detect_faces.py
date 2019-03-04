#I'm following this tutorial by Adrian Rosenbrock
#https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/?__s=k1tfi5xcxncrpsppkeop

#We are going to perform a fast, accurate face detection with open CV using a
#pre trained deep learning face detector model shipped with the library.

#In August 2017 OpenCV was officially released, including a number of DL
#frameworks like Caffe, TensorFlow and PyTorch

#The Caffe based face detector can be found in the face_detector sub-directory
#of the dnn samples (https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
#When using OpenCVs deep neural network module with Caffe models, you'll need
#two sets of files
#1. .prototxt file(s), which define the model architecture (i.e. the
    #hidden layers themselves)
#2. .caffemodel file which contains the weights for the actual layers

#The weigt files are not included in the openCV samples direcotry and it
#requires a bit more digging to find them

#Thanks to the hard work of Aleksandr Rybnikov and the other
#contributors to OpenCV’s dnn  module, we can enjoy these more accurate OpenCV face detectors in our own applications.

#In this first example we'll learn how to apply face detection with openCV to
#single input images

# USAGE
# cd '/Users/adrian/desktop/CompVision'
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
#python detect_faces.py --image Webp.net-resizeimage.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
import numpy as np
import argparse
import cv2

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#We have three required arguments
#--image: The path to the input image
ap.add_argument("-i", "--image", required=True, help ="path to input image")
#--prototxt: The path to the Caffe prototxt file
ap.add_argument("-p", "--prototxt", required=True, help ="Path to Caffe 'deploy' prototxt file")
#--model: The path to the pretrained Caffe Model
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
#an optional argument, --confidence, can overwrite default treshold of 05. if you wish
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum propability to filter weak decisions")
args=vars(ap.parse_args())

#From now let's load our model and create a blob from our image:
#load our serialized model from disk
print("[Info] loading model...")
#First, we load our model using our --prototxt and --model file paths
#we store the model as 'net'
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#We then load the input image, extract the dimensions (Line 54),and
#construct an input blop for the image by resizing to
#a fixed 300x300 pixels and then nomalizing it (Line 59)
image=cv2.imread(args["image"])
(h,w) = image.shape[:2]
#The dnn.blopFromImage takes care of pre-preocessing which includes setting the
#blop domensions and nomalization.
#More about the blopFromImage-Function:
#https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
blop= cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0, (300,300),(104.0, 177.0,123.0))

#Now, we'll apply face detection:
#pass the blop through the network and obtain the detecions and predictions
#To detect faces, we pass the blop through the net
print("[INFO] computing object detecions...")
net.setInput(blop)
detections=net.forward()

#And from there we'll loop over the detections and draw boxes around the detected
#faces

#Loop over the detections
for i in range (0, detections.shape[2]):
    #extract the confidence (i.e., propability) associated with the prediction
    confidence = detections[0,0,i,2]

    #filter out weak detections by ensuring the 'confidence' is greater than
    #the minimum confidence
    #We perform this check to filter out weak detections
    if confidence > args["confidence"]:
        #If the confidence meets the minumum treshild, we proceed to draw
        #a recatngle and along with the propability of the detection. To
        #Accomplish this, we first compute the (x-y)coordinates of the bounding
        #box for the object
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX, startY, endX,endY) = box.astype("int")

        #We now build the confidence text string which contains the propbility
        #of the detecion
        #draw the bounding box of the face along with the associated propability
        text = "{:.2f}%".format(confidence*100)
        #In case our text would go off-image borders, we shift it down by 10 pixels
        y= startY -10 if startY-10 > 10 else startY +10
        #Our face rectangle and confidence text is drawn on the image
        cv2.rectangle(image,(startX, startY), (endX, endY), (0,0,255),2)
        cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        #From here we loop back for additional detections following the process again.
        #if no detections remain, we're ready to show our output image in the screen
#Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
