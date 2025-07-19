import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os
import tensorflow as tf
# Set up webcam
cap = cv2.VideoCapture(0)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# Initialize hand detector
detector = HandDetector(maxHands=1)
imgSize=400
offset=30
counter = 0
labels= ["A","C"]
folder = "Data/C"
os.makedirs(folder, exist_ok=True)


while True:
    success, img = cap.read()
    if not success:
        break  # Stop if the camera fails

    # Detect hands
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    #Crop image
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        
        
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        
        imgCropShape = imgCrop.shape
        #Aspect ratio for the size
        aspect_ratio = h/w
        #conditions to resize the image 
        if aspect_ratio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wGap+wCal]=imgResize
            prediction,index = classifier.getPrediction(imgWhite)
            print(prediction, index)
            
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hGap+hCal, :]=imgResize
            prediction,index = classifier.getPrediction(imgWhite)
        
        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (0, 255, 0), 2)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the image
    cv2.imshow("Image", imgOutput)

    # Exit when 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    key = cv2.waitKey(1)
    if key == ord('s'):
        counter +=1
        cv2.imwrite(f"{folder}/Image_{int(time.time())}.jpg", imgWhite)
        print(counter)
        
    if key == ord('q'):
        break
    
# Release the camera and destroy the window
cap.release()
cv2.destroyAllWindows()
