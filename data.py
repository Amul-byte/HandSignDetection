import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
# Set up webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)
imgSize=400
offset=30

while True:
    success, img = cap.read()
    if not success:
        break  # Stop if the camera fails

    # Detect hands
    hands, img = detector.findHands(img)
    #Crop image
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        
        
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        
        imgCropShape = imgCrop.shape
        imgWhite[0:imgCropShape[0],0:imgCropShape[1]]=imgCrop
        # imgCrop.resize(imgWhite)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the image
    cv2.imshow("Image", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy the window
cap.release()
cv2.destroyAllWindows()
