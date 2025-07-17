import cv2
from cvzone.HandTrackingModule import HandDetector

# Set up webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        break  # Stop if the camera fails

    # Detect hands
    hands, img = detector.findHands(img)

    # Display the image
    cv2.imshow("Image", img)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy the window
cap.release()
cv2.destroyAllWindows()
