import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import pyautogui, sys

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
screenSize = pyautogui.size()



while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    #img = cv2.flip(img,1)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  
        bbox1 = hand1["bbox"]  
        centerPoint1 = hand1['center'] 
        handType1 = hand1["type"]  
        print(centerPoint1)

        fingers = detector.fingersUp(hand1)
        totalFingers = fingers.count(1)
        pyautogui.moveTo(screenSize[0]-centerPoint1[0], centerPoint1[1])


        if totalFingers == 4:
            pyautogui.click()
            pyautogui.click()

    
    cv2.imshow("Image", img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
