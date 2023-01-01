import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
import cv2
import mediapipe as mp
from cvzone.PoseModule import PoseDetector
import face_recognition
import numpy as np



cap = cv2.VideoCapture(0)
faceDetector = FaceMeshDetector(maxFaces=4)
detector = HandDetector(detectionCon=0.8, maxHands=2)
poseDetector = PoseDetector()
faceLayer = False
handLayer = False
poseLayer = False
handLayer2 = False
handlayer3 = False
handlayer4 = False
facialRecLayer = False

justin_image = face_recognition.load_image_file("justin.jpg")
justin_face_encoding = face_recognition.face_encodings(justin_image)[0]

omar_image = face_recognition.load_image_file("omar.jpg")
omar_face_encoding = face_recognition.face_encodings(omar_image)[0]
known_face_encodings = [
    justin_face_encoding,
    omar_face_encoding
]
known_face_names = [
    "Justin Chuang",
    "Omar"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    
    rgbFrame = img[:, :, ::-1]
    
    
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  
        bbox1 = hand1["bbox"]  
        centerPoint1 = hand1['center'] 
        handType1 = hand1["type"]  

        fingers1 = detector.fingersUp(hand1)
        totalFingersRight = fingers1.count(1)

        if faceLayer is True or totalFingersRight == 1:
            faceLayer = True
            img, faces = faceDetector.findFaceMesh(img)

            if faces:
                print(faces[0])

        if poseLayer is True or totalFingersRight == 4:
            poseLayer = True
            faceLayer = False
            mg = poseDetector.findPose(img)
            lmList, bboxInfo = poseDetector.findPosition(img)
    
            if bboxInfo:
                center = bboxInfo["center"]
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            fingers2 = detector.fingersUp(hand2)
            lmList2 = hand2["lmList"] 
            bbox2 = hand2["bbox"] 
            centerPoint2 = hand2['center']  
            handType2 = hand2["type"]  
            faceLayer = False
            poseLayer = False
            facialRecLayer = False
        


            #index index distance

            totalFingersLeft = fingers2.count(1)

            if totalFingersRight == 1 or totalFingersLeft == 1:
                fingers2 = detector.fingersUp(hand2)
                length, info, img = detector.findDistance(lmList1[8], lmList2[8], img) 
                cv2.putText(img, f'Dist:{int(length)}', (bbox2[0] + 400, bbox2[1] - 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        else:
            #Thumb index distance
            if handLayer is True or totalFingersRight == 3:
                handlayer3 = True
                length, info, img = detector.findDistance(lmList1[8], lmList1[4], img)
                cv2.putText(img, f'Dist:{int(length)}', (bbox1[0] + 400, bbox1[1] - 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            #Finger Count
            if handLayer2 == True or totalFingersRight ==2:
                handLayer2 = True
                totalFingers = fingers1.count(1)
                cv2.putText(img, f'Fingers:{totalFingers}', (bbox1[0] + 200, bbox1[1] - 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
        if facialRecLayer is True or totalFingersRight == 5:
            faceLayer = False
            handLayer = False
            poseLayer = False
            handLayer2 = False
            handlayer3 = False
            handlayer4 = False

        

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgbFrame)
                face_encodings = face_recognition.face_encodings(rgbFrame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "A lil Bitch"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)
                
                
                for (top, right, bottom, left), name in zip(face_locations, face_names):

                    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 255), 2)
                    cv2.rectangle(img, (left, bottom - 35), (right, bottom), (255, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)



    cv2.imshow("Image", img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()