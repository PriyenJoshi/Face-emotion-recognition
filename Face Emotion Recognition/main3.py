from pickle import FALSE
import cv2
from deepface import DeepFace 

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(1)

#check if webcam is opened correctly
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

while True:
    ret,frame=cap.read() #read one image from video
    result= DeepFace.analyze(frame, actions = ['emotion'],enforce_detection=FALSE)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    
    faces = faceCascade.detectMultiScale(gray, 1.1,4)
    # faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    #draw rectangle around face
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX


    #use putText() method for inserting text on video
    cv2.putText(frame,
                result["dominant_emotion"],
                (50, 50),
                font, 3,
                (0, 0, 255),
                2,
                cv2.LINE_4)

    cv2.imshow("Demo Video",frame)
    
    if cv2.waitKey(2) & 0xFF==27:
        break

    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break    

cap.release()
cv2.destroyAllWindows()
    
    
    
