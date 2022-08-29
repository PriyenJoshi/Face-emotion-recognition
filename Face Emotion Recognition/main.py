import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace 

#drawing rectangle around face
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('happyboy.jpg') #variable in which image is stored
plt.imshow(img) #BGR Color

predictions =  DeepFace.analyze(img)

#print(predictions)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #print(faceCascade.empty())

faces = faceCascade.detectMultiScale(gray, 1.1,4)

# draw rectangle around face
for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)

font = cv2.FONT_HERSHEY_SIMPLEX

#use putText() method for inserting text on video
cv2.putText(img,
            predictions['dominant_emotion'],
            (0,50),
            font, 1,
            (0,0,255),
            2,
            cv2.LINE_4)

#Converts BGR(Default) to RGB
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


