from imp import new_module
import cv2
from deepface import DeepFace
import numpy as np 

path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

#set the rectangle background to white
rectangle_bgr = (255,255,255)
#make a black image
img = np.zeros((500,500))
#set some text
text = "some text in box"
#get width and height of text box
(text_width, text_heigth) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
#set text start position
text_offset_x = 10
text_offset_y = img.shape[0]-25
#make cords of the box with a small padding of 2px
box_coords = ((text_offset_x, text_offset_y)), ((text_offset_x + text_width + 2, text_offset_y - text_heigth - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text,(text_offset_x,text_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=1)


cap=cv2.VideoCapture(1)

#check if webcam is opened correctly
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")


while True:
    ret,frame=cap.read() #read one image from video
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = faceCascade.detectMultiScale(gray, 1.1,4)

    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color  = frame[y:y+h,x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("face not detected")
        else:
            for(ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex + ew] #cropping the face
            
    final_image = cv2.resize(face_roi, (224,224))
    final_image  = np.expand_dims(final_image,axis= 0)#need fourth dimension
    final_image = final_image/255.0

    font = cv2.FONT_HERSHEY_SIMPLEX

    Predictions = new_module.predict(final_image)

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    if(np.argmax(Predictions)==1):
        status = "Disgust"

        x1,y1,w1,h1 = 0, 0,175, 75
        #draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1+int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.putText(frame, status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))


    elif(np.argmax(Predictions)==2):
        status = "Fear"

        x1,y1,w1,h1 = 0, 0,175, 75
        #draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1+int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.putText(frame, status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))


    elif(np.argmax(Predictions)==3):
        status = "Happy"

        x1,y1,w1,h1 = 0, 0,175, 75
        #draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1+int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.putText(frame, status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))


    elif(np.argmax(Predictions)==4):
        status = "Sad"

        x1,y1,w1,h1 = 0, 0,175, 75
        #draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1+int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.putText(frame, status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))

    elif(np.argmax(Predictions)==5):
        status = "Surprise"

        x1,y1,w1,h1 = 0, 0,175, 75
        #draw black background rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1+int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.putText(frame, status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))
    else:
        status = "Neutral"

        x1,y1,w1,h1 = 0,0,175,75

        #draw black bg rectangle
        cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1)
        #add text
        cv2.putText(frame, status, (x1+int(w1/10),y1 + int(h1/2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.putText(frame, status, (100,150),font,3,(0,0,255),2,cv2.LINE_4)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255))

        cv2.imshow("face emoji recog", frame)

        if cv2.waitKey(2)& 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

        








    

    # faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    # result= DeepFace.analyze(frame, actions = ['emotion'],enforce_detection=FALSE)
    #draw rectangle around face


