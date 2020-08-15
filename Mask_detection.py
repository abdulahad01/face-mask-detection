import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import face_recognition
from imutils import paths
import datetime
from zipfile import ZipFile
import messaging as msg

import h5py
# cv2.__version__

print('loading the model ..')
model =tf.keras.models.load_model('mask_model.h5') #loading the pretrained model from directory

labels ={0 :' mask', 1 :'No Mask'}
color = {1 : (0,0,255),0 :(0,255,0)}

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #path to face classifier
source = cv2.VideoCapture(0) #defining video source

face_1_image = face_recognition.load_image_file("face1.jpg") #Add the directory to your face image
face_1_encoding = face_recognition.face_encodings(face_1_image)[0]

# Load a second sample picture and learn how to recognize it.
sample_image = face_recognition.load_image_file("sample.jpg")
martin_face_encoding = face_recognition.face_encodings(sample_image)[0]

# Create list of known face encodings and their names
known_face_encodings = [
     azeem_face_encoding,
     martin_face_encoding
]
known_face_names = {
     "Azeem":[0,0],
     "Martin":[0,0]
}

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while(source.isOpened()):

    _,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    

    thres, b_w =  cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5) 
    faces_b_w=face_clsfr.detectMultiScale(b_w,1.5,5)
    
    if len(faces) ==0 and len(faces_b_w) == 0 :
        cv2.putText(img,'No face found', (30,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    elif len(faces) == 0 and len(faces_b_w) ==1 :
        label = 0
        cv2.putText(img,labels[label], (30, 30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    else :
        for (x,y,w,h) in faces:
            face_img=img[y:y+h,x:x+w]
            resized=cv2.resize(face_img,(224,224))

            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,224,224,3))


            result=model.predict(reshaped)
            label=np.argmax(result,axis=1)[0]
            probs = round((np.max(result)*100),2)

            if label == 0:
                   cv2.rectangle(img,(x,y),(x+w,y+h),color[label],2)
                   cv2.rectangle(img,(x,y-40),(x+w,y),color[label],-1)
                   cv2.putText(img, labels[label]+str(probs), (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            if label == 1:
    # Grab a single frame of video
    # Resize frame of video to 1/4 size for faster face recognition processing
                   small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                   rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
                   if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
                        face_locations = face_recognition.face_locations(rgb_small_frame)
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                        face_names = []
                        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
                             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                             name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
                             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                             best_match_index = np.argmin(face_distances)
                             if matches[best_match_index]:
                                  name = list(known_face_names.keys())[best_match_index]
                                  if datetime.datetime.now().minute - known_face_names[name][1] >= 1:
                                       known_face_names[name][0] = known_face_names[name][0] + 1
                                       known_face_names[name][1] = datetime.datetime.now().minute
                                       print(name)
                                       print(known_face_names[name][0])
                                       print(known_face_names[name][1])
                                       if known_face_names[name][0] == 1:
                                            msg.sendmessage(name,"First Warning",datetime.datetime.now().hour,datetime.datetime.now().minute+1)
                                       elif known_face_names[name][0] == 2:
                                            msg.sendmessage(name,"Final Warning",datetime.datetime.now().hour,datetime.datetime.now().minute+1)
                                       elif known_face_names[name][0] == 3:
                                            msg.sendmessage(name,"Is Fined",datetime.datetime.now().hour,datetime.datetime.now().minute+1)
                             face_names.append(name)

                   process_this_frame = not process_this_frame


    # Display the results
                   for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                                  top *= 4
                                  right *= 4
                                  bottom *= 4
                                  left *= 4

        # Draw a label with a name below the face
                                  cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                                  cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                                  cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)

    # Display the resulting image

    # Hit 'q' on the keyboard to quit
                   if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    if datetime.datetime.now().minute == 0:
         for i in list(known_face_names.keys()):
              known_face_names[i][0] = 0
              known_face_names[i][1] = 0 
# Release handle to the webcam
                   
    cv2.imshow('LIVE',img)
    #cv2.imshow('face',face_img)
    
    
    key=cv2.waitKey(1)
    
    if(key==27):
        break
cv2.destroyAllWindows()
source.release()
