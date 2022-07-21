# %%
from unittest import result
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace
from retinaface import RetinaFace
import numpy as np
import os
import face_recognition
import time

backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

# %%
def get_identity(path):
    file = os.path.abspath(str(path))
    class_name = os.path.dirname(file)
    class_name = os.path.basename(class_name)
    return class_name


db_path = r"..\\mix_set"

camera_id = 0
cap = cv2.VideoCapture(camera_id)
# Check if camera opened successfully
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Frame',frame)
    
        tic = time.time()
        result = DeepFace.find(frame, db_path, model_name='Facenet', detector_backend='ssd', enforce_detection=False)
        if len(result['identity']) >0:
            name = get_identity(str(result['identity'][0])) 
            cv2.putText(frame, 'Hello '+str(name), (50,50), fontFace= cv2.FONT_HERSHEY_DUPLEX,fontScale=1.0, color=(255,255,0), thickness=5 )
        else:
            cv2.putText(frame, 'NO face detected '+str(name), (50,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,0), 5)
        cv2.putText(frame, 'Time: ' +str(time.time() - tic), (50,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,0),1) 
        cv2.imshow('Frame',frame)
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


