import cv2
# from facenet_pytorch import MTCNN
# import torch
import numpy as np
import time
from mtcnn import MTCNN

# device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
prev_frame_time = 0
new_frame_time = 0

mtcnn = MTCNN()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while cap.isOpened():
    isSuccess, frame = cap.read()
    if isSuccess:
        faces = mtcnn.detect_faces(frame)
        if faces is not None:
            for face in faces:
                bbox = face['box'] # x, y ,w, h
                x, y, w, h = bbox
                # frame = cv2.rectangle(frame,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,0,255),6)
                cv2.rectangle(frame,(x, y),(x+w, y+h),(0,155,255),2)
        
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(int(fps))
    cv2.putText(frame, 'FPS: '+ fps, (30,30), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()