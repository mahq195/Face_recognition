import cv2
from facenet_pytorch import MTCNN
import torch
import time

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device, factor=0.6)

cap = cv2.VideoCapture('nhatanh2.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
count = -1
boxes = None
fps = 0
while cap.isOpened():
    tic = time.time()
    isSuccess, frame = cap.read()
    
    count += 1
    if isSuccess:
        if count%1==0:
            boxes,_ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),1)
            toc = time.time()
            fps = 1/(toc - tic)
            print('detect FPS:', fps)
            # cv2.putText(frame, 'FPS: ' + str(fps), (10,70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,200,0), 1 )
        else:
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),1)
            toc = time.time()
            toc = time.time()
            # fps = 1/(toc - tic)
            # print('skip FPS:', fps)
            # cv2.putText(frame, 'FPS: ' + str(int(1/(toc-tic))), (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,200,0), 1 )

    
    # cv2.putText(frame, 'FPS: ' + str(int(1/(toc-tic))), (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,200,0), 1 )
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
