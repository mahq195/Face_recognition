from itertools import count
from operator import concat, index
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from matplotlib.pyplot import axis
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

frame_size = (640,480)
IMG_PATH = './data/test_images'
DATA_PATH = './data'

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])

mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8],margin=20, keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_faceslist():
    embeds = torch.load(DATA_PATH+'/faceslist.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names

def recognize(face, threshold: float=1.0):
    local_ebds, names = load_faceslist()

    img_embedding = resnet(face.unsqueeze(0))
    # print('image_ebd:', img_embedding.size())
    diff = img_embedding - local_ebds
    norm = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))
    print('norm: ', norm)

    min_dist, idx = torch.min(norm, dim=0)
    print(min_dist, idx)
    if min_dist > threshold:
        name = 'Unknown'
        return min_dist, name 
    else:
        name = names[idx]
        return min_dist, name


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

count = -1
score = 0
name = 0
bbox = 0
faces = None

while cap.isOpened():
    ret, frame = cap.read()
    frame2 = frame
    count += 1
    if ret:
        if count%1 ==0:
            faces, boxes = mtcnn(frame)
            # print('boxes: ', boxes)
            if faces is not None:
                for face, box in zip(faces, boxes):
                    score, name = recognize(face)
                    bbox = list(map(int,box.tolist()))
                    cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                    cv2.putText(frame, name + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_8)

        else:
            if faces is not None:
                cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                cv2.putText(frame, str(name) + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_8 )

    cv2.imshow('Face Recognition', frame2)
    if cv2.waitKey(1)&0xFF == 27:
        break