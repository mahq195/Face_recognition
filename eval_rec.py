import cv2
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
import numpy as np
import mediapipe as mp
import time
import os
import glob
from PIL import Image
from recognize_face import load_faceslist, recognize

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face_detection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.75)

TEST_SET = 'lfw1000'
IMG_PATH = './data/test_images'
DATA_PATH = './data'

count = 0
list_img = glob.glob(r'lfw1000\**\*.jpg')
total = len(list_img)

for i, img_path in enumerate(list_img) :

    ground_truth = os.path.basename(os.path.dirname(img_path))
    # print(ground_truth)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img)
    if results.detections:
        # print(results.detections)
        detection = results.detections[0]
        # for id, detection in enumerate(results.detections):
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, ic = img.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
        # print(bbox)
        face = img[bbox[1]:(bbox[1] +bbox[3] +1) , bbox[0]: (bbox[0] +bbox[2] +1)]
        face = cv2.resize(face, (160,160))
    # img = img.resize((160,160))
        score, name = recognize(face)


        # score, name = recognize(img)
        print(str(i)+'/'+str(total), score , name , ground_truth)
        
        if name == ground_truth:
            count += 1

print(count)
print(count / len(list_img)) 
# accuracy = 91.68%