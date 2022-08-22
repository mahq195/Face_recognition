import cv2
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
import numpy as np
import mediapipe as mp
import time

frame_size = (640,480)
IMG_PATH = './data/test_images'
DATA_PATH = './data'

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])

# mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8],margin=20, keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face_detection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.75)

def load_faceslist():
    embeds = torch.load(DATA_PATH+'/faceslist.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names

def recognize(face, threshold: float=1.0):
    local_ebds, names = load_faceslist()

    img_embedding = resnet(transform(face).unsqueeze(0))
    # print('image_ebd:', img_embedding.size())
    diff = img_embedding - local_ebds
    norm = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))
    # print('norm: ', norm)

    min_dist, idx = torch.min(norm, dim=0)
    # print(min_dist, idx)
    if min_dist > threshold:
        name = 'Unknown'
        return min_dist, name 
    else:
        name = names[idx]
        return min_dist, name

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    count = -1
    score = 0
    name = 0
    bbox = 0
    faces = None

    while cap.isOpened():
        time_start = time.time()
        ret, frame = cap.read()
        if ret:
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(imgRGB)

            if results.detections:
                for id, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = frame.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                    face = frame[bbox[1]:(bbox[1] +bbox[3] +1) , bbox[0]: (bbox[0] +bbox[2] +1)]
                    face = cv2.resize(face, (160,160))
                    start_re = time.time()
                    score, name = recognize(face)
                    print("recognition time: ", time.time() - start_re)

                    cv2.rectangle(frame, bbox, (255, 0, 255), 2)
                    cv2.putText(frame, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
                    cv2.putText(frame, name + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 1, cv2.LINE_8)
                # print('boxes: ', boxes)
            #     if faces is not None:
            #         for face, box, prob in zip(faces, boxes, probs):
            #             print("prob: ", prob)
            #             score, name = recognize(face)
            #             bbox = list(map(int,box.tolist()))
            #             cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
            #             cv2.putText(frame, name + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_8)

            # else:
            #     if faces is not None:
            #         cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
            #         cv2.putText(frame, str(name) + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_8 )
        time_end = time.time()
        cv2.putText(frame, 'FPS: ' + str(int(1/float(time_end-time_start))), (20,35), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1)&0xFF == 27:
            break