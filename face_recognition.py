from itertools import count
from operator import concat
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from matplotlib.pyplot import axis
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import time
from deepface import DeepFace

frame_size = (640,480)
IMG_PATH = './data/test_images'
DATA_PATH = './data'

def trans(img):
    transform = transforms.Compose([transforms.ToTensor(),fixed_image_standardization])
    return transform(img)

def load_faceslist():
    embeds = torch.load(DATA_PATH+'/faceslist.pth')
    names = np.load(DATA_PATH+'/usernames.npy')
    return embeds, names

def inference(model, face, local_embeds, threshold = 2):
    #local: [n,512] voi n la so nguoi trong faceslist
    embeds = []
    # print(trans(face).unsqueeze(0).shape)
    # embed = DeepFace.represent(img_path = face, detector_backend = 'skip', enforce_detection=False) #return a list
    
    embeds.append(model(trans(face).to('cpu').unsqueeze(0)))
    # embeds.append(embed) 
    detect_embeds = torch.cat(embeds) #[1,512]
    # print(detect_embeds.shape)
                    #[1,512,1]                                      [1,512,n]
    # norm_diff = embed - local_embeds
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    print('norm_diff: ',norm_diff.size())
    # norm_score = np.sum(np.power(norm_diff, 2), axis=1) #(1,n), moi cot la tong khoang cach euclide so vs embed moi
    norm_score = torch.sqrt(torch.sum(torch.pow(norm_diff, 2), dim=1)) #(1,n)
    print('norm_score: ', norm_score)
    print('names: ', names)
    # min_dist = np.min(norm_score)
    # embed_idx = np.argmin(norm_score)
    min_dist, embed_idx = torch.min(norm_score, dim = 1)
    print('min_dist: ', min_dist)
    print('embed_idx: ', embed_idx)
    # min_dist, embed_idx = torch.min(norm_score, dim = 1)
    # print(min_dist*power, names[embed_idx])
    # print(min_dist.shape)
    if min_dist > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()

def extract_face(box, img, margin=20):
    face_size = 160
    # img_size = frame_size
    # margin = [
    #     margin * (box[2] - box[0]) / (face_size - margin),
    #     margin * (box[3] - box[1]) / (face_size - margin),
    # ] #tạo margin bao quanh box cũ
    # box = [
    #     int(max(box[0] - margin[0] / 2, 0)),
    #     int(max(box[1] - margin[1] / 2, 0)),
    #     int(min(box[2] + margin[0] / 2, img_size[0])),
    #     int(min(box[3] + margin[1] / 2, img_size[1])),
    # ]
    t, l, b, r = box

    img = img[t:b, l:r]
    # face = cv2.resize(img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    # face = Image.fromarray(face)
    
    return img

if __name__ == "__main__":
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = InceptionResnetV1(
        classify=False,
        pretrained="vggface2"
    ).to(device)
    model.eval()

    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    embeddings, names = load_faceslist()


    # count = -1
    # idx, score = -1, -1
    # boxes = None
    while cap.isOpened():
        isSuccess, frame = cap.read()
        # count += 1
        if isSuccess:
            # if count %2 ==0:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    face = extract_face(bbox, frame)
                    idx, score = inference(model, face, embeddings)
                    # embed = DeepFace.represent(img_path = face, detector_backend = 'skip', enforce_detection=False)

                    if idx != -1:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        score = torch.Tensor.cpu(score[0]).detach().numpy()
                        frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_8)
                    else:
                        frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
                        frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_8)
            # else:
            #     if boxes is not None:
            #         for box in boxes:
            #             bbox = list(map(int,box.tolist()))
            #             if idx is not None:
            #                 frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
            #                 # score = torch.Tensor.cpu(score[0]).detach().numpy()*(10**6)
            #                 frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_8)
            #             else:
            #                 frame = cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
            #                 frame = cv2.putText(frame,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, cv2.LINE_8)

            # new_frame_time = time.time()
            # fps = 1/(new_frame_time-prev_frame_time)
            # prev_frame_time = new_frame_time
            # fps = str(int(fps))
            # cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1)&0xFF == 27:
            break

    # cap.release()
    # cv2.destroyAllWindows()
    # #+": {:.2f}".format(score)
    # img = cv2.imread('chudinhnghiem.jpg')
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # embeddings, names = load_faceslist()
    # power = 10**6

    # mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = 'cpu')

    # model = InceptionResnetV1(
    #     classify=False,
    #     pretrained="casia-webface"
    # ).to('cpu')
    # model.eval()

    # boxes, _ = mtcnn.detect(img)
    # print(boxes)
    # if boxes is not None:
    #     for box in boxes:
    #         bbox = list(map(int,box.tolist()))
    #         face = extract_face(bbox, img)
    #         face.show()
    #         # cv2.imshow('face', face[0])
    #         # cv2.waitKey(0)
    #         idx, score = inference(img, face, embeddings)
    #         if idx != -1:
    #             img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
    #             score = torch.Tensor.cpu(score[0]).detach().numpy()*power
    #             img = cv2.putText(img, names[idx] + '_{:.2f}'.format(score), (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
    #         else:
    #             img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,0,255), 6)
    #             img = cv2.putText(img,'Unknown', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2, cv2.LINE_8)
    # print('DONE')  

    



