from unicodedata import decimal
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import glob2


device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)


dst_path = r'./data/test_images'
src_path = os.path.normpath(r'./raw_data')

list_image = glob2.glob(src_path + '\**\*.*') 
for (i, image) in enumerate(list_image):
    img = Image.open(image)

    usr_name = os.path.basename(os.path.dirname(image))
    USR_PATH = os.path.join(dst_path, usr_name)
    path = os.path.join(USR_PATH, '{}.jpg'.format(str(i%1) + str(usr_name.replace(' ', ''))))
    if mtcnn(img) is not None:
        face = mtcnn(img, save_path=path)
        print(f'Save image {i} successful')



