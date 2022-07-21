import cv2
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import os
import glob2

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def face_capture(update_mode: bool = False):

    dst_path = r'./data/test_images'
    src_path = os.path.normpath(r'./raw_data')

    # mtcnn = MTCNN(margin = 20, keep_all=False, post_process=False, device = device)
    mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

    if update_mode:

        print("On update mode: ")
        usr_name = str(input("Name of updated folder: "))
        list_image = glob2.glob(os.path.join(src_path, usr_name) + "\\*")

        for (i, image) in enumerate(list_image):
            img = cv2.imread(image)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes,_ = mtcnn.detect(img)
            # img = cv2.resize(img, (640,480))
            USR_PATH = os.path.join(dst_path, usr_name)
            os.mkdir(USR_PATH)
            
            if boxes[0] is not None:                
                l,t,r,b = list(map(int,boxes[0]))
                cropped = img[t:b, l:r]
                # face = cv2.resize(cropped, (160,160))
                cv2.imwrite(os.path.join(USR_PATH, '{}.jpg'.format(str(i%1) + str(usr_name.replace(' ', '')))), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))  
                print(f'Save image {i} successful')


    else:

        print('Scan the whole \'raw_data\' folder: ')
        list_image = glob2.glob(src_path + '\**\*.*')

        for (i, image) in enumerate(list_image):
            img = cv2.imread(image)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes,_ = mtcnn.detect(img)
            # img = cv2.resize(img, (640,480))
            usr_name = os.path.basename(os.path.dirname(image))
            USR_PATH = os.path.join(dst_path, usr_name)        
            
            if boxes[0] is not None:                
                l,t,r,b = list(map(int,boxes[0]))                
                cropped = img[t:b, l:r]
                # face = cv2.resize(cropped, (160,160))                     
                cv2.imwrite(os.path.join(USR_PATH, '{}.jpg'.format(str(i%1) + str(usr_name.replace(' ', '')))), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                print(f'Save image {i} successful')

        # for (i, image) in enumerate(list_image):
        #     img = cv2.imread(image)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     # img = cv2.resize(img, (640,480))
        #     usr_name = os.path.basename(os.path.dirname(image))
        #     USR_PATH = os.path.join(dst_path, usr_name)
        #     if mtcnn(img) is not None:
        #         path = str(USR_PATH+'/{}.png'.format(str(i%1) + str(usr_name.replace(' ', ''))))
        #         face_img = mtcnn(img, save_path = path)
        #         print(f'Save image {i} successful')


face_capture()