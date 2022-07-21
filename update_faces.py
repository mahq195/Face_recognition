import glob
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np
from deepface import DeepFace
import cv2

IMG_PATH = './data/test_images'
DATA_PATH = './data'

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def trans(img):
    transform = transforms.Compose([transforms.ToTensor(), fixed_image_standardization])
    return transform(img)
    
model = InceptionResnetV1(classify=False, pretrained="vggface2").to(device).eval()

embeddings = []
names = []

for i, usr in enumerate(os.listdir(IMG_PATH)):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, str(usr))+'/*'):
        img = Image.open(file)
        print(file)
        # with torch.no_grad():
            # print('smt')
        embeds.append(model(trans(img).to(device).unsqueeze(0))) #1 anh, kich thuoc [1,512]
        print(embeds)
        # embed = DeepFace.represent(img_path = file, detector_backend = 'skip', enforce_detection=False)
        # print(len(embed))
        # print(type(embed))
        # embeds.append(embed)
        # print(np.shape(embeds))
        # print(type(embeds))
    # print(embeds) # embeds gom nhieu ebd, moi ebd cua 1 anh mat 
    # embedding = np.mean(embeds, 0, keepdims=False) #dua ra trung binh cua 30 anh, kich thuoc [1,...]
    
    if len(embeds) ==0:
        continue
    embedding = torch.cat(embeds).mean(0, keepdim=True)
    embeddings.append(embedding) # 1 cai list n cai [1,512]
    # print(embedding)
    names.append(usr)
    print('Done user ', str(usr))
    
embeddings = torch.cat(embeddings) #[n,512]
names = np.array(names)


# np.save(DATA_PATH+"/faceslist", embeddings)
torch.save(embeddings, DATA_PATH+"/faceslist.pth")
np.save(DATA_PATH+"/usernames", names)
print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))