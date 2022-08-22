import glob
import torch 
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os
from PIL import Image
import numpy as np
import cv2

IMG_PATH = './data/test_images'
DATA_PATH = './data'

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

embeddings = []
names = []

for i, usr in enumerate(os.listdir(IMG_PATH)):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, str(usr))+'/*'):
        face = Image.open(file)
        face = face.resize((160,160))
        # face.show()
        tensor_face = transform(face)
        img_embedding = resnet(tensor_face.unsqueeze(0))
        embeds.append(img_embedding)

    embedding = torch.cat(embeds).mean(0, keepdim=True)
    embeddings.append(embedding) 
    names.append(usr)
    print('Done user ', str(usr))

print('start cat ebd ')
embeddings = torch.cat(embeddings) #[n,512]
names = np.array(names)
torch.save(embeddings, DATA_PATH+"/faceslist.pth")
np.save(DATA_PATH+"/usernames", names)
print('Update Completed! There are {0} people in FaceLists'.format(names.shape[0]))