
import os
import glob
import mediapipe as mp
import cv2


# device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(device=device)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face_detection = mpFaceDetection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


dst_path = r'./data/test_images'
src_path = r'lfw1000'

list_user = os.listdir(src_path)

for user in list_user:
    user_path = os.path.join(src_path, user)
    user_images = os.listdir(user_path)
    
# print(list_image)

    image = os.path.join(user_path, user_images[0])
# for (i, image) in enumerate(list_image):
    img = cv2.imread(image)
    # print(image)
    results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    usr_name = os.path.basename(os.path.dirname(image))
    print(usr_name)
    # print(usr_name)
    USR_PATH = os.path.join(dst_path, usr_name)
    # path = os.path.join(USR_PATH, '{}.jpg'.format(str(i%1) + str(usr_name.replace(' ', ''))))

    if results.detections:
        print(results.detections)
        detection = results.detections[0]
        # for id, detection in enumerate(results.detections):
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, ic = img.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
        # print(bbox)
        face = img[bbox[1]:(bbox[1] +bbox[3] +1) , bbox[0]: (bbox[0] +bbox[2] +1)]
        face = cv2.resize(face, (160,160))

        if not os.path.exists(USR_PATH):
            os.mkdir(USR_PATH)

        cv2.imwrite(os.path.join(USR_PATH, os.path.basename(image)), face)
        print( usr_name + ' done ' + str(id) + '\n')
    
    



