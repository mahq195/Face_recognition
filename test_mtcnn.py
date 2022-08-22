import cv2
import time
from mtcnn.mtcnn import MTCNN
detector = MTCNN(min_face_size=85, steps_threshold=[0.6, 0.6, 0.7])

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH,640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

if (video.isOpened() == False):
    print("Web Camera not detected")
while (True):
    ret, frame = video.read()
    start_time = time.time()
    if ret == True:
        start_time = time.time()
        location = detector.detect_faces(frame)
        print(time.time()- start_time)
        print('detect done in: ', time.time() - start_time)
        if len(location) > 0:
            for face in location:
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 4)
        cv2.imshow("Output",frame)
        end_time = time.time()
        # fps = 1/(end_time - start_time)
        # print('FPS :', fps)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()