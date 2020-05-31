import cv2
from tensorflow import keras
from collections import deque
from utils import batch_generate as bg

import numpy as np

INPUT_VIDEO = "../Data/demo/d2_demo.MOV"
MODEL_PATH = "models/C3D"
HIT_AREA_SIZE = 300
MODE_INPUT_SIZE = 224
INPUT_FRAME_NUMBER = 5
MODEL = keras.models.load_model(MODEL_PATH)

LEFT_REC_POSITION = (160,160)
RIGHT_REC_POSITION = (820,160)

SHAPE = (1280,720)

def main(input_path = None):
    if input_path == None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    print("video size:", width, height)
    
    size_h = HIT_AREA_SIZE
    size_w = HIT_AREA_SIZE
    queue_l = deque()
    queue_r = deque()
    frame_number = INPUT_FRAME_NUMBER
    prev_frame = None
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.resize(frame, SHAPE)
            # left
            left_start_point = LEFT_REC_POSITION
            left_end_point = (left_start_point[0]+size_w,left_start_point[1]+size_h)
            cv2.rectangle(frame, left_start_point, left_end_point, (0,255,0),3)

            left_image = frame[left_start_point[1]:left_end_point[1],left_start_point[0]:left_end_point[0]]

            # right
            right_start_point = RIGHT_REC_POSITION
            right_end_point = (right_start_point[0]+size_w,right_start_point[1]+size_h)
            cv2.rectangle(frame, right_start_point, right_end_point, (0,255,0),3)

            right_image = frame[right_start_point[1]:right_end_point[1],right_start_point[0]:right_end_point[0]]

            if prev_frame is not None:
                queue_l.append(bg.opticalFlow(prev_frame[0], left_image))
                queue_r.append(bg.opticalFlow(prev_frame[1], right_image))
            prev_frame = [left_image,right_image]

            l_status = actionStatus(queue_l,frame_number)
            r_status = actionStatus(queue_r,frame_number)

            if len(queue_l) >= frame_number:
                queue_l.popleft()
            if len(queue_r) >= frame_number:
                queue_r.popleft()

            print(l_status,r_status)

            # font = cv2.FONT_HERSHEY_SIMPLEX

            # cv2.putText(frame,"left:{},right:{}".format(str(l_status),str(r_status)),(10,30), font, 1,(2,255,255),3,cv2.LINE_AA)
            
            # cv2.imshow('Video', frame)

            if cv2.waitKey(1) == 27:
                break
        else:
            break
        
    cv2.destroyAllWindows()
    cap.release()


def actionStatus(queue,frame_number):
    status = 0
    if len(queue) >= frame_number:
        status = MODEL.predict_classes(np.array([queue]))
        queue.popleft()
    return status

if __name__ == "__main__":
    print("Run AirDrum")
    main(INPUT_VIDEO)
    # model = C3D()
    
    # model.getData()
    # model.train()