import cv2
from tensorflow import keras
from collections import deque
from utils import batch_generate as bg

import numpy as np

INPUT_VIDEO = "../Data/demo/d1_demo.MOV"
OUTPUT_PATH = "../Data/demo/d1_demo_output.MOV"
MODEL_PATH = "models/C3D"
HIT_AREA_SIZE = 300
MODE_INPUT_SIZE = (224,224)
INPUT_FRAME_NUMBER = 5
MODEL = keras.models.load_model(MODEL_PATH)

LEFT_REC_POSITION = (160,160)
RIGHT_REC_POSITION = (820,160)

# LEFT_REC_POSITION = (160,230)
# RIGHT_REC_POSITION = (820,230)
SHAPE = (1280,720)
COLOR = [(255,0,0),(0,255,0)] #0:no hit, 1:hit
EXPORT = True

def main(input_path = None,isOutput = EXPORT):
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
    
    l_status = 0
    r_status = 0
    count = 0
    if isOutput:
        if OUTPUT_PATH is None:
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter(OUTPUT_PATH,fourcc, fps, SHAPE)
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.resize(frame, SHAPE)
            # left
            left_start_point = LEFT_REC_POSITION
            left_end_point = (left_start_point[0]+size_w,left_start_point[1]+size_h)

            left_image = frame[left_start_point[1]:left_end_point[1],left_start_point[0]:left_end_point[0]]
            cv2.rectangle(frame, left_start_point, left_end_point, COLOR[int(l_status)],3)

            left_image = cv2.resize(left_image, MODE_INPUT_SIZE)
            
            # right
            right_start_point = RIGHT_REC_POSITION
            right_end_point = (right_start_point[0]+size_w,right_start_point[1]+size_h)

            right_image = frame[right_start_point[1]:right_end_point[1],right_start_point[0]:right_end_point[0]]
            cv2.rectangle(frame, right_start_point, right_end_point, COLOR[int(r_status)],3)

            right_image = cv2.resize(right_image, MODE_INPUT_SIZE)

            if prev_frame is not None:
                queue_l.append(bg.opticalFlow(prev_frame[0], left_image))
                queue_r.append(bg.opticalFlow(prev_frame[1], right_image))
            prev_frame = [left_image,right_image]

            l_status = actionStatus(queue_l,frame_number)
            r_status = actionStatus(queue_r,frame_number)

            print(count,l_status,r_status)

            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(frame,"left:{},right:{}".format(str(l_status),str(r_status)),(10,30), font, 1,(2,255,255),3,cv2.LINE_AA)
            
            if isOutput:
                out.write(frame)
            else:
                cv2.imshow('Video', frame)

            
            if cv2.waitKey(1) == 27:
                break
            count += 1
        else:
            break
    
    if isOutput:
        out.release()
    else:
        cv2.destroyAllWindows()
        
    cap.release()


def playCroppedVideo(input_path,output_path):
    if input_path is None:
        return
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    

    print("video size:", width, height)
    
    size_h = HIT_AREA_SIZE
    size_w = HIT_AREA_SIZE
    queue = deque()
    frame_number = INPUT_FRAME_NUMBER
    prev_frame = None
    
    status = 0
    count = 0

    if output_path is None:
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(output_path,fourcc, fps, (width,height))

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:            
            framse_resized = cv2.resize(frame, MODE_INPUT_SIZE)
            cv2.rectangle(frame, (10,10), (size_w-10,size_h-10), COLOR[int(status)],5)

            if prev_frame is not None:
                queue.append(bg.opticalFlow(prev_frame, framse_resized))
            prev_frame = framse_resized

            status = actionStatus(queue,frame_number)
            
            print(count,status)

            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(frame,"count:{},status:{}".format(str(count),str(status)),(10,30), font, 1,(2,255,255),3,cv2.LINE_AA)
            
            out.write(frame)
            # cv2.imshow('Video', frame)
            if cv2.waitKey(1) == 27:
                break
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    out.release()
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
    # playCroppedVideo(INPUT_VIDEO,OUTPUT_PATH)
    # model = C3D()
    
    # model.getData()
    # model.train()