import cv2
from tensorflow import keras
from collections import deque
import numpy as np

INPUT_VIDEO = ""
MODEL_PATH = "models/C3M"
HIT_AREA_SIZE = 280
MODE_INPUT_SIZE = 224
INPUT_FRAME_NUMBER = 5
MODEL = keras.models.load_model(MODEL_PATH)


def main(input_path = None):
    if input_path == None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    size_h = HIT_AREA_SIZE
    size_w = HIT_AREA_SIZE
    queue_b_l = deque()

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret:
            
            # b
            dis_to_center = (280,80)
            start_point = (width//2-dis_to_center[0]-size_w, height//2+dis_to_center[1])
            end_point = (start_point[0]+size_w,start_point[1]+size_h)
            cv2.rectangle(frame, start_point, end_point, (0,255,0),3)

            b_l = frame[start_point[1]:end_point[1],start_point[0]:end_point[0]]
            b_l = cv2.resize(b_l, (224,224))
            # start_point2 = (width//2+dis_to_center[0], height//2+dis_to_center[1])
            # end_point2 = (start_point2[0]+size_w,start_point2[1]+size_h)
            # image = cv2.rectangle(frame, start_point2, end_point2, (0,255,0),3) 

            # # m
            # dis_to_center = (300,-20)
            # start_point1 = (width//2-dis_to_center[0]-size_w, height//2+dis_to_center[1]-size_h//2)
            # end_point1 = (start_point1[0]+size_w,start_point1[1]+size_h)
            # image = cv2.rectangle(frame, start_point1, end_point1, (0,255,0),3)

            # start_point2 = (width//2+dis_to_center[0], height//2+dis_to_center[1]-size_h//2)
            # end_point2 = (start_point2[0]+size_w,start_point2[1]+size_h)
            # image = cv2.rectangle(frame, start_point2, end_point2, (0,255,0),3) 

            # # t

            # dis_to_center = (220,-120)
            # start_point1 = (width//2-dis_to_center[0]-size_w, height//2+dis_to_center[1]-size_h)
            # end_point1 = (start_point1[0]+size_w,start_point1[1]+size_h)
            # image = cv2.rectangle(frame, start_point1, end_point1, (0,255,0),3)

            # start_point2 = (width//2+dis_to_center[0], height//2+dis_to_center[1]-size_h)
            # end_point2 = (start_point2[0]+size_w,start_point2[1]+size_h)
            # image = cv2.rectangle(frame, start_point2, end_point2, (0,255,0),3) 
            # cv2.imshow('Video', image)
            
            
            # status = actionStatus(queue_b_l,b_l)
            # print(status)

            # font = cv2.FONT_HERSHEY_SIMPLEX

            # cv2.putText(frame,str(status),(10,30), font, 1,(2,255,255),3,cv2.LINE_AA)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) == 27:
                break
        else:
            break
        
    cv2.destroyAllWindows()
    cap.release()
    
def actionStatus(queue,input_data):
    status = 0
    queue.append(input_data)
    print(len(input_data))
    if len(queue) > INPUT_FRAME_NUMBER:
        queue.popleft()
    if len(queue) == INPUT_FRAME_NUMBER:

        status = MODEL.predict_classes(np.array([queue]))

    return status

if __name__ == "__main__":
    print("Run AirDrum")
    main()
    # model = C3D()
    
    # model.getData()
    # model.train()