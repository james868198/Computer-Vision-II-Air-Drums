import numpy as np
import cv2
import time
import os

DATA_ROOT = "../../Data/input/"
LABEL_ROOT = "labels/"
SHAPE = (224, 224)
INPUT_FILE_TYPE = {
    "mp4":1,
    "avi":1
}
INPUT_FRAME_NUMBER = 5

def generateBatches(directory,shape = SHAPE):
    print("[generateBatchs]")
    entries = os.listdir(directory)
    print(entries)
    batchs = []
    for file in entries:
        data = generateBatch(directory,file,shape)
        if data == None:
            continue
        batchs.extend(data)
    return batchs

def generateBatch(directory,file, shape):

    batches = []
    input_path = directory+file

    # validate input file
    if input_path == None or input_path == "":
        return 
    if not os.path.exists(input_path):
        return 
    dot_position = file.rfind(".")

    if dot_position <0:
        return 
    if file[dot_position+1:] not in INPUT_FILE_TYPE:
        return 
    
    label_path = directory+LABEL_ROOT+file[:dot_position]+".csv"
    
    # get labels
    if not os.path.exists(label_path):
        print("[error] get no label:", label_path,shape)
        return 
    
    labels = None
    with open(label_path, "r") as file_data:
        labels = file_data.read().split(',')
    
    if labels == None or len(labels) == 0:
        return 
    
    # start parsing
    print("[generateBatchs] start parsing")
    cap = cv2.VideoCapture(input_path)

    queue = []
    count = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if frame is not None:
            # print(str(count))
            if len(queue) == INPUT_FRAME_NUMBER:
                queue.pop(0)
            # check = isDrum(frame)
            if shape is not None: 
                frame = cv2.resize(frame, shape)
            queue.append(frame)
            batch = queue.copy()
            if len(batch) == INPUT_FRAME_NUMBER:
                batches.append((batch, labels[count]))
            count += 1
        else:
            break
    print("Sampled batches size:", len(batches))
    # When everything done, release the capture
    cap.release()
    # cv2.destroyAllWindows()
    # print("count: ", len(batches))
    return batches

def playVideo(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            cv2.imshow('Origin',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

def playHit(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            if isDrum(frame):
                frame = printText(frame, "hit")
            cv2.imshow('Hit',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break    
       
def printText(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (50, 50) 
    fontScale = 1
    color = (0, 0, 255) 
    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method
    out = cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    return out   

if __name__ == "__main__":
    
    
    # filename = "labeledvideo5.mp4"

    # ---- generate batchs ----
    batches = generateBatches(DATA_ROOT)
    print("batches:",len(batches))
   
    # ---- plot batches ----
    # i = 1
    # for batch, hit in batches:
    #     # if hit is True:
    #         # print(count, hit, len(batch))
    #     for frame in batch:
    #         if isDrum(frame) is True:
    #             # printText(frame, "hit")
    #             img = printText(frame, str(i) + ". hit")
    #         else:
    #             img = printText(frame, str(i) + ".")
    #         cv2.imshow('Batch',img)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
                
    #     i += 1
    #     # time.sleep(1)