import numpy as np
import cv2
import time
import os

DATA_ROOT = "../Data/input/"
SHAPE = (224, 224)
INPUT_FILE_TYPE = {
    "mp4":1,
    "avi":1
}
def isDrum(frame):
    
    def matchColor(pixel, color, thresh):
        pB, pG, pR = pixel[0], pixel[1], pixel[2]
        cB, cG, cR = color[0], color[1], color[2]
        if(abs(pB - cB)<thresh and abs(pG - cG)<thresh and abs(pR - cR)<thresh):
            return True
        else:
            return False
            
    if matchColor(frame[0][0], [0,0,0], 20) and matchColor(frame[0][-1], [0,0,255], 20) and matchColor(frame[-1][0], [255,0,0], 20) and matchColor(frame[-1][-1], [0,255,0], 20):
        return True
    else:
        return False
        
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
    
# frames = []
# for i in range(22):
#     ret, frame = cap.read()
#     frames.append(frame)
#     if(isDrum(frame)):
#         print(frames[i][0][0])

# print(frames[0].shape)

def batch_generate(filename, shape=SHAPE):
    cap = cv2.VideoCapture(filename)
    batches = []
    queue = []
    count = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if frame is not None:
            # print(str(count))
            if len(queue) == 5:
                queue.pop(0)
            check = isDrum(frame)
            if shape is not None: 
                frame = cv2.resize(frame, shape)
            queue.append(frame)
            batch = queue.copy()
            if len(batch) == 5:
                batches.append((batch, check))
            count += 1
        else:
            break
    print("Sampled " + str(count) + "frames.")
    # When everything done, release the capture
    cap.release()
    # cv2.destroyAllWindows()
    # print("count: ", len(batches))
    return batches

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

def readFiles(folder_path):
    entries = os.listdir(folder_path)
    print(entries)
    for file in entries:
        batch_generate(folder_path+"/"+file,SHAPE)

def generateLabels(directory):
    entries = os.listdir(directory)
    
    for file in entries:
        generateLabel(directory+file)

def generateLabel(input_path):
    print("[generateLabel] file:", input_path)
    
    # validate input file
    if input_path == None or input_path == "":
        return
    if not os.path.exists(input_path):
        return
    dot_position = input_path.rfind(".")

    if dot_position <0:
        return
    if input_path[dot_position+1:] not in INPUT_FILE_TYPE:
        return
    
    output_path = input_path[:dot_position]+".csv"
    
    # is label existed
    if os.path.exists(output_path):
        return
    labels = getLabelFromVideo(input_path)
    if labels == None:
        return
    if len(labels) == 0:
        return
    with open(output_path, "w") as output_file:
        for i, label in enumerate(labels):
            output_file.write(str(label))
            if i < len(labels)-1:
                output_file.write(",")

def getLabelFromVideo(filename):
    cap = cv2.VideoCapture(filename)
    labels = []
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            # print(str(count))
            labels.append(isDrum(frame))
            count += 1
        else:
            break
    print("Sampled " + str(count) + "frames.")
    # When everything done, release the capture
    cap.release()
    # cv2.destroyAllWindows()
    # print("count: ", len(batches))
    return labels

if __name__ == "__main__":
    
    
    # filename = "labeledvideo5.mp4"
    filename = "v_ApplyEyeMakeup_g01_c01.avi"
    
    # playVideo(DATA_ROOT + filename)
    # playHit(DATA_ROOT + filename)
    
    # ---- generate batchs ----
    # batches = batch_generate(DATA_ROOT + filename, shape)
    # print("batches:")
    # count = len(batches)
    # print("Generated: " + str(count) + " batches.")

    # ---- write labels for all video in a directory. output file name = [input_file_name}.csv ----
    generateLabels(DATA_ROOT)

    # ---- read directory ----
    # readFiles(DATA_ROOT+"UCF-101/HammerThrow")

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