
# python video_process.py -i wefw -o wefwe -s 135

import cv2
import numpy as np
import sys
from augmentation import denoise,flip,blurring,increase_brightness
import shutil
import os
ROOT =  "/Users/james/Pictures/opencv_test/record2/"
OUTPUT_ROOT =  "/Users/james/Pictures/opencv_test/input3/"


INPUT_DIRECTORY = "d2/"
INPUT_FILE = "d_2_t.mp4"
SIZE_H = 280 
SIZE_W = 280 

OFFSET = 50
DIS_TO_CENTER = (200,-240) # for top
P_COPY_NUM = 5 # 0~4
Filter_COPY_NUM = 4 #~0~4

OUTPUT_FILE = "_r"

# DIS_TO_CENTERS = [(200,-240)]


STATUS = 1 # 1 output, 0 check

fgbg = cv2.createBackgroundSubtractorMOG2()

# (x, y, w, h) = cv2.boundingRect(c)
# cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 20)
# roi = frame[y:y+h, x:x+w]

def parseArgs():
    if sys.argv:
        i = 0
        while i<len(sys.argv): 
            arg = sys.argv[i]
            if arg == "-i":
                if i == len(sys.argv)-1:
                    continue
                i+= 1
                INPUT_FILE = sys.argv[i]
            elif arg == "-o":
                if i == len(sys.argv)-1:
                    continue
                i+= 1
                OUTPUT_FILE = sys.argv[i]
            elif arg == "-sh":
                if i == len(sys.argv)-1:
                    continue
                i+= 1
                SIZE_H = sys.argv[i]
            elif arg == "-sw":
                if i == len(sys.argv)-1:
                    continue
                i+= 1
                SIZE_W = sys.argv[i]
            i+= 1
    # print(INPUT_FILE,OUTPUT_FILE,SIZE)
def cropVideo(type = 0,filter = 0):
    
    input_path = ROOT+INPUT_DIRECTORY+INPUT_FILE
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    size_h = SIZE_H
    size_w = SIZE_W
    dis_to_center = getDis(DIS_TO_CENTER,OFFSET,type)
    dot_position = INPUT_FILE.rfind(".")
    print(dis_to_center)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:",fps)

    filename = "{}{}_p{}_f{}.mp4".format(INPUT_FILE[:dot_position],OUTPUT_FILE,str(type),str(filter))
    output_path = "{}{}{}".format(OUTPUT_ROOT,INPUT_DIRECTORY,filename)

    dot_position = filename.rfind(".")
    output_labelname = "{}.csv".format(filename[:dot_position])
    dot_position = input_path.rfind(".")
    input_label_path = "{}.csv".format(input_path[:dot_position])
    output_label_path = "{}{}labels/{}".format(OUTPUT_ROOT,INPUT_DIRECTORY,output_labelname)
    
    # print(input_path)
    # print(output_path)
    # print(input_label_path)
    # print(output_label_path)

    out = cv2.VideoWriter(output_path,fourcc, fps, (SIZE_W, SIZE_H))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # b1
        if OUTPUT_FILE == "_l":
            start_point = (width//2-dis_to_center[0]-size_w, height//2+dis_to_center[1])
            end_point = (start_point[0]+size_w,start_point[1]+size_h)
        elif OUTPUT_FILE == "_r":
            start_point = (width//2+dis_to_center[0], height//2+dis_to_center[1])
            end_point = (start_point[0]+size_w,start_point[1]+size_h)
        else:
            print("?")
        
        image = frame[start_point[1]:end_point[1],start_point[0]:end_point[0]]
        # cv2.imshow('Video', image)
        if (len(image[0]),len(image)) != (SIZE_W, SIZE_H):
            print("[error] wrong size")
            print(len(image),len(image[0]),len(image[0][0]), (SIZE_W, SIZE_H))

            break
        image = applyFilter(filter, image)
        out.write(image)
        if cv2.waitKey(1) == 27:
            break
    out.release()
    cv2.destroyAllWindows()

    # copy label
    copyLabel(input_label_path,output_label_path)




def showRec():

    width = int(cap.get(3))
    height = int(cap.get(4))
    size_h = SIZE_H
    size_w = SIZE_W
    dis_to_center = DIS_TO_CENTER
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # b1
        if OUTPUT_FILE == "_l":
            start_point = (width//2-dis_to_center[0]-size_w, height//2+dis_to_center[1])
            end_point = (start_point[0]+size_w,start_point[1]+size_h)
        elif OUTPUT_FILE == "_r":
            start_point = (width//2+dis_to_center[0], height//2+dis_to_center[1])
            end_point = (start_point[0]+size_w,start_point[1]+size_h)
        else:
            print("?")
        
        image = cv2.rectangle(frame, start_point, end_point, (0,255,0),3) 

        
        cv2.imshow('Video', image)
       
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def drawAllRec():
    width = int(cap.get(3))
    height = int(cap.get(4))
    size_h = SIZE_H
    size_w = SIZE_W
    
    while True:
        ret, frame = cap.read()
        # frame = fgbg.apply(frame)

        # b
        dis_to_center = (220,80)
        start_point1 = (width//2-dis_to_center[0]-size_w, height//2+dis_to_center[1])
        end_point1 = (start_point1[0]+size_w,start_point1[1]+size_h)
        image = cv2.rectangle(frame, start_point1, end_point1, (0,255,0),3)

        start_point2 = (width//2+dis_to_center[0], height//2+dis_to_center[1])
        end_point2 = (start_point2[0]+size_w,start_point2[1]+size_h)
        image = cv2.rectangle(frame, start_point2, end_point2, (0,255,0),3) 

        # m
        dis_to_center = (300,-20)
        start_point1 = (width//2-dis_to_center[0]-size_w, height//2+dis_to_center[1]-size_h//2)
        end_point1 = (start_point1[0]+size_w,start_point1[1]+size_h)
        image = cv2.rectangle(frame, start_point1, end_point1, (0,255,0),3)

        start_point2 = (width//2+dis_to_center[0], height//2+dis_to_center[1]-size_h//2)
        end_point2 = (start_point2[0]+size_w,start_point2[1]+size_h)
        image = cv2.rectangle(frame, start_point2, end_point2, (0,255,0),3) 

        # t

        dis_to_center = (220,-120)
        start_point1 = (width//2-dis_to_center[0]-size_w, height//2+dis_to_center[1]-size_h)
        end_point1 = (start_point1[0]+size_w,start_point1[1]+size_h)
        image = cv2.rectangle(frame, start_point1, end_point1, (0,255,0),3)

        start_point2 = (width//2+dis_to_center[0], height//2+dis_to_center[1]-size_h)
        end_point2 = (start_point2[0]+size_w,start_point2[1]+size_h)
        image = cv2.rectangle(frame, start_point2, end_point2, (0,255,0),3) 
        cv2.imshow('Video', image)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

def copyLabel(input_path,output_root):
    print("[generateDataWithAug]:",input_path,output_root)
    if not os.path.exists(input_path):
        return
    shutil.copy2(input_path,output_root)
    print("Completed copying label: ",input_path)

def getDis(position,offset,type):
    if type == 4:
        return (position[0]+offset,position[1]+offset) # for 4
    elif type == 3:
        return (position[0]-offset,position[1]+offset) # for 3
    elif type == 2:
        return (position[0]+offset,position[1]-offset) # for 2
    elif type == 1:
        return (position[0]-offset,position[1]-offset) # for 1
    else:
        return position

def applyFilter(type,frame):
    if type == 1:
        return flip(frame)
    elif type == 2:
        return blurring(frame)
    elif type == 3:
        return increase_brightness(frame)
    else:
        return frame
def generateDataWithAug():
    print("generateDataWithAug")
    for i in range(P_COPY_NUM):
        for j in range(Filter_COPY_NUM):
            cropVideo(i,j)

def runningScript():
    print("runningScript")
if __name__ == "__main__":

    # parseArgs()
    # drawRec()
    if STATUS:
        # cropVideo()
        generateDataWithAug()
    else:
        showRec()