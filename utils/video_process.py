
# python video_process.py -i wefw -o wefwe -s 135

import cv2
import numpy as np
import sys
from augmentation import denoise,flip,blurring,increase_brightness,colorModify,rotation
import shutil
import os
ROOT =  "/Users/james/Pictures/opencv_test/record3/"
OUTPUT_ROOT =  "/Users/james/Pictures/opencv_test/record3/output_d2"


INPUT_DIRECTORY = "d2/"
INPUT_FILE = "d2_1.mp4"
# SIZE_H = 280 
# SIZE_W = 280 

SIZE_H = 320 
SIZE_W = 320 

OFFSET = 50

DIS_TO_CENTER = (180,80) # for bot r

REG_POSITION = (480,200)
P_COPY_NUM = 5 # 0~4
Filter_COPY_NUM = 4 #~0~4

OUTPUT_FILE = "_"

# DIS_TO_CENTERS = [(200,-240)]


STATUS = 1 # 1 output, 0 check

fgbg = cv2.createBackgroundSubtractorMOG2()

# --------------- FOR CROP VIDEO 2

INPUT_ROOT2 = "/Users/james/Pictures/opencv_test/record3/d2"
INPUT_FILE_Name2 = "d2_1.mp4"
OUTPUT_ROOT2 =  "/Users/james/Pictures/opencv_test/record3/output_d2_1"

ROTATIONS = [0,10,30,-10,-30]
COLOR_LEVEL = [(1,0),(0.5,40),(0.67,-20),(1.6,10),(1.8,30)]
# ROTATIONS = [0,10]
# COLOR_LEVEL = [(1,0),(0.5,40)]

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

#out of updated  
def cropVideo(type = 0,filter = 0, replace = False):
    
    input_path = ROOT+INPUT_DIRECTORY+INPUT_FILE
   
    size_h = SIZE_H
    size_w = SIZE_W
    dis_to_center = getDis(DIS_TO_CENTER,OFFSET,type)
    dot_position = INPUT_FILE.rfind(".")
    print(dis_to_center)
   
    filename = "{}{}_p{}_f{}.mp4".format(INPUT_FILE[:dot_position],OUTPUT_FILE,str(type),str(filter))
    output_path = "{}{}{}".format(OUTPUT_ROOT,INPUT_DIRECTORY,filename)

    dot_position = filename.rfind(".")
    output_labelname = "{}.csv".format(filename[:dot_position])
    dot_position = input_path.rfind(".")
    input_label_path = "{}.csv".format(input_path[:dot_position])
    output_label_path = "{}{}labels/{}".format(OUTPUT_ROOT,INPUT_DIRECTORY,output_labelname)
    
    if os.path.exists(output_path) and not replace:
        print("video already existed")
        return
    
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(3))
    height = int(cap.get(4))
   
    # print(input_path)
    # print(output_path)
    # print(input_label_path)
    # print(output_label_path)

    if OUTPUT_FILE == "_l":
        start_point = (width//2-dis_to_center[0]-size_w, height//2+dis_to_center[1])
        end_point = (start_point[0]+size_w,start_point[1]+size_h)
    elif OUTPUT_FILE == "_r":
        start_point = (width//2+dis_to_center[0], height//2+dis_to_center[1])
        end_point = (start_point[0]+size_w,start_point[1]+size_h)
    else:
        start_point = REG_POSITION
        end_point = (start_point[0]+size_w,start_point[1]+size_h)

    print(start_point,end_point)
    if start_point[0]<0 or start_point[0]> width:
        print("out of border")
        return
    if start_point[1]<0 or start_point[1]> height:
        print("out of border")
        return
    if end_point[1]<0 or end_point[1]> width:
        print("out of border")
        return
    if end_point[1]<0 or end_point[1]> height:
        print("out of border")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(output_path,fourcc, fps, (SIZE_W, SIZE_H))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        image = frame[start_point[1]:end_point[1],start_point[0]:end_point[0]]
        # cv2.imshow('Video', image)
        # print(image)
        if (len(image[0]),len(image)) != (SIZE_W, SIZE_H):
            print("[error] wrong size")
            print(len(image),len(image[0]),len(image[0][0]), (SIZE_W, SIZE_H))

            break
        image = applyFilter(filter, image)
        out.write(image)
        if cv2.waitKey(1) == 27:
            break
    out.release()
    # copy label
    copyLabel(input_label_path,output_label_path)

def cropVideo2(file_name, input_root,output_root, reg_p, ro, cl):
    dot_position = file_name.rfind(".")
    
    input_path = "{}/{}".format(input_root,file_name)
    input_label_path = "{}/labels/{}.csv".format(input_root,file_name[:dot_position])
    
    output_file_name = file_name[:dot_position]
    output_file_name+= "_r{}_c{}".format(str(ro),str(cl))
    output_file_name += ".mp4"


    dot_position = output_file_name.rfind(".")

    labelname = "{}.csv".format(output_file_name[:dot_position])

    output_path = "{}/{}".format(output_root,output_file_name)
    output_label_path = "{}/labels/{}".format(output_root,labelname)
    
    print(input_path)
    print(input_label_path)
    print(output_path)
    print(output_label_path)

    if os.path.exists(output_path):
        print("video already existed")
        return
    
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    size_h = SIZE_H
    size_w = SIZE_W

    start_point = reg_p
    end_point = (start_point[0]+size_w,start_point[1]+size_h)

    print(start_point,end_point)
    if start_point[0]<0 or start_point[0]> width:
        print("out of border")
        return
    if start_point[1]<0 or start_point[1]> height:
        print("out of border")
        return
    if end_point[1]<0 or end_point[1]> width:
        print("out of border")
        return
    if end_point[1]<0 or end_point[1]> height:
        print("out of border")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(output_path,fourcc, fps, (SIZE_W, SIZE_H))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if ro != 0:
            frame = rotation(frame,ROTATIONS[ro])
        if cl != 0:
            frame = colorModify(frame,COLOR_LEVEL[cl][0],COLOR_LEVEL[cl][1])
        image = frame[start_point[1]:end_point[1],start_point[0]:end_point[0]]
        out.write(image)
        if cv2.waitKey(1) == 27:
            break
    out.release()
    # copy label
    copyLabel(input_label_path,output_label_path)


def showRec():
    input_path = ROOT+INPUT_DIRECTORY+INPUT_FILE
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    size_h = SIZE_H
    size_w = SIZE_W
    dis_to_center = DIS_TO_CENTER
    
    if OUTPUT_FILE == "_l":
        start_point = (width//2-dis_to_center[0]-size_w, height//2+dis_to_center[1])
        end_point = (start_point[0]+size_w,start_point[1]+size_h)
    elif OUTPUT_FILE == "_r":
        start_point = (width//2+dis_to_center[0], height//2+dis_to_center[1])
        end_point = (start_point[0]+size_w,start_point[1]+size_h)
    else:
        start_point = REG_POSITION
        end_point = (start_point[0]+size_w,start_point[1]+size_h)

    print(start_point,end_point)
    if start_point[0]<0 or start_point[0]> width:
        return
    if start_point[1]<0 or start_point[1]> height:
        return
    if end_point[1]<0 or end_point[1]> width:
        return
    if end_point[1]<0 or end_point[1]> height:
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
       
        
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
        print("label exist")
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

def applyFilter(type,frame, angle = 0, parameterForlEVEL = (1,0)):
    if type == 1:
        return flip(frame)
    elif type == 2:
        return blurring(frame)
    elif type == 3:
        return increase_brightness(frame)
    elif type == 4:
        return colorModify(frame,alpha=parameterForlEVEL[0], beta=parameterForlEVEL[1])
    elif type == 5:
        return rotation(frame,angle)
    else:
        return frame

def generateDataWithAug():
    print("generateDataWithAug")
    for i in range(P_COPY_NUM):
        for j in range(Filter_COPY_NUM):
            cropVideo(i,j)
        # cropVideo(i,3,replace = True)

def runningScript():
    for i in range(len(ROTATIONS)):
        for j in range(len(COLOR_LEVEL)):
            cropVideo2(file_name = INPUT_FILE_Name2, input_root = INPUT_ROOT2, output_root=OUTPUT_ROOT2,reg_p = REG_POSITION, ro = i, cl = j)

if __name__ == "__main__":
    # parseArgs()
    # drawRec()
    if STATUS:
        # cropVideo()
        # generateDataWithAug()
        runningScript()
    else:
        showRec()