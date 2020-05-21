
# python video_process.py -i wefw -o wefwe -s 135

import cv2
import numpy as np
import sys

ROOT =  "/Users/james/Pictures/opencv_test/record2/"
OUTPUT_DIRECTORY = "d2/edited/"
INPUT_DIRECTORY = "d2/"
INPUT_FILE = "d_2_t.mp4"
SIZE_H = 280 
SIZE_W = 280 

DIS_TO_CENTER = (200,-240) # for top

OUTPUT_FILE = "_r"
STATUS = 1 # 1 output, 0 check

fgbg = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture(ROOT+INPUT_DIRECTORY+INPUT_FILE)

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
def cropVideo():
    

    width = int(cap.get(3))
    height = int(cap.get(4))
    size_h = SIZE_H
    size_w = SIZE_W
    dis_to_center = DIS_TO_CENTER
    dot_position = INPUT_FILE.rfind(".")
    
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:",fps)
    out = cv2.VideoWriter(ROOT+OUTPUT_DIRECTORY+INPUT_FILE[:dot_position]+OUTPUT_FILE+'.mp4',fourcc, fps, (SIZE_W, SIZE_H))
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
        out.write(image)
        if cv2.waitKey(1) == 27:
            break
    out.release()
    cv2.destroyAllWindows()

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

if __name__ == "__main__":

    # parseArgs()
    # drawRec()
    if STATUS:
        cropVideo()
    else:
        showRec()