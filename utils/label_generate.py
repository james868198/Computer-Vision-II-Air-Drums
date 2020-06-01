import numpy as np
import cv2
import time
import os
from check import isDrum

DATA_ROOT = "../../Data/labeledvideos/"

OUTPUTFILE_NAMES = []

CLASS = {
    "d0":1,
    "d1":2,
    "d2":3
}

INPUT_FILE_TYPE = {
    "mp4":1,
    "avi":1
}

# data in seperated fiies defined by different label
def generateLabelsByClass(directory = DATA_ROOT):
     
    classes = os.listdir(directory)
    # if len(classes)!= len(CLASS.keys()):
    #     print("classes number doesn't match")
    #     return False
    for i in range(len(classes)):
        if classes[i] not in CLASS:
            continue
        generateLabelsInDir(directory+classes[i]+"/",CLASS[classes[i]])

    return True

def generateLabelsInDir(directory,type):
    entries = os.listdir(directory)
    for file in entries:
        dot_position = file.rfind(".")
        if file[dot_position+1:] not in INPUT_FILE_TYPE:
            continue
        generateLabel(directory,file,type)
    return True

def generateLabel(directory,file,type):
   
    input_path = directory+file
    label_path = directory+"labels/"
    

    print("[generateLabel] file:", input_path)

    # validate input file
    if file == None or file == "":
        return 
    if not os.path.exists(input_path):
        print("[generateLabel] file isn't existed")
        return

    # auto generate label file
    dot_position = file.rfind(".")
    if dot_position <0:
        return
    if file[dot_position+1:] not in INPUT_FILE_TYPE:
        return
    output_path = label_path+file[:dot_position]+".csv"

    # auto creake directory for labels if not existed
    if not os.path.exists(label_path):
        try:
            os.mkdir(label_path)
        except OSError:
            print ("Creation of the directory %s failed" % label_path)

    # is label existed
    if os.path.exists(output_path):
        return
    labels = getLabelFromVideo(input_path,type)
    if labels == None:
        return
    if len(labels) == 0:
        return
    with open(output_path, "w") as output_file:
        for i, label in enumerate(labels):
            output_file.write(str(label))
            if i < len(labels)-1:
                output_file.write(",")
    return True

def getLabelFromVideo(filename,type):
    cap = cv2.VideoCapture(filename)
    labels = []
    count = 0
    count_hit  = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            # print(str(count))
            hit = isDrum(frame)
            if hit:
                count_hit += 1
                labels.append(type)
            else:
                labels.append(0)

            count += 1
        else:
            break
    print("Sampled " + str(count) + "frames.")
    print("hit:",count_hit)
    # When everything done, release the capture
    cap.release()
    return labels

if __name__ == "__main__":

    test_directory = "/Users/james/Pictures/opencv_test/record3/"
    
    generateLabelsByClass(test_directory)
    pass