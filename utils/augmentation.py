import os
import cv2
import numpy as np
from skimage import io 
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte
import os
from skimage.util import random_noise
from matplotlib import pyplot as plt
import shutil
import imutils

# from batch_generate import generateBatches

input_path = "../../Data/testing/d2_15_1.jpg"
output_path = "../../Data/testing/hand_vsh.png"


INPUT_ROOT = "../../Data/test_input"
OUTPUT_ROOT = "../../Data/testing/output"

INPUT_FILE_TYPE = {
    "mp4":1,
    "avi":1
}

# ------------------

def rotation_90_clockwise(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 

def rotation(img,angle):
    return imutils.rotate(img, angle)
    
def blurring(img):
    return cv2.GaussianBlur(img, (11,11),0)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def denoise(img):
    # return random_noise(img)
    return cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

def flip(img):
    return cv2.flip(img, 1)

def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def generateData(input_root,output_root,islabel=True):
    entries = os.listdir(input_root)
    print(entries)

    label_root = input_root+ "/labels"
    for file in entries:

        dot_position = file.rfind(".")
        if dot_position <0:
            continue 
        if file[dot_position+1:] not in INPUT_FILE_TYPE:
            continue 
        
        input_path = "{}/{}".format(input_root,file)
        output_path = "{}/edited_{}".format(output_root,file)
        cap = cv2.VideoCapture(input_path)

        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter(output_path,fourcc,fps,(width,height))

        print("start augementation: ",input_path)
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                img = horizontal_shift(frame,0.3)
                out.write(img)
                # out.write(frame)
            else:
                break
        out.release()
        cap.release()
        print("Completed augementation: ",input_path)
       
        #copy labels
        if islabel:
            label_path ="{}/{}".format(label_root,file[:dot_position]+".csv")
            if not os.path.exists(label_path):
                continue
            new_label_path ="{}/labels/edited_{}".format(output_root,file[:dot_position]+".csv")
            print(label_path,new_label_path)
            shutil.copy2(label_path,new_label_path)
            print("Completed copying label: ",input_path)

def colorModify(img, alpha, beta):
    return cv2.convertScaleAbs(img,alpha=alpha, beta=beta)

def script_testing(input_path,output_path):
    img = cv2.imread(input_path)
    new_image = channel_shift(img)
    cv2.imwrite(output_path, new_image)
    # plt.imshow(new_image)
    # plt.show()

def play_effect(input_path):
    if input_path:
        cap = cv2.VideoCapture(input_path)
    else:
        cap = cv2.VideoCapture(0)
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            img = colorModify(frame,0.67,-20)
            cv2.imshow('frame',img)
            # cv2.imshow('frame',frame)
      
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
if __name__ == "__main__":
    print("run aug")
     
    # generateData(INPUT_ROOT,OUTPUT_ROOT,False)
    # play_effect("/Users/james/Pictures/opencv_test/record3/d2/d2_1.mp4")