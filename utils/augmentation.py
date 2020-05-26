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

# from batch_generate import generateBatches

# input_path = "../../Data/testing/d2_15/d2_15_1.jpg"
input_path = "../../Data/testing/d2_15_1.jpg"

# output_path = "../../Data/testing/output/output_flp.png"
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

def blurring(img):
    return cv2.GaussianBlur(img, (11,11),0)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def zoom(img, value=0.7):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img

def denoise(img):
    # return random_noise(img)
    return cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

def flip(img):
    return cv2.flip(img, 1)

def horizontal_shift(img,ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img
    
def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    return img

def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def fill(img, h, w):
        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
        return img

def channel_shift(img, value = 100):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
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

def script_testing(input_path,output_path):
    img = cv2.imread(input_path)
    new_image = channel_shift(img)
    cv2.imwrite(output_path, new_image)
    # plt.imshow(new_image)
    # plt.show()

def play_effect():
    cap = cv2.VideoCapture(0)
    
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            img = horizontal_shift(frame)
            cv2.imshow('frame',img)
            # cv2.imshow('frame',frame)
      
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
if __name__ == "__main__":
    print("run aug")
     
    generateData(INPUT_ROOT,OUTPUT_ROOT,False)
    # play_effect()