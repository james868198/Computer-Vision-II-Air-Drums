from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt


import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from batch_generate import generateBatches
import numpy as np

input_path = "../../Data/testing/d2_15/d2_15_1.jpg"
output_path = "../../Data/testing/output/output.jpg"
output_root = "../../Data/testing/d2_15/dd1"

img = cv2.imread(input_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)

print(img.shape)

img = img.reshape((1,) + img.shape)

print(img.shape)

datagen = ImageDataGenerator(    
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
i = 0

datagen.flow(img, batch_size=10,
    save_to_dir=output_root, save_prefix='cat', save_format='jpeg')

for batch in datagen.flow(img, batch_size=10,
    save_to_dir=output_root, save_prefix='cat', save_format='jpeg'):
    # print(batch)

    # plt.subplot(5,4,1 + i)
    # plt.axis("off")

    # augImage = batch[0]
    # augImage = augImage.astype('float32')
    # augImage /= 255
    # plt.imshow(augImage)
    i += 1
    if i > 20:
        break

# def saveBatch(type,seq):
#     dir_path = TEST_ROOT + "d{}/".format(type)+"d{}_{}".format(type,seq)
#     if not os.path.exists(dir_path):
#         try:
#             os.mkdir(dir_path)
#         except OSError:
#             print ("Creation of the directory %s failed" % dir_path)

#         for i in range(INPUT_FRAME_NUMBER):
#             filename = dir_path + "/d{}_{}_{}.jpg".format(type,seq,i)
#             cv2.imwrite(filename, batch[i])