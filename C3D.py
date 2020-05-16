from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv3D, Flatten, MaxPooling3D, Dropout,Activation
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np
import batch_generate as bg
import sys
# conv_num = 8
# pool_num = 5
# fc_num = 2
epoch_num = 5
conv_kernel_shape = (3,3,3)
pool_kernel_shape1 = (1,2,2)
# pool_kernel_shape2 = (2,2,2)

batch_size = 4
input_data_shape =(5,224, 224, 3)
directory = "../Data/input/"
file_path = "../Data/labeledvideo5.mp4"


class C3D:
    def __init__(self,input = directory):
        self.input = input
    def getData(self):
        print("\n[C3D][getData] start...")
        batches = bg.generateBatches(directory = self.input)
        # batches = bg.generateBatch(filename = self.input, shape = (224, 224))
        data, labels = zip(*batches)
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
        trainX = np.array(trainX)
        trainY = np.array(trainY)

        # check hit and not hit data distribution
        # count = [0,0]
        # for hit in trainY:
        #     if hit:
        #         count[0] += 1
        #     else:
        #         count[1] += 1
        # print("[training]hit,not hit: ", count)
        # count = [0,0]
        # for hit in testY:
        #     if hit:
        #         count[0] += 1
        #     else:
        #         count[1] += 1
        # print("[test]hit,not hit: ", count)
        testX = np.array(testX)
        testY = np.array(testY)
        trainY = keras.utils.to_categorical(trainY)
        # sys.exit()
        testY = keras.utils.to_categorical(testY)
        print("\n[C3D][getData] end...")
        return (trainX, testX, trainY, testY)
    def train(self):
        print("[C3D][train] start")

        model = keras.Sequential(
            [
                Conv3D(64,conv_kernel_shape, activation='relu', input_shape=input_data_shape),
                MaxPooling3D(pool_size=pool_kernel_shape1, strides=None, padding="valid", data_format=None),
                Conv3D(128,conv_kernel_shape, activation='relu'),
                MaxPooling3D(pool_size=pool_kernel_shape1, strides=None, padding="valid", data_format=None),
                Flatten(),
                Dense(64, activation="relu", name="fc_layer1"),
                Dense(64, activation="relu", name="fc_layer2"),
                Dense(2, activation="softmax", name="fc_layer3")
            ]
        )
        model.compile('adam', loss='categorical_crossentropy')
        # get data 
        (trainX, testX, trainY, testY) = self.getData()

        print("\n[C3D][train] training network...")
        H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=batch_size, epochs=epoch_num, verbose=1)
        predictions = model.predict(testX, batch_size=batch_size)
        print("\n[C3D][train] evaluating network...")
        print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=["hit", "others"]))
        print("[C3D][train] end")

if __name__ == "__main__":
    model = C3D()
    model.train()