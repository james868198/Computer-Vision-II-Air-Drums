from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, \
    Flatten, TimeDistributed, Conv2D, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from utils import batch_generate as bg
from utils.check import plotResult

DATA_ROOT = "../Data/input/"

MODEL_PATH = "models/LRCN"
CP_PATH = "checkpoints/LRCN/LRCN_cp.ckpt"
CP_DIR = os.path.dirname(CP_PATH)

class LRCN():
    def __init__(self,input = DATA_ROOT):
        self.input = input
        self.epoch_num = 100
        self.batch_size = 32
        self.optical_flow = False
        self.input_data_shape =(5,224, 224, 3)
        self.targets = ["None","fist", "one finger", "stick"]
        self.output_size = 4
        self.frame_number = 5
        self.binary_output = False
        self.class_weight = {0: 1.,
        1: 10.,
        2: 10.,
        3: 10.
        }

    def getData(self):
        print("\n[LRCN][getData] start...")
        batches = bg.generateBatches(directory = self.input, of = self.optical_flow, binary = self.binary_output, frame_number = self.frame_number)
        # batches = bg.generateBatch(filename = self.input, shape = (224, 224))
        data, labels = zip(*batches)
        # classTotals = labels.sum(axis=0)        
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)
        testY = np.array(testY)

        trainY = keras.utils.to_categorical(trainY)
        testY = keras.utils.to_categorical(testY)

        # classTotals = trainY.sum(axis=0)
        # self.classWeight = classTotals.max()
        print("\n[LRCN][getData] end...")

        return (trainX, testX, trainY, testY)
    def train(self):

        # build model
        vgg = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_data_shape[1:]
        )
        model = Sequential()
        model.add(
            TimeDistributed(vgg, input_shape= self.input_data_shape)
        )
        model.add(
            TimeDistributed(
                Flatten()
            )
        )
        model.add(LSTM(256, activation='relu', return_sequences=False))
        # finalize with standard Dense, Dropout...
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(.5))
        model.add(Dense(self.output_size, activation='softmax'))
        
        model.save(MODEL_PATH)
        model.compile('adam', loss='categorical_crossentropy',metrics=["accuracy"])
       
        print(model.summary())

        # callbacks: early_stop, check_point
        es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=2)
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=CP_PATH,save_weights_only=True,verbose=1)

        (trainX, testX, trainY, testY) = self.getData()
        # train the network
        print("[INFO] training network...")

        trainX = np.array(trainX)
        trainY = np.array(trainY)
        testX = np.array(testX)
        testY = np.array(testY)
       
        H = model.fit(trainX, trainY, validation_data=(testX, testY),
        batch_size=self.batch_size, epochs=self.epoch_num,
        class_weight=self.class_weight,callbacks=[es_callback,cp_callback], verbose=1)

        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = model.predict(testX, batch_size=self.batch_size)
        print(classification_report(testY.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=self.targets))
        
        (loss,val_loss,accuracy,val_accuracy) = (H.history["loss"],H.history["val_loss"],H.history["accuracy"],H.history["val_accuracy"])
        plotResult((loss,val_loss,accuracy,val_accuracy))

    

if __name__ == "__main__":
    print("Run LRCN")
    # model = LRCN()
    # model.train()