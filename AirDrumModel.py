

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, \
    Flatten, TimeDistributed, Conv2D, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np
# import sys
import os
from utils import batch_generate as bg


class modelFramewrok():
    def __init__(self,input, frame_number, model_path, checkpoint_path):
        #read config

        #set parameters
        self.input = input
        self.epoch_num = 100
        self.batch_size = 32
        self.optical_flow = True
        self.frame_number = frame_number
        self.labelBalance = False
        self.binary_output = True
        self.model_path = model_path
        self.targets = ["not hit","hit"]
        self.shape = (224,224)
        self.input_data_shape =(5, 224,224, 3)
        self.cp_path = checkpoint_path
     
        # self.class_weight = {0: 1.,
        #     1: 3.
        # }
    def updateInputShape(shape = self.shape, frame_number = self.frame_number):
        self.frame_number = frame_number
        self.shape = shape
        self.input_data_shape =(frame_number, self.shape[0], self.shape[1], 3)
    
    def getData(self):
        print("[getData] start")
        batches = bg.generateBatches(directory = self.input, shape = self.shape, of = self.optical_flow, binary = self.binary_output, 
        frame_number = self.frame_number,labelBalance = self.labelBalance)
        # batches = bg.generateBatch(filename = self.input, shape = self.shape)
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
        print("[getData] end")

        return (trainX, testX, trainY, testY)

    def build(self):
        pass
    
    def run(self,train = False):
        print("[train] start")

        print("\nshow parameters\n")
        print("directory path:",  self.input)
        print("enable optical flow:",  self.optical_flow)
        print("binary label:",  self.binary_output)
        print("frame_number:",  self.frame_number)
        print("label balance:",  self.labelBalance)
        print("\n")

        #get model
        if self.model_path is None:
            print("[train] doesn't get model from path:", self.model_path)
            return 
        model = keras.models.load_model(self.model_path)
        
        # load checkpoint
        # if self.cp_path is not None:
        #     ck_dir = os.path.dirname(self.cp_path)
        #     if os.path.exists(ck_dir): 
        #         latest_ck = tf.train.latest_checkpoint(ck_dir)
        #         model.load_weights(latest_ck)
        
        #get data
        if self.input is None:
            print("[train] doesn't get data from directory:", self.input)
            return 
        (trainX, testX, trainY, testY) = self.getData()

        # if self.cp_path is not None:
        #     model.save_weights(self.cp_path.format(epoch=0))

        if train:
            # callbacks: early_stop, check_point
            es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.01, patience=5)
            cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.ck_path,save_weights_only=True,verbose=1)
            callback_list = [es_callback,cp_callback]

            # H = model.fit(trainX, trainY, validation_data=(testX, testY),
            # batch_size=self.batch_size, epochs=self.epoch_num,
            # class_weight=self.class_weight, callbacks=callback_list, verbose=1)

            H = model.fit(trainX, trainY, validation_data=(testX, testY),
            batch_size=self.batch_size, epochs=self.epoch_num, callbacks=callback_list, verbose=1)
            
            model.save(self.model_path)

        predictions = model.predict(testX, batch_size=self.batch_size)
        print("\n[train] evaluating network...")
        print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=self.targets))
        print("[train] end")
    
    def test(self,input_path,file_name):
        batches = generateBatch(input_path,file_name, shape = self.shape, self.optical_flow,self.binary_output,5,self.labelBalance)
        input_data, output_data = zip(*batches)
        i = np.array(input_data)
        o = np.array(output_data) 

        #get model
        if self.model_path is None:
            return 
        model = keras.models.load_model(self.model_path)
        pred = model.predict_classes(i)

        print(pred)
        print(len(pred))
        print("")
        print(o)
        print(len(o))

        total = len(pred)
        correct_count = 0
        for i in range(total):
            if pred[i] == 1:
                if o[i] == '1':
                    correct_count += 1
            else:
                if o[i] == '0':
                    correct_count += 1

        print("accuracy:", correct_count/total)

if __name__ == "__main__":
    print("Run AirDrumModel")


    pass