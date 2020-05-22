from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv3D, Flatten, MaxPooling3D, Dropout,Activation
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

import numpy as np
from utils import batch_generate as bg
from utils.check import plotResult


import sys
import os

# conv_num = 8
# pool_num = 5
# fc_num = 2



DATA_ROOT = "../Data/input/"
# DATA_ROOT = "../Data/test_input/"

MODEL_PATH = "models/C3M"
CP_PATH = "checkpoints/C3M/C3M_cp.ckpt"
CP_DIR = os.path.dirname(CP_PATH)

class C3D:
    def __init__(self,input = DATA_ROOT):
        self.input = input
        self.epoch_num = 100
        self.output_size = 4
        self.batch_size = 32
        self.conv_kernel_shape = (3,3,3)
        self.pool_kernel_shape1 = (1,2,2)
        self.input_data_shape =(5,224, 224, 3)
        self.targets = ["None","fist", "one finger", "stick"]
        self.pool_kernel_shape2 = (2,2,2)
        self.class_weight = {0: 1.,
        1: 10.,
        2: 10.,
        3: 10.
        }

    def getData(self):
        print("\n[C3D][getData] start...")
        batches = bg.generateBatches(directory = self.input)
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
        print("\n[C3D][getData] end...")

        return (trainX, testX, trainY, testY)
    def train(self):
        print("[C3D][train] start")

        # callbacks: early_stop, check_point
        es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=2)
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=CP_PATH,save_weights_only=True,verbose=1)

        model = keras.Sequential(
            [
                Conv3D(64,self.conv_kernel_shape, activation='relu', input_shape=self.input_data_shape),
                MaxPooling3D(pool_size=self.pool_kernel_shape1, strides=None, padding="valid", data_format=None),
                Conv3D(128,self.conv_kernel_shape, activation='relu'),
                MaxPooling3D(pool_size=self.pool_kernel_shape1, strides=None, padding="valid", data_format=None),
                Flatten(),
                Dense(64, activation="relu", name="fc_layer1"),
                Dense(64, activation="relu", name="fc_layer2"),
                Dense(self.output_size, activation="softmax", name="fc_layer3")
            ]
        )
        model.save(MODEL_PATH)
        model.compile('adam', loss='categorical_crossentropy',metrics=["accuracy"])
       
        print(model.summary())
        # get data 
        (trainX, testX, trainY, testY) = self.getData()
        # model.save_weights(checkpoint_path.format(epoch=0))
      

        print("\n[C3D][train] training network...")
        H = model.fit(trainX, trainY, validation_data=(testX, testY),
            batch_size=self.batch_size, epochs=self.epoch_num,class_weight=self.class_weight,
            verbose=1,callbacks=[es_callback,cp_callback])
        
        predictions = model.predict(testX, batch_size=self.batch_size)
        print("\n[C3D][train] evaluating network...")
        print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=self.targets))
        print("[C3D][train] end")

        (loss,val_loss,accuracy,val_accuracy) = (H.history["loss"],H.history["val_loss"],H.history["accuracy"],H.history["val_accuracy"])
       
        plotResult((loss,val_loss,accuracy,val_accuracy))

    
    def loadModel():
        print("[loadModel]")

        # new_model = tf.keras.models.load_model('saved_model/my_model')

        # cp = tf.train.latest_checkpoint(CP_DIR)
        # # Create a new model instance
        # model = create_model()

        # # Load the previously saved weights
        # model.load_weights(cp)

        # # Re-evaluate the model
        # loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
        # print("Restored model, accuracy: {:5.2f}%".format(100*acc))

if __name__ == "__main__":
    print("Run CSD")
    # model = C3D()
    
    # model.getData()
    # model.train()