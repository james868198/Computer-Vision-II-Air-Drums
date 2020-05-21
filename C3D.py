from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv3D, Flatten, MaxPooling3D, Dropout,Activation
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import numpy as np
import batch_generate as bg
import sys
import os

# conv_num = 8
# pool_num = 5
# fc_num = 2
epoch_num = 1
output_size = 4
conv_kernel_shape = (3,3,3)
pool_kernel_shape1 = (1,2,2)
# pool_kernel_shape2 = (2,2,2)

batch_size = 4
input_data_shape =(5,224, 224, 3)
directory = "../Data/test_input/"


CLASS = ["None","fist", "one finger", "stick"]
# MODEL_DIRECTORY = "models/"
MODEL_PATH = "models/C3M"
CP_PATH = "checkpoints/C3M/C3M_cp.ckpt"
CP_DIR = os.path.dirname(CP_PATH)



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
        
        testY = keras.utils.to_categorical(testY)
        for data in testY:
            print(data)
        print("\n[C3D][getData] end...")
        return (trainX, testX, trainY, testY)
    def train(self):
        print("[C3D][train] start")

        # callbacks: early stop
        es_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=2)
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=CP_PATH,save_weights_only=True,verbose=1)

        model = keras.Sequential(
            [
                Conv3D(64,conv_kernel_shape, activation='relu', input_shape=input_data_shape),
                MaxPooling3D(pool_size=pool_kernel_shape1, strides=None, padding="valid", data_format=None),
                Conv3D(128,conv_kernel_shape, activation='relu'),
                MaxPooling3D(pool_size=pool_kernel_shape1, strides=None, padding="valid", data_format=None),
                Flatten(),
                Dense(64, activation="relu", name="fc_layer1"),
                Dense(64, activation="relu", name="fc_layer2"),
                Dense(output_size, activation="softmax", name="fc_layer3")
            ]
        )
        model.save(MODEL_PATH)

        model.compile('adam', loss='categorical_crossentropy')
        print(model.summary())
        # get data 
        (trainX, testX, trainY, testY) = self.getData()
        # model.save_weights(checkpoint_path.format(epoch=0))
      

        print("\n[C3D][train] training network...")
        H = model.fit(trainX, trainY, validation_data=(testX, testY),batch_size=batch_size, epochs=epoch_num, verbose=1,callbacks=[es_callback,cp_callback])
        predictions = model.predict(testX, batch_size=batch_size)
        print("\n[C3D][train] evaluating network...")
        print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=CLASS))
        print("[C3D][train] end")
    
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
    model = C3D()
    
    # model.getData()
    model.train()