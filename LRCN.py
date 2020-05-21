from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, \
    Flatten, TimeDistributed, Conv2D, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
import tools/batch_generate as bg
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

batches = bg.generateBatches(directory = "../Data/input/")
data, labels = zip(*batches)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

trainY = to_categorical(trainY)
testY = to_categorical(testY)
# create a VGG16 "model", we will use
# image with shape (224, 224, 3)
vgg = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
# do not train first layers, I want to only train
# the 4 last layers (my own choice, up to you)
# for layer in vgg.layers[:-4]:
#     layer.trainable = False
# create a Sequential model
model = Sequential()
# add vgg model for 5 input images (keeping the right shape
model.add(
    TimeDistributed(vgg, input_shape=(5, 224, 224, 3))
)
# now, flatten on each output to send 5 
# outputs with one dimension to LSTM
model.add(
    TimeDistributed(
        Flatten()
    )
)
model.add(LSTM(256, activation='relu', return_sequences=False))
# finalize with standard Dense, Dropout...
model.add(Dense(64, activation='relu'))
# model.add(Dropout(.5))
# model.add(Dense(3, activation='softmax'))
model.add(Dense(2, activation='softmax'))
model.compile('adam', loss='categorical_crossentropy')

# train the network
print("[INFO] training network...")

trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)

H = model.fit(trainX, trainY, validation_data=(testX, testY),
batch_size=8, epochs=2, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=8)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=["not hit", "hit"]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()