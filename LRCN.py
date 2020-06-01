from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, \
    Flatten, TimeDistributed, Conv2D, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16


from AirDrumModel import modelFramewrok

DATA_ROOT = "../Data/input/"
MODEL_PATH = "models/LRCN"
CP_PATH = "checkpoints/LRCN/LRCN_cp.ckpt"

class LRCN(modelFramewrok):
    def build(self):
        print("[build] start")
        # validation
        if len(self.targets) <= 1:
            print("[error]")
            return
        output_size = len(self.targets)


        #initalize VGG16
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
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(.5))
        model.add(Dense(output_size, activation='softmax'))
        opt = keras.optimizers.Adam(learning_rate=0.01, decay=0.01)
        model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=["accuracy"])
        
        print(model.summary())
        model.save(self.model_path)

   
if __name__ == "__main__":
    print("Run LRCN")
    # model = LRCN(input=DATA_ROOT,frame_number = 5,model_path = MODEL_PATH,checkpoint_path = CP_PATH)
    # model.build()
    pass

