from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv3D, Flatten, MaxPooling3D, Dropout,Activation
from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16

from AirDrumModel import modelFramewrok

DATA_ROOT = "../Data/input/"
MODEL_PATH = "models/C3D"
CP_PATH = "checkpoints/C3D/C3D_cp.ckpt"

class C3D(modelFramewrok):
    def build(self):
        print("[build] start")

        # validation
        if len(self.targets) <= 1:
            print("[error]")
            return
        output_size = len(self.targets)
        conv_kernel_shape = (2,3,3)
        pool_kernel_shape = (1,2,2)
        
        model = keras.Sequential(
            [
                Conv3D(64,conv_kernel_shape, activation='relu', input_shape=self.input_data_shape),
                MaxPooling3D(pool_size=pool_kernel_shape, strides=None, padding="valid", data_format=None),
                Conv3D(128,conv_kernel_shape, activation='relu'),
                MaxPooling3D(pool_size=pool_kernel_shape, strides=None, padding="valid", data_format=None),
                Conv3D(256,conv_kernel_shape, activation='relu'),
                Conv3D(256,conv_kernel_shape, activation='relu'),
                MaxPooling3D(pool_size=pool_kernel_shape, strides=None, padding="valid", data_format=None),
                Flatten(),
                Dense(512, activation="relu", name="fc_layer1"),
                Dropout(.5),
                Dense(512, activation="relu", name="fc_layer2"),
                Dense(output_size, activation="softmax", name="fc_layer3")
            ]
        )
        model.compile('adam', loss='categorical_crossentropy',metrics=["accuracy"])
       
        print(model.summary())
        model.save(self.model_path)

        print("[build] end")

if __name__ == "__main__":
    print("Run C3D")
    # model = C3D(input=DATA_ROOT,frame_number = 5,model_path = MODEL_PATH,checkpoint_path = CP_PATH)
    # model.getData()
    
    # model.getData()
    pass