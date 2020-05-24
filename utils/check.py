import matplotlib.pyplot as plt
import numpy as np

def isDrum(frame):
    
    def matchColor(pixel, color, thresh):
        pB, pG, pR = pixel[0], pixel[1], pixel[2]
        cB, cG, cR = color[0], color[1], color[2]
        if(abs(pB - cB)<thresh and abs(pG - cG)<thresh and abs(pR - cR)<thresh):
            return 1
        else:
            return 0

    # print(frame[0,0],frame[-1,0],frame[0,-1],frame[-1,-1])
    threash = 40
    
    if matchColor(frame[0][0], [0,0,0], threash) or matchColor(frame[0][-1], [0,0,255], threash) or matchColor(frame[-1][0], [255,0,0], threash) or matchColor(frame[-1][-1], [0,255,0], threash):
        return 1
    else:
        return 0

def plotResult(data):
    (loss,val_loss,accuracy,val_accuracy) = data
    training_number = len(loss)
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, training_number), loss, label="train_loss")
    plt.plot(np.arange(0, training_number), val_loss, label="val_loss")
    plt.plot(np.arange(0, training_number), accuracy, label="train_acc")
    plt.plot(np.arange(0, training_number), val_accuracy, label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    pass
    # print("run utils")