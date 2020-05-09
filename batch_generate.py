import numpy as np
import cv2
import time

def isDrum(frame):
    
    def matchColor(pixel, color, thresh):
        pB, pG, pR = pixel[0], pixel[1], pixel[2]
        cB, cG, cR = color[0], color[1], color[2]
        if(abs(pB - cB)<thresh and abs(pG - cG)<thresh and abs(pR - cR)<thresh):
            return True
        else:
            return False
            
    if matchColor(frame[0][0], [0,0,0], 20) and matchColor(frame[0][-1], [0,0,255], 20) and matchColor(frame[-1][0], [255,0,0], 20) and matchColor(frame[-1][-1], [0,255,0], 20):
        return True
    else:
        return False
        
def printText(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (50, 50) 
    fontScale = 1
    color = (0, 0, 255) 
    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method
    out = cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    return out
    
# frames = []
# for i in range(22):
#     ret, frame = cap.read()
#     frames.append(frame)
#     if(isDrum(frame)):
#         print(frames[i][0][0])

# print(frames[0].shape)

def batch_generate(filename):
    cap = cv2.VideoCapture(filename)
    batches = []
    queue = []
    count = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        if frame is not None:
            # print(str(count))
            if len(queue) == 5:
                queue.pop(0)
            
            check = isDrum(frame)
            
            queue.append(frame)
            batch = queue.copy()
            if len(batch) == 5:
                batches.append((batch, check))
            count += 1
        else:
            break
    print("Sampled " + str(count) + "frames.")
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    # print("count: ", len(batches))
    return batches

def playHit(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            if isDrum(frame):
                frame = printText(frame, "hit")
            cv2.imshow('Hit',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

def playVideo(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            cv2.imshow('Origin',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    
if __name__ == "__main__":
    
    dataroot = "../Data/"
    filename = "labeledvideo5.mp4"
    
    # playVideo(dataroot + filename)
    # playHit(dataroot + filename)
    
    batches = batch_generate(dataroot + filename)
    
    count = len(batches)
    print("Generated: " + str(count) + " batches.")
    i = 1
    for batch, hit in batches:
        # if hit is True:
            # print(count, hit, len(batch))
        for frame in batch:
            if isDrum(frame) is True:
                # printText(frame, "hit")
                img = printText(frame, str(i) + ". hit")
            else:
                img = printText(frame, str(i) + ".")
            cv2.imshow('Batch',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        i += 1
        # time.sleep(1)