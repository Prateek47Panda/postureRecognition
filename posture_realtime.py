import cv2
import os
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_testing_model
import api
#import dlib

tic=0
detections=[]
dirforkneeling = "E:/NeosoftResearchProjects/PostureRecognition/kneelingpose"
dirforEvery4sec = "E:/NeosoftResearchProjects/PostureRecognition/frameEvery4sec"
# visualize with the colors
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def process (input_image, params, model_params):
    ''' Start with finding the Key points of full body using Open Pose.'''
    oriImg = input_image # B,G,R order  
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))  #part affinity field
    for m in range(1):
        scale = multiplier[m]
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
        output_blobs = model.predict(input_img)
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = [] #To store all the key points which are detected.
    peak_counter = 0
    
    #prinfTick(1) #prints time required till now.

    for part in range(18):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    #prinfTick(2) #prints time required till now.


    canvas = frame# B,G,R order

    for i in range(18): #drawing all the detected key points.
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    print()
    position = checkPosition(all_peaks)
    checkKneeling(all_peaks)
    #checkHandFold(all_peaks)
    print()
    print()
    return canvas

def checkPosition(all_peaks):
    try:
        f = 0
        if (all_peaks[16]):
            a = all_peaks[16][0][0:2] #Right Ear
            f = 1
        else:
            a = all_peaks[17][0][0:2] #Left Ear
        b = all_peaks[11][0][0:2] # Hip
        angle = calcAngle(a,b)
        degrees = round(math.degrees(angle))
        if (f):
            degrees = 180 - degrees
        if (degrees<70):
            return 1
        elif (degrees > 110):
            return -1
        else:
            return 0
    except Exception as e:
        print("person not in lateral view and unable to detect ears or hip")

def calcAngle(a, b):
    try:
        ax, ay = a
        bx, by = b
        if (ax == bx):
            return 1.570796
        return math.atan2(by-ay, bx-ax)
    except Exception as e:
        print("unable to calculate angle")


def checkHandFold(all_peaks):
    try:
        if (all_peaks[3][0][0:2]):
            try:
                if (all_peaks[4][0][0:2]):
                    distance  = calcDistance(all_peaks[3][0][0:2],all_peaks[4][0][0:2]) #distance between right arm-joint and right palm.
                    armdist = calcDistance(all_peaks[2][0][0:2], all_peaks[3][0][0:2]) #distance between left arm-joint and left palm.
                    if (distance < (armdist + 100) and distance > (armdist - 100) ): #this value 100 is arbitary. this shall be replaced with a calculation which can adjust to different sizes of people.
                        print("Not Folding Hands")
                    else: 
                        print("Folding Hands")
            except Exception as e:
                print("Folding Hands")
    except Exception as e:
        try:
            if(all_peaks[7][0][0:2]):
                distance  = calcDistance( all_peaks[6][0][0:2] ,all_peaks[7][0][0:2])
                armdist = calcDistance(all_peaks[6][0][0:2], all_peaks[5][0][0:2])
                if (distance < (armdist + 100) and distance > (armdist - 100)):
                    print("Not Folding Hands")
                else: 
                    print("Folding Hands")
        except Exception as e:
            print("Unable to detect arm joints")
        

def calcDistance(a,b): #calculate distance between two points.
    try:
        x1, y1 = a
        x2, y2 = b
        return math.hypot(x2 - x1, y2 - y1)
    except Exception as e:
        print("unable to calculate distance")

def checkKneeling(all_peaks):
    f = 0
    if (all_peaks[16]):
        f = 1
    try:
        if(all_peaks[10][0][0:2] and all_peaks[13][0][0:2]):
            rightankle = all_peaks[10][0][0:2]
            leftankle = all_peaks[13][0][0:2]
            hip = all_peaks[11][0][0:2]
            leftangle = calcAngle(hip,leftankle)
            leftdegrees = round(math.degrees(leftangle))
            rightangle = calcAngle(hip,rightankle)
            rightdegrees = round(math.degrees(rightangle))
        if (f == 0):
            leftdegrees = 180 - leftdegrees
            rightdegrees = 180 - rightdegrees
        if (leftdegrees > 60  and rightdegrees > 60): # 60 degrees is trial and error value here. We can tweak this accordingly and results will vary.
            print ("Both Legs are in Kneeling")
            #os.chdir(directory)
            cv2.imwrite(dirforkneeling, canvas) #saving the kneeling pose pictures from the frame to a folder

            #get embeddings of images of kneeling people
            imgknee = canvas
            def human_vector(imgknee):
                resize_img = cv2.resize(imgknee, (128, 256))
                resize_img = np.expand_dims(resize_img, axis=0)
                emb = sess.run(endpoints['emb'], feed_dict={imgknee: resize_img})
                return emb

            #return a cropped image of kneeling people
            def get_cropped(imgknee, y_min_a, x_min_a, y_max_a, x_max_a):
                height, width, channel = imgknee.shape
                left = int(x_min_a * width)
                right = int(x_max_a * width)
                top = int(y_min_a * height)
                if top < 0:
                    top = 0
                bottom = int(y_max_a * height)
                cropped = img[top:bottom, left:right]
                return cropped, imgknee

            return canvas


#tracking the object detected.. commented for now --
            #img = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
            #imgRGB = cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)
            #cv2.imshow('img',img)
            #cv2.imshow("imgrgb",imgRGB)
            #x, y, w, h = cv2.boundingRect(img)  #x,y coordinates; width and height of the object
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  #  the rectangle
            #bottomRightx= x+w
            #bottomRighty= y+h
            #bbox = dlib.rectangle(x,y,bottomRightx,bottomRighty)
            #tracker = dlib.correlation_tracker() #create the tracker
            #tracker.start_track(img,bbox)


           # _,contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
           #for cnt in contours:
           #    cv2.drawContours(img,[cnt],-1 , (0,255,0),2)
           #     x,y,w,h = cv2.boundingRect(cnt)     #x,y coordinates; width and height of the object
           #     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3) # drawing the rectangle
                
                #detections.append([x,y,w,h])
        elif (rightdegrees > 60):
            print ("Right leg is kneeling")
            #return canvas
        elif (leftdegrees > 60):
            print ("Left leg is kneeling")
            #return canvas
        else:
            print ("Not kneeling")

    except IndexError as e:
        try:
            if (f):
                a = all_peaks[10][0][0:2] # if only one leg (right leg) is detected
            else:
                a = all_peaks[13][0][0:2] # if only one leg (left leg) is detected
            b = all_peaks[11][0][0:2] #location of hip
            angle = calcAngle(b,a)
            degrees = round(math.degrees(angle))
            if (f == 0):
                degrees = 180 - degrees
            if (degrees > 60):
                print ("Both Legs Kneeling")
                return canvas
            else:
                print("Not Kneeling")
        except Exception as e:
            print("legs not detected")


#create the embeddings from images
def get_embed(img):
    return api.human_vector(img)[0]



def get_cropped(img, y_min_a, x_min_a, y_max_a, x_max_a):
    height, width, channel = img.shape
    left = int(x_min_a * width)
    right = int(x_max_a * width)
    top = int(y_min_a * height)
    if top < 0:
        top = 0
    bottom = int(y_max_a * height)
    cropped = img[top:bottom, left:right]
    return cropped, img

def display(img):
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.imshow("test", img)
    cv2.waitKey(0)



def prinfTick(i):
    toc = time.time()
    print ('processing time%d is %.5f' % (i,toc - tic))        

if __name__ == '__main__':

    tic = time.time()
    print('start processing...')
    model = get_testing_model()
    model.load_weights('./model/keras/model.h5')
    
    cap=cv2.VideoCapture(0)
    vi=cap.isOpened()

    if(vi == True):
        cap.set(100,160)
        cap.set(200,120)
        time.sleep(2)
    
        while(1):
            tic = time.time()
        
            ret,frame=cap.read()

            # save frame from video every 4 seconds
            while ret:

                cnt = 0

                cap.set(cv2.CAP_PROP_POS_MSEC, (cnt * 1000))
                frame_copy = frame
                cv2.imwrite(dirforEvery4sec + "\\frame%d.jpg" % cnt, frame_copy)

                #get embeddings of all images every 4seconds
                imgall = frame_copy
                def human_vector(imgall):
                    resize_img = cv2.resize(imgall, (128, 256))
                    resize_img = np.expand_dims(resize_img, axis=0)
                    emb = sess.run(endpoints['emb'], feed_dict={imgall: resize_img})
                    return emb
                cv2.imshow("frame_copy", frame_copy)
                cnt = cnt + 4


            # the euclidean distance between the two embeddings
            def compare(emb1, emb2):
                return np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))

            params, model_params = config_reader()
            canvas = process(frame, params, model_params)    
            cv2.imshow("capture",canvas) 
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        cap.release()
    else:
        print("unable to open camera")
cv2.destroyAllWindows()    