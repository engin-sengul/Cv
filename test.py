import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier # ClassificationModule modülü, Classifier kütüphanesinin bir parçasıdır ve görüntü sınıflandırma için kullanılan araçları sağlar.
import numpy as np
import math
import time

cap =cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifer=Classifier("Model/keras_model.h5","Model/labels.txt")
offset=20
imgSize=300
folder = "Data/C"
counter =0
labels =["A","B","C","D","E"]
password = 'E'
lockCase='D'
lock=True    

while True:
    success,img = cap.read()
    imgOutput=img.copy()
    hands,img= detector.findHands(img)
    try :
        if hands:
            hand = hands[0]
            x,y,w,h=hand['bbox']
            imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
            imgCrop= img[y-offset:y+offset+h,x-offset:x+offset+w]
            imgCropShape = imgCrop.shape
            imgWhite[0:imgCropShape[0],0:imgCropShape[1]]=imgCrop        
        aspectRatio=h/w 

        if aspectRatio>1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)            
            imgWhite[:,wGap:wCal+wGap]=imgResize
            prediction,index =classifer.getPrediction(imgWhite,draw=False)
            print(prediction,index)            
        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)            
            imgWhite[hGap:hCal+hGap,:]=imgResize
            prediction,index =classifer.getPrediction(imgWhite,draw=False)

        if lock and labels[index]==password:            
            lock = not lock
        if not lock and labels[index]==lockCase:            
            lock = not lock
        if lock:
            cv2.putText(imgOutput,"Locked",(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),2)
            print('locked')
        else:
            cv2.putText(imgOutput,"Unlocked",(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,0),2)
            print('unlocked')
    except:
        print('no hand')
    


              
    cv2.imshow("Image",imgOutput)
    cv2.waitKey(1)
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    