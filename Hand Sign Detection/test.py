import cv2
import HandTrackingModule 
import numpy as np
import math
import time
import  tensorflow as tf

from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)

detector = HandTrackingModule.handDetector(maxHands=1)
classifier = Classifier('model/keras_model.h5', 'model/labels.txt')

offset = 20
img_size = 300
labels = ['A','B','C','D','E']

while True:
    success, img = cap.read()
    img_output = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['b_box']

        img_white = np.ones((img_size, img_size, 3), np.uint8)*255
        img_crop = img[y - offset:y+h+ offset, x-offset:x + w + offset]


        aspect_ratio = h/w
        if aspect_ratio > 1:
            k= img_size/h
            w_cal = math.ceil(k*w)
            img_resize = cv2.resize(img_crop, (w_cal, img_size))
            img_resize_shape = img_resize.shape
            w_gap = math.ceil((img_size-w_cal)/2)
            img_white[:, w_gap:w_cal+w_gap] = img_resize
            prediction, index =classifier.getPrediction(img_white)
            # print(prediction, index)

        else:
            k= img_size/w
            h_cal = math.ceil(k*h)
            img_resize = cv2.resize(img_crop, (img_size, h_cal))
            img_resize_shape = img_resize.shape
            h_gap = math.ceil((img_size-h_cal)/2)
            img_white[h_gap:h_cal+h_gap,:] = img_resize
            prediction, index = classifier.getPrediction(img_white)
            # print(prediction, index)

        cv2.rectangle(img_output, (x-offset, y- offset-50), (x-offset+120, y- offset-50+50), (255,0,255),cv2.FILLED)
        cv2.putText(img_output, labels[index], (x,y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 3)
        cv2.rectangle(img_output, (x-offset,y-offset), (x+w+offset,y+h+offset), (255,0,255),2)

        cv2.imshow('img_show',img_crop)
        cv2.imshow('img_white',img_white)

    cv2.imshow('img', img_output)
    key = cv2.waitKey(1)
    if key==27:
        break


cv2.destroyAllWindows()
