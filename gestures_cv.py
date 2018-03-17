import cv2
import numpy as np
import os

minValue = 70

x0 = 400
y0 = 200
height = 600
width = 600

    # global guessGesture, visualize, mod, lastgesture, saveImg
    # HSV values
kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

low_range = np.array([0, 50, 80])
upper_range = np.array([30, 200, 255])

cap =cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    
    # mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    cv2.imshow("Blur", mask)
    
    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(res,100,200)
    cv2.imshow('edges', edges)
    cv2.imshow('res',res)


    k = cv2.waitKey(5) & 0xFF
    if k==27:
        break

cap.release()
cv2.releaseAllWindows()
