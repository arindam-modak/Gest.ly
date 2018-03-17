import cv2
import numpy as np
import os

minValue = 70

x0 = 400
y0 = 200
height = 600
width = 600
PATH = os.getcwd()
data_path = PATH+'/data'
data_dir_list = os.listdir(data_path)
    # global guessGesture, visualize, mod, lastgesture, saveImg
    # HSV values
kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

low_range = np.array([0, 50, 80])
upper_range = np.array([30, 200, 255])


# folder = 'test2'   
# os.mkdir(folder)
cap =cv2.VideoCapture(0)
count = 0



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
    mask1 = cv2.GaussianBlur(mask, (15,15), 1)
    cv2.imshow("Blur", mask1)

    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask1)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    cv2.imshow('res',res)


    #Used to create gesture images
    # cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), res)     # save frame as JPEG file
    # count += 1
    k = cv2.waitKey(5) & 0xFF
    if k==27:
        break

# print("{} images are extacted in {}.".format(count,folder))

img_data_list = []
for filename in data_dir_list:
    img_list = os.listdir(data_path+'/'+ filename)
    for im in img_list:
        inp = cv2.imread(data_path+'/' + filename + '/' + im)
        resized_image = cv2.resize(inp, (100,100))
        img_data_list.append(resized_image) 

cap.release()
cv2.releaseAllWindows()
