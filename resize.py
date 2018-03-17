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

cap =cv2.VideoCapture(0)

for filename in data_dir_list:
    img_list = os.listdir(data_path+'/'+ filename)
    for im in img_list:
        inp = cv2.imread(data_path+'/' + filename + '/' + im)
        resized_image = cv2.resize(inp, (100,100))
        cv2.imwrite(data_path + '/'+ filename + '/' + im,resized_image) 