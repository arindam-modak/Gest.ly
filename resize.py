import cv2
import numpy as np
import os

PATH = os.getcwd()
data_path = PATH+'/gestures'
data_dir_list = os.listdir(data_path)


for filename in data_dir_list:
    inp = cv2.imread(data_path+'/' + filename)
    resized_image = cv2.resize(inp, (100,100))
    cv2.imwrite(data_path + '/'+ filename,resized_image) 