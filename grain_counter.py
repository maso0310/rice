from cv2 import cv2 as cv 
import numpy as np 
from scipy import stats
import csv 
import time 
import os 

image_list = os.listdir('./image')

for image_name in image_list:
    print(image_name)
    img = cv.imread('./image/'+image_name)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,127,255,cv.THRESH_BINARY)

    copy = img.copy()
    contours,hierarchy=cv.findContours(binary,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    copy=cv.drawContours(copy,contours,-1,(0,0,255),3)

    kernel = np.ones((10,10),np.uint8)
    erosion = cv.erode(binary,kernel,iterations = 1)#腐蝕

    cv.imwrite('./1_binary.png',binary)
    cv.imwrite('./2_erosion.png',erosion)

    HSV = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    H,S,V = cv.split(HSV)

    H_mode=[]
    for i in range(1,H.shape[0]):
        for j in range(1,H.shape[1]):
            if H[i,j]==0:
                pass
            elif H[i,j]>0:
                H_mode.append(H[i,j])
    H_mode = stats.mode(H_mode)[0][0]
    print("H眾數：",H_mode)
    copy2 = img.copy()
    copy3 = copy2[:,:,1]+copy2[:,:,2]
    copy2[H[:,:]>H_mode+4]=0
    cv.imwrite('./3_removal_branch.png',copy2)
    cv.imwrite('./4_original_gray.png',gray)
    gray = cv.cvtColor(copy2,cv.COLOR_BGR2GRAY)
    cv.imwrite('./5_removal_branch_gray.png',gray)
    rg=210
    gray[gray[:,:]<rg]=0
    cv.imwrite('./6_removal_branch_gray_removal_below_'+str(rg)+'.png',gray)



    break