import cv2
import matplotlib.pyplot as plt 
import os
import pandas as pd
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
position = (5, 20)
fontScale = 0.7
fontColor = (255,255,255)
thickness = 2
lineType = 2

cap = cv2.VideoCapture(0) #webcam

while(True):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    
    cv2.putText(src,"Sem equalizacao", 
            position, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
    
    cv2.putText(eq,"Com equalizacao", 
            position, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

    hh, ww = frame.shape[:2]
    max_size = max(hh, ww)

    # illumination normalize
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # separate channels
    y, cr, cb = cv2.split(ycrcb)

    # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size frame)
    # account for size of input vs 300
    sigma = int(5 * max_size / 300)
    gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)

    # subtract background from Y channel
    y = (y - gaussian + 100)

    # merge channels back
    ycrcb = cv2.merge([y, cr, cb])

    #convert to BGR
    output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    cv2.putText(frame,"Original", 
            position, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

    cv2.putText(src,"correcao de luz", 
            position, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)


    res = np.hstack((frame, output))
    cv2.imshow('equalized', res)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # esc
        break