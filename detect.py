from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt 
import os
import pickle
import pandas as pd
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))
path = path + '/'

# modelos utilizados para o reconhecimento e a detecção facial do deepface
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

# modelo do detector facial para a extração da face a ser classificada
detector = cv2.dnn.readNetFromCaffe(path + "deploy.prototxt.txt", path + "res10_300x300_ssd_iter_140000.caffemodel")
font = cv2.FONT_HERSHEY_SIMPLEX

if os.path.isfile(path + "dataset/representations_facenet.pkl"):
        os.remove(path + "dataset/representations_facenet.pkl")

cap = cv2.VideoCapture(0) 

# a imagem da webcam deve ser redimensionada para a detecção facial (300x300)
width  = int(cap.get(3))  
height = int(cap.get(4))

original_size = (width, height)
target_size = (300, 300)
aspect_ratio_x = (original_size[0] / target_size[0])
aspect_ratio_y = (original_size[1] / target_size[1])

while(True):
    i = 0
    ret, frame = cap.read()
    target_frame = cv2.resize(frame, target_size)

    imageBlob = cv2.dnn.blobFromImage(image = target_frame)
    detector.setInput(imageBlob)
    detections = detector.forward()
    detections_df = pd.DataFrame(detections[0][0]
    , columns = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
    
    detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
    detections_df = detections_df[detections_df['confidence'] >= 0.90]

    for i, instance in detections_df.iterrows():
        left = int((instance["left"] * 300) * aspect_ratio_x)
        bottom = int((instance["bottom"] * 300) * aspect_ratio_y)
        right = int((instance["right"] * 300) * aspect_ratio_x)
        top = int((instance["top"] * 300) * aspect_ratio_y)

        roi = frame[top:bottom, left:right]
        cv2.imwrite(path +  str(i) + ".jpg", roi) 
        result = DeepFace.find( 
            img_path = path + str(i) + ".jpg", 
            db_path = path + "dataset", 
            model_name = models[1], 
            detector_backend= backends[1], 
            enforce_detection=False) 
        os.remove(path + str(i) + ".jpg") 

        try:
            target = result[f"{models[1]}_cosine"].idxmin()
            name = result.loc[target]["identity"].split("/")[-2]
            precision = result.loc[target][models[1] + "_cosine"]
            cv2.rectangle(frame, (left,top), (right, bottom), (0, 255, 0), 1) #draw rectangle to main image

            cv2.putText(
                        frame, 
                        name + " " + ("{:.1f}%".format((1 - precision) * 100)), 
                        (left+5,top-5), 
                        font, 
                        1, 
                        (255,255,255), 
                        2
                    )
        except:
            continue

    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # esc
        break
