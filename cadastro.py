import cv2
import os
import pandas as pd

path = os.path.dirname(os.path.realpath(__file__))
path = path + '/'

name = input("nome: ") 

direcoes = ["a esquerda", "a direita", "cima", "baixo"]

cap = cv2.VideoCapture(0) #webcam

detector = cv2.dnn.readNetFromCaffe(path + "deploy.prototxt.txt", path + "res10_300x300_ssd_iter_140000.caffemodel")

width  = int(cap.get(3))  # float `width`
height = int(cap.get(4))  # float `height`

original_size = (width, height)
target_size = (300, 300)
aspect_ratio_x = (original_size[0] / target_size[0])
aspect_ratio_y = (original_size[1] / target_size[1])

font = cv2.FONT_HERSHEY_SIMPLEX
position = (5, 20)
fontScale = 0.7
fontColor = (255,255,255)
thickness = 2
lineType = 2

i = 0
while(i < 5):
    ret, frame = cap.read()

    target_frame = cv2.resize(frame, target_size)

    imageBlob = cv2.dnn.blobFromImage(image = target_frame)
    detector.setInput(imageBlob)
    detections = detector.forward()
    detections_df = pd.DataFrame(detections[0][0]
    , columns = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
    detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
    detections_df = detections_df[detections_df['confidence'] >= 0.90]

    for j, instance in detections_df.iterrows():
        left = int((instance["left"] * 300) * aspect_ratio_x)
        bottom = int((instance["bottom"] * 300) * aspect_ratio_y)
        right = int((instance["right"] * 300) * aspect_ratio_x)
        top = int((instance["top"] * 300) * aspect_ratio_y)
        roi = frame[top:bottom, left:right]
        cv2.rectangle(frame, (left,top), (right, bottom), (0, 255, 0), 1)
    
    if(i == 0):
        cv2.putText(frame,"Posicione o rosto de frente para a camera", 
            position, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
    
    else:
        cv2.putText(frame,"Vire o rosto ligeiramente para " + direcoes[i-1], 
            position, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # esc
        break

    if not os.path.isdir(path + f"dataset/{name}"):
        os.mkdir(path + f"dataset/{name}")
    
    elif k == 32:  # espaço
        cv2.imwrite(path + f"dataset/{name}/{i}.jpg", roi)
        print(f"Foto numero {i+1}")
        i += 1

if os.path.isfile(path + "dataset/representations_facenet.pkl"):
    os.remove(path + "dataset/representations_facenet.pkl")
print("Cadastro realizado com sucesso.")
