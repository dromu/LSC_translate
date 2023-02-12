import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt


def get_points(image, label_per, label, name_image, previous):

    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands          = mp.solutions.hands 
    # Contenedor vacio
    points0 = []

    # Organizacion para la ultimas columnas de del CSV
    data_image = (label_per, label, name_image)

    # Puntos que deseamos obtener
    index = list(range(0,22))

    # Inicialamos mediapipe
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        
        # Giramos la imagen a espejo
        image = cv2.flip(image, 1)

        # Cambiamos el mapa de color para trabajar con medipipe 
        image_rgb =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # llamamos a mediapipe trabaje con nuestra imagen
        results = hands.process(image_rgb)      
        
        # Si detecta alguna mano en la imagen
        if results.multi_hand_landmarks is not None:    
            # Realiza todo la extraccion de los puntos 
            for hand_landmarks in results.multi_hand_landmarks:
                for (i, points) in enumerate(hand_landmarks.landmark):
                    # Si encontramos el indice extramos la informacion espacial
                    if i in index:
                        x0 = points.x
                        y0 = points.y
                        z0 = points.z
                       
                        # Guardamos los puntos X Y
                        points0.append((x0,y0,z0))
                        
           
            points0.append(data_image)
            # print(points0)
            points0 = list(np.concatenate(points0).flat)
            # points0.insert(36,results.multi_handedness[0].classification[0].label)
            return points0


        else:
            
            if previous[-2] == label:
                points0 = list(previous[:-3])
                points0.append(list(map(lambda dato: dato, data_image)) )
                # points0.insert(36,results.multi_handedness[0].classification[0].label)
                
            else:
                zero = np.zeros(63)
                points0.append(list(zero))
                points0.append(list(map(lambda dato: dato, data_image)) )
                points0 = sum(points0, [])
                # points0.insert(36,"results.multi_handedness[0].classification[0].label")

            return points0
        

def mano_image(image, anterior_image,RL=20):

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # modificacion de los modos de trabajo
    with mp_pose.Pose(
        static_image_mode=False) as pose:

        # lectura de imagen  
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # inicio de proceso pose
        results = pose.process(image_rgb)

        # Extraemos las coordenadas del punto 20 RIGHT_INDEX
        if results.pose_landmarks is not None:
            x20 = int(results.pose_landmarks.landmark[RL].x * width)
            y20 = int(results.pose_landmarks.landmark[RL].y * height)

            # Recortamos las imagenes de la mano
            tam = 60
            mano = image_rgb[ y20-tam : y20+tam,
                        x20-tam : x20+tam]

            # Redimensionamos para que todas las imagenes tenga el mismo tamaño
            resize_img = cv2.resize(mano, (120,120), interpolation = cv2.INTER_AREA)
            return resize_img
        
        else:
            return anterior_image