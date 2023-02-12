import cv2
import mediapipe as mp
from funcion import get_points
import numpy as np

#Variables de mediapipe


def mano_pose(imagen, anterior):
    mp_pose     = mp.solutions.pose

    with mp_pose.Pose(
                static_image_mode=False) as pose:

                # lectura de imagen  
                height, width, _ = imagen.shape
                image_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

                # inicio de proceso pose
                results = pose.process(image_rgb)

                # Extraemos las coordenadas del punto 20 RIGHT_INDEX
                if results.pose_landmarks is not None:
                    x20 = int(results.pose_landmarks.landmark[20].x * width)
                    y20 = int(results.pose_landmarks.landmark[20].y * height)

                    if (y20>0 and  y20<480  ) and (x20>0 and x20 < 640  ):
                    # Recortamos las imagenes de la mano
                        tam = 60
                        mano = image_rgb[ y20-tam : y20+tam,
                                    x20-tam : x20+tam]


                        x_sh = mano.shape[0]
                        y_sh = mano.shape[1]
                        # Redimensionamos para que todas las imagenes tenga el mismo tamaño
                        
                        if x_sh ==120 and y_sh == 120:
                            resize_img = cv2.resize(mano, (120,120), interpolation = cv2.INTER_AREA)
                            resize_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2BGR)
                            info = get_points(resize_img,"","","",anterior)

                            print(x20,y20)
                            cv2.circle(imagen,(x20,y20),10,(255,0,255),3)

                            return info[:63]
                        else:
                            info = anterior
                            return info[:63]

                    else:
                        info = anterior
                        return info[:63]

    
                else:
                    info = anterior
                    return info[:63]