import cv2
import cv2
import mediapipe as mp
from funcion import get_points, mano_image
from keras.models import load_model
import numpy as np
import streamlit as st

captura = cv2.VideoCapture(1)

x = []

anterior = np.zeros(63)
dato = [] # almacena todos los valores de puntos

names = ['1', '10','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','L','M','MIL', 'MILLON','N','NN','O', 'P','Q','R','S','T','U','V','W',
		'X','Y','Z']
sal = 0
name = "NaN"
porce = 0
model = load_model(r"C:\Users\Alejandro\Universidad\Tesis\Pruebas\mediapipeLSTM2_85.h5")
ubicacion = (100,100)
font = cv2.FONT_HERSHEY_SIMPLEX
tamañoLetra = 1
colorLetra = (0,0,255)
grosorLetra = 3
name2 = []
limpiar = 0
salto = 0
pos = []
vez1 = True
AntImage = []
flag = False

fps = 0

cuadro = [2,4,5,6,7,8]

inicio = cv2.imread("inicio.jpg")

while (captura.isOpened()):
  ret, imagen = captura.read()
  if ret == True:
    # Obtenemos los puntos a clasificar
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if fps in cuadro:
       
        # modificacion de los modos de trabajo
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

                # Recortamos las imagenes de la mano
                tam = 60
                mano = image_rgb[ y20-tam : y20+tam,
                            x20-tam : x20+tam]

                # Redimensionamos para que todas las imagenes tenga el mismo tamaño
                resize_img = cv2.resize(mano, (120,120), interpolation = cv2.INTER_AREA)

                resize_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2BGR)
                info = get_points(resize_img,"","","",anterior)

            else: 

                info = get_points(inicio,"","","",anterior)

    
        # Agregamos los 6 puntos
        dato.append(info[:63])

    
    # Cuando los tengamos realizamos la interpretacion
    if len(dato) == 6:
        fps = 0   
        dato1 = np.array(dato,dtype = np.float32).reshape((1,6,63))
        predictions = model.predict(dato1)
        y_pred = np.argmax(predictions, axis=1)
        sal = round(y_pred[0])
        porce = predictions[0][sal]

        print(porce)
        if porce >= 0.7:
            
            name = str(names[sal])
            if vez1 == True:
               name2.append(name)
               flag = True
               vez1 = False
               
            elif name != name2[-1]:
               name2.append(name)
               flag = True

        dato = []


 
    if flag == True:
       salto = salto + 30
       pos.append(salto) 
    #    limpiar = 0
       flag = False
       
       if salto >= 500:
          salto = 0
          name2 = []
          vez1 = True
          pos = []
       
    if len(name2)!=0:
        for i in range(0, len(pos)):
            cv2.putText(imagen,name2[i], (pos[i],100), font, tamañoLetra, colorLetra, grosorLetra, bottomLeftOrigin = False)
    # cv2.putText(imagen,str(porce), (100,150), font, tamañoLetra, colorLetra, grosorLetra, bottomLeftOrigin = False)
    
    

    if sum(anterior) == 0:
        anterior = np.zeros(63)
    else:
        anterior = info.copy()

    fps = fps + 1

    # cv2.imshow("img", resize_img)
    cv2.imshow('video', imagen)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
  else: break
captura.release()
cv2.destroyAllWindows()