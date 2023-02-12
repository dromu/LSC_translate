from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import mediapipe as mp
from funcion import get_points
from function2 import mano_pose
from keras.models import load_model
import numpy as np
import streamlit as st

# Numero de cuadros a usar
cuadro      = [2,4,5,6,7,8]
fotograma   = 0


# Variables de uso
anterior    = np.zeros(63)
dato        = []
name2       = []
names       = ['1', '10','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','L','M','MIL', 'MILLON','N','NN','O', 'P','Q','R','S','T','U','V','W',
		        'X','Y','Z']
flag        = False
vez1        = True
salto       = 0
pos         = []

model = load_model(r"C:\Users\Alejandro\Universidad\Tesis\Pruebas\mediapipeLSTM2_85.h5")
# model = load_model(r"C:\Users\Alejandro\Universidad\Tesis\Resultados\mp_point_hand_BI.h5")
class VideoProcessor:
	
	def recv(self, frame):
		global model,cuadro,fotograma,anterior,dato,name2,name,flag,vez1,salto,pos
		
		imagen = frame.to_ndarray(format="bgr24")

		if fotograma in cuadro:
			point = mano_pose(imagen, anterior)
			anterior = point
			dato.append(point)

			if len(dato) == 6: 
				fotograma = 0
                # TRansformamo y predecimos los datos
				dato = np.array(dato,dtype = np.float32).reshape((1,6,63))
				predictions = model.predict(dato)
				# Cambiamos los datos de codificacion a label
				y_pred = np.argmax(predictions, axis=1)
				sal = round(y_pred[0])
				porce = predictions[0][sal]
				# Vaciamos la varible que almacena los datos
				dato      = []

				if porce >= 0.7:
					name = str(names[sal])
					print(porce)
                   
					if vez1 == True:
						name2.append(name)
						flag = True
						vez1 = False

					elif name != name2[-1]:
						name2.append(name)
						flag = True

		if flag == True:
			salto = salto + 30
			pos.append(salto) 
			flag = False

			if salto >= 500:
				salto = 0
				name2 = []
				vez1 = True
				pos = []

		if len(name2)!=0:
			for i in range(0, len(pos)):
				cv2.putText(imagen,name2[i], (pos[i],100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, bottomLeftOrigin = False)      
		
		fotograma+=1

		return av.VideoFrame.from_ndarray(imagen, format='bgr24')

webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
				rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					)
	)