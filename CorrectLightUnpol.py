import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT")
from src.pyCRT import PCRT  

# Função para aplicar um filtro Butterworth no sinal
def FilterButterworth(data, cutoff, order=5):
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Caminho base para os arquivos do projeto
#base_path = "C:/Users/RaquelPantojo/Desktop/ElasticidadePele" 
base_path = "C:/Users/RaquelPantojo/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5" #PC lab
folder_name = "teste1"
video_name = "corrected_v7_gamma=1.mp4"

gammaROI1 = 1
gammaROI2 = 1

# Verifica o caminho do vídeo
video_path = os.path.join(base_path, folder_name, video_name)
if not os.path.exists(video_path):
    print(f"Vídeo {video_name} não encontrado!")
    sys.exit(1)

# Inicializa a captura de vídeo
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    sys.exit(1)

# Variáveis para as ROIs
roi1, roi2 = None, None
frame_count = 0
fps = cap.get(cv.CAP_PROP_FPS)

# Função para selecionar ROIs com o vídeo rodando
def select_rois():
    global roi1, roi2
    print("Pressione ENTER para pausar e selecionar ROI1.")
    while True:
        ret, frame = cap.read()
        frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if not ret:
            print("Fim do vídeo ou erro na leitura do frame.")
            sys.exit(1)

        cv.imshow("Selecionar ROIs", frame)
        key = cv.waitKey(30) & 0xFF
        if key == 13:  # Tecla Enter
            roi1 = cv.selectROI("Selecionar ROI1", frame)
            print("Pressione ENTER novamente para selecionar ROI2.")
            print(roi1)
            roi2 = cv.selectROI("Selecionar ROI2", frame)
            print(roi2)
            cv.destroyAllWindows()
            break

# Selecionar as ROIs
#select_rois()
roi1=(551, 109, 87, 71)
roi2=(868, 363, 61, 50)

# Reinicia o vídeo
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Listas para armazenar intensidades e timestamps
green_roi2, green_roir, green_roig ,green_roib, time_stamps = [], [], [],[],[]

# Processa o vídeo frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)    
    # Extrai as ROIs e calcula a média do canal verde
    roi1_frame = frame[int(roi1[1]):int(roi1[1] + roi1[3]), int(roi1[0]):int(roi1[0] + roi1[2])]
    roi2_frame = frame[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])]
    
    green_roir.append(np.mean(roi1_frame[:, :, 0]))
    green_roig.append(np.mean(roi1_frame[:, :, 1]))
    green_roib.append(np.mean(roi1_frame[:, :, 2]))
    
    green_roi2.append(np.mean(roi2_frame[:, :, 1]))
    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()

# Normalização e processamento das intensidades
green_roig = np.array(green_roig) ** (gammaROI1)
#mean_roig = np.mean(green_roig[:30])
#green_roig /= mean_roig

green_roi2 = np.array(green_roi2) ** (gammaROI2)

time_stamps = np.array(time_stamps)

#mean_roi2= np.mean(green_roi2[:30])
#green_roi2/=mean_roi2

# Aplica filtro Butterworth na ROI2
filtered_roi2 = FilterButterworth(green_roi2, cutoff=0.15)
noise=green_roi2-filtered_roi2


ratios= (green_roig/filtered_roi2)
meanRatios = np.mean(ratios[:30])
ratios=(ratios/meanRatios)*255




# Plotagem dos resultados
plt.figure(figsize=(10, 5))

# Intensidade ROI1
plt.subplot(4, 1, 1)
plt.plot(time_stamps, green_roig, label='G - ROI1 ', color='g', linewidth=2)
plt.xlabel('Tempo (s)')
plt.ylabel('Intensidade Normalizada')
plt.legend()

# Intensidade ROI2
plt.subplot(4, 1, 2)
plt.plot(time_stamps, green_roi2, label='G - ROI2', color='g',linewidth=2)
plt.plot(time_stamps, filtered_roi2, label='Butterworth Filter order=5', color='r',linewidth=1)
plt.xlabel('Tempo (s)')
plt.ylabel('Intensidade Normalizada')
plt.legend()

# Razão entre intensidades
plt.subplot(4, 1, 3)
plt.plot(time_stamps, ratios, label='Razão ROI1/FilterROI2', color='b',linewidth=2)
plt.xlabel('Tempo (s)')
plt.ylabel('Razão')
plt.legend()


plt.tight_layout()
plt.show()



"""
pcrt = PCRT(time_stamps, green_roi1)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()


pcrt = PCRT(time_stamps, ratios)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()

Keyword arguments:
argument -- description
Return: return_description
"""

