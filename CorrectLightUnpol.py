import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

#sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT") # PC lab
sys.path.append("C:/Users/Fotobio/Documents/GitHub/pyCRT") #PC casa
from src.pyCRT import PCRT  

# Função para aplicar um filtro Butterworth no sinal
def FilterButterworth(data, cutoff, order=5):
    b, a = butter(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Caminho base para os arquivos do projeto
#base_path = "C:/Users/RaquelPantojo/Desktop/ElasticidadePele"
base_path = "C:/Users/Fotobio/Desktop/Estudo_ElasticidadePele"
folder_name = "DespolarizadoP3"
video_name = "v6.mp4"
#gamma = 0.829
gammaROI1 = 0.467
gammaROI2 = 0.616

#gammaROI1 = 1
#gammaROI2 = 1

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
            roi2 = cv.selectROI("Selecionar ROI2", frame)
            cv.destroyAllWindows()
            break

# Selecionar as ROIs
select_rois()

# Reinicia o vídeo
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Listas para armazenar intensidades e timestamps
green_roi2,RoiRed,RoiGreen,RoiBlue,time_stamps = [], [], [],[],[]

# Processa o vídeo frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)    
    # Extrai as ROIs e calcula a média do canal verde
    roi1_frame = frame[int(roi1[1]):int(roi1[1] + roi1[3]), int(roi1[0]):int(roi1[0] + roi1[2])]
    roi2_frame = frame[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])]
    
    # ROI 1
    RoiRed.append(np.mean(roi1_frame[:, :, 0]))
    RoiGreen.append(np.mean(roi1_frame[:, :, 1]))
    RoiBlue.append(np.mean(roi1_frame[:, :, 2]))
    
    # ROI 2
    green_roi2.append(np.mean(roi2_frame[:, :, 1]))
    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()

# Usei isso so para conseguir calcular o CRT depois 
RoiRed=np.array(RoiRed)
RoiGreen=np.array(RoiGreen)
RoiBlue= np.array(RoiBlue)


green_roi1 = np.array(RoiGreen) ** (1/gammaROI1)
green_roi2 = np.array(green_roi2) ** (1/gammaROI2)

time_stamps = np.array(time_stamps)

# Aplica filtro Butterworth na ROI2
filtered_roi2 = FilterButterworth(green_roi2, cutoff=0.15)
#mean_roi2 = np.mean(filtered_roi2[:30])
#filtered_roi2 /= mean_roi2
#Meangreen_roi2 = green_roi2/mean_roi2


# Calcula razão entre as intensidades normalizadas - desconsidere o ratiosr e ratiosb não vou usa-los no futuro 
ratiosr = green_roi1 / filtered_roi2
ratiosg = green_roi1 / filtered_roi2
ratiosb = green_roi1 / filtered_roi2

# Plotagem dos resultados
plt.figure(figsize=(10, 5))

# Intensidade ROI1
plt.subplot(3, 1, 1)
plt.plot(time_stamps, green_roi1, label='G - ROI1 ', color='g', linewidth=2)
plt.xlabel('Tempo (s)')
plt.ylabel('Intensidade Normalizada')
plt.legend()

# Intensidade ROI2
plt.subplot(3, 1, 2)
plt.plot(time_stamps, green_roi2, label='G - ROI2', color='g',linewidth=2)
plt.plot(time_stamps, filtered_roi2, label='Butterworth Filter order=5', color='r',linewidth=1)
plt.xlabel('Tempo (s)')
plt.ylabel('Intensidade Normalizada')
plt.legend()

# Razão entre intensidades
plt.subplot(3, 1, 3)
plt.plot(time_stamps, ratiosg, label='Razão ROI1/FilterROI2', color='b',linewidth=2)
plt.xlabel('Tempo (s)')
plt.ylabel('Razão')
plt.legend()

plt.tight_layout()
plt.show()



channelsAvgIntensArr= np.column_stack((RoiRed, RoiGreen, RoiBlue))
pcrt = PCRT(time_stamps,channelsAvgIntensArr,exclusionMethod='best fit',exclusionCriteria=999)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()


ratios=np.column_stack((ratiosr,ratiosg,ratiosb))
pcrt = PCRT(time_stamps, ratios,exclusionMethod='best fit',exclusionCriteria=999 )
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()



