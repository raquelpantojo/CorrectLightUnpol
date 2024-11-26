import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# função aplica um filtro butterworth no sinal da ROI2
def FilterButterworth(data, cutoff, order=5):
    b, a = butter(order, cutoff, btype='low', analog=False)
    signalGFiltered = filtfilt(b, a, data)
    return signalGFiltered




# Caminho base para os arquivos do projeto
base_path = "C:/Users/RaquelPantojo/Desktop/ElasticidadePele"

# Nome da pasta onde o vídeo está localizado
folder_name = "DespolarizadoP3"
folder_path = os.path.join(base_path, folder_name)
gamma = 0.58 

# Verifica se a pasta existe
if not os.path.exists(folder_path):
    print(f"Pasta {folder_name} não encontrada!")
    sys.exit(1)

# Nome do vídeo a ser processado
video_name = "v1.mp4"
video_path = os.path.join(folder_path, video_name)

# Inicializa a captura de vídeo
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    sys.exit(1)

# Variáveis para as ROIs
roi1 = None
roi2 = None

# Exibe o primeiro frame para o usuário selecionar as ROIs
ret, frame = cap.read()
if not ret:
    print("Erro ao ler o primeiro quadro.")
    sys.exit(1)

print("Selecione a ROI1 e pressione ENTER.")
roi1 = cv.selectROI("Seleção de ROIs", frame)
print("Selecione a ROI2 e pressione ENTER.")
roi2 = cv.selectROI("Seleção de ROIs", frame)
cv.destroyAllWindows()

# Reinicia o vídeo para reprocessar com as ROIs selecionadas
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Listas para armazenar as intensidades dos canais verdes das ROIs
green_channel_intensities_roi1 = []
green_channel_intensities_roi2 = []
timeStamps = []

# Lê o vídeo novamente quadro a quadro
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Extrai as regiões das ROIs no quadro
    roi1_frame = frame[int(roi1[1]):int(roi1[1] + roi1[3]), int(roi1[0]):int(roi1[0] + roi1[2])]
    #roi1_frame=frame(459, 137, 128, 98)
    roi2_frame = frame[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])]

    # Calcula a média da intensidade do canal verde (índice 1) para cada ROI
    green_intensity_roi1 = np.mean(roi1_frame[:, :, 1])
    green_intensity_roi2 = np.mean(roi2_frame[:, :, 1])
    
    
    
    green_channel_intensities_roi1.append(green_intensity_roi1)
    green_channel_intensities_roi2.append(green_intensity_roi2)
    sampling_rate = cap.get(cv.CAP_PROP_FPS)
    timeStamps.append(frame_count / sampling_rate)  # Tempo em segundos
    
    frame_count += 1
    
cap.release()



# Converte as listas em arrays NumPy
green_channel_intensities_roi1 = np.array(green_channel_intensities_roi1)
green_channel_intensities_roi2 = np.array(green_channel_intensities_roi2)

green_channel_intensities_roi1**1/gamma 
green_channel_intensities_roi2**1/gamma 

timeStamps = np.array(timeStamps)

#passar um filtro butterworth no sinal da ROI 2
 
filtered_green_channel_roi2 = FilterButterworth(green_channel_intensities_roi2, 0.15)


# calcula a média dos 30 frames iniciais do sinla da ROI 2
meanG_roi2 = np.mean(filtered_green_channel_roi2[:30])
print (meanG_roi2)
filtered_green_channel_roi2 /= meanG_roi2

green_channel_intensities_roi2/= meanG_roi2

# função ganho 
funcGainROI2=green_channel_intensities_roi2/meanG_roi2


normalized_roi1 = green_channel_intensities_roi1
normalized_roi2= green_channel_intensities_roi2

"""
# Normaliza as intensidades (entre 0 e 1)
normalized_roi1 = (green_channel_intensities_roi1 - np.min(green_channel_intensities_roi1)) / \
                  (np.max(green_channel_intensities_roi1) - np.min(green_channel_intensities_roi1))
normalized_roi2 = (green_channel_intensities_roi2 - np.min(green_channel_intensities_roi2)) / \
                  (np.max(green_channel_intensities_roi2) - np.min(green_channel_intensities_roi2))

"""


# Calcula a razão entre as intensidades normalizadas
ratios_over_time =( normalized_roi1 / normalized_roi2)

# Plota os gráficos de intensidades e da razão ao longo do tempo
plt.figure(figsize=(10, 5))

# Gráfico da intensidade do canal verde na ROI1
plt.subplot(3, 1, 1)
plt.plot(timeStamps, normalized_roi1, label='G - ROI1 (Normalizado)', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Intensity')
plt.legend()

# Gráfico da intensidade do canal verde na ROI2
plt.subplot(3, 1, 2)
plt.plot(timeStamps, normalized_roi2, label='G - ROI2 (Normalizado pela média )', color='g')
plt.plot(timeStamps, filtered_green_channel_roi2,label=' filtro', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Intensity')
plt.legend()

# Gráfico da razão entre as intensidades das ROIs
plt.subplot(3, 1, 3)
plt.plot(timeStamps, ratios_over_time, label='Channel G ROI1 Correct', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Ratio')
plt.legend()

plt.figure(figsize=(10, 5))
plt.plot(timeStamps, ratios_over_time/ratios_over_time, label='filter/filter', color='b')


# Exibe os gráficos
plt.tight_layout()
plt.show()


