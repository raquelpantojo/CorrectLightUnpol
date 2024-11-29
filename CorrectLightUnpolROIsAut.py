# o programa seleciona Diferentes ROIs em todo o video e mostra os gráficos da razão entre essas ROIs 


import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

#sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT") # PC lab
sys.path.append("C:/Users/Fotobio/Documents/GitHub/pyCRT") #PC casa
from src.pyCRT import PCRT  

# Função para criar ROIs automáticas com tamanhos fixos
def create_automatic_rois(frame, num_rois):
    height, width = frame.shape[:2]
    rois = []
    for i in range(num_rois):
        x = i * width // num_rois
        roi = (x, 0, width // num_rois, height)
        rois.append(roi)
    return rois

# Caminho base para os arquivos do projeto
#base_path = "C:/Users/RaquelPantojo/Desktop/ElasticidadePele"
#base_path = "C:/Users/Fotobio/Desktop/Estudo_ElasticidadePele"
base_path ="C:/Users/Fotobio/Documents/GitHub/CorrectLightUnpol"# PC casa
folder_name = "DespolarizadoP5"
video_name = "v8.mp4"

gammaROI1 = 0.467
gammaROI2 = 0.616

num_rois = 10


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





all_green_rois = [[] for _ in range(num_rois)]
time_stamps = []
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    sys.exit(1)

frame_count = 0
fps = cap.get(cv.CAP_PROP_FPS)
frames =[]

# Processa o vídeo frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frames.append(frame_resized)
    
    rois = create_automatic_rois(frame_resized, num_rois)
    
    for i, roi in enumerate(rois):
        roi_frame = frame_resized[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        all_green_rois[i].append(np.mean(roi_frame[:, :, 1]))
    
    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()


time_stamps = np.array(time_stamps)
all_green_rois = [np.array(roi) for roi in all_green_rois]


ratios = np.array([all_green_rois[0] / all_green_rois[i] for i in range(1, num_rois)])


# Função para plotar as ROIs e os gráficos de intensidade
def plot_rois_and_ratios(frames, ratios, fps, num_rois):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plotar a imagem com as ROIs
    if frames:  # Verifica se a lista frames não está vazia
        axs[0].imshow(frames[0])
        axs[0].set_title('Imagem com ROIs')
        
        rois = create_automatic_rois(frames[0], num_rois)
        for roi in rois:
            # Cria um objeto Rectangle do Matplotlib
            rect = plt.Rectangle((roi[0], roi[1]), roi[2], roi[3], edgecolor='g', facecolor='none')
            axs[0].add_patch(rect)
    
    # Plotar a razão das intensidades
    time_stamps = np.arange(len(ratios[0])) / fps
    for i in range(ratios.shape[0]):
        axs[1].plot(time_stamps, ratios[i], label=f'Razão ROI1/ROI{i+1}', linewidth=2)
    axs[1].set_xlabel('Tempo (s)')
    axs[1].set_ylabel('Razão')
    axs[1].legend()
    axs[1].set_title('Razão entre intensidades das ROIs')
    
    plt.tight_layout()
    plt.show()

# ... [restante do código] ...


  

if frames: 
    plot_rois_and_ratios(frames, ratios, fps, num_rois)



