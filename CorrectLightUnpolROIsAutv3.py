import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import combinations  # Para gerar combinações de ROIs

# Configurações das ROIs
roi_width = 60
roi_height = 50
num_rois = 120  # Número de ROIs a serem criadas

# Função para criar ROIs fixas dinamicamente
def create_dynamic_rois(frame, num_rois, roi_width, roi_height):
    height, width = frame.shape[:2]
    rois = []
    for i in range(num_rois):
        x = (i * roi_width) % (width - roi_width)
        y = ((i * roi_width) // (width - roi_width)) * roi_height
        if y + roi_height <= height:
            rois.append((x, y, roi_width, roi_height))
    return rois

# Função para calcular as razões com validação
def calculate_ratios(all_green_rois, num_rois):
    roi_combinations = list(combinations(range(num_rois), 2))
    ratios = {}
    for i, j in roi_combinations:
        if len(all_green_rois[i]) == len(all_green_rois[j]) > 0:
            ratios[f"ROI{i+1}/ROI{j+1}"] = all_green_rois[i] / all_green_rois[j]
    return ratios

# Caminho base para os arquivos do projeto
base_path = "C:/Users/Fotobio/Documents/GitHub/CorrectLightUnpol"  # PC casa
folder_name = "DespolarizadoP5"
video_name = "v8.mp4"

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

# Inicialização das variáveis
all_green_rois = [[] for _ in range(num_rois)]
time_stamps = []
frames = []
frame_count = 0
fps = cap.get(cv.CAP_PROP_FPS)

# Processa o vídeo frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frames.append(frame_resized)
    
    rois = create_dynamic_rois(frame_resized, num_rois, roi_width, roi_height)
    for i, roi in enumerate(rois):
        x, y, w, h = roi
        roi_frame = frame_resized[y:y+h, x:x+w]
        if roi_frame.size > 0:  # Certifica-se de que a ROI não está vazia
            all_green_rois[i].append(np.mean(roi_frame[:, :, 1]))
    
    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()

# Converte os dados de ROIs para numpy arrays
time_stamps = np.array(time_stamps)
all_green_rois = [np.array(roi) for roi in all_green_rois]

# Calcula as razões com validação
ratios = calculate_ratios(all_green_rois, num_rois)

# Define um limite para considerar a média da razão próxima de 1
threshold = 0.02
close_to_one_ratios = {key: value for key, value in ratios.items() if abs(np.median(value) - 1) < threshold}

# Função para plotar as ROIs e os gráficos de intensidade
def plot_rois_and_ratios(frames, ratios, fps, title="Razão entre intensidades das ROIs"):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plotar a imagem com as ROIs
    if frames:  # Verifica se a lista frames não está vazia
        axs[0].imshow(frames[0])
        axs[0].set_title('Imagem com ROIs')
        
        rois = create_dynamic_rois(frames[0], num_rois, roi_width, roi_height)
        for roi in rois:
            rect = plt.Rectangle((roi[0], roi[1]), roi[2], roi[3], edgecolor='g', facecolor='none')
            axs[0].add_patch(rect)
    
    # Plotar as razões entre as ROIs
    if ratios:
        time_stamps = np.arange(len(next(iter(ratios.values())))) / fps
        for label, ratio in ratios.items():
            axs[1].plot(time_stamps, ratio, label=label, linewidth=2)
        axs[1].set_xlabel('Tempo (s)')
        axs[1].set_ylabel('Razão')
        axs[1].legend()
        axs[1].set_title(title)
    else:
        axs[1].text(0.5, 0.5, 'Nenhuma razão válida encontrada.', fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.show()

# Plota os gráficos principais
if frames: 
    plot_rois_and_ratios(frames, ratios, fps)

# Plota os gráficos adicionais para razões próximas de 1
if close_to_one_ratios:
    plot_rois_and_ratios(frames, close_to_one_ratios, fps, title="Razões próximas de 1")
else:
    print("Nenhuma razão próxima de 1 foi encontrada.")
