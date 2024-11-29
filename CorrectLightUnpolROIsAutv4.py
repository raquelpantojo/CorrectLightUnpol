import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import combinations  


roi_width = 60
roi_height = 60
num_rois = 200  # Número de ROIs a serem criadas
gamma=1.53


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
#base_path = "C:/Users/Fotobio/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"  # PC casa
base_path = "C:/Users/RaquelPantojo/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"  # PC USP
folder_name = "teste1"
video_name = "corrected_v7_gamma=1.53.mp4"

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
        roi_frame = roi_frame ** (gamma)
        if roi_frame.size > 0:  # Certifica-se de que a ROI não está vazia
            all_green_rois[i].append(np.mean(roi_frame[:, :, 1]))
    
    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()

time_stamps = np.array(time_stamps)
all_green_rois = [np.array(roi) for roi in all_green_rois]
ratios = calculate_ratios(all_green_rois, num_rois)


threshold_median = 0.005  
threshold_std = 0.01     

# Filtra razões mais próximas de 1 com base na mediana e na consistência (desvio padrão)
close_to_one_ratios = {
    key: value for key, value in ratios.items()
    if abs(np.median(value) - 1) < threshold_median and np.std(value) < threshold_std
}


# # Função para plotar as ROIs e as razões filtradas
def plot_filtered_ratios(frames, close_to_one_ratios, fps, title="Razões próximas de 1"):
    if not close_to_one_ratios:
        print("Nenhuma razão próxima de 1 foi encontrada.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plotar a imagem com as ROIs que atenderam ao critério
    if frames:  # Verifica se a lista frames não está vazia
        frame_rgb = cv.cvtColor(frames[0], cv.COLOR_BGR2RGB)  # Converte BGR para RGB
        axs[0].imshow(frame_rgb)
        axs[0].set_title('Imagem com ROIs filtradas')
        axs[0].axis('off')
        
        rois = create_dynamic_rois(frames[0], num_rois, roi_width, roi_height)
        
        included_rois = set(
            int(key.split('/')[0][3:]) - 1  # Extrai o índice da primeira ROI (numerador)
            for key in close_to_one_ratios.keys()
        ).union(
            int(key.split('/')[1][3:]) - 1  # Extrai o índice da segunda ROI (denominador)
            for key in close_to_one_ratios.keys()
        )
        
        for idx, roi in enumerate(rois):
            if idx in included_rois:
                x, y, w, h = roi
                # Desenhar retângulo da ROI
                rect = plt.Rectangle((x, y), w, h, edgecolor='b', facecolor='none', linewidth=2)
                axs[0].add_patch(rect)
                # Adicionar o número da ROI
                axs[0].text(x + w / 2, y + h / 2, f"{idx + 1}", color='blue',
                            ha='center', va='center', fontsize=8, weight='bold')
    
    # Plotar as razões filtradas
    time_stamps = np.arange(len(next(iter(close_to_one_ratios.values())))) / fps
    for label, ratio in close_to_one_ratios.items():
        axs[1].plot(time_stamps, ratio, label=label, linewidth=2)
    axs[1].set_xlabel('Tempo (s)')
    axs[1].set_ylabel('Razão')
    axs[1].legend()
    axs[1].set_title(title)
    
    plt.tight_layout()
    plt.show()

# Plota o gráfico filtrado
if close_to_one_ratios:
    plot_filtered_ratios(frames, close_to_one_ratios, fps)
else:
    print("Nenhuma razão próxima de 1 foi encontrada.")
