import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import combinations  

roi_width = 60
roi_height = 60
num_rois = 200  # Número de ROIs a serem criadas
gamma = 1.53

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
        if roi_frame.size > 0: 
            all_green_rois[i].append(np.mean(roi_frame[:, :, 1]))
    
    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()

time_stamps = np.array(time_stamps)
all_green_rois = [np.array(roi) for roi in all_green_rois]
ratios = calculate_ratios(all_green_rois, num_rois)

# Função para ajustar os valores de a e b
def find_best_a_b(ratios, all_green_rois, num_rois):
    best_a, best_b = 0, 0
    min_diff = float('inf')
    best_ratios = {}

    for a in np.arange(0, 10.1, 0.1):
        for b in np.arange(0, 10.1, 0.1):
            adjusted_ratios = {}
            for i, j in combinations(range(num_rois), 2):
                if len(all_green_rois[i]) == len(all_green_rois[j]) > 0:
                    adjusted_ratio = (a + b * all_green_rois[i]) / all_green_rois[j]
                    adjusted_ratios[f"ROI{i+1}/ROI{j+1}"] = adjusted_ratio

            # Calcula a diferença média entre as razões ajustadas e 1
            avg_diff = np.mean([abs(np.median(ratio) - 1) for ratio in adjusted_ratios.values()])
            if avg_diff < min_diff:
                min_diff = avg_diff
                best_a, best_b = a, b
                best_ratios = adjusted_ratios

    return best_a, best_b, best_ratios

# Encontra os melhores valores de a e b
best_a, best_b, best_ratios = find_best_a_b(ratios, all_green_rois, num_rois)

# Filtra razões próximas de 1 com base nos melhores valores de a e b
threshold_median = 0.005  
threshold_std = 0.01      
close_to_one_ratios = {
    key: value for key, value in best_ratios.items()
    if abs(np.median(value) - 1) < threshold_median and np.std(value) < threshold_std
}

# Função para plotar os gráficos
def plot_filtered_ratios(frames, ratios, best_ratios, fps, best_a, best_b):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plotar o gráfico original
    axs[0].set_title("Gráfico Original")
    time_stamps = np.arange(len(next(iter(ratios.values())))) / fps
    for label, ratio in ratios.items():
        axs[0].plot(time_stamps, ratio, label=label, linewidth=2)
    axs[0].set_xlabel('Tempo (s)')
    axs[0].set_ylabel('Razão')
    axs[0].legend()

    # Plotar o gráfico ajustado
    axs[1].set_title(f"Gráfico Ajustado (a={best_a}, b={best_b})")
    for label, ratio in close_to_one_ratios.items():
        axs[1].plot(time_stamps, ratio, label=label, linewidth=2)
    axs[1].set_xlabel('Tempo (s)')
    axs[1].set_ylabel('Razão')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Plota os gráficos
plot_filtered_ratios(frames, ratios, best_ratios, fps, best_a, best_b)