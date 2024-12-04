import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize, curve_fit
from sklearn.metrics import r2_score

sys.path.append("C:/Users/Fotobio/Documents/GitHub/pyCRT") #PC casa 
#sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT") # PC lab

from src.pyCRT import PCRT  


# Caminho base para os arquivos do projeto
#base_path = "C:/Users/RaquelPantojo/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"  # PC USP
base_path="C:/Users/Fotobio/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5" #PC casa

folder_name = "teste1"
video_name="corrected_v7_gamma=1.53.mp4"
#video_name = "corrected_v7_gamma=1.mp4"
roi_width = 80 
roi_height = 80
num_rois = 200  # Número de ROIs a serem criadas


# Funções de modelo para ajuste de curva
def logarithmic(x, a, b, c):
    return a + b * np.log(c * x + 1)

def power_law(x, a, b, p):
    return a + b * np.power(x, p)

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def quadratic(x, a, b, c):
    return a + b * x + c * x**2

# Função para calcular o erro
def calculate_error(adjusted_ratio):
    return np.mean(np.abs(adjusted_ratio - 1))

# Função para encontrar os melhores valores de a e b
def find_best_a_b(roi2_normalized, models):
    best_model_name = None
    best_params = None
    min_error = float('inf')

    for model_name, model_func in models.items():
        try:
            # Ajuste de curva
            x_data = np.arange(len(roi2_normalized))
            y_data = roi2_normalized
            params, _ = curve_fit(model_func, x_data, y_data, maxfev=5000)

            # Avaliação do erro
            adjusted_ratio = model_func(x_data, *params)
            error = calculate_error(adjusted_ratio)

            if error < min_error:
                min_error = error
                best_model_name = model_name
                best_params = params

            if min_error < 0.01:
                break
        except Exception as e:
            continue  # Ignora erros em modelos específicos

    return best_model_name, best_params, min_error

# Função para calcular as razões com validação
def calculate_ratios(all_green_rois, num_rois):
    roi_combinations = list(combinations(range(num_rois), 2))
    ratios = {}
    for i, j in roi_combinations:
        if len(all_green_rois[i]) == len(all_green_rois[j]) > 0:
            # Calculate the ratio of the pixel intensities for the two ROIs
            # Assuming you're interested in the mean intensity or another metric
            ratio = np.mean(all_green_rois[i]) / np.mean(all_green_rois[j])
            ratios[f"ROI{i+1}/ROI{j+1}"] = ratio
    return ratios


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

# Função principal para encontrar o melhor modelo
def find_best_model(image, video_frames, models):
    if image is None:
        raise ValueError("A imagem fornecida é None. Certifique-se de que o frame foi carregado corretamente.")
    
    rois = create_dynamic_rois(image, 200, 60, 60)  # Passo 1: Gerar ROIs principais
    best_combination = find_combinations_with_ratio_close_to_one(rois)  # Passo 2: Combinações
    
    if best_combination is None:
        raise ValueError("Nenhuma combinação de ROIs encontrada. Verifique os parâmetros de entrada.")

    i, j = best_combination
    roi2_normalized = normalize_roi(rois[i])  # Passo 3: Normalizar ROI2

    best_model, best_params, min_error = find_best_a_b(roi2_normalized, models)  # Passo 4: Iteração para ajuste

    if min_error > 0.01:  
        return find_best_model(image, video_frames, models)

    adjust_and_plot_roi1(rois[i], best_model, best_params, models)  # Passo 5: Ajustar e plotar

    return best_model, best_params, min_error, best_combination


# Função para encontrar combinações de ROIs com razão próxima de 1
def find_combinations_with_ratio_close_to_one(rois):
    best_combination = None
    min_diff = float('inf')

    for i, roi_a in enumerate(rois):
        for j, roi_b in enumerate(rois):
            if i != j:
                ratio = np.mean(roi_a) / np.mean(roi_b)
                diff = abs(ratio - 1)

                if diff < min_diff:
                    min_diff = diff
                    best_combination = (i, j)

    return best_combination

# Função para normalizar uma ROI
def normalize_roi(roi):
    mean_initial_frames = np.mean(roi[:40])  # Média dos primeiros 40 frames
    return roi / mean_initial_frames


def adjust_and_plot_roi1(roi1, best_model_name, best_params,models):
    model_func = models[best_model_name]
    x_data = np.arange(len(roi1))
    y_data = roi1

    # Ajustar na região do pico máximo
    max_point = np.argmax(y_data)
    fit_region_x = x_data[max_point - 10:max_point + 10]
    fit_region_y = y_data[max_point - 10:max_point + 10]

    fitted_curve = model_func(fit_region_x, *best_params)

    # Avaliar Qui-quadrado e R²
    chi_square = np.sum((fit_region_y - fitted_curve) ** 2 / fitted_curve)
    r_squared = 1 - np.sum((fit_region_y - fitted_curve) ** 2) / np.sum((fit_region_y - np.mean(fit_region_y)) ** 2)

    if r_squared > 0.9:  # Critério de aceitação
        plt.figure(figsize=(10, 6))
        plt.plot(fit_region_x, fit_region_y, 'o', label="ROI1 Data")
        plt.plot(fit_region_x, fitted_curve, label=f"Best Fit: {best_model_name}")
        plt.title("Adjusted Curve")
        plt.legend()
        plt.show()
    else:
        raise ValueError("Model criteria not met, reiterating process.")

# Função para plotar os gráficos de comparação
def plot_image_and_ratios(frames, best_combination, best_a, best_b, all_green_rois, time_stamps, fps, green_roi2):
    if best_combination is None:
        print("Nenhuma combinação de ROIs encontrada.")
        return
    
    i, j = best_combination
    ratio_key = f"ROI{i+1}/ROI{j+1}"
    original_ratio = ratios[ratio_key]
    adjusted_ratio = (best_a + best_b * (green_roi2))

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    
    axs[0, 0].imshow(cv.cvtColor(frames[0], cv.COLOR_BGR2RGB))  
    axs[0, 0].set_title('Imagem Original')
    axs[0, 0].axis('off')  
    
    rois = create_dynamic_rois(frames[0], num_rois, roi_width, roi_height)
    x_i, y_i, w_i, h_i = rois[i]
    x_j, y_j, w_j, h_j = rois[j]
    
    axs[0, 0].add_patch(plt.Rectangle((x_i, y_i), w_i, h_i, edgecolor='blue', facecolor='none', linewidth=2))
    axs[0, 0].add_patch(plt.Rectangle((x_j, y_j), w_j, h_j, edgecolor='red', facecolor='none', linewidth=2))
    axs[0, 0].text(x_i + w_i / 2, y_i + h_i / 2, f"{i+1}", color='blue', ha='center', va='center', fontsize=8, weight='bold')
    axs[0, 0].text(x_j + w_j / 2, y_j + h_j / 2, f"{j+1}", color='red', ha='center', va='center', fontsize=8, weight='bold')
    
    # Remove blank subplots from the first column
    axs[1, 0].axis('off')
    axs[2, 0].axis('off')

    axs[0, 1].plot(time_stamps, all_green_rois[i], label=f"Canal Verde ROI{i+1} - Original", color='blue')
    axs[0, 1].set_xlabel('Tempo (s)')
    axs[0, 1].set_ylabel('Intensidade do Canal Verde')
    axs[0, 1].legend()

    axs[1, 1].plot(time_stamps, all_green_rois[j], label=f"Canal Verde ROI{j+1} - Original", color='red')
    axs[1, 1].set_xlabel('Tempo (s)')
    axs[1, 1].set_ylabel('Intensidade do Canal Verde')
    axs[1, 1].legend()

    axs[2, 1].plot(time_stamps, original_ratio, label=f"Razão Original ROI{i+1} / ROI{j+1}", color='orange')
    axs[2, 1].set_xlabel('Tempo (s)')
    axs[2, 1].set_ylabel('Razão Original')
    axs[2, 1].legend()

    axs[0, 2].plot(time_stamps, adjusted_ratio, label=f"Canal Verde ROI{i+1} a={best_a:.2f} b={best_b:.2f}", color='blue')
    axs[0, 2].set_title(f"a={best_a} b={best_b}")
    axs[0, 2].set_xlabel('Tempo (s)')
    axs[0, 2].set_ylabel('Intensidade do Canal Verde')
    axs[0, 2].legend()

    plt.tight_layout()
    plt.show()

# Função para filtrar razões próximas de 1
def filter_ratios(ratios, threshold_median=0.005, threshold_std=0.01):
    return {
        key: value for key, value in ratios.items()
        if abs(np.median(value) - 1) < threshold_median and np.std(value) < threshold_std
    }





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

roi1 = None
frame_count = 0
fps = cap.get(cv.CAP_PROP_FPS)

print(f"Frames no vídeo: {frame_count} FPS: {fps}")

# Inicializa variáveis
all_green_rois = [[] for _ in range(num_rois)]
time_stamps = []
frames = []
frame_count = 0
fps = cap.get(cv.CAP_PROP_FPS)
video_frame =[]

cap.set(cv.CAP_PROP_POS_FRAMES, 0)
# Processa os frames do vídeo
# Processa os frames do vídeo
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Falha ao ler o frame {frame_count}.")
        break
    
    frame_resized = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frames.append(frame_resized)

    # Garante que pelo menos um frame foi lido
    if len(frames) == 1:  # Apenas no primeiro frame
        rois = create_dynamic_rois(frame_resized, num_rois, roi_width, roi_height)

    # Analisa o frame
    for i, roi in enumerate(rois):
        x, y, w, h = roi
        roi_frame = frame_resized[y:y+h, x:x+w]
        if roi_frame.size > 0:  # Certifica-se de que a ROI não está vazia
            all_green_rois[i].append(np.mean(roi_frame[:, :, 1]))

    time_stamps.append(len(frames) / fps)
    video_frame.append(frame)

cap.release()


models = {
        "Exponential": exponential,
        "Logarithmic": logarithmic,
        "Power Law": power_law,
        "Quadratica":quadratic
    }

all_green_rois = [np.array(roi) for roi in all_green_rois]
ratios = calculate_ratios(all_green_rois, num_rois)
ratios_filtered = filter_ratios(ratios)
best_model, best_params, min_error = find_best_model(frame, video_frame, models)

# Exibir resultados
print(f"Modelo {best_model} com parâmetros {best_params}, erro mínimo: {min_error:.5f}")
