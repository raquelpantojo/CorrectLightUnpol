import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter


# Caminho base para os arquivos do projeto
base_path = "C:/Users/Fotobio/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"  # PC casa
folder_name = "teste1"
video_name = "corrected_v7_gamma=1.mp4."
roi_width = 60
roi_height = 60
num_rois = 200  # Número de ROIs a serem criadas

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

def normalize(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)

def plot_and_save_model(model_name, params, RoiGreen1, fitted_curve1, fitted_curve1_exp, RoiGreen2, r_squared, output_filename):
  
    # Criando o gráfico
    plt.figure()
    
    # Subplot 1: ROI 2
    plt.subplot(211)
    plt.plot(RoiGreen2, label="ROI 2", color='green')
    plt.title(f"Model: {model_name} r^2:{r_squared}")
    plt.xlabel("Time",fontsize=12)
    plt.ylabel("Intensity of green channel",fontsize=12)
    plt.legend(fontsize=10)

    # Subplot 2: ROI 1 vs Fit Exponencial
    plt.subplot(212)
    plt.plot(fitted_curve1, label="ROI 1 - corrigida", color='green')
    plt.plot(fitted_curve1_exp, label="Exponential Fit", color='red')
    plt.xlabel("Time",fontsize=12)
    plt.ylabel("Intensity of green channel",fontsize=12)
    plt.legend(fontsize=10)

    """   # Subplot 3: Fitted Curve and Equation
    plt.subplot(313)
    plt.plot(RoiGreen1, color='blue')
    plt.xlabel("Time",fontsize=12)
    plt.ylabel("Intensity of green channel",fontsize=12)
    plt.legend(fontsize=10)

    """
 

    # Salvando o gráfico em um arquivo
    plt.savefig(output_filename, dpi=600)
    plt.close()


# Função para garantir que nenhum parâmetro seja igual a zero
def avoid_zero(param):
    while param == 0:
        param = np.random.uniform(0.1, 5.0)  # Defina o intervalo de valores conforme necessário
    return param


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
        
        if roi_frame.size > 0:  # Certifique-se de que a ROI não está vazia
            green_mean = np.mean(roi_frame[:, :, 1])
            #print(f"ROI {i+1} - Green mean: {green_mean}")  # Verifique os valores de verde
            all_green_rois[i].append(green_mean)

    time_stamps.append(frame_count / fps)
    frame_count += 1

time_stamps = np.array(time_stamps)
all_green_rois = [np.array(roi) for roi in all_green_rois]
ratios = calculate_ratios(all_green_rois, num_rois)

# Define thresholds para mediana e desvio padrão
threshold_median = 0.005  
threshold_std = 0.01

# Filtra razões mais próximas de 1 com base na mediana e na consistência (desvio padrão)
close_to_one_ratios = {
    key: value for key, value in ratios.items()
    if abs(np.median(value) - 1) < threshold_median and np.std(value) < threshold_std
}

# Encontrar as ROIs 1, 2 e 3
roi1 = (553, 113, 91, 88)  # Exemplo de ROI 1

for roi_pair in close_to_one_ratios.keys():
    # Extrai os índices das ROIs da chave
    i, j = map(lambda x: int(x.split('/')[0][3:]) - 1, roi_pair.split('/'))
    best_combination = (i, j) 

rois = create_dynamic_rois(frames[0], num_rois, roi_width, roi_height)
i, j = best_combination
x_i, y_i, w_i, h_i = rois[i]
x_j, y_j, w_j, h_j = rois[j]
roi2 = (x_i, y_i, w_i, h_i )  
roi3 = (x_j, y_j, w_j, h_j) 


# Reinicia o vídeo para obter os valores de ROI
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Listas para armazenar intensidades e timestamps
RoiRed, RoiGreen, RoiBlue,Roi2,Roi3, time_stamps = [], [], [], [],[],[]

# Processa o vídeo frame a frame para capturar os valores das ROIs
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    roi1_frame = frame_resized[int(roi1[1]):int(roi1[1] + roi1[3]), int(roi1[0]):int(roi1[0] + roi1[2])]
    roi2_frame = frame_resized[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])]
    roi3_frame = frame_resized[int(roi3[1]):int(roi3[1] + roi3[3]), int(roi3[0]):int(roi3[0] + roi3[2])]
    
    RoiGreen.append(np.mean(roi1_frame[:, :, 1]))
    Roi2.append(np.mean(roi2_frame[:, :, 1]))
    Roi3.append(np.mean(roi3_frame[:, :, 1]))
    
    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()

RoiGreen = np.array(RoiGreen)
Roi2 = np.array(Roi2)
Roi3 = np.array(Roi3)

time_stamps = np.array(time_stamps)

# Normaliza a Curva 2 pela média dos primeiros 40 pontos
normalization_factor = np.mean(Roi2[:30])
if normalization_factor == 0:
    print("Erro: O fator de normalização é zero.")
else:
    normalized_curve2 = Roi2 / normalization_factor




# Função para ajuste exponencial, logarítmico, etc.
def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def logarithmic(x, a, b, c):
    return a * np.log(b * (x - c))

def power_law(x, a, b, c):
    return a * (x - c) ** b

def quadratic(x, a, b, c):
    return a * (x - c) ** 2 + b * (x - c) + c

def gaussian_model(x, a, b, c):
    return a * np.exp(-(x - c)**2 / (2 * b**2))

def linear(x, a, b):
    return a + b * x 

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c
   
def polynomial_4th_order(x, a, b, c, d, e):
    return a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x + e

# Dicionário dos modelos
models = {
    #"Exponential": exponential,
    "Linear": linear,
    #"Polynomial_4th_Order": polynomial_4th_order ,
    #"Quadratic":quadratic
}

# Função para calcular o erro
def calculate_error(adjusted_ratio):
    return np.mean(np.abs(adjusted_ratio - 1))

def calculate_r2(observed, fitted):
    residuals = observed - fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    return 1 - (ss_res / ss_tot)

def initialize_params(model_name):
    if model_name == "Exponential":
        return [
            avoid_zero(np.random.uniform(0.5, 10)),  
            avoid_zero(np.random.uniform(-0.5, 10.5)), 
            avoid_zero(np.random.uniform(0.1, 5.0))    
        ]
    elif model_name == "Linear":
        return [
            avoid_zero(np.random.uniform(0.1, 10.0)),  
            avoid_zero(np.random.uniform(0.1, 10.0))   
        ]
    elif model_name == "Quadratic":
        # Parâmetros para um modelo quadrático: ax^2 + bx + c
        return [
            avoid_zero(np.random.uniform(-5.0, 5.0)),  # Coeficiente a
            avoid_zero(np.random.uniform(-5.0, 5.0)),  # Coeficiente b
            avoid_zero(np.random.uniform(-5.0, 5.0))   # Coeficiente c
        ]
    else:
        raise ValueError("Modelo não suportado.")

def initialize_params_exponential(data):
    # Inicializa a partir de uma estimativa dos dados
    a = np.max(data)  # Usar o máximo dos dados para 'a'
    b = -0.01  # Um valor inicial para o decaimento (valor negativo para uma exponencial decrescente)
    c = np.min(data)  # Usar o mínimo dos dados para 'c'
    return [a, b, c]

def bestfit_models(RoiGreen1, RoiGreen2, models, max_attempts=400):
    best_model = None
    best_r2 = -np.inf
    best_params = None
    best_fitted_curve1 = None
    fitted_curve1_exp = None

    if len(RoiGreen1) > 0:
        max_idx = np.argmax(RoiGreen1)
        RoiGreen1 = RoiGreen1[max_idx:]
        RoiGreen2 = RoiGreen2[max_idx:]

        
    else:
        print("Erro: RoiGreen1 está vazio.")
        return None, None, None, None, None, None

    for model_name, model in models.items():
        attempt = 0
        found_good_fit = False

        while attempt < max_attempts and not found_good_fit:
            attempt += 1
            try:
                params_init = initialize_params(model_name)
                params, _ = curve_fit(model, np.arange(len(RoiGreen2)), RoiGreen2, p0=params_init)
                fittedROI2 = model(np.arange(len(RoiGreen2)), *params)
                error = calculate_error(fittedROI2)

                if error < 0.01:  
                    print("Erros:",error)
                    fitted_curve1 = model(np.arange(len(RoiGreen1)), *params)
                    #tentativa de usar a ROI1 mesmo
                    params_init_exp = initialize_params_exponential(fitted_curve1)
                    params_exp, _ = curve_fit(exponential, np.arange(len(fitted_curve1)), fitted_curve1, p0=params_init_exp)

                    #params_exp, _ = curve_fit(exponential, np.arange(len(RoiGreen1)), RoiGreen1)
                    fitted_curve1_exp = exponential(np.arange(len(fitted_curve1)), *params_exp)
                    r_squared_exp = calculate_r2(fitted_curve1, fitted_curve1_exp)
                    
                    plt.figure(figsize=(8, 6))  # Ajusta o tamanho da figura
                    plt.plot(fitted_curve1_exp, label="Exponential Fit", color='red')
                    plt.close()
                    print("Valores de R^2:", r_squared_exp)



                    if r_squared_exp > 0.6:
                        print(f"Modelo {model_name} com R² = {r_squared_exp:.4f} encontrado!")
                        found_good_fit = True
                        best_model = model_name
                        best_r2 = r_squared_exp
                        best_params = params
                        fitted_curve_best = model(np.arange(len(RoiGreen1)), *best_params)
                        params_exp, _ = curve_fit(exponential, np.arange(len(fitted_curve1)), fitted_curve1)
                        fitted_curve1_exp_best = exponential(np.arange(len(fitted_curve1)), *params_exp)

              

            except Exception as e:
                print(f"Erro durante o ajuste do modelo {model_name} na tentativa {attempt}: {e}")
                continue

        if found_good_fit:
            output_filename = f"{model_name}_final_plot.png"
            plot_and_save_model(model_name, best_params, RoiGreen1, fitted_curve_best, fitted_curve1_exp_best, RoiGreen2, best_r2, output_filename)

    if best_model is None:
        print("Nenhum ajuste adequado foi encontrado.")
        return None, None, None, None, None, None

    return best_model, best_r2, best_params, best_fitted_curve1, fitted_curve1_exp, best_r2



# Chama a função de melhor ajuste
bestfit_models(RoiGreen, normalized_curve2,models)
#print(f"RoiGreen length: {len(RoiGreen)}")
#print(f"normalized_curve2 length: {len(normalized_curve2)}")


