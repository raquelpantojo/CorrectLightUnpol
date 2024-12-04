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
video_name = "corrected_v7_gamma=1.53.mp4"
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

    # Gerando a equação formatada para a legenda com base no modelo
    if model_name == "exponential":
        eq = f"{model_name}: $y = {params[0]:.2f}e^{{({params[1]:.2f}(x-{params[2]:.2f}))}}$, $R^2 = {r_squared:.4f}$"
    elif model_name == "power_law":
        eq = f"{model_name}: $y = {params[0]:.2f}x^{params[1]:.2f}$, $R^2 = {r_squared:.4f}$ "
    elif model_name == "logarithmic":
        eq = f"{model_name}: $y = {params[0]:.2f} \log(x) + {params[1]:.2f}$, $R^2 = {r_squared:.4f}$"
    elif model_name =="Quadratic":
        eq = f"{model_name}: $y = {params[0]:.2f}x^2 + {params[1]:.2f}x + {params[2]:.2f}$"
    elif model_name == "gaussiano":
        eq = f"{model_name}: $y = {params[0]:.2f} \exp\left(-\\frac{{(x - {params[2]:.2f})^2}}{{2{params[1]:.2f}^2}}\\right)$"
    elif model_name =="Linear":
        eq = f"{model_name}: $y = {params[0]:.2f} + {params[1]:.2f}x + {params[2]:.2f}$"
    else:
         eq = f"{model_name}: Não reconhecido"
    
    # Criando o gráfico
    plt.figure()
    
    # Subplot 1: ROI 2
    plt.subplot(311)
    plt.plot(RoiGreen2, label="ROI 2", color='green')
    plt.title(f"Model: {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Intensity of green channel")
    plt.legend()

    # Subplot 2: ROI 1 vs Fit Exponencial
    plt.subplot(312)
    plt.plot(RoiGreen1, label="ROI 1", color='green')
    plt.plot(fitted_curve1_exp, label="Exponential Fit", color='red')
    plt.xlabel("Time")
    plt.ylabel("Intensity of green channel")
    plt.legend()

    # Subplot 3: Fitted Curve and Equation
    plt.subplot(313)
    plt.plot(fitted_curve1, label=eq, color='blue')
    plt.xlabel("Time")
    plt.ylabel("Intensity of green channel")
    plt.legend(fontsize=14)

    # Salvando o gráfico em um arquivo
    plt.savefig(output_filename, dpi=600)
    plt.close()


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
RoiRed, RoiGreen, RoiBlue,Roi2, time_stamps = [], [], [], [],[]

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
    
    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()

RoiGreen = np.array(RoiGreen)
time_stamps = np.array(time_stamps)

# Normaliza a Curva 2 pela média dos primeiros 40 pontos
normalization_factor = np.mean(Roi2[:40])
if normalization_factor == 0:
    print("Erro: O fator de normalização é zero.")
else:
    normalized_curve2 = Roi2 / normalization_factor




# Função para ajuste exponencial, logarítmico, etc.
def exponential(x, a, b, c):
    return a * np.exp(b * (x - c))

def logarithmic(x, a, b, c):
    return a * np.log(b * (x - c))

def power_law(x, a, b, c):
    return a * (x - c) ** b

def quadratic(x, a, b, c):
    return a * (x - c) ** 2 + b * (x - c) + c

def gaussian_model(x, a, b, c):
    return a * np.exp(-(x - c)**2 / (2 * b**2))

def linear(x, a, b,c):
    return a + b * x + c

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c
   

# Dicionário dos modelos
models = {
    "Exponential": exponential,
    "Linear": linear,
    "quadratic" :quadratic,
}

def bestfit_models(RoiGreen1, RoiGreen2, models, max_attempts=400):
    best_model = None
    best_r2 = -np.inf
    best_params = None
    best_fitted_curve1 = None
    fitted_curve1_exp = None  # Defina a variável aqui, antes de usá-la.

    if len(RoiGreen1) > 0:
        max_idx = np.argmax(RoiGreen1)
    else:
        print("Erro: RoiGreen1 está vazio.")
        return None, None, None, None

    # Deixa somente a parte do decaimento
    RoiGreen1 = RoiGreen1[max_idx:]
    RoiGreen2 = RoiGreen2[max_idx:]

    for model_name, model in models.items():
        attempt = 0
        found_good_fit = False
        while attempt < max_attempts and not found_good_fit:
            attempt += 1
            try:
                # Parâmetros de inicialização específicos para cada modelo
                if model_name == "exponential":
                    a_init, b_init, c_init = np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5), np.random.uniform(0.0, 5.0)
                elif model_name == "linear":
                    a_init, b_init, c_init = np.random.uniform(0.5, 2.0), np.random.uniform(0.1, 1.0), np.random.uniform(0.0, 5.0)
                elif model_name == "quadratic":
                    a_init, b_init, c_init = np.random.uniform(0.5, 1.5), np.random.uniform(1.0, 2.0), np.random.uniform(0.0, 5.0)
                else:
                    a_init, b_init, c_init = np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5), np.random.uniform(0.0, 5.0)

                # Ajuste o modelo aos dados RoiGreen2
                params, _ = curve_fit(model, np.arange(len(RoiGreen2)), RoiGreen2, p0=[a_init, b_init, c_init])
                fitted_curve2 = model(np.arange(len(RoiGreen2)), *params)
                error = np.mean(np.abs(fitted_curve2 - 1))

                # Verifica se o erro é suficientemente pequeno
                if error < 0.01:
                    print(f"Modelo {model_name} ajustado com erro = {error:.4f}")
                    fitted_curve1 = model(np.arange(len(RoiGreen1)), *params)

                    # Ajuste para o modelo exponencial de RoiGreen1
                    a_init_curv1 = np.max(fitted_curve1)  
                    b_init_curv1 = -0.1  
                    c_init_curv1 = np.median(fitted_curve1) 
                    params_exp, _ = curve_fit(exponential, np.arange(len(RoiGreen1)), RoiGreen1, p0=[a_init_curv1, b_init_curv1, c_init_curv1])
                    fitted_curve1_exp = exponential(np.arange(len(RoiGreen1)), *params_exp)  # Garanta que fitted_curve1_exp seja inicializado
                    
                    residuals = RoiGreen1 - fitted_curve1_exp
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((RoiGreen1 - np.mean(RoiGreen1))**2)
                    r_squared = 1 - (ss_res / ss_tot)

                    # Verifica se o ajuste é bom o suficiente
                    if r_squared > 0.8:
                        print(f"Modelo {model_name} com R² = {r_squared:.4f} encontrado no {attempt}º tentativa!")
                        found_good_fit = True
                        best_r2 = r_squared
                        best_model = model_name
                        best_params = params
                        best_fitted_curve1 = fitted_curve2
                    else:
                        print(f"Tentativa {attempt} do modelo {model_name} com R² = {r_squared:.4f} falhou. Tentando novamente...")
                else:
                    print(f"Erro {error:.4f} do modelo {model_name} não é suficiente. Tentando novos parâmetros...")

                    # Caso o erro não seja suficientemente baixo, calcule novos parâmetros
                    if model_name == "exponential":
                        a_init, b_init, c_init = np.random.uniform(0.5, 2.0), np.random.uniform(0.5, 2.0), np.random.uniform(0.0, 5.0)
                    elif model_name == "linear":
                        a_init, b_init, c_init = np.random.uniform(0.5, 2.0), np.random.uniform(0.1, 1.0), np.random.uniform(0.0, 5.0)
                    elif model_name == "quadratic":
                        a_init, b_init, c_init = np.random.uniform(0.5, 1.5), np.random.uniform(1.0, 2.0), np.random.uniform(0.0, 5.0)
                    params, _ = curve_fit(model, np.arange(len(RoiGreen2)), RoiGreen2, p0=[a_init, b_init, c_init])
                    fitted_curve2 = model(np.arange(len(RoiGreen2)), *params)
                    error = np.mean(np.abs(fitted_curve2 - 1))
                    print(f"Nova tentativa com erro = {error:.4f}")

            except Exception as e:
                print(f"Erro ao ajustar modelo {model_name} na tentativa {attempt}: {e}")
                continue

        if found_good_fit:
            print(f"Modelo {model_name} ajustado com sucesso após {attempt} tentativas.")
        else:
            print(f"Modelo {model_name} não conseguiu R² > 0.7 após {max_attempts} tentativas.")

        # Gerar e salvar o gráfico, independente do resultado
        for model_name, model_func in models.items():
            params = [params[0], params[1], params[2]]  
            fitted_curve1 = model_func(RoiGreen1, *params)  
            output_filename = f"{model_name}_plot.png"  
            plot_and_save_model(model_name, params, RoiGreen1, fitted_curve1, fitted_curve1_exp, RoiGreen2, r_squared, output_filename,)

    return best_model, best_params, best_r2, best_fitted_curve1




# Chama a função de melhor ajuste
bestfit_models(RoiGreen, normalized_curve2,models)
#print(f"RoiGreen length: {len(RoiGreen)}")
#print(f"normalized_curve2 length: {len(normalized_curve2)}")


