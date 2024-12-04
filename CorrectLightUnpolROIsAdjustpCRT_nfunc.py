# Teste 2- 03-12-2024
# 
#Obter duas ROIs (ROI2 e ROI3) fora da região de interesse
#Crio 200 ROIs dentro da imagem com 60x60 px
#Faz diferentes combinações 2 a 2 
#Encontra o conjunto de combinações que possua a razão mais próxima de 1
#Seleciona a ROI 2 (Roi que não teve alteração pela penumbra)
#Normaliza esse ROI pelo valor médio dos primeiros 40 frames do vídeo
#Divide a ROI1/ ROI2
#Encontra os valores de a e b (0,1 – 100 passos de 0,1) ajustando a equação forçando a ter valores igual a 1: 
#𝑎+𝑏∗𝐼(𝑅𝑂𝐼_2 )^𝛾
#Ajusta a curva da ROI1 : 𝑎+𝑏∗𝐼(𝑅𝑂𝐼_1 )^𝛾


import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import curve_fit
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



# Função de plotagem com a razão ajustada
def plot_image_and_ratios(frames, best_combination, best_a, best_b, all_green_rois, time_stamps, fps,green_roi2):
    if best_combination is None:
        print("Nenhuma combinação de ROIs encontrada.")
        return
    
    i, j = best_combination
    ratio_key = f"ROI{i+1}/ROI{j+1}"
    original_ratio = ratios[ratio_key]
    adjusted_ratio = (best_a + best_b *(green_roi2))

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
    
    #Remover subplots em branco da primeira coluna
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




# Funções Candidatas
def logarithmic(x, a, b, c):
    return a + b * np.log(c * x + 1)

def sinusoidal(x, a, b, c, d):
    return a + b * np.sin(c * x + d)

def power_law(x, a, b, p):
    return a + b * np.power(x, p)

def quadratic(x, a, b, c):
    return a + b * x + c * x**2

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

# Função para calcular o chi-quadrado
def calculate_chi_squared(y_true, y_pred):
    residuals = y_true - y_pred
    return np.sum((residuals**2) / y_pred)

# Ajuste para garantir que valores logarítmicos sejam válidos
def safe_log(x, c=1e-10):
    return np.log(x + c)

# Função para calcular o erro
def calculate_error(adjusted_ratio):
    return np.mean(np.abs(adjusted_ratio - 1))


# Função para encontrar os melhores valores de a e b para a curva green_roi2
def find_best_a_b(green_roi2, best_func):
    best_a, best_b, best_c = None, None, None  # Agora adicionando o parâmetro c
    best_error = float('inf')  # Inicializa o erro com um valor muito grande

    # Defina os valores possíveis de a, b e c
    a_values = np.arange(0.1, 10.01, 0.01)
    b_values = np.arange(0.1, 10.01, 0.01)
    c_values = np.arange(0.1, 10.01, 0.01)  # Agora também testando valores para c

    # Iteração sobre todas as combinações possíveis de a, b e c
    for a in a_values:
        for b in b_values:
            for c in c_values:  # Adicionando iteração para c
                try:
                    # Ajusta a razão com a função encontrada
                    if best_func == logarithmic:
                        adjusted_ratio = best_func(green_roi2, a, b, 1)  
                    else:
                        adjusted_ratio = best_func(green_roi2, a, b, c)  

                    # Calcula o erro (diferença média absoluta entre a razão ajustada e 1)
                    error = np.mean(np.abs(adjusted_ratio - 1))

                    # Se o erro for menor que o erro anterior, armazena a nova combinação de a, b, c
                    if error < best_error:
                        best_error = error
                        best_a, best_b, best_c = a, b, c  # Armazena os três parâmetros
                        print(f"Novo melhor ajuste encontrado: a={best_a:.2f}, b={best_b:.2f}, c={best_c:.2f}, erro={best_error:.4f}")
                except Exception as e:
                    print(f"Erro ao ajustar {best_func.__name__} para green_roi2: {e}")

    return best_a, best_b, best_c  # Retorna também o parâmetro c


# modelos testados 

models = {
        "Exponential": exponential,
        "Logarithmic": logarithmic,
        "Power Law": power_law,
        "Quadratica":quadratic
    }

#{{
def find_best_model(green_roi1, green_roi2, models):
    best_model = None
    best_r2 = -np.inf
    best_params = None
    best_func = None
    best_error = np.inf  # Inicializa o melhor erro com um valor grande

    # Encontrar o pico máximo de green_roi1
    max_idx = np.argmax(green_roi1)
    x = np.arange(len(green_roi1))
    
    # Dados após o pico para ROI1
    x_after = x[max_idx:]
    y_after = green_roi1[max_idx:]
    
    # Teste dos modelos para green_roi2
    for name, func in models.items():
        try:
            print(f"Testando o modelo {name} para green_roi2...")

            # Ajuste de cada modelo para green_roi2
            if func == logarithmic:
                params, _ = curve_fit(func, x_after, green_roi2[max_idx:], maxfev=5000, bounds=(0, [10, 10, 10]))
            else:
                params, _ = curve_fit(func, x_after, green_roi2[max_idx:], maxfev=5000)

            adjusted_ratio = func(x_after, *params)  # Curva ajustada para ROI2
            error = calculate_error(adjusted_ratio)  # Calcula o erro

            # Se o erro for suficientemente pequeno, considere o ajuste bom
            if error < 0.005:
                print(f"Modelo {name} ajustado com erro = {error:.4f}")
                if error < best_error:
                    best_error = error
                    best_model = name
                    best_params = params
                    best_func = func

        except Exception as e:
            print(f"Erro ao ajustar o modelo {name} para green_roi2: {e}")

    # Se encontrou um bom modelo para ROI2, ajuste para ROI1 usando o melhor modelo encontrado
    if best_model:
        print(f"Melhor modelo encontrado: {best_model} com erro = {best_error:.4f}")

        # Ajuste para green_roi1 com os parâmetros do melhor modelo
        x_before = x[:max_idx]
        y_before = green_roi1[:max_idx]
        
        try:
            # Ajuste o modelo escolhido para ROI1
            if best_func == logarithmic:
                params_roi1, _ = curve_fit(best_func, x_before, y_before, maxfev=5000, bounds=(0, [10, 10, 10]))
            else:
                params_roi1, _ = curve_fit(best_func, x_before, y_before, maxfev=5000)
            
            # Predição para ROI1 com o melhor modelo
            y_pred = best_func(x_before, *params_roi1)
            r2 = r2_score(y_before, y_pred)
            chi2 = np.sum((y_before - y_pred)**2)  # Chi-quadrado

            # Verificar se os critérios de R2 e Qui-quadrado são os melhores
            print(f"Ajuste para ROI1: R2={r2:.4f}, Chi2={chi2:.4f}")

            if r2 > best_r2 and chi2 < best_error:
                best_r2 = r2
                best_model = name
                best_params = params_roi1
                best_func = best_func  # Atualiza a função com o melhor ajuste

            # Retorna a curva ajustada para ROI1
            adjusted_curve_roi1 = best_func(x, *params_roi1)
            return best_model, best_params, best_r2, best_error, best_func, adjusted_curve_roi1,best_combination

        except Exception as e:
            print(f"Erro ao ajustar o modelo {best_model} para ROI1: {e}")

    return best_model, best_params, best_r2, best_error, best_func, None , best_combination


#}}




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
        roi_frame = roi_frame 

        if roi_frame.size > 0:  # Certifica-se de que a ROI não está vazia
            all_green_rois[i].append(np.mean(roi_frame[:, :, 1]))
    
    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()

time_stamps = np.array(time_stamps)
all_green_rois = [np.array(roi) for roi in all_green_rois]
ratios = calculate_ratios(all_green_rois, num_rois)

# Definir thresholds para mediana e desvio padrão
threshold_median = 0.005  
threshold_std = 0.01      

# Filtra razões mais próximas de 1 com base na mediana e na consistência (desvio padrão)
close_to_one_ratios = {
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

# Variáveis para as ROIs
roi1 = None
frame_count = 0
fps = cap.get(cv.CAP_PROP_FPS)

# Função para selecionar ROIs com o vídeo rodando
def select_rois():
    global roi1
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
            print(roi1)
            cv.destroyAllWindows()
            break

# Selecionar as ROIs
# select_rois()
# usa uma ROI conhecida 
roi1=(553, 113, 91, 88)

# Reinicia o vídeo
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Listas para armazenar intensidades e timestamps
RoiRed,RoiGreen,RoiBlue,time_stamps = [], [],[],[]

# Processa o vídeo frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)    
    # Extrai as ROIs e calcula a média do canal verde
    roi1_frame = frame[int(roi1[1]):int(roi1[1] + roi1[3]), int(roi1[0]):int(roi1[0] + roi1[2])]
    
    # ROI 1
    RoiRed.append(np.mean(roi1_frame[:, :, 0]))
    RoiGreen.append(np.mean(roi1_frame[:, :, 1]))
    RoiBlue.append(np.mean(roi1_frame[:, :, 2]))

    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()

time_stamps = np.array(time_stamps)




# Encontrar a melhor combinação de ROIs e ajustar a e b
best_a, best_b = None, None
best_combination = None

for roi_pair in close_to_one_ratios.keys():
    # Extrai os índices das ROIs da chave
    i, j = map(lambda x: int(x.split('/')[0][3:]) - 1, roi_pair.split('/'))
    best_combination = {i,j}

    roi_values = all_green_rois[i]  
    green_roi2=roi_values/np.mean(roi_values[:40]) # corrige pelo inicio da curva 
    #print(np.mean(roi_values[:40]))

    RoiRed=np.array(RoiRed)
    RoiGreen=np.array(RoiGreen)
    RoiBlue= np.array(RoiBlue)


    RoiRed = np.array(RoiRed) 
    RoiGreen = np.array(RoiGreen)
    RoiBlue = np.array(RoiBlue)

    green_roi1 = RoiGreen




    # Exemplo de uso
    best_model, best_params, best_r2, best_a, best_b,adjusted_curve_roi1,best_combination = find_best_model(green_roi1, green_roi2, models)
    print(f"Melhor modelo: {best_model} com R^2 = {best_r2:.3f}")

    


ratiosr = adjusted_curve_roi1 
ratiosg = adjusted_curve_roi1 
ratiosb = adjusted_curve_roi1 


plot_image_and_ratios(frames, best_combination, best_a, best_b, all_green_rois, time_stamps, fps,green_roi2)



# Testa o CRT:
channelsAvgIntensArr= np.column_stack((RoiRed, RoiGreen, RoiBlue))
pcrt = PCRT(time_stamps,channelsAvgIntensArr,exclusionMethod='best fit',exclusionCriteria=999)
#pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()


ratios=np.column_stack((ratiosr,ratiosg,ratiosb))
pcrt = PCRT(time_stamps, ratios,exclusionMethod='best fit',exclusionCriteria=999 )
#pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()



# Agora, plotamos os dados e o ajuste exponencial
x = np.arange(len(green_roi1))
plt.plot(x, green_roi1, 'bo', label='Dados green_roi1')

# Plotamos o ajuste exponencial para o melhor modelo
if best_model == 'Exponential':
    plt.plot(x, exponential(x, *best_params), 'r-', label='Ajuste Exponencial')

# Plotamos os ajustes para todos os outros modelos também
for name, func in models.items():
    try:
        params, _ = curve_fit(func, x, green_roi1, maxfev=5000)
        plt.plot(x, func(x, *params), label=f'Ajuste {name}')
    except Exception as e:
        print(f"Erro ao ajustar o modelo {name}: {e}")

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ajuste de Modelos para green_roi1')
plt.legend()
plt.show()



