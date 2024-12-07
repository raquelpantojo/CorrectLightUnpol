import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


sys.path.append("C:/Users/Fotobio/Documents/GitHub/pyCRT") #PC casa 
#sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT") # PC lab

from src.pyCRT import PCRT  


# Caminho base para os arquivos do projeto
base_path = "C:/Users/Fotobio/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"  # PC casa
folder_name = "teste1"
video_name = "v7.mp4"
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



def plot_and_save_model(model_name, RoiDecayGreen1, fitted_curve1, RoiDecayGreen2, output_filename):
 
    plt.figure()
    #  ROI 1
    plt.subplot(311)
    plt.plot(RoiDecayGreen1, label="ROI 1 ", color='green')
    plt.title(f"Model: {model_name} ")
    #plt.xlabel("Time",fontsize=10)
    plt.legend(fontsize=10)

    #  ROI 2
    plt.subplot(312)
    plt.plot(RoiDecayGreen2, label="ROI 2", color='blue')
    #plt.xlabel("Time",fontsize=10)
    plt.ylabel("Intensity of G-channel normalized",fontsize=12)
    plt.legend(fontsize=10)



    plt.subplot(313)
    x = range(len(fitted_curve1))
    plt.scatter(x,fitted_curve1,label="ROI 1 - corrigida",  color='lightgreen', s=30)
    #plt.plot(fitted_curve1_exp, label="Exponential Fit", color='red')
    #plt.xlabel("Time",fontsize=10)
    plt.legend(fontsize=10)


    """
        plt.subplot(414)
            x = range(len(fittedROI2))
            plt.scatter(x,fittedROI2,label="ROI 2 - corrigida",  color='blue', s=30)
            plt.xlabel("Time",fontsize=10)
            plt.legend(fontsize=10)
    """
        

    # Salvando o gráfico em um arquivo
    plt.tight_layout()
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
frame_count = 0
# Processa o vídeo frame a frame para capturar os valores das ROIs
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame_resized = np.clip(frame_resized, 0, 255).astype(np.uint8)

    time_stamps.append(frame_count / fps)
    
    # Durante o processamento de cada frame, garanta que as ROIs sejam uint8
    roi1_frame = frame_resized[int(roi1[1]):int(roi1[1] + roi1[3]), int(roi1[0]):int(roi1[0] + roi1[2])]
    roi2_frame = np.clip(frame_resized[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])], 0, 255).astype(np.uint8)
    roi3_frame = np.clip(frame_resized[int(roi3[1]):int(roi3[1] + roi3[3]), int(roi3[0]):int(roi3[0] + roi3[2])], 0, 255).astype(np.uint8)

   
    #roi2_frame = frame_resized[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])]
    #roi3_frame = frame_resized[int(roi3[1]):int(roi3[1] + roi3[3]), int(roi3[0]):int(roi3[0] + roi3[2])]
    
    RoiGreen.append(np.mean(roi1_frame[:, :, 1]))
    Roi2.append(np.mean(roi2_frame[:, :, 1]))
    Roi3.append(np.mean(roi3_frame[:, :, 1]))
    
    
    frame_count += 1

cap.release()

RoiGreen = np.array(RoiGreen)
Roi2 = np.array(Roi2)
Roi3 = np.array(Roi3)
time_stamps = np.array(time_stamps)


""""
plt.figure()
plt.plot(Roi2)
plt.show()

print(f"Máximo em RoiGreen: {np.max(RoiGreen)}, Mínimo: {np.min(RoiGreen)}")
print(f"Máximo em roi2: {np.max(Roi2)}, Mínimo: {np.min(Roi2)}")
print(f"Máximo em roi3: {np.max(Roi3)}, Mínimo: {np.min(Roi3)}")

"""


# Normaliza a Curva 1 pela média dos primeiros 30 pontos
#RoiGreen = RoiGreen / np.mean(RoiGreen[:30])

# Normaliza a Curva 2 pela média dos primeiros 30 pontos
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

def linear(x, b, c):
    epsilon = 1e-10  # Pequeno valor para evitar divisão por zero ou log de número negativo
    return  (np.abs(b * x) + epsilon)**c


def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c
   
def polynomial_4th_order(x, a, b, c, d, e):
    return a * (x ** 4) + b * (x ** 3) + c * (x ** 2) + d * x + e

def polynomial_3rd_order(x, a, b, c, d):
    return a * (x ** 3) + b * (x ** 2) + c * x + d

def double_exp_decay(x, a1, b1, a2, b2, c):
    return a1 * np.exp(-b1 * x) + a2 * np.exp(-b2 * x) + c


# Dicionário dos modelos
models = {
   #"Exponential": exponential,
    "Linear": linear,
   #"Quadratic": quadratic,
    #"Logarithmic": logarithmic,
    #"Power Law": power_law,
   #"Gaussian": gaussian_model,
    #"Exp Decay": exp_decay,
    #"Polynomial 4th Order": polynomial_4th_order,
    #"Polynomial 3rd Order": polynomial_3rd_order
    #"Double Exponential Decay":double_exp_decay,
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
            np.random.uniform(0.1, 10),   # 'a' positivo para amplitude
            -np.random.uniform(0.1, 1),  # 'b' negativo para decaimento
            np.random.uniform(0, 1)      # 'c' pequeno para deslocamento
        ]
    elif model_name == "Linear":
        return [1, 
            np.random.uniform(0, 2.5) 
                ]
        
    elif model_name == "Quadratic":
        return [
            np.random.uniform(0, 10),  # Coeficiente quadrático
            np.random.uniform(-10, 10), # Coeficiente linear
            np.random.uniform(0, 10)  # Termo constante
            
        ]
    elif model_name == "Logarithmic":
        return [
            np.random.uniform(0.1, 10),  # Amplitude
            np.random.uniform(0.1, 2),  # Fator de escala
            np.random.uniform(-10, 10)  # Deslocamento
        ]
    elif model_name == "Power Law":
        return [
            np.random.uniform(0.1, 5),   # Coeficiente
            np.random.uniform(0.1, 3),  # Expoente
            np.random.uniform(-10, 10)  # Deslocamento
        ]
    elif model_name == "Gaussian":
        return [
            np.random.uniform(0.1, 10),  # Altura do pico
            np.random.uniform(1, 5),    # Largura (desvio padrão)
            np.random.uniform(-10, 10)  # Posição do centro
        ]
    elif model_name == "Exp Decay":
        return [
            np.random.uniform(1, 10),   # Amplitude inicial
            np.random.uniform(0.1, 1), # Taxa de decaimento
            np.random.uniform(0, 1)    # Deslocamento vertical
        ]
    elif model_name == "Polynomial 4th Order":
        return [
            np.random.uniform(-1, 1),   # Coeficiente x⁴
            np.random.uniform(-1, 1),  # Coeficiente x³
            np.random.uniform(-1, 1),  # Coeficiente x²
            np.random.uniform(-1, 1),  # Coeficiente x
            np.random.uniform(0, 10)   # Constante
        ]
    elif model_name == "Polynomial 3rd Order":
        return [
            np.random.uniform(-1, 1),   # Coeficiente x³
            np.random.uniform(-1, 1),   # Coeficiente x²
            np.random.uniform(-1, 1),   # Coeficiente x
            np.random.uniform(0, 10)     # Constante
        ]
    
    elif model_name == "Double Exponential Decay":
        return [
            np.random.uniform(0.1, 1),   # 'a1' para a primeira amplitude
            np.random.uniform(0.1, 1),    # 'b1' para a primeira taxa de decaimento
            np.random.uniform(0.1, 10),   # 'a2' para a segunda amplitude
            np.random.uniform(0.1, 1),    # 'b2' para a segunda taxa de decaimento
            np.random.uniform(0, 10)       # 'c' para o deslocamento
        ]
    else:
        raise ValueError("Modelo não suportado.")



def initialize_params_exponential(data):
    a = np.max(data)  
    b = -0.01  
    c = np.min(data) 
    return [a, b, c]


from scipy.optimize import curve_fit
import numpy as np

def bestfit_models(RoiGreen1, RoiGreen2,RoiGreen3, models, max_attempts=100):
    best_model = None
    best_r2 = -np.inf
    best_params = None

    #print(f"Máximo em RoiGreen: {np.max(RoiGreen1)}, Mínimo: {np.min(RoiGreen1)}")
    #print(f"Máximo em roi2: {np.max(RoiGreen2)}, Mínimo: {np.min(RoiGreen2)}")
    #print(f"Máximo em roi3: {np.max(RoiGreen3)}, Mínimo: {np.min(RoiGreen3)}")

    if len(RoiGreen1) == 0:
            print("Erro: RoiGreen1 está vazio.")
            return None
            
    max_idx = np.argmax(RoiGreen1)
    RoiDecayGreen1 = RoiGreen1[max_idx:]
    RoiDecayGreen2 = RoiGreen2[max_idx:]

  
    scaler = MinMaxScaler()
    RoiGreen1 = scaler.fit_transform(RoiGreen1.reshape(-1, 1)).flatten()
    RoiGreen2 = scaler.fit_transform(RoiGreen2.reshape(-1, 1)).flatten()
    RoiGreen3 = scaler.fit_transform(RoiGreen3.reshape(-1, 1)).flatten()


    
    for model_name, model in models.items():
        for attempt in range(max_attempts):
            try:
               
                params_init = initialize_params(model_name)

                # Ajustar o modelo
                params, _ = curve_fit(model, np.arange(len(RoiGreen2)), RoiGreen2, p0=params_init,maxfev=10000)
                fittedROI2 = model(np.arange(len(RoiGreen2)), *params)
                fittedROI3 = model(np.arange(len(RoiGreen3)), *params)
                z=fittedROI2/fittedROI3
                error = calculate_error(z)

                if error < 0.01:  
                    print("gamma:",params[2])
                    #print("Erros:",error)
                    fitted_curve = RoiDecayGreen1**params[2]
                    #melhorando os parametroos iniciais de ajuste 
                    params_init_exp = initialize_params_exponential(fitted_curve)
                    params_exp, _ = curve_fit(exponential, np.arange(len(fitted_curve)), fitted_curve, p0=params_init_exp,maxfev=10000)
                    fitted_curve1_exp = exponential(np.arange(len(fitted_curve)), *params_exp)

                    # Calcular R^2
                    r_squared = calculate_r2(fitted_curve, fitted_curve1_exp)

                    # Verificar se é o melhor ajuste
                    if r_squared > best_r2:
                        best_model = model_name
                        best_r2 = r_squared
                        best_params = params
                        best_curve = fitted_curve
                    
                    
                    

            except Exception as e:
                print(f"Erro durante o ajuste do modelo {model_name} na tentativa {attempt + 1}: {e}")
                continue

    if best_model:
        print(f"Melhor modelo: {best_model} com R² = {best_r2:.4f}")
        fittedROI1Complet=RoiGreen1**best_params[2]
        return model_name,params, RoiGreen1,RoiDecayGreen1, fitted_curve, RoiGreen2, RoiDecayGreen2,fittedROI2,fitted_curve1_exp,fittedROI1Complet
    else:
        print("Nenhum ajuste aceitável encontrado.")
        return None



# Chama a função de melhor ajuste
#model_name,params, RoiGreen1,RoiDecayGreen1, fitted_curve1, fitted_curve1_exp, RoiGreen2, RoiDecayGreen2,fittedROI2, r_squared_exp,fittedROI1Complet = bestfit_models(RoiGreen, normalized_curve2,models)
model_name,params, RoiGreen1,RoiDecayGreen1, fitted_curve, RoiGreen2, RoiDecayGreen2,fittedROI2,fitted_curve1_exp,fittedROI1Complet= bestfit_models(RoiGreen, Roi2, Roi3,models)
#print(f"RoiGreen length: {len(RoiGreen)}")
#print(f"normalized_curve2 length: {len(normalized_curve2)}")



output_filename = f"{model_name}_final_plot.png"
plot_and_save_model(model_name, RoiDecayGreen1, fitted_curve, RoiDecayGreen2, output_filename)


#testando o pCRT 
# sem ajuste 
"""
channelsAvgIntensArr= np.column_stack((RoiGreen1, RoiGreen1, RoiGreen1))
pcrt = PCRT(time_stamps,channelsAvgIntensArr,exclusionMethod='best fit',exclusionCriteria=999,fromTime =7)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()





print(len(fittedROI1Complet))


for model_name, model in models.items():
   
    

"""

output_filename2 = f"{model_name}ROI1Corrigidav2.png"

plt.figure()
plt.plot(time_stamps, fittedROI1Complet, label="ROI1 corrigida")
plt.legend()
plt.tight_layout()  
plt.savefig(output_filename2, dpi=600)
#plt.show() 


# não levar em consideração que todos os valores RGB estão somente com a curva do verde
# aqui só estou preocupada com a curva verde mesmo 



channelsAvgIntensArr= np.column_stack((fittedROI1Complet, fittedROI1Complet, fittedROI1Complet))
pcrtCorrigido = PCRT(time_stamps,channelsAvgIntensArr,exclusionMethod='best fit',exclusionCriteria=999999)
#pcrt.showAvgIntensPlot()
pcrtCorrigido.showPCRTPlot()
pcrtCorrigido.savePCRTPlot(f"Corrigido{model_name}.png")

# salvando os dados 

# dados da região do decaaimento 
outputDataDecay = []

for i in range(len(RoiDecayGreen1)):  
    outputDataDecay.append({
        "Model Name": model_name,  
        "Best Params": params,  
        "Roi Decay Green 1": RoiDecayGreen1[i],
        "Fitted Curve 1": fitted_curve[i],
        "Fitted Curve 1 Exp": fitted_curve1_exp[i],
        "pcrtCorrigido": pcrtCorrigido.pCRT[0],
        "incerteza":pcrtCorrigido.pCRT[1],

    })

# Convertendo para DataFrame
df = pd.DataFrame(outputDataDecay)

# Salvando no Excel
outputCompleteData = f"resultadosCompletos{model_name}.xlsx"
df.to_excel(outputCompleteData, index=False)

print(f"Dados salvos em: {outputCompleteData}")



