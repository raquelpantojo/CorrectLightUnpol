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
gamma = 1.53

# Função de plotagem com a razão ajustada
def plot_image_and_ratios(frames, best_combination, best_a, best_b, all_green_rois, time_stamps, fps,green_roi2):
    if best_combination is None:
        print("Nenhuma combinação de ROIs encontrada.")
        return
    
    i, j = best_combination
    ratio_key = f"ROI{i+1}/ROI{j+1}"
    original_ratio = ratios[ratio_key]
    adjusted_ratio = (best_a + best_b * green_roi2) 

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


# Função para ajustar os valores de a e b
import numpy as np

def find_best_a_b(green_roi2):
    best_a, best_b = None, None
    best_error = float('inf')  # Inicializa o erro com um valor muito grande

    # Defina os valores possíveis de a e b
    a_values = np.arange(0.1, 10.01, 0.1)
    b_values = np.arange(0.1, 10.01, 0.1)

    # Iteração sobre todas as combinações possíveis de a e b
    for a in a_values:
        for b in b_values:
            # Ajusta a razão com o valor atual de a e b
            adjusted_ratio = a + b * (green_roi2)  
            # Calcula o erro (diferença média absoluta entre a razão ajustada e 1)
            error = np.mean(np.abs(adjusted_ratio - 1))
            
            # Se o erro for menor que o erro anterior, armazena a nova combinação de a, b 
            if error < best_error:
                best_error = error
                best_a, best_b = a, b

    return best_a, best_b





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
#select_rois()
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





# Encontrar a melhor combinação de ROIs e ajustar a e b
best_a, best_b = None, None
best_combination = None

for roi_pair in close_to_one_ratios.keys():
    # Extrai os índices das ROIs da chave
    i, j = map(lambda x: int(x.split('/')[0][3:]) - 1, roi_pair.split('/'))

    roi_values = all_green_rois[i]  
    green_roi2=roi_values/np.mean(roi_values[:40]) # corrige pelo inicio da curva 
    #print(np.mean(roi_values[:40]))
    a, b = find_best_a_b(green_roi2)

    if a is not None and b is not None:
        if best_a is None or best_b is None or (a and b):  
            best_a, best_b = a, b
            best_combination = (i, j) 


if best_a is not None and best_b is not None:
    print(f"Melhor a: {best_a}, Melhor b: {best_b}, ROI utilizada: {best_combination}")
else:
    print("Não foi possível encontrar valores válidos de a, b e gamma.")



plot_image_and_ratios(frames, best_combination, best_a, best_b, all_green_rois, time_stamps, fps,green_roi2)


# Usei isso so para conseguir calcular o CRT depois 
RoiRed=np.array(RoiRed)
RoiGreen=np.array(RoiGreen)
RoiBlue= np.array(RoiBlue)


RoiRed = np.array(RoiRed) 
RoiGreen = np.array(RoiGreen)
RoiBlue = np.array(RoiBlue)

green_roi1 = (best_a + best_b * (RoiGreen/green_roi2))

time_stamps = np.array(time_stamps)

# Calcula razão entre as intensidades normalizadas - desconsidere o ratiosr e ratiosb não vou usa-los no futuro -
#  é somente para poder processa usando o pyCRT
ratiosr = green_roi1 
ratiosg = green_roi1 
ratiosb = green_roi1 



# Plotagem dos resultados
plt.figure(figsize=(10, 5))

# Intensidade ROI1
plt.subplot(3, 1, 1)
plt.plot(time_stamps, RoiGreen, label='G ROI1 ', color='g', linewidth=2)
plt.xlabel('Tempo (s)')
plt.ylabel('Intensidade Canal Verde')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time_stamps, ratiosg, label=f'G ROI1  ({best_a:.2f} + {best_b:.2f} * (ROI1 G/ROI2 G))', color='b',linewidth=2)
plt.xlabel('Tempo (s)')
plt.ylabel('Intensidade Canal Verde')
plt.legend()
plt.tight_layout()
plt.show()



channelsAvgIntensArr= np.column_stack((RoiRed, RoiGreen, RoiBlue))
pcrt = PCRT(time_stamps,channelsAvgIntensArr,exclusionMethod='best fit',exclusionCriteria=999)
#pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()


ratios=np.column_stack((ratiosr,ratiosg,ratiosb))
pcrt = PCRT(time_stamps, ratios,exclusionMethod='best fit',exclusionCriteria=999 )
#pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()






