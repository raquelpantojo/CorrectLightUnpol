# atualizado dia 09-01
# Encontra a função ganho para o sinal de intensidade do canal verde



# atualização dia 07/01/2024 - implemntação do histograma canal verde 

import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import combinations

import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.signal import argrelextrema, butter, filtfilt


#sys.path.append("C:/Users/Fotobio/Documents/GitHub/pyCRT") #PC casa 
#sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT") # PC lab
sys.path.append("C:/Users/raque/OneDrive/Documentos/GitHub/pyCRT") # note

from src.pyCRT import PCRT  



#base_path="C:/Users/raque/OneDrive/Documentos/GitHub/CorrectLightUnpol/DadosExperimentoRugosoELiso" # note
base_path="C:/Users/raque/OneDrive/Documentos/GitHub" # note
folder_name="CorrectLightUnpol"
#video_name = "NatanLedDesp6.mp4"
#video_name = "EduLedDesp4.mp4"
#video_name = "SeiyLedDesp4.mp4"
video_name="EduLedDesp4.mp4"
#video_name = "SeiyLedPol6.mp4"
#video_name= "Desp1.mp4"
#video_name="v7.mp4"

roi_width = 80 
roi_height = 80
num_rois = 200  # Número de ROIs a serem criadas

#v7
#roi1=(1121, 222, 154, 154)

##video_name= "Desp1.mp4"
#roi1=(339, 855, 176, 137)

#Seyi Despolarizado
#roi1=(523, 246, 180, 147)

#Seyi Polarizado
#roi1=(594, 183, 124, 131)

# Natan despolarizado
#roi1= (457, 571, 165, 215)

#Edu Desp
roi1=(710, 121, 131, 150)


# Função de plotagem com a razão ajustada
def plot_image_and_ratios(frames, best_combination, all_green_rois, time_stamps, 
                          RoiGreen, ROI1Corrigida, folder_name, roi1):
    """
    Função para plotar a imagem original, os canais verdes e as razões ajustadas/corrigidas.
    
    Args:
        frames: Lista de frames de vídeo.
        best_combination: Melhor combinação de ROIs (índices i e j).
        all_green_rois: Lista de intensidades do canal verde para todas as ROIs.
        time_stamps: Lista de timestamps para os frames.
        RoiGreen: Intensidades do canal verde da ROI1.
        ROI1Corrigida: Canal verde corrigido para ROI1.
        adjusted_ratio: Razão ajustada entre as ROIs selecionadas.
        folder_name: Nome da pasta para salvar o arquivo de saída.
        roi1: Coordenadas da ROI1 no formato (x, y, largura, altura).
    """
    if best_combination is None:
        print("Nenhuma combinação de ROIs encontrada.")
        return
    
    i, j = best_combination
    ratio_key = f"ROI{i+1}/ROI{j+1}"
    original_ratio = ratios[ratio_key]
     
    # Extrair coordenadas da ROI1
    x_roi1, y_roi1, w_roi1, h_roi1 = roi1

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    
    axs[0, 0].imshow(cv.cvtColor(frames[0], cv.COLOR_BGR2RGB))  
    axs[0, 0].set_title('Imagem Original')
    axs[0, 0].axis('off')  
    
    # Criar ROIs dinâmicas
    rois = create_dynamic_rois(frames[0], num_rois, roi_width, roi_height)
    x_i, y_i, w_i, h_i = rois[i]
    x_j, y_j, w_j, h_j = rois[j]
   
    # Adicionar retângulos das ROIs na imagem
    axs[0, 0].add_patch(plt.Rectangle((x_roi1, y_roi1), w_roi1, h_roi1, edgecolor='green', facecolor="none", linewidth=2))
    axs[0, 0].add_patch(plt.Rectangle((x_i, y_i), w_i, h_i, edgecolor='blue', facecolor='none', linewidth=2))
    axs[0, 0].add_patch(plt.Rectangle((x_j, y_j), w_j, h_j, edgecolor='red', facecolor='none', linewidth=2))
    axs[0, 0].text(x_roi1 + w_roi1 / 2, y_roi1 + h_roi1 / 2, "ROI1", color='green', ha='center', va='center', fontsize=8, weight='bold')
    axs[0, 0].text(x_i + w_i / 2, y_i + h_i / 2, f"{i+1}", color='blue', ha='center', va='center', fontsize=8, weight='bold')
    axs[0, 0].text(x_j + w_j / 2, y_j + h_j / 2, f"{j+1}", color='red', ha='center', va='center', fontsize=8, weight='bold')
    
    # Remover subplots em branco da primeira coluna
    axs[1, 0].axis('off')
    axs[2, 0].axis('off')

    axs[0, 1].plot(time_stamps, all_green_rois[i], label=f"Canal Cinza ROI {i+1} ", color='blue')
    axs[0, 1].set_xlabel('Tempo (s)')
    axs[0, 1].set_ylabel('Intensidade do Canal Cinza')
    axs[0, 1].legend()

    axs[1, 1].plot(time_stamps, all_green_rois[j], label=f"Canal Cinza ROI {j+1} ", color='red')
    axs[1, 1].set_xlabel('Tempo (s)')
    axs[1, 1].set_ylabel('Intensidade do Canal Cinza')
    axs[1, 1].legend()

    axs[2, 1].plot(time_stamps, original_ratio, label=f"Razão ROI {i+1} / ROI {j+1}", color='orange')
    axs[2, 1].set_xlabel('Tempo (s)')
    axs[2, 1].set_ylabel('Razão')
    axs[2, 1].legend()

    axs[0, 2].plot(time_stamps, RoiGreen, label=f"Canal Cinza ROI 1 ", color='darkgreen')
    axs[0, 2].set_xlabel('Tempo (s)')
    axs[0, 2].set_ylabel('Intensidade do Canal Cinza')
    axs[0, 2].legend()

    axs[1, 2].plot(time_stamps, ROI1Corrigida, label=f"Canal Cinza Equalizado", color='orange')
    axs[1, 2].set_xlabel('Tempo (s)')
    axs[1, 2].set_ylabel('Intensidade do Canal Cinza')
    axs[1, 2].legend()



    plt.tight_layout()
    output_filename = f"teste_{folder_name}.png"
    plt.savefig(output_filename, dpi=600)
    plt.show()
    plt.close()

def plot_graph_curves(time_stamps,RoiGreen, RoiGray,RoiGreen1Equ,RoiGray1Equ,outputImageGraph):
    
    plt.subplot(311)
    plt.plot(time_stamps, RoiGreen, label=f"Canal Verde ROI 1 ", color='darkgreen')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Verde')
    plt.legend()

    plt.plot(time_stamps, RoiGray, label=f"Canal Cinza ROI 1", color='gray')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza', fontsize="12")
    plt.legend()
    
    plt.subplot(312)
    plt.plot(time_stamps, RoiGreen1Equ, label=f"Equalizado verde", color='green')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Verde')
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(313)
    plt.plot(time_stamps, RoiGray1Equ, label=f"Equalizado Cinza", color='green')
    #plt.title(f"a={a:.2f}_b={b:.2f} $\gamma$={gamma:.2f}")
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    plt.legend()
    plt.tight_layout()
    
    
   #pastaSalvar= "C:/Users/Fotobio/Desktop/ResultadosAbgamma"
    plt.savefig(outputImageGraph, dpi=300)
    plt.show()
    plt.close()



def plotCurveFGanho(time_stamps,RoiGray1, RoiGray2, RoiGray3, funçãoGanhoGray1,funçãoGanhoGray2,funçãoGanhoGray3,
                  IntensityGrayCorrect1,IntensityGrayCorrect2,IntensityGrayCorrect3, outputImageGraph):
    plt.figure()
    plt.subplot(331)
    plt.plot(time_stamps, RoiGray1, label=f"ROI 1 - CRT ", color='darkgreen')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    plt.legend()
    
    plt.subplot(332)
    plt.plot(time_stamps, RoiGray2, label=f"ROI 2", color='red')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza', fontsize="12")
    plt.legend()
    
    plt.subplot(333)
    plt.plot(time_stamps, RoiGray3, label=f"ROI 3", color='orange')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza', fontsize="12")
    plt.legend()
    plt.tight_layout()
    
    plt.subplot(334)
    plt.plot(time_stamps, funçãoGanhoGray1, label=f"Função Ganho ROI 1", color='darkgreen')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    plt.title("Função ganho ponto máximo ")
    #plt.legend()
    
    plt.subplot(335)
    plt.plot(time_stamps, funçãoGanhoGray2, label=f"Função Ganho ROI 2", color='red')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    plt.title("Função ganho ponto máximo ")
    #plt.legend()
    
    
    plt.subplot(336)
    plt.plot(time_stamps, funçãoGanhoGray3, label=f"Função Ganho ROI 3", color='orange')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    plt.title("Função ganho ponto máximo ")
    #plt.legend()
    
    
    plt.subplot(337)
    plt.plot(time_stamps, IntensityGrayCorrect1, label=f"ROI 1 corrigida", color='darkgreen')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    plt.title("Intensidade Corrigida ")
    #plt.legend()
    
    plt.subplot(338)
    plt.plot(time_stamps, IntensityGrayCorrect2, label=f"ROI 1 corrigida pela ROI 2", color='red')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    plt.title("Intensidade Corrigida ")
    #plt.legend()
    
    
    plt.subplot(339)
    plt.plot(time_stamps, IntensityGrayCorrect3, label=f"ROI 1 corrigida pela ROI 3", color='orange')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    plt.title("Intensidade Corrigida ")
    #plt.legend()
    
    plt.savefig(outputImageGraph, dpi=300)
    plt.show()
    plt.close()


def plotCurveFGanhoJuntos(time_stamps,RoiGray1, RoiGray2, RoiGray3, funçãoGanhoGray1,funçãoGanhoGray2,funçãoGanhoGray3,
                  IntensityGrayCorrect1,IntensityGrayCorrect2,IntensityGrayCorrect3, outputImageGraph):
    plt.figure()
    plt.subplot(131)
    plt.plot(time_stamps, RoiGray1, label=f"ROI 1 - CRT ", color='darkgreen')
    plt.plot(time_stamps, IntensityGrayCorrect1, label=f"ROI 1 corrigida", color='darkgreen')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    #plt.legend()
    
    plt.subplot(132)
    plt.plot(time_stamps, RoiGray1, label=f"ROI 1 - CRT ", color='darkgreen')
    plt.plot(time_stamps, IntensityGrayCorrect2, label=f"ROI 1 corrigida pela ROI 2", color='red')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    plt.title("Intensidade Corrigida ")
    #plt.legend()
    
    
    plt.subplot(133)
    plt.plot(time_stamps, RoiGray1, label=f"ROI 1 - CRT ", color='darkgreen')
    plt.plot(time_stamps, IntensityGrayCorrect3, label=f"ROI 1 corrigida pela ROI 3", color='orange')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Cinza')
    plt.title("Intensidade Corrigida ")
    plt.legend()
    
    plt.savefig(outputImageGraph, dpi=300)
    plt.show()
    plt.close()


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


# Função para selecionar ROIs com o vídeo rodando
def selectROI():

    global roi1
    print("Pressione ENTER para pausar e selecionar ROI1.")
    while True:
        ret, frame = cap.read()
        #frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if not ret:
            print("Fim do vídeo ou erro na leitura do frame.")
            sys.exit(1)

        cv.imshow("Selecionar ROIs", frame)
        key = cv.waitKey(30) & 0xFF
        if key == 13:  # Tecla Enter
           
            roi1 = cv.selectROI("Selecionar ROI1", frame)
            print(roi1)
            cv.destroyAllWindows()
            
            """
            
            roi2 = cv.selectROI("Selecionar ROI2", frame)
            print(roi2)
            cv.destroyAllWindows()
            roi3 = cv.selectROI("Selecionar ROI3", frame)
            print(roi3)
            cv.destroyAllWindows()
            roi4 = cv.selectROI("Selecionar ROI4", frame)
            print(roi4)
            cv.destroyAllWindows()
         
            roi5 = cv.selectROI("Selecionar ROI5", frame)
            print(roi5)
            cv.destroyAllWindows()
            """
         
            
            break


def DerivadaDoSinal(sinal, time_stamps):
    """
    Calcula a derivada do sinal de CRT, identifica os picos positivos e negativos e plota 
    tanto o sinal original quanto sua derivada com os picos máximos destacados.

    :param sinal: Lista ou array contendo os valores do sinal.
    :param dt: Intervalo de tempo entre os pontos do sinal (assumido como constante).
    """
    dt = time_stamps[1] - time_stamps[0] 
    derivada = np.diff(sinal) / dt
    
  
    t = np.arange(0, len(sinal) * dt, dt)
    t_derivada = t[:-1]  
    
   
    picosPositivos = argrelextrema(derivada, np.greater)[0]
    picosNegativos = argrelextrema(derivada, np.less)[0]
    
    # Encontra o pico máximo positivo e o pico mínimo negativo
    if len(picosPositivos) > 0:
        maxPicoPositivo = picosPositivos[np.argmax(derivada[picosPositivos])]
    else:
        maxPicoPositivo = None 
    
    if len(picosNegativos) > 0:
        maxPicoNegativo = picosNegativos[np.argmin(derivada[picosNegativos])]
    else:
        maxPicoNegativo = None 
    
   
    plt.subplot(211)
    plt.plot(t, sinal)
    plt.title('Sinal Original')
    plt.xlabel('Tempo')
    plt.ylabel('Amplitude')
   
    plt.subplot(212)
    plt.plot(t_derivada, derivada, label='Derivada')
    plt.title('Derivada do Sinal')
    plt.xlabel('Tempo')
    plt.ylabel('Derivada da Amplitude')
    
   
    if maxPicoPositivo is not None:
        plt.plot(t_derivada[maxPicoPositivo], derivada[maxPicoPositivo], 'ro', label='Máximo Pico Positivo')
    
    if maxPicoNegativo is not None:
        plt.plot(t_derivada[maxPicoNegativo], derivada[maxPicoNegativo], 'go', label='Máximo Pico Negativo')
    
    plt.legend()  
    plt.tight_layout()
    plt.show()
    
    return maxPicoPositivo, maxPicoNegativo




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
    frames.append(frame)
    
    rois = create_dynamic_rois(frame, num_rois, roi_width, roi_height)
    for i, roi in enumerate(rois):
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
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

#Select ROI
#selectROI()

# Listas para armazenar intensidades e timestamps
RoiRed, RoiGreen, RoiBlue,Roi1, Roi2,Roi3,RoiGreen1Equ, EqualizadeGreenRoi1, EqualizedGrayRoi1,time_stamps = [], [], [], [],[],[],[],[],[],[]
frame_count = 0
# Processa o vídeo frame a frame para capturar os valores das ROIs
while True:
    ret, frame = cap.read()
    if not ret:
        break

   

    time_stamps.append(frame_count / fps)
    
    # Durante o processamento de cada frame, garanta que as ROIs sejam uint8
    roi1_frame = frame[int(roi1[1]):int(roi1[1] + roi1[3]), int(roi1[0]):int(roi1[0] + roi1[2])]
    roi2_frame = np.clip(frame[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])], 0, 255).astype(np.uint8)
    roi3_frame = np.clip(frame[int(roi3[1]):int(roi3[1] + roi3[3]), int(roi3[0]):int(roi3[0] + roi3[2])], 0, 255).astype(np.uint8)

    
    RoiGray1=cv.cvtColor(roi1_frame, cv.COLOR_BGR2GRAY)
    RoiGray2=cv.cvtColor(roi2_frame, cv.COLOR_BGR2GRAY)
    RoiGray3=cv.cvtColor(roi3_frame, cv.COLOR_BGR2GRAY)
    


    
    Roi1.append(np.mean(RoiGray1))
    Roi2.append(np.mean(RoiGray2))
    Roi3.append(np.mean(RoiGray3))
    
    
    frame_count += 1

cap.release()


RoiGray1 = np.array(Roi1)
RoiGray2 = np.array(Roi2)
RoiGray3 = np.array(Roi3)

time_stamps = np.array(time_stamps)


maxPicoPositivo, maxPicoNegativo = DerivadaDoSinal(RoiGray1,time_stamps)
print(maxPicoPositivo)
maxPicoPositivo=maxPicoPositivo+5
tempoMaxPicoPositivo = time_stamps[maxPicoPositivo]

# Teste 1 
#considerando I(0) no pico

funçãoGanhoGray1=RoiGray1/RoiGray1[maxPicoPositivo]
funçãoGanhoGray2=RoiGray2/RoiGray2[maxPicoPositivo]
funçãoGanhoGray3=RoiGray3/RoiGray3[maxPicoPositivo]

IntensityGrayCorrect1=RoiGray1/funçãoGanhoGray1
IntensityGrayCorrect2=RoiGray1/funçãoGanhoGray2
IntensityGrayCorrect3=RoiGray1/funçãoGanhoGray3

outputImageGraph = f"Teste1GrayscaleFGanho{folder_name}{video_name}.png"
plotCurveFGanho(time_stamps,RoiGray1, RoiGray2, RoiGray3, funçãoGanhoGray1,funçãoGanhoGray2,funçãoGanhoGray3,
                  IntensityGrayCorrect1,IntensityGrayCorrect2,IntensityGrayCorrect3, outputImageGraph)

outputImageGraphJuntos = f"Teste1GrayscaleFGanhoCombinado{folder_name}{video_name}.png"
plotCurveFGanhoJuntos(time_stamps,RoiGray1, RoiGray2, RoiGray3, funçãoGanhoGray1,funçãoGanhoGray2,funçãoGanhoGray3,
                  IntensityGrayCorrect1,IntensityGrayCorrect2,IntensityGrayCorrect3, outputImageGraphJuntos)


try:
    IntesityGrayROI1 = np.column_stack((RoiGray1, RoiGray1, RoiGray1))
    pcrtComp = PCRT(time_stamps, IntesityGrayROI1, exclusionMethod='best fit', exclusionCriteria=9999, fromTime=tempoMaxPicoPositivo)
    pcrtComp.showPCRTPlot()
    outputFilePCRT = f"pCRTGrayTeste1ROI1{video_name}.png"
    pcrtComp.savePCRTPlot(outputFilePCRT)
    print(pcrtComp)

except Exception as e:
    print("Erro no Teste 1 - ROI1: {e}")
    

try:
    IntesityGrayROI1CorrectROI2 = np.column_stack((IntensityGrayCorrect2, IntensityGrayCorrect2, IntensityGrayCorrect2))
    pcrtComp = PCRT(time_stamps, IntesityGrayROI1CorrectROI2, exclusionMethod='best fit', exclusionCriteria=9999, fromTime=tempoMaxPicoPositivo)
    pcrtComp.showPCRTPlot()
    outputFilePCRT = f"pCRTGrayTeste1ROI1CorrectROI2{video_name}.png"
    pcrtComp.savePCRTPlot(outputFilePCRT)
    print(pcrtComp)
except Exception as e:
    print(f"Erro no Teste 1 - ROI2: {e}")

try:
    IntesityGrayROI1CorrectROI3 = np.column_stack((IntensityGrayCorrect3, IntensityGrayCorrect3, IntensityGrayCorrect3))
    pcrtComp = PCRT(time_stamps, IntesityGrayROI1CorrectROI3, exclusionMethod='best fit', exclusionCriteria=9999, fromTime=tempoMaxPicoPositivo)
    pcrtComp.showPCRTPlot()
    outputFilePCRT = f"pCRTGrayTeste1ROI1CorrectROI3{video_name}.png"
    pcrtComp.savePCRTPlot(outputFilePCRT)
    print(pcrtComp)
except Exception as e:
    print(f"Erro no Teste 1 - ROI3: {e}")


#outputImageGraph = f"TesteGrayscale{folder_name}{video_name}.png"
#plot_graph_curves(time_stamps,RoiGray1, RoiGray2,  outputImageGraph)

# Teste 2
#considerando I(0) no inicio do vídeo

funçãoGanhoGray1=RoiGray1/RoiGray1[10]
funçãoGanhoGray2=RoiGray2/RoiGray2[10]
funçãoGanhoGray3=RoiGray3/RoiGray3[10]

IntensityGrayCorrect1=RoiGray1/funçãoGanhoGray1
IntensityGrayCorrect2=RoiGray1/funçãoGanhoGray2
IntensityGrayCorrect3=RoiGray1/funçãoGanhoGray3

outputImageGraph = f"Teste2GrayscaleFGanho{folder_name}{video_name}.png"
plotCurveFGanho(time_stamps,RoiGray1, RoiGray2, RoiGray3, funçãoGanhoGray1,funçãoGanhoGray2,funçãoGanhoGray3,
                  IntensityGrayCorrect1,IntensityGrayCorrect2,IntensityGrayCorrect3, outputImageGraph)

try:
    IntesityGrayROI1 = np.column_stack((RoiGray1, RoiGray1, RoiGray1))
    pcrtComp = PCRT(time_stamps, IntesityGrayROI1, exclusionMethod='best fit', exclusionCriteria=9999, fromTime=tempoMaxPicoPositivo)
    pcrtComp.showPCRTPlot()
    outputFilePCRT = f"pCRTGrayTeste2ROI1{video_name}.png"
    pcrtComp.savePCRTPlot(outputFilePCRT)
    print(pcrtComp)

except Exception as e:
    print("Erro no Teste 2 - ROI1: {e}")

try:
    IntesityGrayROI1CorrectROI2 = np.column_stack((IntensityGrayCorrect2, IntensityGrayCorrect2, IntensityGrayCorrect2))
    pcrtComp = PCRT(time_stamps, IntesityGrayROI1CorrectROI2, exclusionMethod='best fit', exclusionCriteria=9999, fromTime=tempoMaxPicoPositivo)
    pcrtComp.showPCRTPlot()
    outputFilePCRT = f"pCRTGrayTeste2ROI1CorrectROI2{video_name}.png"
    pcrtComp.savePCRTPlot(outputFilePCRT)
    print(pcrtComp)
except Exception as e:
    print(f"Erro no Teste 2 - ROI2: {e}")

try:
    IntesityGrayROI1CorrectROI3 = np.column_stack((IntensityGrayCorrect3, IntensityGrayCorrect3, IntensityGrayCorrect3))
    pcrtComp = PCRT(time_stamps, IntesityGrayROI1CorrectROI3, exclusionMethod='best fit', exclusionCriteria=9999, fromTime=tempoMaxPicoPositivo)
    pcrtComp.showPCRTPlot()
    outputFilePCRT = f"pCRTGrayTeste2ROI1CorrectROI3{video_name}.png"
    pcrtComp.savePCRTPlot(outputFilePCRT)
    print(pcrtComp)
except Exception as e:
    print(f"Erro no Teste 2 - ROI3: {e}")


# Teste 3
#considerando I(0) no final do video 

funçãoGanhoGray1=RoiGray1/RoiGray1[400]
funçãoGanhoGray2=RoiGray2/RoiGray2[400]
funçãoGanhoGray3=RoiGray3/RoiGray3[400]

IntensityGrayCorrect1=RoiGray1/funçãoGanhoGray1
IntensityGrayCorrect2=RoiGray1/funçãoGanhoGray2
IntensityGrayCorrect3=RoiGray1/funçãoGanhoGray3

outputImageGraph = f"Teste3GrayscaleFGanho{folder_name}{video_name}.png"
plotCurveFGanho(time_stamps,RoiGray1, RoiGray2, RoiGray3, funçãoGanhoGray1,funçãoGanhoGray2,funçãoGanhoGray3,
                  IntensityGrayCorrect1,IntensityGrayCorrect2,IntensityGrayCorrect3, outputImageGraph)

try:
    IntesityGrayROI1 = np.column_stack((RoiGray1, RoiGray1, RoiGray1))
    pcrtComp = PCRT(time_stamps, IntesityGrayROI1, exclusionMethod='best fit', exclusionCriteria=9999, fromTime=tempoMaxPicoPositivo)
    pcrtComp.showPCRTPlot()
    outputFilePCRT = f"pCRTGrayTeste3ROI1{video_name}.png"
    pcrtComp.savePCRTPlot(outputFilePCRT)
    print(pcrtComp)

except Exception as e:
 
    print("Erro no Teste 3 - ROI1: {e}")

try:
    IntesityGrayROI1CorrectROI2 = np.column_stack((IntensityGrayCorrect2, IntensityGrayCorrect2, IntensityGrayCorrect2))
    pcrtComp = PCRT(time_stamps, IntesityGrayROI1CorrectROI2, exclusionMethod='best fit', exclusionCriteria=9999, fromTime=tempoMaxPicoPositivo)
    pcrtComp.showPCRTPlot()
    outputFilePCRT = f"pCRTGrayTeste3ROI1CorrectROI2{video_name}.png"
    pcrtComp.savePCRTPlot(outputFilePCRT)
    print(pcrtComp)
except Exception as e:
    print(f"Erro no Teste 3 - ROI2: {e}")

try:
    IntesityGrayROI1CorrectROI3 = np.column_stack((IntensityGrayCorrect3, IntensityGrayCorrect3, IntensityGrayCorrect3))
    pcrtComp = PCRT(time_stamps, IntesityGrayROI1CorrectROI3, exclusionMethod='best fit', exclusionCriteria=9999, fromTime=tempoMaxPicoPositivo)
    pcrtComp.showPCRTPlot()
    outputFilePCRT = f"pCRTGrayTeste3ROI1CorrectROI3{video_name}.png"
    pcrtComp.savePCRTPlot(outputFilePCRT)
    print(pcrtComp)
except Exception as e:
    print(f"Erro no Teste 3 - ROI3: {e}")

