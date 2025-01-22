# atualização dia 06/01/2024 - implemntação do histograma e derivada da função

import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import combinations

import pandas as pd

from scipy.optimize import minimize
from scipy.signal import argrelextrema


#sys.path.append("C:/Users/Fotobio/Documents/GitHub/pyCRT") #PC casa 
#sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT") # PC lab
sys.path.append("C:/Users/raque/OneDrive/Documentos/GitHub/pyCRT") # note

from src.pyCRT import PCRT  

# Caminho base para os arquivos do projeto

#base_path = "C:/Users/RaquelPantojo/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"  # PC USP
#base_path="C:/Users/Fotobio/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"#PC casa 
#base_path="C:/Users/Fotobio/Documents/GitHub"
base_path="C:/Users/raque/OneDrive/Documentos/GitHub" # note
folder_name = "CorrectLightUnpol"


#video_name="v5.mp4"
#video_name="v7.mp4"
#video_name = "corrected_v7_gamma=1.mp4"

#video_name = "SeiyLedDesp4.mp4"
#video_name ="SeiyLedPol6.mp4"

#video_name = "NatanLedDesp6.mp4"
#video_name="NatanledPol5.mp4"


video_name = "EduLedDesp4.mp4"

roi1=(710, 121, 131, 150)

roi_width = 80 
roi_height = 80
num_rois = 200  # Número de ROIs a serem criadas
#gamma = 1.53

def plot_image_and_ratios(frames,roi1, roi2, roi3, roi4, time_stamps,time_stamps_RoiGreen2,time_stamps_RoiGreen3,time_stamps_RoiGreen4, RoiGreen1_selected,RoiGreen2_aligned, RoiGreen3_aligned,RoiGreen4_aligned, folder_name):

#def plot_image_and_ratios(frames, roi1, roi2, roi3, roi4, time_stamps, RoiGreen1, RoiGreen2, RoiGreen3, RoiGreen4, folder_name):
    # Extrair coordenadas das ROIs
    x_roi1, y_roi1, w_roi1, h_roi1 = roi1
    x_i, y_i, w_i, h_i = roi2
    x_j, y_j, w_j, h_j = roi3
    x_k, y_k, w_k, h_k = roi4

    # Criar figura e subplots
    fig, axs = plt.subplots(4, 2, figsize=(18, 16))
    axs = axs.flatten()  # Transformar em lista para acesso linear

    # Subplot (4, 2, 1) - Imagem original com ROIs
    axs[0].imshow(cv.cvtColor(frames[0], cv.COLOR_BGR2RGB))
    axs[0].set_title('Imagem Original')
    axs[0].axis('off')
    axs[0].add_patch(plt.Rectangle((x_roi1,y_roi1),w_roi1,h_roi1,edgecolor='green',facecolor='none',linewidth=2))
    axs[0].add_patch(plt.Rectangle((x_i, y_i), w_i, h_i, edgecolor='blue', facecolor='none', linewidth=2))
    axs[0].add_patch(plt.Rectangle((x_j, y_j), w_j, h_j, edgecolor='red', facecolor='none', linewidth=2))
    axs[0].add_patch(plt.Rectangle((x_k, y_k), w_k, h_k, edgecolor='orange', facecolor='none', linewidth=2))

    axs[0].text(x_roi1 + w_roi1 / 2, y_roi1 + h_roi1 / 2, f"1", color='green', ha='center', va='center', fontsize=8, weight='bold')
    axs[0].text(x_i + w_i / 2, y_i + h_i / 2, f"2", color='blue', ha='center', va='center', fontsize=8, weight='bold')
    axs[0].text(x_j + w_j / 2, y_j + h_j / 2, f"3", color='red', ha='center', va='center', fontsize=8, weight='bold')
    axs[0].text(x_k + w_k / 2, y_k + h_k / 2, f"4", color='orange', ha='center', va='center', fontsize=8, weight='bold')

    # Subplot (4, 2, 2) - ROI 1
    axs[1].plot(time_stamps, RoiGreen1_selected, label="ROI 1", color='green')
    #axs[1].set_xlabel('Tempo (s)')
    #axs[1].set_ylim(100, 255)
    axs[1].set_ylabel('Intensidade do Canal Verde')
    axs[1].legend()

    # Subplot (4, 2, 4) - ROI 2
    axs[3].plot(time_stamps_RoiGreen2, RoiGreen2_aligned, label="ROI 2", color='blue')
    #axs[3].set_xlabel('Tempo (s)')
    #axs[3].set_ylim(100, 255)
    axs[3].set_ylabel('Intensidade do Canal Verde')
    axs[3].legend()

    # Subplot (4, 2, 6) - ROI 3
    axs[5].plot(time_stamps_RoiGreen3, RoiGreen3_aligned, label="ROI 3", color='red')
    #axs[5].set_xlabel('Tempo (s)')
    #axs[5].set_ylim(100, 255)
    axs[5].set_ylabel('Intensidade do Canal Verde')
    axs[5].legend()

    # Subplot (4, 2, 8) - ROI corrigida
    axs[7].plot(time_stamps_RoiGreen4, RoiGreen4_aligned, label="ROI 4", color='orange')
    axs[7].set_xlabel('Tempo (s)')
    axs[7].set_ylabel('Intensidade do Canal Verde')
    axs[7].legend()

    # Desativar subplots extras
    for i in range(len(axs)):
        if i not in [0, 1, 3, 5, 7]:
            axs[i].axis('off')

    # Ajustar layout e salvar
    plt.tight_layout()
    #output_filename = f"ROI={a:.2f}_b={b:.2f}_g={gamma:.2f}_{folder_name}.png"
    #plt.savefig(output_filename, dpi=600)
    plt.show()
    plt.close()


def plot_graph_curves(RoiGreen, ROI1Corrigida,SinalFinal,outputImageGraph):
    
    plt.subplot(311)
    plt.plot(time_stamps, RoiGreen, label=f"Canal Verde ROI 1 ", color='darkgreen')
    plt.xlabel('Tempo (s)')
    #plt.ylabel('Intensidade do Canal Verde')
    plt.legend()

    plt.subplot(312)
    plt.plot(time_stamps, ROI1Corrigida, label=f"G(t)", color='green')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Verde', fontsize="12")
    plt.legend()
    
    plt.subplot(313)
    plt.plot(time_stamps, SinalFinal, label=f"Canal Corrigido", color='green')
    #plt.title(f"a={a:.2f}_b={b:.2f} $\gamma$={gamma:.2f}")
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal ')
    plt.legend()
    plt.tight_layout()
   #pastaSalvar= "C:/Users/Fotobio/Desktop/ResultadosAbgamma"
    plt.savefig(outputImageGraph, dpi=300)
    plt.show()
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


# Função para calcular as razões 
def calculate_ratios(roi2, roi3):
    roi2_array = np.array(roi2)
    roi3_array = np.array(roi3)
    
    ratios = roi2_array / roi3_array
    #print(ratios)
    return ratios.tolist() 



def find_best_a_b(green_roi2, green_roi3):
    """
    Encontra os melhores parâmetros a, b e gamma que minimizam o erro absoluto médio
    através de uma busca exaustiva.

    Args:
        green_roi2 (array-like): Vetor de intensidades da ROI 2.
        green_roi3 (array-like): Vetor de intensidades da ROI 3.
        a_range (tuple): Intervalo para 'a' (min, max, step).
        b_range (tuple): Intervalo para 'b' (min, max, step).
        gamma_range (tuple): Intervalo para 'gamma' (min, max, step).

    Returns:
        tuple: a_otimizado, b_otimizado, gamma_otimizado, adjusted_ratio.
    """
    a_range = (0, 255, 1)
    b_range = (0.1, 100.1, 1)
    gamma_range = (0.2, 2.2, 0.2)
    
    # Inicializar os melhores valores
    best_a, best_b, best_gamma = None, None, None
    best_value = float('inf')

    # Gerar os valores discretos para 'a', 'b' e 'gamma'
    a_values = np.arange(*a_range)
    b_values = np.arange(*b_range)
    gamma_values = np.arange(*gamma_range)

    # Iterar sobre todos os valores possíveis de a, b e gamma
    for a in a_values:
        for b in b_values:
            for gamma in gamma_values:
                # Calcular a razão ajustada
                adjusted_ratio = (a + b * (green_roi2 ** gamma)) / (a + b * (green_roi3 ** gamma))
                # Calcular o erro absoluto médio
                #mean_abs_error = np.mean(np.abs(adjusted_ratio - 1))
                std_error = np.std(adjusted_ratio)
                #print(f"a={a}, b={b}, gamma={gamma}, std_error={std_error}")

                # Verificar se este é o melhor valor encontrado
                if std_error < best_value:
                    best_value = std_error
                    best_a, best_b, best_gamma = a, b, gamma

    # Calcular a razão ajustada final com os melhores parâmetros
    adjusted_ratio = (best_a + best_b * (green_roi2 ** best_gamma)) / (best_a + best_b * (green_roi3 ** best_gamma))
    return best_a, best_b, best_gamma, adjusted_ratio


def find_best_gamma(green_roi2, green_roi3):
    
    #Encontra o melhor parâmetro gamma, fixando a = 0 e b = 1.
    
    def funcao_objetivo(params, green_roi2, green_roi3):
        gamma = params[0]  # Apenas gamma será otimizado
        
        # Calcular o adjusted_ratio com a = 0 e b = 1
        adjusted_ratio = (0 + 1 * (green_roi2 ** gamma)) / (0 + 1 * (green_roi3 ** gamma))
        
        # Calcular o erro absoluto médio
        mean_abs_error = np.mean(np.abs(adjusted_ratio - 1))
        
        return mean_abs_error

    def restricao_std_error(params, green_roi2, green_roi3):
        gamma = params[0]  # Apenas gamma será otimizado
        
        # Calcular o adjusted_ratio com a = 0 e b = 1
        adjusted_ratio = (0 + 1 * (green_roi2 ** gamma)) / (0 + 1 * (green_roi3 ** gamma))
        
        # Calcular o erro padrão
        std_error = np.std(adjusted_ratio)
        
        return  std_error  # erro padrão deve ser menor que 0.5


    x0 = np.array([1.0])  # Apenas o valor de gamma será otimizado 0.1 e 5 

    # Definindo os limites para gamma, com valor inicial maior que 0
    bounds = [(0.1, 10)]  # gamma > 0

    restricoes = [
        {'type': 'ineq', 'fun': restricao_std_error, 'args': (green_roi2, green_roi3)},  
        #{'type': 'ineq', 'fun': restricao_mean_abs_error, 'args': (green_roi2, green_roi3)}
    ]

    # Executando a otimização
    resultado = minimize(funcao_objetivo, x0, args=(green_roi2, green_roi3), method='SLSQP', bounds=bounds, constraints=restricoes)

    # Exibindo os resultados
    gamma_otimizado = resultado.x[0]

    # Calcular o adjusted_ratio com gamma otimizado
    adjusted_ratio = (0 + 1 * (green_roi2 ** gamma_otimizado)) / (0 + 1 * (green_roi3 ** gamma_otimizado))
    aFixo = 0
    Bfixo = 1
    return aFixo,Bfixo, gamma_otimizado, adjusted_ratio




# Função para selecionar ROIs com o vídeo rodando
def select_rois():

    global roi1,roi2,roi3,roi4,roi5
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
            
            roi2 = cv.selectROI("Selecionar ROI2", frame)
            print(roi2)
            cv.destroyAllWindows()
            roi3 = cv.selectROI("Selecionar ROI3", frame)
            print(roi3)
            cv.destroyAllWindows()
            roi4 = cv.selectROI("Selecionar ROI4", frame)
            print(roi4)
            cv.destroyAllWindows()
            """
            roi5 = cv.selectROI("Selecionar ROI5", frame)
            print(roi5)
            cv.destroyAllWindows()
            """
         
            
            break
  


############################ Inicalizando o programa ####################################
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
#roi1, roi2, roi3, roi4, roi5 = None


frame_count = 0
fps = cap.get(cv.CAP_PROP_FPS)



cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Listas para armazenar intensidades e timestamps
RoiGreen1, RoiGreen2, RoiGreen3, RoiGreen4,RoiGreen5, RoiGray1,RoiRed1,RoiBlue1,time_stamps = [], [], [], [], [], [], [],[], []
frames = []
# Processa o vídeo frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    # frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)   
    frames.append(frame) 
    
    
    # Extrai as ROIs e calcula a média do canal verde
    roi1_frame = frame[int(roi1[1]):int(roi1[1] + roi1[3]), int(roi1[0]):int(roi1[0] + roi1[2])] # ROI CRT
    #roi2_frame = frame[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])]
    #roi3_frame = frame[int(roi3[1]):int(roi3[1] + roi3[3]), int(roi3[0]):int(roi3[0] + roi3[2])]
    #roi4_frame = frame[int(roi4[1]):int(roi4[1] + roi4[3]), int(roi4[0]):int(roi4[0] + roi4[2])]  

    RoiRed1.append(np.mean(roi1_frame[:,:,0]))
    RoiGreen1.append(np.mean(roi1_frame[:,:,1]))
    RoiBlue1.append(np.mean(roi1_frame[:,:,2]))
    
    # ROI Gray
    roiGray1 = cv.cvtColor(roi1_frame, cv.COLOR_BGR2GRAY)
    RoiGray1.append(np.mean(roiGray1))
    
    #roiGray2 = cv.cvtColor(roi2_frame, cv.COLOR_BGR2GRAY)
    #RoiGreen2.append(np.mean(roiGray2))
    
    # ROI fora 
    #roiGray3 = cv.cvtColor(roi3_frame, cv.COLOR_BGR2GRAY)
    #RoiGreen3.append(np.mean(roiGray3))

    # ROI fora 
    #roiGray4 = cv.cvtColor(roi4_frame, cv.COLOR_BGR2GRAY)
    #RoiGreen4.append(np.mean(roiGray4))
    

    #YUV
    #roi1YUV = cv.cvtColor(roi1_frame, cv.COLOR_BGR2YUV)
    #roi1YUV.append(np.mean(roi1YUV[:,:,1]))
    
    

    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()




RoiGreen1=np.array(RoiGreen1)
#RoiGreen2=np.array(RoiGreen2)
#RoiGreen3=np.array(RoiGreen3)
#RoiGreen4=np.array(RoiGreen4)
roiGray1=np.array(RoiGray1)

#roi1YUV = np.array(roi1YUV)

# g=G/(R+B+G)
RoiRed1 = np.array(RoiRed1)
RoiBlue1 = np.array(RoiBlue1)

SomeRGB = RoiRed1+RoiGreen1+RoiBlue1
IntensityGNormalized= RoiGreen1/roiGray1
IntensityGNormalized=np.array(IntensityGNormalized)

time_stamps = np.array(time_stamps)







def DerivadaMeiaAltura(IntesityChannel, time_stamps, roi1YUV,NameFigDerivada):
    """
    Calcula a derivada do sinal, identifica os picos positivos e negativos, 
    e encontra o ponto de meia altura do pico positivo deslocado para frente.
    
    :param sinal: Lista ou array contendo os valores do sinal.
    :param time_stamps: Lista ou array com os timestamps correspondentes ao sinal.
    :return: Índice do pico positivo, índice do ponto de meia altura.
    """
    # Calcula a derivada
    dt = time_stamps[1] - time_stamps[0]
    derivada = np.diff(IntesityChannel) / dt
    
    # Ajusta os eixos de tempo
    t = time_stamps
    t_derivada = t[:-1]
    
    # Identifica picos locais positivos e negativos
    picosPositivos = argrelextrema(derivada, np.greater)[0]
    
    # Determina o pico máximo positivo
    if len(picosPositivos) > 0:
        maxPicoPositivo = picosPositivos[np.argmax(derivada[picosPositivos])]
    else:
        maxPicoPositivo = None
    
    # Calcula o ponto de meia altura do pico máximo positivo
    meio_altura_index = None
    if maxPicoPositivo is not None:
        altura_meio = 0.5 * derivada[maxPicoPositivo]
        
        # Busca o primeiro ponto após o pico com valor igual ou menor que a meia altura
        for i in range(maxPicoPositivo, len(derivada)):
            if derivada[i] <= altura_meio:
                meio_altura_index = i
                break
    
    # Plota o sinal e a derivada
    plt.figure(figsize=(10, 6))
    
    plt.subplot(311)
    plt.plot(t, IntesityChannel,label='Canal Verde ROI 1')
    plt.plot(t[maxPicoPositivo],IntesityChannel[maxPicoPositivo],'ro', label='Pico Derivada')
    plt.plot(t[meio_altura_index],IntesityChannel[meio_altura_index],'bo', label='Pico derivada deslocado')
    plt.xlabel('Time')
    plt.ylabel('Average intensity')
    plt.legend()
    
    plt.subplot(312)
    plt.plot(time_stamps,roi1YUV, label='Corrigido')
    plt.xlabel('Time')
    plt.ylabel('Average intensity')
    plt.legend()
    
    plt.subplot(313)
    plt.plot(t_derivada, derivada, label='Derivada')
    plt.title('Derivada do Sinal')
    plt.xlabel('Tempo')
    plt.ylabel('Derivada da Amplitude')
    plt.plot(t_derivada[maxPicoPositivo], derivada[maxPicoPositivo], 'ro', label='Máximo Pico Positivo')
    plt.plot(t_derivada[meio_altura_index], derivada[meio_altura_index], 'bo', label='Meia Altura (Deslocado)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(NameFigDerivada, dpi=300) 
    plt.show()
    
    return maxPicoPositivo, meio_altura_index

    
 
    
# os picos positivos e negativos da derivada do sinal de CRT
# o pico positivo é o momento que se tem a retirado do dedo
# o pico negativo é o momento que se aplica a compressão
NameFigDerivada =f"DerivadaverdeGray{video_name}.png"
maxPicoPositivo, meio_altura_index = DerivadaMeiaAltura(RoiGreen1,time_stamps,IntensityGNormalized,NameFigDerivada)
tempoMaxPicoPositivo = time_stamps[meio_altura_index]


plt.figure(figsize=(10, 6))   
plt.subplot(311)
plt.plot(time_stamps, RoiGreen1, label='Canal Verde ROI 1', color='darkgreen')
plt.xlabel('Time')
plt.ylabel('Average intensity')
plt.legend()

plt.subplot(312)
plt.plot(time_stamps,RoiGray1, label='Canal Cinza ROI 1', color='gray')
plt.xlabel('Time')
plt.ylabel('Average intensity')
plt.legend()

plt.subplot(313)
plt.plot(time_stamps, IntensityGNormalized, label='Green/Gray', color='orange')
plt.xlabel('Time')
plt.ylabel('Average intensity')
plt.legend()


plt.tight_layout()
NameFigDerivada =f"verdeGray{video_name}.png"
plt.savefig(NameFigDerivada, dpi=300)
plt.show()



#print(maxPicoPositivo)
#print(meio_altura_index)


ratiosC=np.column_stack((RoiGreen1,RoiGreen1,RoiGreen1))
pcrtComp = PCRT(time_stamps, ratiosC,exclusionMethod='best fit',exclusionCriteria=9999,fromTime=tempoMaxPicoPositivo)

pcrtComp.showPCRTPlot()
outputFilePCRT = f"verdeGrayPCRT{video_name}.png"
pcrtComp.savePCRTPlot(outputFilePCRT)
print(pcrtComp)

