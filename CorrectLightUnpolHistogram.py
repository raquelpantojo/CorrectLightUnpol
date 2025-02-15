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

#folder_name = "teste1"
#folder_name="CorrectLightUnpol"
#video_name="v5.mp4"
#video_name = "SeiyLedDesp4.mp4"
#
video_name = "NatanLedDesp6.mp4"
#video_name ="SeiyLedPol6.mp4"
#video_name="NatanledPol5.mp4"
#video_name = "corrected_v7_gamma=1.mp4"



folder_name = "CorrectLightUnpol"
#video_name="corrected_v7_gamma=1.mp4"
#video_name="raqueltestecasa.mp4"
#video_name="v7.mp4"
#video_name ="SeiyLedDesp4.mp4"


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


#roi1=(1121, 222, 154, 154) #v7 original
#roi2=(1810, 9, 41, 93)
#roi3=(43, 22, 94, 82)
#roi4=(1794, 744, 76, 84)
#roi5=(453, 199, 84, 82)

roi1= (457, 571, 165, 215)
roi2= (209, 81, 102, 97)
roi3=(694, 139, 83, 73)
roi4=(1409, 22, 121, 70)


cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Listas para armazenar intensidades e timestamps
RoiGreen1, RoiGreen2, RoiGreen3, RoiGreen4,RoiGreen5, RoiGray,time_stamps = [], [], [], [], [], [], []
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
    roi2_frame = frame[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])]
    roi3_frame = frame[int(roi3[1]):int(roi3[1] + roi3[3]), int(roi3[0]):int(roi3[0] + roi3[2])]
    roi4_frame = frame[int(roi4[1]):int(roi4[1] + roi4[3]), int(roi4[0]):int(roi4[0] + roi4[2])]  
    #roi5_frame = frame[int(roi5[1]):int(roi5[1] + roi5[3]), int(roi5[0]):int(roi5[0] + roi5[2])] 

    # ROI Gray
    roiGray1 = cv.cvtColor(roi1_frame, cv.COLOR_BGR2GRAY)
    RoiGreen1.append(np.mean(roiGray1))
    
    roiGray2 = cv.cvtColor(roi2_frame, cv.COLOR_BGR2GRAY)
    RoiGreen2.append(np.mean(roiGray2))
    
    # ROI fora 
    roiGray3 = cv.cvtColor(roi3_frame, cv.COLOR_BGR2GRAY)
    RoiGreen3.append(np.mean(roiGray3))

    # ROI fora 
    roiGray4 = cv.cvtColor(roi4_frame, cv.COLOR_BGR2GRAY)
    RoiGreen4.append(np.mean(roiGray4))
    

    # ROI fora 
    
    #
    #RoiGreen1.append(np.mean(roi1_frame[:, :, 1]))
    #RoiGreen2.append(np.mean(roi2_frame[:, :, 1]))
    #RoiGreen3.append(np.mean(roi3_frame[:, :, 1]))
    #RoiGreen4.append(np.mean(roi4_frame[:, :, 1]))   
    #RoiGreen5.append(np.mean(roi5_frame[:, :, 1]))    


    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()




RoiGreen1=np.array(RoiGreen1)
RoiGreen2=np.array(RoiGreen2)
RoiGreen3=np.array(RoiGreen3)
RoiGreen4=np.array(RoiGreen4)

# ponto B de máxima intensidade 
max_index = np.argmax(RoiGreen1)
#print(max_index)

# função ganho 
#fGanho=RoiGreen2/RoiGreen2[max_index]
#print(fGanho)

# função ganho com a ROI2 deslocada - depois da retirada do dedo  
#max_index = np.argmax(RoiGreen1[300:])
#print(max_index)

#print(RoiGreen2[max_index+300])
#fGanho=RoiGreen2/RoiGreen2[max_index+300]

# função ganho aplicado na ROI 1 
#SinalFinal=RoiGreen1/fGanho

maxindex = np.argmax(RoiGreen1)  # Índice com maior intensidade em RoiGreen1
final_index = min(maxindex + 300, len(RoiGreen1))  # Garantir que não ultrapassa o comprimento da lista

# Selecionar apenas os valores relevantes
RoiGreen1_selected = RoiGreen1[maxindex:final_index]
RoiGreen2_selected = RoiGreen2[maxindex:final_index]
RoiGreen3_selected = RoiGreen3[maxindex:final_index]
RoiGreen4_selected = RoiGreen4[maxindex:final_index]
time_stamps = time_stamps[maxindex:final_index]




def align_curve_from_peak(curve):
    # Encontra o índice do pico máximo
    peak_index = np.argmax(curve)
    
    # Retorna a curva a partir do pico máximo
    return curve[peak_index:]

def adjust_time_stamps_from_peak(curve, original_time_stamps):
    peak_index = np.argmax(curve)
    
    # Ajusta os timestamps para começar a partir do pico máximo
    adjusted_time_stamps = original_time_stamps[peak_index:]
    
    return adjusted_time_stamps


# Alinha as outras curvas com base no pico de RoiGreen1
RoiGreen2_aligned = align_curve_from_peak(RoiGreen2_selected)
RoiGreen3_aligned = align_curve_from_peak(RoiGreen3_selected)
RoiGreen4_aligned = align_curve_from_peak(RoiGreen4_selected)

# Ajusta os timestamps a partir de seus picos máximos
time_stamps_RoiGreen2 = adjust_time_stamps_from_peak(RoiGreen2_selected, time_stamps)
time_stamps_RoiGreen3 = adjust_time_stamps_from_peak(RoiGreen3_selected, time_stamps)
time_stamps_RoiGreen4 = adjust_time_stamps_from_peak(RoiGreen4_selected, time_stamps)





#plot_image_and_ratios(frames, a, b, gamma, roi1, roi2, roi3, roi4, time_stamps, RoiGreen1, RoiGreen2, RoiGreen3, RoiGreen4, RoiGreen1Corrigido, folder_name)


#(sinal - ruido)/branco
#SinalFinal = (np.array(RoiGreen1)-np.array(RoiGreen3))/meanRatiosRoi1
#SinalFinal = (np.array(RoiGreen1)-ruido)/meanRatiosRoi1
#SinalFinal=((RoiGreen1-(-1*(np.array(RoiGreen4))))/np.array(RoiGreen2))
#SinalFinal=(RoiGreen1Corrigido/RoiGreen3Corrigido)
#SinalFinal=RoiGreen1Corrigido-(-1*RoiGreen4Corrigido)/branco
#SinalFinal=RoiGreen1Corrigido/RoiGreen4Corrigido

#print(f"a={a} b={b} gamma={gamma}")
#ROICorrigida = np.array(RoiGreen1) 
#time_stamps=np.array(time_stamps)

#outputImageGraph = f"TesteGrayFuncaoGanho{folder_name}.png"
#plot_graph_curves(RoiGreen2, fGanho, SinalFinal, outputImageGraph)


def apply_fourier_transform(time_stamps, roi_data, cutoff_frequency=None):
    """
    Aplica a Transformada de Fourier (DFT) e filtra as frequências com base em um corte de frequência.
    
    Args:
        time_stamps: Lista de timestamps (tempo em segundos).
        roi_data: Dados da ROI (intensidade do canal verde ao longo do tempo).
        cutoff_frequency: Frequência de corte para o filtro passa-baixa.
        
    Returns:
        Sinal filtrado no domínio do tempo.
    """
    # Número de pontos no sinal
    N = len(roi_data)
    
    # Calcular a Transformada de Fourier (DFT)
    X = np.fft.fft(roi_data)
    
    # Plotar o espectro de frequências
    frequencies = np.fft.fftfreq(N, time_stamps[1] - time_stamps[0])  # Frequências associadas
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies[:N//2], np.abs(X)[:N//2])  # Apenas as frequências positivas
    plt.title("Espectro de Frequências")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()
    
    # Se uma frequência de corte for especificada, aplicar o filtro passa-baixa
    if cutoff_frequency is not None:
        # Atenuar frequências acima do corte
        X[np.abs(frequencies) > cutoff_frequency] = 0

    # Calcular a Transformada Inversa de Fourier (IDFT)
    roi_filtered = np.fft.ifft(X)
    
    return np.real(roi_filtered)  # Retornar apenas a parte real

def plot_signals(time_stamps, original_signal, filtered_signal, roi_label):
    """
    Plota o sinal original e o sinal filtrado no domínio do tempo.
    
    Args:
        time_stamps: Lista de timestamps (tempo em segundos).
        original_signal: Sinal original (domínio do tempo).
        filtered_signal: Sinal filtrado (domínio do tempo).
        roi_label: Rótulo para a ROI.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time_stamps, original_signal, label="Sinal Original", color='blue', alpha=0.7)
    plt.plot(time_stamps, filtered_signal, label="Sinal Filtrado", color='red', alpha=0.7)
    plt.title(f"Filtragem por Transformada de Fourier para {roi_label}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Intensidade")
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_roi_with_fourier(RoiGreen1, RoiGreen2, RoiGreen3, RoiGreen4, time_stamps, cutoff_frequency=None):
    """
    Aplica a Transformada de Fourier e filtra as frequências de ruído em cada ROI.
    
    Args:
        RoiGreen1, RoiGreen2, RoiGreen3, RoiGreen4: Dados das ROIs no domínio do tempo.
        time_stamps: Lista de timestamps (tempo em segundos).
        cutoff_frequency: Frequência de corte para o filtro passa-baixa.
    """
    # Aplicar a transformada de Fourier e filtro para cada ROI
    RoiGreen1_filtered = apply_fourier_transform(time_stamps, RoiGreen1, cutoff_frequency)
    #RoiGreen2_filtered = apply_fourier_transform(time_stamps, RoiGreen2, cutoff_frequency)
    #RoiGreen3_filtered = apply_fourier_transform(time_stamps, RoiGreen3, cutoff_frequency)
    #RoiGreen4_filtered = apply_fourier_transform(time_stamps, RoiGreen4, cutoff_frequency)
    
    # Plotar os resultados
    plot_signals(time_stamps, RoiGreen1, RoiGreen1_filtered, "ROI 1")
    #plot_signals(time_stamps, RoiGreen2, RoiGreen2_filtered, "ROI 2")
    #plot_signals(time_stamps, RoiGreen3, RoiGreen3_filtered, "ROI 3")
    #plot_signals(time_stamps, RoiGreen4, RoiGreen4_filtered, "ROI 4")



# Chamar a função para analisar as ROIs com uma frequência de corte (ex: 1 Hz)
#analyze_roi_with_fourier(SinalFinal, RoiGreen2, RoiGreen3, RoiGreen4, time_stamps, cutoff_frequency=12)









######################################################
#plot_image_and_ratios(frames,roi1, roi2, roi3, roi4, time_stamps,time_stamps_RoiGreen2,time_stamps_RoiGreen3,time_stamps_RoiGreen4, RoiGreen1_selected,RoiGreen2_aligned, RoiGreen3_aligned,RoiGreen4_aligned, folder_name)



"""

max_index = np.argmax(SinalFinal)
shift_frame = 0
max_index_shifted = min(max_index + shift_frame, len(SinalFinal))

timeStampsShiftFrame=time_stamps[max_index_shifted:]
ROICorrigidaShiftFrame = SinalFinal[max_index_shifted:]

ratiosrC = ROICorrigidaShiftFrame
ratiosgC = ROICorrigidaShiftFrame
ratiosbC = ROICorrigidaShiftFrame

ratiosC=np.column_stack((ratiosrC,ratiosgC,ratiosbC))
pcrtComp = PCRT(timeStampsShiftFrame, ratiosC,exclusionMethod='best fit',exclusionCriteria=999)
outputFilePCRTRGB = f"pCRTDeslocadoRGB{shift_frame}{video_name}.png"
#pcrtComp.showAvgIntensPlot()
pcrtComp.showPCRTPlot()
outputFilePCRT = f"pCRTDeslocado{shift_frame}{video_name}.png"
pcrtComp.savePCRTPlot(outputFilePCRT)
"""









#print(f"a= {a} b = {b} gamma = {gamma}")
#plot_image_and_ratios(frames, a, b, gamma, roi1, roi2, roi3, roi4, time_stamps,RoiGreen1,RoiGreen2, RoiGreen3,RoiGreen4, ROI4Corrigida, folder_name)

"""






# Calculando média, desvio padrão e coeficiente de variação
def calculate_stats(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    cv = (1000)*(std_dev / mean) if mean != 0 else 0  # Evita divisão por zero
    return mean, std_dev, cv

# Aplicando a função para cada curva
stats_Corrigida = calculate_stats(ROI4Corrigida)
stats_RoiGreen2 = calculate_stats(RoiGreen2)
stats_RoiGreen3 = calculate_stats(RoiGreen3)
stats_RoiGreen4 = calculate_stats(RoiGreen4)
stats_RoiGreen5 = calculate_stats(RoiGreen5)

# Criando um DataFrame com os resultados
data = {
    'RoiGreen2': stats_RoiGreen2,
    'RoiGreen3': stats_RoiGreen3,
    'RoiGreen4': stats_RoiGreen4,
    'RoiGreen5': stats_RoiGreen5,
    'ROICorrigida': stats_Corrigida,
}



df_stats = pd.DataFrame(data, index=['Média', 'Desvio Padrão', 'Coeficiente de Variação']).T

# Salvando o DataFrame em um arquivo Excel
excel_filename = "estatisticas_roisCaso7.xlsx"
df_stats.to_excel(excel_filename, index=True)

print(f"Os dados foram salvos no arquivo {excel_filename}")


"""
##### histograma


def histogram(video_path, roi1, roi2, roi3, roi4, firstFrame,lastFrame, output_image_path):
    # Tenta abrir o vídeo
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        sys.exit(1)

    # Lê os frames 10 e 400
    cap.set(cv.CAP_PROP_POS_FRAMES, firstFrame)
    ret, frame_10 = cap.read()
    if not ret:
        print(f"Não foi possível ler o frame {firstFrame}.")
        sys.exit(1)

    cap.set(cv.CAP_PROP_POS_FRAMES, lastFrame)
    ret, frame_400 = cap.read()
    if not ret:
        print(f"Não foi possível ler o frame {lastFrame}.")
        sys.exit(1)

    # Extrai as ROIs para cada frame
    roi_10_1 = frame_10[roi1[1]:roi1[1]+roi1[3], roi1[0]:roi1[0]+roi1[2]]
    roi_10_2 = frame_10[roi2[1]:roi2[1]+roi2[3], roi2[0]:roi2[0]+roi2[2]]
    roi_10_3 = frame_10[roi3[1]:roi3[1]+roi3[3], roi3[0]:roi3[0]+roi3[2]]
    roi_10_4 = frame_10[roi4[1]:roi4[1]+roi4[3], roi4[0]:roi4[0]+roi4[2]]

    roi_400_1 = frame_400[roi1[1]:roi1[1]+roi1[3], roi1[0]:roi1[0]+roi1[2]]
    roi_400_2 = frame_400[roi2[1]:roi2[1]+roi2[3], roi2[0]:roi2[0]+roi2[2]]
    roi_400_3 = frame_400[roi3[1]:roi3[1]+roi3[3], roi3[0]:roi3[0]+roi3[2]]
    roi_400_4 = frame_400[roi4[1]:roi4[1]+roi4[3], roi4[0]:roi4[0]+roi4[2]]

    # Converte as ROIs para escala de cinza
    gray_roi_10_1 = cv.cvtColor(roi_10_1, cv.COLOR_BGR2GRAY)
    gray_roi_10_2 = cv.cvtColor(roi_10_2, cv.COLOR_BGR2GRAY)
    gray_roi_10_3 = cv.cvtColor(roi_10_3, cv.COLOR_BGR2GRAY)
    gray_roi_10_4 = cv.cvtColor(roi_10_4, cv.COLOR_BGR2GRAY)

    gray_roi_400_1 = cv.cvtColor(roi_400_1, cv.COLOR_BGR2GRAY)
    gray_roi_400_2 = cv.cvtColor(roi_400_2, cv.COLOR_BGR2GRAY)
    gray_roi_400_3 = cv.cvtColor(roi_400_3, cv.COLOR_BGR2GRAY)
    gray_roi_400_4 = cv.cvtColor(roi_400_4, cv.COLOR_BGR2GRAY)

    # Calcula os histogramas para as ROIs em escala de cinza
    hist_10_1 = cv.calcHist([gray_roi_10_1], [0], None, [256], [0, 256])
    hist_10_2 = cv.calcHist([gray_roi_10_2], [0], None, [256], [0, 256])
    hist_10_3 = cv.calcHist([gray_roi_10_3], [0], None, [256], [0, 256])
    hist_10_4 = cv.calcHist([gray_roi_10_4], [0], None, [256], [0, 256])

    hist_400_1 = cv.calcHist([gray_roi_400_1], [0], None, [256], [0, 256])
    hist_400_2 = cv.calcHist([gray_roi_400_2], [0], None, [256], [0, 256])
    hist_400_3 = cv.calcHist([gray_roi_400_3], [0], None, [256], [0, 256])
    hist_400_4 = cv.calcHist([gray_roi_400_4], [0], None, [256], [0, 256])

    # Plota os histogramas com sobreposição
    plt.figure(figsize=(12, 6))

    plt.subplot(221)
    plt.title('Histograma ROI 1')
    plt.plot(hist_10_1, label=f'Frame{firstFrame}')
    plt.plot(hist_400_1, label=f'Frame{lastFrame}')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.legend()

    plt.subplot(222)
    plt.title('Histograma ROI 2')
    plt.plot(hist_10_2, label=f'Frame{firstFrame}')
    plt.plot(hist_400_2, label=f'Frame{lastFrame}')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.legend()

    plt.subplot(223)
    plt.title('Histograma ROI 3')
    plt.plot(hist_10_3, label=f'Frame{firstFrame}')
    plt.plot(hist_400_3, label=f'Frame{lastFrame}')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.legend()

    plt.subplot(224)
    plt.title('Histograma ROI 4')
    plt.plot(hist_10_4, label=f'Frame{firstFrame}')
    plt.plot(hist_400_4, label=f'Frame{lastFrame}')
    plt.xlabel('Intensidade')
    plt.ylabel('Frequência')
    plt.legend()

    plt.tight_layout()
    
    
    plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
    
    plt.show()
    cap.release()





def DerivadaMeiaAltura(sinal, time_stamps):
    """
    Calcula a derivada do sinal, identifica os picos positivos e negativos, 
    e encontra o ponto de meia altura do pico positivo deslocado para frente.
    
    :param sinal: Lista ou array contendo os valores do sinal.
    :param time_stamps: Lista ou array com os timestamps correspondentes ao sinal.
    :return: Índice do pico positivo, índice do ponto de meia altura.
    """
    # Calcula a derivada
    dt = time_stamps[1] - time_stamps[0]
    derivada = np.diff(sinal) / dt
    
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
    plt.plot(t_derivada[maxPicoPositivo], derivada[maxPicoPositivo], 'ro', label='Máximo Pico Positivo')
    plt.plot(t_derivada[meio_altura_index], derivada[meio_altura_index], 'bo', label='Meia Altura (Deslocado)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return maxPicoPositivo, meio_altura_index

    
 
    
# os picos positivos e negativos da derivada do sinal de CRT
# o pico positivo é o momento que se tem a retirado do dedo
# o pico negativo é o momento que se aplica a compressão
maxPicoPositivo, meio_altura_index = DerivadaMeiaAltura(RoiGreen1,time_stamps)


print(maxPicoPositivo)
print(meio_altura_index)


"""


firstFrame = 10
lastFrame = meio_altura_index # ponto de retirada do dedo 

###################### Histograna da imagem ####################
# função para calcular o histograma de uma imagem
output_image_path = f"FigureHistogram{firstFrame}{lastFrame}.png"
histogram(video_path, roi1, roi2, roi3, roi4, firstFrame, lastFrame, output_image_path)
"""
