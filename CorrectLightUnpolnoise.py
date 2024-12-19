# Teste usando ROIS silimares
# Teste com 4 ROis 

import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
import pandas as pd
import cv2
import numpy as np


#sys.path.append("C:/Users/Fotobio/Documents/GitHub/pyCRT") #PC casa 
#sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT") # PC lab
sys.path.append("C:/Users/raque/OneDrive/Documentos/GitHub/pyCRT")

from src.pyCRT import PCRT  

# Caminho base para os arquivos do projeto
#base_path = "C:/Users/RaquelPantojo/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"  # PC USP
#base_path="C:/Users/Fotobio/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"
#base_path="C:/Users/Fotobio/Documents/GitHub"
#folder_name = "teste1"
#folder_name="CorrectLightUnpol"
#video_name="v5.mp4"
#video_name = "SeiyLedDesp4.mp4"
#video_name = "NatanLedDesp6.mp4"
#video_name ="SeiyLedPol6.mp4"
#video_name="NatanledPol5.mp4"
#video_name = "corrected_v7_gamma=1.mp4"


base_path="C:/Users/raque/OneDrive/Documentos/GitHub"
folder_name = "CorrectLightUnpol"
#video_name="corrected_v7_gamma=1.mp4"
video_name="raqueltestecasa.mp4"
#video_name="v7.mp4"
#video_name ="SeiyLedDesp4.mp4"


roi_width = 80 
roi_height = 80
num_rois = 200  # Número de ROIs a serem criadas
#gamma = 1.53

def plot_image_and_ratios(frames, a, b, gamma, roi1, roi2, roi3, roi4, time_stamps, RoiGreen1, RoiGreen2, RoiGreen3, RoiGreen4, ROI4Corrigida, folder_name):
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
    axs[1].plot(time_stamps, RoiGreen1, label="ROI 1", color='green')
    #axs[1].set_xlabel('Tempo (s)')
    #axs[1].set_ylim(100, 255)
    axs[1].set_ylabel('Intensidade do Canal Verde')
    axs[1].legend()

    # Subplot (4, 2, 4) - ROI 2
    axs[3].plot(time_stamps, RoiGreen2, label="ROI 2", color='blue')
    #axs[3].set_xlabel('Tempo (s)')
    #axs[3].set_ylim(100, 255)
    axs[3].set_ylabel('Intensidade do Canal Verde')
    axs[3].legend()

    # Subplot (4, 2, 6) - ROI 3
    axs[5].plot(time_stamps, RoiGreen3, label="ROI 3", color='red')
    #axs[5].set_xlabel('Tempo (s)')
    #axs[5].set_ylim(100, 255)
    axs[5].set_ylabel('Intensidade do Canal Verde')
    axs[5].legend()

    # Subplot (4, 2, 8) - ROI corrigida
    axs[7].plot(time_stamps, RoiGreen4, label="ROI 4", color='orange')
    axs[7].set_xlabel('Tempo (s)')
    axs[7].set_ylabel('Intensidade do Canal Verde')
    axs[7].legend()

    # Desativar subplots extras
    for i in range(len(axs)):
        if i not in [0, 1, 3, 5, 7]:
            axs[i].axis('off')

    # Ajustar layout e salvar
    plt.tight_layout()
    output_filename = f"ROI={a:.2f}_b={b:.2f}_g={gamma:.2f}_{folder_name}.png"
    plt.savefig(output_filename, dpi=600)
    plt.show()
    plt.close()


def plot_graph_curves(RoiGreen, ROI1Corrigida,outputImageGraph):
    
    plt.subplot(311)
    plt.plot(time_stamps, RoiGreen, label=f"Canal Verde ROI 1 ", color='darkgreen')
    plt.xlabel('Tempo (s)')
    #plt.ylabel('Intensidade do Canal Verde')
    plt.legend()

    plt.subplot(312)
    plt.plot(time_stamps, ROI1Corrigida, label=f"Canal Verde corrigido", color='green')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Verde', fontsize="12")
    plt.legend()
    
    #plt.subplot(313)
    #plt.plot(time_stamps, ROICorrigidaCompleta, label=f"Canal Verde Equação", color='green')
    #plt.title(f"a={a:.2f}_b={b:.2f} $\gamma$={gamma:.2f}")
    #plt.xlabel('Tempo (s)')
    #plt.ylabel('Intensidade do Canal ')
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
    print(ratios)
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
  


############################Inicalizando o programa####################################
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
roi2 = None
roi3 = None
roi4 = None
roi5 = None

frame_count = 0
fps = cap.get(cv.CAP_PROP_FPS)


# Selecionar as ROIs

#roi1=(553, 113, 91, 88) #v7 com resize
#roi1=(41, 287, 106, 83) #v6

select_rois()

 


#roi1= (1091, 623, 188, 181) #v5
#roi1=(1121, 222, 154, 154) #v7 original
#roi2=(1810, 9, 41, 93)
#roi3=(43, 22, 94, 82)
#roi4=(1794, 744, 76, 84)
#roi5=(453, 199, 84, 82)




# rois do video do Seyi Despolarizado
#roi1= (545, 247, 155, 151)
#roi2= (1320, 622, 72, 80)
#roi3=(834, 582, 170, 42)
#roi4=(1317, 873, 53, 80)

"""
# rois do video do Seyi Despolarizado teste2
roi1=(523, 246, 180, 147)
roi2=(667, 953, 46, 51)
roi3=(1004, 949, 51, 52)
roi4=(996, 677, 51, 56)
"""

# rois do video do Seyi Polarizado
#roi1=(594, 183, 124, 131)
#roi2=(555, 799, 51, 48)
#roi3=(993, 926, 38, 48)
#roi4=(1088, 775, 43, 86)


# rois do video do Natan depsolarizado
#roi1= (457, 571, 165, 215)
#roi2= (209, 81, 102, 97)
#roi3=(694, 139, 83, 73)
#roi4=(1409, 22, 121, 70)

# rois do video do Natan polarizado
#roi1= 
#roi2= 
#roi3=
#roi4=

cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Listas para armazenar intensidades e timestamps
RoiGreen1, RoiGreen2, RoiGreen3, RoiGreen4,RoiGreen5, time_stamps = [], [], [], [], [],[]
frames = []
# Processa o vídeo frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    # frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)   
    frames.append(frame) 
    
    # Extrai as ROIs e calcula a média do canal verde
    roi1_frame = frame[int(roi1[1]):int(roi1[1] + roi1[3]), int(roi1[0]):int(roi1[0] + roi1[2])]
    roi2_frame = frame[int(roi2[1]):int(roi2[1] + roi2[3]), int(roi2[0]):int(roi2[0] + roi2[2])]
    roi3_frame = frame[int(roi3[1]):int(roi3[1] + roi3[3]), int(roi3[0]):int(roi3[0] + roi3[2])]
    roi4_frame = frame[int(roi4[1]):int(roi4[1] + roi4[3]), int(roi4[0]):int(roi4[0] + roi4[2])]  
    #roi5_frame = frame[int(roi5[1]):int(roi5[1] + roi5[3]), int(roi5[0]):int(roi5[0] + roi5[2])] 

    # ROI 1
   
    RoiGreen1.append(np.mean(roi1_frame[:, :, 1]))
    RoiGreen2.append(np.mean(roi2_frame[:, :, 1]))
    RoiGreen3.append(np.mean(roi3_frame[:, :, 1]))
    RoiGreen4.append(np.mean(roi4_frame[:, :, 1]))   
    #RoiGreen5.append(np.mean(roi5_frame[:, :, 1]))    


    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()


#for roi_pair in close_to_one_ratios.keys():
    # Extrai os índices das ROIs da chave
#    i, j = map(lambda x: int(x.split('/')[0][3:]) - 1, roi_pair.split('/'))


   
#a, b, gamma,adjusted_ratio= find_best_a_b(all_green_rois[i] ,all_green_rois[j])
a, b, gamma, adjusted_ratio= find_best_a_b(RoiGreen2,RoiGreen3)
#a, b, gamma, adjusted_ratio= find_best_gamma(RoiGreen5,RoiGreen4)


#ROICorrigida = (a + b * (RoiGreen1 ** gamma)) / (a + b * (RoiGreen3 ** gamma))

# branco 
RoiGreen1Corrigido=(a + b * (RoiGreen1 ** gamma))
RoiGreen2Corrigido=(a + b * (RoiGreen2 ** gamma))
RoiGreen3Corrigido = (a + b* ( RoiGreen3 **gamma))
RoiGreen4Corrigido = (a + b * (RoiGreen4 **gamma))/(a + b * (RoiGreen2 **gamma))

branco=np.mean(RoiGreen2Corrigido[:30])

#ruido 
meanRatiosRoi4 = np.mean(RoiGreen4Corrigido[:30])
ruido=(RoiGreen4Corrigido/meanRatiosRoi4)

#sinal

meanRatiosRoi1 = np.mean(RoiGreen1Corrigido[:30])
#RoiGreen1Corrigido=(RoiGreen1Corrigido/meanRatiosRoi1)

RoiGreen1=np.array(RoiGreen1)


#plot_image_and_ratios(frames, a, b, gamma, roi1, roi2, roi3, roi4, time_stamps, RoiGreen1, RoiGreen2, RoiGreen3, RoiGreen4, RoiGreen1Corrigido, folder_name)


#(sinal - ruido)/branco
#SinalFinal = (np.array(RoiGreen1)-np.array(RoiGreen3))/meanRatiosRoi1
#SinalFinal = (np.array(RoiGreen1)-ruido)/meanRatiosRoi1
#SinalFinal=((RoiGreen1-(-1*(np.array(RoiGreen4))))/np.array(RoiGreen2))
SinalFinal=(RoiGreen1/(np.array(RoiGreen4)))
#SinalFinal=RoiGreen1Corrigido-(-1*RoiGreen4Corrigido)/branco
#SinalFinal=RoiGreen1Corrigido/RoiGreen4Corrigido

print(f"a={a} b={b} gamma={gamma}")
#ROICorrigida = np.array(RoiGreen1) 
time_stamps=np.array(time_stamps)

outputImageGraph = f"CasoFototipoII{folder_name}.png"
plot_graph_curves(RoiGreen1, SinalFinal,outputImageGraph)


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
#plot_image_and_ratios(frames, a, b, gamma, roi1, roi2, roi3, roi4, time_stamps,RoiGreen1,RoiGreen2, RoiGreen3,RoiGreen4, ROICorrigida, folder_name)





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
