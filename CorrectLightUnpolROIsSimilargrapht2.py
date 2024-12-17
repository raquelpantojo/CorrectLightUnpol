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

import numpy as np


sys.path.append("C:/Users/Fotobio/Documents/GitHub/pyCRT") #PC casa 
#sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT") # PC lab

from src.pyCRT import PCRT  

# Caminho base para os arquivos do projeto
#base_path = "C:/Users/RaquelPantojo/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"  # PC USP
base_path="C:/Users/Fotobio/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"
folder_name = "teste1"
video_name="v7.mp4"
#video_name = "corrected_v7_gamma=1.mp4"
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
    axs[0].add_patch(plt.Rectangle((x_i, y_i), w_i, h_i, edgecolor='blue', facecolor='none', linewidth=2))
    axs[0].add_patch(plt.Rectangle((x_j, y_j), w_j, h_j, edgecolor='red', facecolor='none', linewidth=2))
    axs[0].add_patch(plt.Rectangle((x_k, y_k), w_k, h_k, edgecolor='orange', facecolor='none', linewidth=2))

    axs[0].text(x_i + w_i / 2, y_i + h_i / 2, f"1", color='blue', ha='center', va='center', fontsize=8, weight='bold')
    axs[0].text(x_j + w_j / 2, y_j + h_j / 2, f"2", color='red', ha='center', va='center', fontsize=8, weight='bold')
    axs[0].text(x_k + w_k / 2, y_k + h_k / 2, f"3", color='orange', ha='center', va='center', fontsize=8, weight='bold')

    # Subplot (4, 2, 2) - ROI 1
    axs[1].plot(time_stamps, RoiGreen2, label="ROI 1", color='blue')
    #axs[1].set_xlabel('Tempo (s)')
    #axs[1].set_ylim(100, 255)
    axs[1].set_ylabel('Intensidade do Canal Verde')
    axs[1].legend()

    # Subplot (4, 2, 4) - ROI 2
    axs[3].plot(time_stamps, RoiGreen3, label="ROI 2", color='red')
    #axs[3].set_xlabel('Tempo (s)')
    #axs[3].set_ylim(100, 255)
    axs[3].set_ylabel('Intensidade do Canal Verde')
    axs[3].legend()

    # Subplot (4, 2, 6) - ROI 3
    axs[5].plot(time_stamps, RoiGreen4, label="ROI 3", color='orange')
    #axs[5].set_xlabel('Tempo (s)')
    #axs[5].set_ylim(100, 255)
    axs[5].set_ylabel('Intensidade do Canal Verde')
    axs[5].legend()

    # Subplot (4, 2, 8) - ROI corrigida
    axs[7].plot(time_stamps, ROI4Corrigida, label="ROI 3 Corrigida/ ROI 1 Corrigida", color='darkgreen')
    axs[7].set_xlabel('Tempo (s)')
    axs[7].set_ylabel('Intensidade do Canal Verde')
    axs[7].legend()

    # Desativar subplots extras
    for i in range(len(axs)):
        if i not in [0, 1, 3, 5, 7]:
            axs[i].axis('off')

    # Ajustar layout e salvar
    plt.tight_layout()
    output_filename = f"ROIEscolhidas_a={a:.2f}_b={b:.2f}_g={gamma:.2f}_{folder_name}menormaior.png"
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



def find_best_a_b(green_roi2, green_roi3,bounds):
    """
    Encontra os melhores parâmetros a, b e gamma que minimizam o erro absoluto médio
    e satisfazem as restrições de erro padrão e erro absoluto médio.

    Args:
        green_roi2, green_roi3: Vetores de intensidades para as duas ROIs.
        error_threshold: Limite para o erro absoluto médio.

    Returns:
        a_otimizado, b_otimizado, gamma_otimizado, adjusted_ratio: Parâmetros otimizados
        e a razão ajustada.
    """

    best_result = None
    best_fun = float('inf')


        # Função objetivo a ser minimizada
    def funcao_objetivo(params, green_roi2, green_roi3):
        a, b, gamma = params
        # Calcular a razão ajustada
        adjusted_ratio = (a + b * (green_roi2 ** gamma)) / (a + b * (green_roi3 ** gamma))
        # Calcular o erro absoluto médio
        mean_abs_error = np.mean(np.abs(adjusted_ratio - 1))
        return mean_abs_error

    # Restrições (erro padrão e erro absoluto médio)
    def restricao_std_error(params, green_roi2, green_roi3):
        a, b, gamma = params
        adjusted_ratio = (a + b * (green_roi2 ** gamma)) / (a + b * (green_roi3 ** gamma))
        std_error = np.std(adjusted_ratio)
        return std_error  

    def restricao_mean_abs_error(params, green_roi2, green_roi3):
        a, b, gamma = params
        adjusted_ratio = (a + b * (green_roi2 ** gamma)) / (a + b * (green_roi3 ** gamma))
        # Calcular o erro absoluto médio
        mean_abs_error = np.mean(np.abs(adjusted_ratio - 1))
        return mean_abs_error  

    # Chute inicial para os parâmetros a, b, gamma
    #x0 = [10,1.5, 1]  # valores iniciais para a, b, gamma

    # Definir limites (exemplo: a e b podem ser negativos, gamma > 0)
    #bounds = [(0, None), (0, None), (0.1, 3)]  # a e b sem limites, gamma > 0

    # Definir as restrições
    restricoes = [
        #{'type': 'ineq', 'fun': restricao_std_error, 'args': (green_roi2, green_roi3)},
        {'type': 'ineq', 'fun': restricao_mean_abs_error, 'args': (green_roi2, green_roi3)}
    ]

    for _ in range(10):  # Teste 10 vezes 
        x0 = np.random.uniform(0, 10, size=3)
        # Minimização
        result = minimize(funcao_objetivo, x0, args=(green_roi2, green_roi3), method='SLSQP',
                            bounds=bounds, constraints=restricoes)
        if result.fun < best_fun:
            best_fun = result.fun
            best_result = result
            a_otimizado, b_otimizado, gamma_otimizado = result.x
            adjusted_ratio = (a_otimizado + b_otimizado * (green_roi2 ** gamma_otimizado)) / \
                            (a_otimizado + b_otimizado * (green_roi3 ** gamma_otimizado))
            return a_otimizado, b_otimizado, gamma_otimizado, adjusted_ratio
        else:
            print("Otimização não convergiu: ", result.message)
            return None



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
            """
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

#roi1=(553, 113, 91, 88) #v7
#roi1=(41, 287, 106, 83) #v6

roi1=(1121, 222, 154, 154)
roi2=(1810, 9, 41, 93)
roi3=(43, 22, 94, 82)
roi4=(1794, 744, 76, 84)
roi5=(453, 199, 84, 82)
#select_rois()

# Reinicia o vídeo

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
    roi5_frame = frame[int(roi5[1]):int(roi5[1] + roi5[3]), int(roi5[0]):int(roi5[0] + roi5[2])] 

    # ROI 1
   
    RoiGreen1.append(np.mean(roi1_frame[:, :, 1]))
    RoiGreen2.append(np.mean(roi2_frame[:, :, 1]))
    RoiGreen3.append(np.mean(roi3_frame[:, :, 1]))
    RoiGreen4.append(np.mean(roi4_frame[:, :, 1]))   
    RoiGreen5.append(np.mean(roi5_frame[:, :, 1]))    


    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()


#for roi_pair in close_to_one_ratios.keys():
    # Extrai os índices das ROIs da chave
#    i, j = map(lambda x: int(x.split('/')[0][3:]) - 1, roi_pair.split('/'))

bounds = [(0.1, 10), (0.1, 10), (0.1, 10)]
   
#a, b, gamma,adjusted_ratio= find_best_a_b(all_green_rois[i] ,all_green_rois[j])
#a, b, gamma, adjusted_ratio= find_best_a_b(RoiGreen3,RoiGreen4,bounds)
a, b, gamma, adjusted_ratio= find_best_gamma(RoiGreen5,RoiGreen4)

gamma1=1
#ROICorrigida = (a + b * (RoiGreen1 ** gamma1)) / (a + b * (RoiGreen5 ** gamma1))
ROICorrigida = np.array(RoiGreen1) /np.array(RoiGreen4)


outputImageGraph = f"Casoextra.png"
plot_graph_curves(RoiGreen1, ROICorrigida,outputImageGraph)

"""

ratiosrC = ROI4Corrigida
ratiosgC = ROI4Corrigida
ratiosbC = ROI4Corrigida

ratiosC=np.column_stack((ratiosrC,ratiosgC,ratiosbC))
pcrtComp = PCRT(time_stamps, ratiosC,exclusionMethod='best fit',exclusionCriteria=999)
pcrtComp.showAvgIntensPlot()
#pcrtComp.showPCRTPlot()
outputFilePCRT = f"pCRTCompletocompletoCaso1.png"
pcrtComp.savePCRTPlot(outputFilePCRT)



print(f"a= {a} b = {b} gamma = {gamma}")
#plot_image_and_ratios(frames, a, b, gamma, roi1, roi2, roi3, roi4, time_stamps,RoiGreen1,RoiGreen2, RoiGreen3,RoiGreen4, ROI4Corrigida, folder_name)




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
