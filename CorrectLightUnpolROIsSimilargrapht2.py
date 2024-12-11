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
def plot_graph_curves(best_a, best_b, best_gamma, RoiGreen, ROI1Corrigida, ROICorrigidaCompleta):
    
    plt.subplot(311)
    plt.plot(time_stamps, RoiGreen, label=f"Canal Verde ROI 1 ", color='darkgreen')
    plt.xlabel('Tempo (s)')
    #plt.ylabel('Intensidade do Canal Verde')
    plt.legend()

    plt.subplot(312)
    plt.plot(time_stamps, ROI1Corrigida, label=f"Canal Verde corrigido", color='green')
    plt.title(r"$\gamma=$" + f"{best_gamma:.2f}")
    plt.xlabel('Tempo (s)')
    plt.ylabel('Intensidade do Canal Verde', fontsize="12")
    plt.legend()
    
    plt.subplot(313)
    plt.plot(time_stamps, ROICorrigidaCompleta, label=f"Canal Verde Equação", color='green')
    plt.title(f"a={best_a:.2f}_b={best_b:.2f} $\gamma$={best_gamma:.2f}")
    plt.xlabel('Tempo (s)')
    #plt.ylabel('Intensidade do Canal ')
    plt.legend()
    
    
    plt.tight_layout()
    pastaSalvar= "C:/Users/Fotobio/Desktop/ResultadosAbgamma"
    outputImageGraph = f"{pastaSalvar}/abPertoGrafico_a={best_a:.2f}_b={best_b:.2f}_g={best_gamma:.2f}_{folder_name}completo.png"
    plt.savefig(outputImageGraph, dpi=300)
    plt.show()
    plt.close()

# Função de plotagem com a razão ajustada
def plot_image_and_ratios(frames, best_a, best_b, best_gamma, roi1, roi2,roi3, time_stamps, 
                          RoiGreen,RoiGreen2, RoiGreen3, ROI1Corrigida, adjusted_ratio, folder_name):
    


    # Converta as listas para arrays do NumPy
    roi_green2_array = np.array(RoiGreen2)
    roi_green3_array = np.array(RoiGreen3)

    # Divida as curvas elemento a elemento
    ratios = roi_green2_array / roi_green3_array
     
    # Extrair coordenadas da ROI1
    x_roi1, y_roi1, w_roi1, h_roi1 = roi1
    x_i, y_i, w_i, h_i = roi2
    x_j, y_j, w_j, h_j = roi3

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    
    axs[0, 0].imshow(cv.cvtColor(frames[0], cv.COLOR_BGR2RGB))  
    axs[0, 0].set_title('Imagem Original')
    axs[0, 0].axis('off')  
    
   
   
    # Adicionar retângulos das ROIs na imagem
    axs[0, 0].add_patch(plt.Rectangle((x_roi1, y_roi1), w_roi1, h_roi1, edgecolor='green', facecolor="none", linewidth=2))
    axs[0, 0].add_patch(plt.Rectangle((x_i, y_i), w_i, h_i, edgecolor='blue', facecolor='none', linewidth=2))
    axs[0, 0].add_patch(plt.Rectangle((x_j, y_j), w_j, h_j, edgecolor='red', facecolor='none', linewidth=2))
    axs[0, 0].text(x_roi1 + w_roi1 / 2, y_roi1 + h_roi1 / 2, "ROI1", color='green', ha='center', va='center', fontsize=8, weight='bold')
    axs[0, 0].text(x_i + w_i / 2, y_i + h_i / 2, f"ROI2", color='blue', ha='center', va='center', fontsize=8, weight='bold')
    axs[0, 0].text(x_j + w_j / 2, y_j + h_j / 2, f"ROI3", color='red', ha='center', va='center', fontsize=8, weight='bold')
    
    # Remover subplots em branco da primeira coluna
    axs[1, 0].axis('off')
    axs[2, 0].axis('off')

    axs[0, 1].plot(time_stamps, RoiGreen2, label=f"Canal Verde ROI2 ", color='blue')
    axs[0, 1].set_xlabel('Tempo (s)')
    axs[0, 1].set_ylabel('Intensidade do Canal Verde')
    axs[0, 1].legend()

    axs[1, 1].plot(time_stamps, RoiGreen3, label=f"Canal Verde ROI3  ", color='red')
    axs[1, 1].set_xlabel('Tempo (s)')
    axs[1, 1].set_ylabel('Intensidade do Canal Verde')
    axs[1, 1].legend()

    axs[2, 1].plot(time_stamps, ratios, label=f"Razão ", color='orange')
    axs[2, 1].set_xlabel('Tempo (s)')
    axs[2, 1].set_ylabel('Razão Original')
    axs[2, 1].legend()

    axs[0, 2].plot(time_stamps, RoiGreen, label=f"Canal Verde ROI 1 ", color='darkgreen')
    axs[0, 2].set_xlabel('Tempo (s)')
    axs[0, 2].set_ylabel('Intensidade do Canal Verde')
    axs[0, 2].legend()

    axs[1, 2].plot(time_stamps, ROI1Corrigida, label=f"Canal Verde corrigido", color='green')
    axs[1, 2].set_title(r"$\gamma=$" + f"{best_gamma:.2f}")
    axs[1, 2].set_xlabel('Tempo (s)')
    axs[1, 2].set_ylabel('Intensidade do Canal Verde')
    axs[1, 2].legend()

    axs[2, 2].plot(time_stamps, adjusted_ratio, label=f"Razão corrigido", color='orange')
    axs[2, 2].set_xlabel('Tempo (s)')
    axs[2, 2].set_ylabel('Intensidade do Canal Verde')
    axs[2, 2].legend()

    plt.tight_layout()
    output_filename = f"Perto_a={best_a:.2f}_b={best_b:.2f}_g={best_gamma:.2f}_{folder_name}ROIcompleto.png"
    plt.savefig(output_filename, dpi=600)
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
        {'type': 'ineq', 'fun': restricao_std_error, 'args': (green_roi2, green_roi3)},
        #{'type': 'ineq', 'fun': restricao_mean_abs_error, 'args': (green_roi2, green_roi3)}
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


def find_best_gamma(green_roi2, green_roi3,bounds):
    
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
        
        return 0.5 - std_error  # erro padrão deve ser menor que 0.5

    def restricao_mean_abs_error(params, green_roi2, green_roi3):
        gamma = params[0]  # Apenas gamma será otimizado
        
        # Calcular o adjusted_ratio com a = 0 e b = 1
        adjusted_ratio = (0 + 1 * (green_roi2 ** gamma)) / (0 + 1 * (green_roi3 ** gamma))
        
        # Calcular o erro absoluto médio
        mean_abs_error = np.mean(np.abs(adjusted_ratio - 1))
        
        return 0.01 - mean_abs_error  # erro absoluto médio deve ser menor que 0.01

    # Inicialização para otimizar apenas gamma
    x0 = np.array([2.0])  # Apenas o valor de gamma será otimizado 0.1 e 5 

    # Definindo os limites para gamma, com valor inicial maior que 0
    bounds = [(0.1, None)]  # gamma > 0

    restricoes = [
        {'type': 'ineq', 'fun': restricao_std_error, 'args': (green_roi2, green_roi3)},  
        {'type': 'ineq', 'fun': restricao_mean_abs_error, 'args': (green_roi2, green_roi3)}
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

    global roi1,roi2,roi3,roi4
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

frame_count = 0
fps = cap.get(cv.CAP_PROP_FPS)


# Selecionar as ROIs
select_rois()
#roi1=(553, 113, 91, 88) #v7
#roi1=(41, 287, 106, 83) #v6
# Reinicia o vídeo
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# Listas para armazenar intensidades e timestamps
RoiRed,RoiGreen,RoiBlue,time_stamps,RoiGreen2,RoiGreen3 = [], [],[],[],[],[]
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
    roi3_frame = frame[int(roi4[1]):int(roi4[1] + roi4[3]), int(roi4[0]):int(roi4[0] + roi4[2])]  

    # ROI 1
   
    RoiGreen.append(np.mean(roi1_frame[:, :, 1]))
    RoiGreen2.append(np.mean(roi2_frame[:, :, 1]))
    RoiGreen3.append(np.mean(roi3_frame[:, :, 1]))
    RoiGreen4.append(np.mean(roi4_frame[:, :, 1]))      

    

    time_stamps.append(frame_count / fps)
    frame_count += 1

cap.release()

if len(frames) == 0:
    raise ValueError("Nenhum frame foi lido do vídeo.")

"""
# usado para encontrar ROIs perto
ratios = calculate_ratios(RoiGreen2,RoiGreen3)


# usado para encontrar ROIs proximas 
# Definir thresholds para mediana e desvio padrão
threshold_median = 0.005  
threshold_std = 0.01      

# Filtra razões mais próximas de 1 com base na mediana e na consistência (desvio padrão)
close_to_one_ratios = {
    key: value for key, value in ratios.items()
    if abs(np.median(value) - 1) < threshold_median and np.std(value) < threshold_std
}



"""

best_a, best_b,best_gamma = None, None, None
best_combination = None
all_outputDataDecay = []

#for roi_pair in close_to_one_ratios.keys():
    # Extrai os índices das ROIs da chave
#    i, j = map(lambda x: int(x.split('/')[0][3:]) - 1, roi_pair.split('/'))

bounds = [(0.1, None), (0.1, None), (0.1, 2.5)]
   
    #a, b, gamma,adjusted_ratio= find_best_a_b(all_green_rois[i] ,all_green_rois[j])
a, b, gamma,adjusted_ratio= find_best_a_b(RoiGreen2,RoiGreen3,bounds)

best_a, best_b, best_gamma = a, b, gamma




    # Usei isso so para conseguir calcular o CRT depois 
    #RoiRed=np.array(RoiRed)
RoiGreen=np.array(RoiGreen)
    #RoiBlue= np.array(RoiBlue)

ROI1Corrigida= ((RoiGreen)**best_gamma)

plot_image_and_ratios(frames, best_a, best_b, best_gamma, roi1, roi2,roi3, time_stamps, 
                          RoiGreen,RoiGreen2, RoiGreen3, ROI1Corrigida, adjusted_ratio, folder_name)

    #time_stamps = np.array(time_stamps)

    # Calcula razão entre as intensidades normalizadas - desconsidere o ratiosr e ratiosb não vou usa-los no futuro -
    #  é somente para poder processa usando o pyCRT
    #ratiosr = ROI1Corrigida
    #ratiosg = ROI1Corrigida
    #ratiosb = ROI1Corrigida

    #plot_image_and_ratios(frames, best_combination, best_a, best_b, best_gamma,all_green_rois, time_stamps,RoiGreen,ROI1Corrigida,adjusted_ratio,folder_name,roi1)

    ### ajusta PCRT 

    #channelsAvgIntensArr= np.column_stack((RoiRed, RoiGreen, RoiBlue))
    #pcrtO = PCRT(time_stamps,channelsAvgIntensArr,exclusionMethod='best fit',exclusionCriteria=999)
    #pcrt.showAvgIntensPlot()
    #pcrtO.showPCRTPlot()
    #pcrtO.savePCRTPlot(f"Perto_pCRTOriginal{folder_name}ROI{i+1}ROI{j+1}.png")


    #ratios=np.column_stack((ratiosr,ratiosg,ratiosb))
    #pcrtCorrigidogamma = PCRT(time_stamps, ratios,exclusionMethod='best fit',exclusionCriteria=999 )
    #pcrt.showAvgIntensPlot()
    #pcrtC.showPCRTPlot()
    #outputFilePCRT = f"Perto_pCRTa={best_a: .2f}b={best_b: .2f}g={best_gamma: .2f}{folder_name}ROI{i+1}ROI{j+1}completo.png"
    #pcrtCorrigidogamma.savePCRTPlot(outputFilePCRT)

ROI1CorrigidaCompleto= best_a+(best_b*(RoiGreen**best_gamma))

    #ratiosrC = ROI1CorrigidaCompleto
    #ratiosgC = ROI1CorrigidaCompleto
    #ratiosbC = ROI1CorrigidaCompleto

    #ratiosC=np.column_stack((ratiosrC,ratiosgC,ratiosbC))
    #pcrtComp = PCRT(time_stamps, ratiosC,exclusionMethod='best fit',exclusionCriteria=999)
    #pcrt.showAvgIntensPlot()
    #pcrtC.showPCRTPlot()
    #outputFilePCRT = f"Perto_pCRTCompletoa={best_a: .2f}b={best_b: .2f}g={best_gamma: .2f}{folder_name}ROI{i+1}ROI{j+1}completo.png"
    #pcrtComp.savePCRTPlot(outputFilePCRT)
    
    
plot_graph_curves(best_a, best_b, best_gamma, RoiGreen, ROI1Corrigida, ROI1CorrigidaCompleto)



all_outputDataDecay.append({
    "Best A": best_a,
    "Best B": best_b,
    "Best Gamma": best_gamma,
    # Adicione outros resultados conforme necessário
})
            # Convertendo para DataFrame
df = pd.DataFrame(all_outputDataDecay)

    # Salvando no Excel
outputCompleteData = f"resultadosCompletosPerto.xlsx"
df.to_excel(outputCompleteData, index=False)

print(f"Dados salvos em: {outputCompleteData}")

"""
all_outputDataDecay.append({
"Best A": best_a,
"Best B": best_b,
"Best Gamma": best_gamma,
"pcrtOriginal": pcrtO.pCRT[0],
"incertezaOriginal":pcrtO.pCRT[1],
"pcrtCorrigido": pcrtCorrigidogamma.pCRT[0],
"incerteza":pcrtCorrigidogamma.pCRT[1],
"pcrtCorrigidoCompleto": pcrtComp.pCRT[0],
"incertezaCorrigidoCompleto":pcrtComp.pCRT[1],
# Adicione outros resultados conforme necessário
})
        # Convertendo para DataFrame
df = pd.DataFrame(all_outputDataDecay)

# Salvando no Excel
outputCompleteData = f"resultadosCompletosPerto.xlsx"
df.to_excel(outputCompleteData, index=False)

print(f"Dados salvos em: {outputCompleteData}")


Keyword arguments:
argument -- description
Return: return_description


    if best_a is not None and best_b is not None:
    print(f"Melhor a: {best_a}, Melhor b: {best_b}, Melhor gamma: {best_gamma}, ROI utilizada: {best_combination}")
    else:
    print("Não foi possível encontrar valores válidos de a, b e gamma.")
"""
