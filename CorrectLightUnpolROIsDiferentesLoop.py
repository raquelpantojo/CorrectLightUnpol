# Teste usando ROIS diferentes 
# A razão entre as curvas 

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
base_path="C:/Users/Fotobio/Desktop/Estudo_ElasticidadePele"
folder_name = "DespolarizadoP3"
video_name="v6.mp4"
#video_name = "corrected_v7_gamma=1.mp4"
roi_width = 80 
roi_height = 80
num_rois = 200  # Número de ROIs a serem criadas
#gamma = 1.53

# Função de plotagem com a razão ajustada
def plot_image_and_ratios(frames, best_combination, best_a, best_b, best_gamma, all_green_rois, time_stamps, 
                          RoiGreen, ROI1Corrigida, adjusted_ratio, folder_name, roi1):
    """
    Função para plotar a imagem original, os canais verdes e as razões ajustadas/corrigidas.
    
    Args:
        frames: Lista de frames de vídeo.
        best_combination: Melhor combinação de ROIs (índices i e j).
        best_a, best_b, best_gamma: Parâmetros do ajuste.
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

    axs[0, 1].plot(time_stamps, all_green_rois[i], label=f"Canal Verde ROI {i+1} ", color='blue')
    axs[0, 1].set_xlabel('Tempo (s)')
    axs[0, 1].set_ylabel('Intensidade do Canal Verde')
    axs[0, 1].legend()

    axs[1, 1].plot(time_stamps, all_green_rois[j], label=f"Canal Verde ROI {j+1} ", color='red')
    axs[1, 1].set_xlabel('Tempo (s)')
    axs[1, 1].set_ylabel('Intensidade do Canal Verde')
    axs[1, 1].legend()

    axs[2, 1].plot(time_stamps, original_ratio, label=f"Razão ROI {i+1} / ROI {j+1}", color='orange')
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

    axs[2, 2].plot(time_stamps, adjusted_ratio, label=f"Razão ROI {i+1} / ROI {j+1} corrigido", color='orange')
    axs[2, 2].set_xlabel('Tempo (s)')
    axs[2, 2].set_ylabel('Intensidade do Canal Verde')
    axs[2, 2].legend()

    plt.tight_layout()
    output_filename = f"Longe_a={best_a:.2f}_b={best_b:.2f}_g={best_gamma:.2f}_{folder_name}ROI{i+1}ROI{j+1}completo.png"
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
def calculate_ratios(all_green_rois, num_rois):
    roi_combinations = list(combinations(range(num_rois), 2))
    max_ratio = -np.inf  
    max_pair = None
    matching_ratios = {}

    for i, j in roi_combinations:
        if i < len(all_green_rois) and j < len(all_green_rois):
            # Verifica se as ROIs têm o mesmo comprimento
            if len(all_green_rois[i]) == len(all_green_rois[j]) > 0:
                # Calcula a razão entre as duas ROIs
                ratio = all_green_rois[i] / all_green_rois[j]
                mean_ratio = np.mean(ratio)  # Usar a média das razões para comparação
                
                #print(f"Ratio média entre ROI{i+1} e ROI{j+1}: {mean_ratio}")
                
                # Atualiza o máximo se a nova razão for maior
                if mean_ratio > max_ratio:
                    max_ratio = mean_ratio
                    max_pair = (i, j)
                    matching_ratios[f"ROI{i+1}/ROI{j+1}"] = ratio

    return max_pair, matching_ratios


def find_best_a_b(green_roi2, green_roi3):
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
        # Calcular o erro padrão
        std_error = np.std(adjusted_ratio)
        return std_error  # restrição: erro padrão deve ser < 0.5

    def restricao_mean_abs_error(params, green_roi2, green_roi3):
        a, b, gamma = params
        adjusted_ratio = (a + b * (green_roi2 ** gamma)) / (a + b * (green_roi3 ** gamma))
        # Calcular o erro absoluto médio
        mean_abs_error = np.mean(np.abs(adjusted_ratio - 1))
        #print(mean_abs_error)
        return mean_abs_error 

    # Chute inicial para os parâmetros a, b, gamma
    x0 = [1, 1, 1]  # valores iniciais para a, b, gamma

    
    bounds = [(0, 1), (0, 1), (1, 3)]  

    # Definir as restrições
    restricoes = [
        {'type': 'ineq', 'fun': restricao_std_error, 'args': (green_roi2, green_roi3)},
        {'type': 'ineq', 'fun': restricao_mean_abs_error, 'args': (green_roi2, green_roi3)}
    ]

    # Minimização
    resultado = minimize(funcao_objetivo, x0, args=(green_roi2, green_roi3), method='SLSQP',
                         bounds=bounds, constraints=restricoes)

    # Obter os parâmetros otimizados
    a_otimizado, b_otimizado, gamma_otimizado = resultado.x

    # Calcular a razão ajustada final usando os parâmetros otimizados
    adjusted_ratio = (a_otimizado + b_otimizado * (green_roi2 ** gamma_otimizado)) / \
                     (a_otimizado + b_otimizado * (green_roi3 ** gamma_otimizado))

    return a_otimizado, b_otimizado, gamma_otimizado, adjusted_ratio



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
        
        return std_error  # erro padrão deve ser menor que 0.5

    def restricao_mean_abs_error(params, green_roi2, green_roi3):
        gamma = params[0]  # Apenas gamma será otimizado
        
        # Calcular o adjusted_ratio com a = 0 e b = 1
        adjusted_ratio = (0 + 1 * (green_roi2 ** gamma)) / (0 + 1 * (green_roi3 ** gamma))
        
        # Calcular o erro absoluto médio
        mean_abs_error = np.mean(np.abs(adjusted_ratio - 1))
        
        return 0.01 - mean_abs_error  # erro absoluto médio deve ser menor que 0.01

    # Inicialização para otimizar apenas gamma
    x0 = np.array([2.0])  # Apenas o valor de gamma será otimizado

    # Definindo os limites para gamma, com valor inicial maior que 0
    bounds = [(0.1, 3)]  # gamma > 0

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
  








# Função principal para rodar a análise com múltiplas combinações de parâmetros
def run_analysis_multiple_params(video_path, roi_width, roi_height, num_rois, param_ranges, output_file):
    results = []
    RoiRed,RoiGreen,RoiBlue,time_stamps = [], [],[],[]
    all_green_rois = [[] for _ in range(num_rois)]

    # Intervalos de parâmetros
    a_values = np.linspace(param_ranges['a_min'], param_ranges['a_max'], param_ranges['a_steps'])
    b_values = np.linspace(param_ranges['b_min'], param_ranges['b_max'], param_ranges['b_steps'])
    gamma_values = np.linspace(param_ranges['gamma_min'], param_ranges['gamma_max'], param_ranges['gamma_steps'])

    # Carregar o vídeo
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Selecionar ROI1 interativamente
    global roi1
    _, frame = cap.read()
    frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    #roi1 = cv.selectROI("Selecionar ROI1", frame, fromCenter=False, showCrosshair=True)
    roi1=(41, 287, 106, 83) #v6
    #roi1=(553, 113, 91, 88) #v7
    roi1_frame = frame[int(roi1[1]):int(roi1[1] + roi1[3]), int(roi1[0]):int(roi1[0] + roi1[2])]
    
    # ROI 1
    RoiRed.append(np.mean(roi1_frame[:, :, 0]))
    RoiGreen.append(np.mean(roi1_frame[:, :, 1]))
    RoiBlue.append(np.mean(roi1_frame[:, :, 2]))
    cv.destroyAllWindows()
    frames= []
    # Loop sobre os parâmetros
    for a in a_values:
        for b in b_values:
            for gamma in gamma_values:
                # Processar o vídeo e calcular métricas
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # Reiniciar o vídeo
                _, frame = cap.read()
                frame_resized = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
                frames.append(frame_resized)
                rois = create_dynamic_rois(frame, num_rois, roi_width, roi_height)
                for i, roi in enumerate(rois):
                    x, y, w, h = roi
                    roi_frame = frame_resized[y:y+h, x:x+w]
                    roi_frame = roi_frame 
                    if roi_frame.size > 0:  # Certifica-se de que a ROI não está vazia
                        all_green_rois[i].append(np.mean(roi_frame[:, :, 1]))

                # Selecionar duas ROIs para análise
                max_pair, ratios = calculate_ratios([frame], num_rois)
                if max_pair is None:
                    continue

                i, j = max_pair
                green_roi2 = frame[rois[i][1]:rois[i][1]+roi_height, rois[i][0]:rois[i][0]+roi_width]
                green_roi3 = frame[rois[j][1]:rois[j][1]+roi_height, rois[j][0]:rois[j][0]+roi_width]

                # Calcular ajuste
                best_a, best_b,best_gamma , adjusted_ratio = find_best_a_b(green_roi2, green_roi3)

                ### ajusta PCRT 

                ROI1Corrigida= best_a+best_b*((RoiGreen)**best_gamma)

                RoiRed=np.array(RoiRed)
                RoiGreen=np.array(RoiGreen)
                RoiBlue= np.array(RoiBlue)


                ROI1Corrigida= best_a+best_b*((RoiGreen)**best_gamma)

                time_stamps = np.array(time_stamps)

                # Calcula razão entre as intensidades normalizadas - desconsidere o ratiosr e ratiosb não vou usa-los no futuro -
                #  é somente para poder processa usando o pyCRT
                ratiosr = ROI1Corrigida
                ratiosg = ROI1Corrigida
                ratiosb = ROI1Corrigida

                channelsAvgIntensArr= np.column_stack((RoiRed, RoiGreen, RoiBlue))
                pcrtO = PCRT(time_stamps,channelsAvgIntensArr,exclusionMethod='best fit',exclusionCriteria=999)
                #pcrt.showAvgIntensPlot()
                pcrtO.showPCRTPlot()
                pcrtO.savePCRTPlot(f"Longe_pCRTOriginal{folder_name}ROI{i+1}ROI{j+1}.png")


                ratios=np.column_stack((ratiosr,ratiosg,ratiosb))
                pcrtC = PCRT(time_stamps, ratios,exclusionMethod='best fit',exclusionCriteria=999 )
                #pcrt.showAvgIntensPlot()
                pcrtC.showPCRTPlot()
                outputFilePCRT = f"Longe_pCRTa={best_a: .2f}b={best_b: .2f}g={best_gamma: .2f}{folder_name}ROI{i+1}ROI{j+1}completo.png"
                pcrtC.savePCRTPlot(outputFilePCRT)




                # Salvar resultados
                results.append({
                    'a': a,
                    'b': b,
                    'gamma': gamma,
                    'a_opt': best_a,
                    'b_opt': best_b,
                    'gamma_opt': best_gamma,
                    'adjusted_mean_ratio': np.mean(adjusted_ratio),
                    'std_error': np.std(adjusted_ratio),
                    "pcrtOriginal": pcrtO.pCRT[0],
                    "incertezaOriginal":pcrtO.pCRT[1],
                    "pcrtCorrigido": pcrtC.pCRT[0],
                    "incerteza":pcrtC.pCRT[1],
                })

    # Salvar em Excel
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"Resultados salvos em {output_file}")



sys.path.append("C:/Users/Fotobio/Documents/GitHub/pyCRT") #PC casa 
#sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT") # PC lab

from src.pyCRT import PCRT  

# Caminho base para os arquivos do projeto
#base_path = "C:/Users/RaquelPantojo/Documents/GitHub/CorrectLightUnpol/DespolarizadoP5"  # PC USP
base_path="C:/Users/Fotobio/Desktop/Estudo_ElasticidadePele"
folder_name = "DespolarizadoP3"
video_name="v6.mp4"
#video_name = "corrected_v7_gamma=1.mp4"
roi_width = 80 
roi_height = 80
num_rois = 200  # Número de ROIs a serem criadas


output_file = "resultados_analise.xlsx"

############################Inicalizando o programa####################################
# Verifica o caminho do vídeo
video_path = os.path.join(base_path, folder_name, video_name)
if not os.path.exists(video_path):
    print(f"Vídeo {video_name} não encontrado!")
    sys.exit(1)

# Configuração dos parâmetros
param_ranges = {
    'a_min': 0,
    'a_max': 1,
    'a_steps': 5,
    'b_min': 0,
    'b_max': 1,
    'b_steps': 5,
    'gamma_min': 1,
    'gamma_max': 3,
    'gamma_steps': 5
}



run_analysis_multiple_params(video_path, roi_width=80, roi_height=80, num_rois=200, param_ranges=param_ranges, output_file=output_file)