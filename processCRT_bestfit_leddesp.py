import os
import pandas as pd
import sys
import matplotlib.pyplot as plt


sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT")
from src.pyCRT import PCRT  

# Corrigir o caminho para a pasta do módulo

#sys.path.append("H:\Meu Drive\AvaliaçãoFamb2024\Code_pCRT\Rugoso\pyCRT")




# Definir o caminho base onde estão as pastas P1, P2, ..., P31
base_path ="C:/Users/RaquelPantojo/Desktop/ElasticidadePele"
output_file = "resultados_polarizado.xlsx"

# Criar uma lista para armazenar os resultados
resultados = []
numero_pastas = 1


"""
#Rugoso
rois = [
    [(787, 560, 124, 200), (757, 550, 162, 220), (780, 550, 130, 190), (795, 565, 115, 205), (785, 555, 125, 198)],
    [(344, 839, 114, 71), (362, 767, 154, 85), (335, 747, 194, 87), (343, 760, 78, 72), (343, 750, 100, 72)],
    [(943, 354, 93, 181), (952, 350, 95, 200), (940, 350, 95, 185), (945, 355, 92, 180), (948, 348, 94, 178)]
]

#Liso
rois = [
    [(339, 855, 176, 137), (386, 934, 114, 68), (387, 799, 147, 203), (341, 853, 177, 136), (338, 857, 175, 139)],
    [(897, 536, 105, 127), (856, 551, 106, 121), (895, 548, 93, 111), (899, 535, 106, 126), (896, 537, 104, 129)],
    [(438, 790, 199, 208), (435, 795, 202, 210), (440, 788, 198, 206), (437, 792, 200, 207), (439, 789, 201, 209)]
]


#700Lux
rois = [
    [(989, 313, 119, 163)],
    [(1043, 434, 141, 145)],
    [(1058, 418, 147, 145)]
]
"""
#rois
rois = [
    [(193, 374, 90, 83)],
    [(107, 402, 87, 81)],
    [(164, 364, 79, 97)],
    [(477, 63, 107, 91)],
    [(459, 137, 128, 98)],
    [(434, 179, 176, 171)],
    [(267, 283, 135, 126)],
    [(197, 349, 102, 97)]
]



for i in range(1, numero_pastas + 1):
    folder_name = f"polarizado"
    folder_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Pasta {folder_name} não encontrada!")
        continue
    
    for j in range(1,9):
        video_name = f"v{j}.mp4"
        video_path = os.path.join(folder_path, video_name)
        
        if not os.path.exists(video_path):
            print(f"Vídeo {video_name} não encontrado na pasta {folder_name}!")
            continue
        
        # Pega as 5 ROIs específicas do vídeo
        video_rois = rois[j - 1]
        
        for roi_index, roi in enumerate(video_rois, start=1):
            try:
                # Usar a ROI específica para o vídeo
                pcrt = PCRT.fromVideoFile(video_path, roi=roi, displayVideo=False,rescaleFactor=0.5, exclusionMethod='best fit', exclusionCriteria=9999)
                print(pcrt)

                # Calcular os valores necessários
                crt_9010 = pcrt.calculate_crt_9010()
                crt_10010 = pcrt.calculate_crt_10010()
                crt_10010exp, uncert_10010exp = pcrt.calculate_crt_10010exp()
                
                
                # Diretório para salvar as imagens
                plot_dir = os.path.join(folder_path, "Gráficos")
                os.makedirs(plot_dir, exist_ok=True)

                # Salvar o gráfico de AvgIntensPlot e PCRTPlot
                avg_intens_plot_path = os.path.join(plot_dir, f"{video_name}_ROI{roi_index}_AvgIntensPlot.png")
                pcrt_plot_path = os.path.join(plot_dir, f"{video_name}_ROI{roi_index}_PCRTPlot.png")

                pcrt.saveAvgIntensPlot(avg_intens_plot_path)  
                pcrt.savePCRTPlot(pcrt_plot_path)  
                plt.close()  


                # Armazenar os resultados
                resultados.append({
                    "Pasta": folder_name,
                    "Video": video_name,
                    "ROI_Index": roi_index,
                    "ROI": roi,
                    "pCRT": pcrt.pCRT[0],
                    "uncert_pCRT": pcrt.pCRT[1],
                    "crt_9010": pcrt.crt_9010,
                    "crt_10010": pcrt.crt_10010,
                    "crt_10010exp": crt_10010exp,
                    "uncert_10010exp": uncert_10010exp
                })
                
                print(f"Processado: {folder_name}/{video_name} com ROI {roi} (ROI# {roi_index})")
            
            except Exception as e:
                print(f"Erro ao processar {folder_name}/{video_name} com ROI {roi} (ROI# {roi_index}): {e}")

# Salvar os resultados
df = pd.DataFrame(resultados)
df.to_excel(output_file, index=False)

print(f"Todos os resultados foram salvos no arquivo {output_file}")
