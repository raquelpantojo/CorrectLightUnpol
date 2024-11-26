import os
import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.append("C:/Users/raque/Documents/GitHub/pyCRT")
from src.pyCRT import PCRT


# Definir o caminho base onde estão as pastas P1, P2, ..., P31
base_path ="J:\Meu Drive\TrabalhoNatan\Code_pCRT\ElasticidadePele"
output_file = "resultados_Pol_Lisov2.xlsx"
# Criar uma lista para armazenar os resultados
resultados = []
numero_pastas=1


"""
#Rugoso
rois = [
    [(310, 892, 108, 65), (315, 890, 105, 68), (305, 895, 110, 63), (312, 887, 107, 67), (308, 890, 106, 64)],
    [(822, 642, 70, 96), (825, 637, 52, 110), (807, 630, 71, 123), (794, 591, 81, 187), (795, 578, 142, 201)],
    [(361, 811, 152, 104), (348, 785, 197, 173), (386, 829, 104, 85), (375, 786, 141, 118), (363, 770, 193, 98)]
]
"""
#Esticado
rois = [
    [(393, 796, 139, 199), (390, 800, 135, 195), (395, 790, 140, 198), (398, 794, 137, 200), (391, 798, 138, 197)],
    [(894, 420, 93, 132), (890, 425, 95, 130), (896, 415, 92, 135), (892, 422, 94, 133), (895, 418, 93, 131)],
    [(505, 847, 145, 130), (510, 845, 143, 128), (502, 850, 147, 132), (507, 849, 144, 129), (503, 846, 146, 131)]
]


# Alterar o loop para iterar sobre cada ROI de cada vídeo
for i in range(1, numero_pastas + 1):
    folder_name = f"LisoPolarizadoP{i}"
    folder_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Pasta {folder_name} não encontrada!")
        continue
    
    for j in range(1, 4):
        video_name = f"Pol{j}.mp4"
        video_path = os.path.join(folder_path, video_name)
        
        if not os.path.exists(video_path):
            print(f"Vídeo {video_name} não encontrado na pasta {folder_name}!")
            continue
        
        # Pega as 5 ROIs específicas do vídeo
        video_rois = rois[j - 1]
        
        for roi_index, roi in enumerate(video_rois, start=1):
            try:
                # Usar a ROI específica para o vídeo
                pcrt = PCRT.fromVideoFile(video_path, roi=roi, displayVideo=False, exclusionMethod='best fit', exclusionCriteria=9999)
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
                
                print(f"Processado: {folder_name}/{video_name} com ROI {roi} (ROI #{roi_index})")
            
            except Exception as e:
                print(f"Erro ao processar {folder_name}/{video_name} com ROI {roi} (ROI #{roi_index}): {e}")

# Salvar os resultados
df = pd.DataFrame(resultados)
df.to_excel(output_file, index=False)

print(f"Todos os resultados foram salvos no arquivo {output_file}")
