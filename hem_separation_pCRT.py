import os
import pandas as pd
import sys
import cv2 as cv
import matplotlib.pyplot as plt

sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT")
from src.pyCRT import PCRT
import hem_separation as hs

# Caminho base e saída
base_path = "C:/Users/RaquelPantojo/Desktop/ElasticidadePele"
#output_file = "resultados_Desp_RugosoHemoglobina.xlsx"
output_file = "resultados_Pol_RugosoHemoglobina.xlsx"

"""
#Rugoso - despolarizado
rois = [
    [(787, 560, 124, 200), (757, 550, 162, 220), (780, 550, 130, 190), (795, 565, 115, 205), (785, 555, 125, 198)],
    [(344, 839, 114, 71), (362, 767, 154, 85), (335, 747, 194, 87), (343, 760, 78, 72), (343, 750, 100, 72)],
    [(943, 354, 93, 181), (952, 350, 95, 200), (940, 350, 95, 185), (945, 355, 92, 180), (948, 348, 94, 178)]
]

#Liso- despolarizado
rois = [
    [(339, 855, 176, 137), (386, 934, 114, 68), (387, 799, 147, 203), (341, 853, 177, 136), (338, 857, 175, 139)],
    [(897, 536, 105, 127), (856, 551, 106, 121), (895, 548, 93, 111), (899, 535, 106, 126), (896, 537, 104, 129)],
    [(438, 790, 199, 208), (435, 795, 202, 210), (440, 788, 198, 206), (437, 792, 200, 207), (439, 789, 201, 209)]
]

"""

#Rugoso - polarizado 
rois = [
    [(310, 892, 108, 65), (315, 890, 105, 68), (305, 895, 110, 63), (312, 887, 107, 67), (308, 890, 106, 64)],
    [(822, 642, 70, 96), (825, 637, 52, 110), (807, 630, 71, 123), (794, 591, 81, 187), (795, 578, 142, 201)],
    [(361, 811, 152, 104), (348, 785, 197, 173), (386, 829, 104, 85), (375, 786, 141, 118), (363, 770, 193, 98)]
]
"""
#Liso - polarizado 
rois = [
    [(393, 796, 139, 199), (390, 800, 135, 195), (395, 790, 140, 198), (398, 794, 137, 200), (391, 798, 138, 197)],
    [(894, 420, 93, 132), (890, 425, 95, 130), (896, 415, 92, 135), (892, 422, 94, 133), (895, 418, 93, 131)],
    [(505, 847, 145, 130), (510, 845, 143, 128), (502, 850, 147, 132), (507, 849, 144, 129), (503, 846, 146, 131)]
]

"""


numero_pastas = 1
resultados = []

for i in range(1, numero_pastas + 1):
    folder_name = f"RugosoPolarizadoP1"
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.exists(folder_path):
        print(f"Pasta {folder_name} não encontrada!")
        continue

    for j in range(1, 4) :
        video_name = f"Pol{j}.mp4"
        video_path = os.path.join(folder_path, video_name)

        if not os.path.exists(video_path):
            print(f"Vídeo {video_name} não encontrado na pasta {folder_name}!")
            continue

        video_rois = rois[j - 1]

        output_dir = os.path.join(folder_path, "ProcessamentoHemoglobina")
        os.makedirs(output_dir, exist_ok=True)

        for roi_index, roi in enumerate(video_rois, start=1): 
            try:
                # Processamento do canal de hemoglobina
                cap = cv.VideoCapture(video_path)

                fps = int(cap.get(cv.CAP_PROP_FPS))
                frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                codec = cv.VideoWriter_fourcc(*'mp4v')  

                x, y, w, h = roi
                if x + w > frame_width or y + h > frame_height:
                    print(f"ROI {roi} está fora dos limites do quadro no vídeo {video_name}!")
                    continue

                hem_video_path = os.path.join(output_dir, f"{video_name}_ROI{roi_index}_Hemoglobina.mp4")
                out = cv.VideoWriter(hem_video_path, codec, fps, (w, h), isColor=False)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    roi_frame = frame[y:y+h, x:x+w]
                    hem = hs.hem_separation(roi_frame)
                    hem_uint8 = (hem * 255).astype('uint8')
                    out.write(hem_uint8)

                cap.release()
                out.release()
                

                # PCRT do vídeo de hemoglobina
                pcrt = PCRT.fromVideoFile(hem_video_path, roi="all", displayVideo=False, exclusionMethod='best fit', exclusionCriteria=9999)
                print(pcrt)

                crt_9010 = pcrt.calculate_crt_9010()
                crt_10010 = pcrt.calculate_crt_10010()
                crt_10010exp, uncert_10010exp = pcrt.calculate_crt_10010exp()

                
                avg_intens_plot_hemo = os.path.join(output_dir, f"{video_name}_ROI{roi_index}_AvgIntensPlotHemo.png")
                pcrt_plot_hemo = os.path.join(output_dir, f"{video_name}_ROI{roi_index}_PCRTPlotHemo.png")

                pcrt.saveAvgIntensPlot(avg_intens_plot_hemo)
                pcrt.savePCRTPlot(pcrt_plot_hemo)
                plt.close()

                
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
                    "uncert_10010exp": uncert_10010exp,
                })

                print(f"Processado: {folder_name}/{video_name} com ROI {roi} (ROI #{roi_index}) usando mapa de hemoglobina")

            except Exception as e:
                print(f"Erro ao processar {folder_name}/{video_name} com ROI {roi} (ROI #{roi_index}): {e}")

df = pd.DataFrame(resultados)
df.to_excel(output_file, index=False)

print(f"Todos os resultados foram salvos no arquivo {output_file}")
