import cv2 as cv
import numpy as np
import os
import sys

sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT")
from src.pyCRT import PCRT


def correct_flat_field_bgr_video(video_path, video_path_white, output_path, gamma):
    # Abre o vídeo principal e o vídeo de referência (imgWhite)
    cap = cv.VideoCapture(video_path)
    cap_white = cv.VideoCapture(video_path_white)

    if not cap.isOpened() or not cap_white.isOpened():
        raise FileNotFoundError("Um ou ambos os vídeos não foram encontrados ou não puderam ser abertos.")

    cap_white.set(cv.CAP_PROP_POS_FRAMES, 10)
    ret_white, frame_white = cap_white.read()
    if not ret_white:
        raise ValueError("Não foi possível ler o quadro do vídeo de referência (imgWhite).")
    cap_white.release()  

    # Normaliza o quadro de referência
    white_f32 = frame_white.astype(np.float32) / 255
    white_gamma = white_f32 ** (gamma)
    white_ycrcb = cv.cvtColor(white_gamma, cv.COLOR_BGR2YCrCb)
    white_y = white_ycrcb[..., 0]

   
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        frame_f32 = frame.astype(np.float32) / 255
        frame_gamma = frame_f32 ** (gamma)
        frame_ycrcb = cv.cvtColor(frame_gamma, cv.COLOR_BGR2YCrCb)
        frame_y = frame_ycrcb[..., 0]

       
        correction = np.divide(
            frame_y,
            white_y,
            out=np.zeros_like(frame_y),
            where=white_y != 0
        )
        correction_mean = np.mean(correction)
        frame_y_mean = np.mean(frame_y)
        C = frame_y_mean / correction_mean  

        frame_y_corrected = C * correction
        frame_ycrcb[..., 0] = frame_y_corrected

        corrected_frame = np.clip(cv.cvtColor(frame_ycrcb, cv.COLOR_YCrCb2BGR), 0, 1)
        corrected_frame = (corrected_frame ** gamma * 255).astype(np.uint8)

        
        out.write(corrected_frame)

    # Libera os recursos
    cap.release()
    out.release()


# Caminho base para os arquivos do projeto
base_path = "C:/Users/RaquelPantojo/Documents/GitHub/CorrectLightUnpol"
folder_name = "DespolarizadoP5"
video_name = "v9.mp4"
video_name_white = "imgWhite2.mp4"
output_video_name = "corrected_v10_gamma=1.mp4"

video_path = os.path.join(base_path, folder_name, video_name)
video_path_white = os.path.join(base_path, folder_name, video_name_white)
output_video_path = os.path.join(base_path, folder_name, output_video_name)

gamma =1

if not os.path.exists(video_path) or not os.path.exists(video_path_white):
    print(f"Um ou ambos os vídeos não foram encontrados!")
    sys.exit(1)

# Aplica a correção ao vídeo
correct_flat_field_bgr_video(video_path, video_path_white, output_video_path, gamma=gamma)


"""

# Usa o vídeo corrigido com o PCRT
pcrt = PCRT.fromVideoFile(output_video_path, exclusionMethod='best fit', exclusionCriteria=9999)

pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()""



"""

