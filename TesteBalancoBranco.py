# teste corrigindo balança de branco 
import cv2
import numpy as np



import moviepy.editor as mp
from moviepy.editor import VideoFileClip

# Função para corrigir o balanço de branco
def corrigir_balanco_branco(frame):
    # Converter de BGR para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Calcular as médias dos canais R, G e B
    medios = np.mean(frame_rgb, axis=(0, 1))  # média ao longo da altura e largura
    
    # Calcular o fator de correção (normalizando o G)
    fator = np.mean(medios) / medios  # usar a média dos 3 canais como base
    
    # Corrigir os canais multiplicando pelos fatores
    frame_corrigido = frame_rgb * fator
    frame_corrigido = np.clip(frame_corrigido, 0, 255)  # manter dentro dos limites de [0, 255]
    
    return cv2.cvtColor(frame_corrigido.astype(np.uint8), cv2.COLOR_RGB2BGR)

# Carregar o vídeo
video = mp.VideoFileClip("EduLedDesp4.mp4")

# Função para processar cada frame
def processar_frame(frame):
    frame = np.array(frame)
    return corrigir_balanco_branco(frame)

# Aplicar a correção no vídeo
video_corrigido = video.fl_image(processar_frame)

# Salvar o vídeo corrigido
video_corrigido.write_videofile("EduLedDesp4WB.mp4", codec="libx264")
