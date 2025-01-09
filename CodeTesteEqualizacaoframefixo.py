# 09-01
# teste equalização em um unico frame 
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd


def equalize_video_with_reference(input_path, output_path, reference_frame_idx):
    # Abre o vídeo de entrada
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    # Obtém informações sobre o vídeo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Formato de saída (MP4)
    
    # Cria o objeto de gravação do vídeo
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)
    
    # Encontra o frame de referência
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame_idx)
    ret, reference_frame = cap.read()
    
    if not ret:
        print(f"Erro ao capturar o frame de referência (frame {reference_frame_idx}).")
        cap.release()
        return
    
    # Converte o frame de referência para escala de cinza e calcula seu histograma
    reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
    reference_hist, bins = np.histogram(reference_gray.flatten(), 256, [0, 256])
    cdf_reference = np.cumsum(reference_hist)
    cdf_reference_normalized = cdf_reference * float(reference_hist.max()) / cdf_reference.max()
    
    # Processa os frames restantes
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Retorna ao início do vídeo
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converte o frame atual para escala de cinza
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplica a equalização baseada no CDF do frame de referência
        hist, bins = np.histogram(gray_frame.flatten(), 256, [0, 256])
        cdf = np.cumsum(hist)
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        
        cdf_m = np.ma.masked_equal(cdf, 0)  # Ignora zeros
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        
        # Aplica a transformação do frame de referência ao frame atual
        equalized_frame = cdf[gray_frame]
        
        # Reconverte para BGR para salvar no vídeo
        equalized_bgr_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)
        out.write(equalized_bgr_frame)
    
    # Libera os objetos
    cap.release()
    out.release()
    print("Vídeo processado e salvo com sucesso!")

# Caminhos de entrada e saída
input_video_path = "SeiyLedDesp4.mp4"

#reference_frame = 30  # Escolha o frame de referência

for reference_frame in range(40, 400, 20):
    output_video_path = "video_equalizadoSeyi.mp4"
    equalize_video_with_reference(input_video_path, output_video_path, reference_frame)
    ############################ Inicalizando o programa ####################################
    # Verifica o caminho do vídeo
    video_path = os.path.join(output_video_path)
    if not os.path.exists(video_path):
        print(f"Vídeo {output_video_path} não encontrado!")
        sys.exit(1)

    # Inicializa a captura de vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        sys.exit(1)



    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)



    roi1= (457, 571, 165, 215)


    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

        #roi5_frame = frame[int(roi5[1]):int(roi5[1] + roi5[3]), int(roi5[0]):int(roi5[0] + roi5[2])] 

        # ROI Gray
        RoiGreen1.append(np.mean(roi1_frame[:, :, 1]))



        time_stamps.append(frame_count / fps)
        frame_count += 1

    cap.release()




    RoiGreen1=np.array(RoiGreen1)
    time_stamps = np.array(time_stamps)

    plt.figure(figsize=(12, 6))
    plt.plot(time_stamps, RoiGreen1, label=f"Equalização Frame {reference_frame}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Intensidade média do canal cinza - histograma Equalizado")
    plt.legend()
    plt.savefig(f"grafico_frame_{reference_frame}.png") 
    plt.show()


