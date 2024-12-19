import cv2 as cv
import numpy as np
import os
import sys

#sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT")
sys.path.append("C:/Users/raque/OneDrive/Documentos/GitHub/pyCRT")
from src.pyCRT import PCRT


base_path = "C:/Users/raque/OneDrive/Documentos/GitHub"
folder_name = "CorrectLightUnpol"


'''
output_video_name = "v5.mp4"
output_video_path = os.path.join(base_path, folder_name, output_video_name)

pcrt = PCRT.fromVideoFile(output_video_path, exclusionMethod='best fit', exclusionCriteria=9999)
pcrt.showPCRTPlot()



output_video_name = "corrected_video_v5.mp4"
output_video_path = os.path.join(base_path, folder_name, output_video_name)
pcrt = PCRT.fromVideoFile(output_video_path, exclusionMethod='best fit', exclusionCriteria=9999)
pcrt.showPCRTPlot()
'''

'''
output_video_name = "v6.mp4"
output_video_path = os.path.join(base_path, folder_name, output_video_name)
pcrt = PCRT.fromVideoFile(output_video_path, exclusionMethod='best fit', exclusionCriteria=9999)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()


output_video_name = "corrected_video_v6.mp4"
output_video_path = os.path.join(base_path, folder_name, output_video_name)
pcrt = PCRT.fromVideoFile(output_video_path, exclusionMethod='best fit', exclusionCriteria=9999)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()


output_video_name = "v7.mp4"
output_video_path = os.path.join(base_path, folder_name, output_video_name)
roi= (1040, 219, 248, 200)
pcrt = PCRT.fromVideoFile(output_video_path, roi=roi, displayVideo=False, exclusionMethod='best fit', exclusionCriteria=9999)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()


output_video_name = "corrected_video_v7.mp4"
output_video_path = os.path.join(base_path, folder_name, output_video_name)
roi= (1040, 219, 248, 200)
pcrt = PCRT.fromVideoFile(output_video_path, roi=roi, displayVideo=False, exclusionMethod='best fit', exclusionCriteria=9999)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()


output_video_name = "corrected_v7_gamma=2.2.mp4"
output_video_path = os.path.join(base_path, folder_name, output_video_name)
roi= (1040, 219, 248, 200)
pcrt = PCRT.fromVideoFile(output_video_path, roi=roi, displayVideo=False, exclusionMethod='best fit', exclusionCriteria=9999)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()


'''

output_video_name = "raqueltestecasa.mp4"
output_video_path = os.path.join(base_path, folder_name, output_video_name)
roi= (698, 346, 375, 376)
pcrt = PCRT.fromVideoFile(output_video_path, roi=roi, displayVideo=False, exclusionMethod='best fit', exclusionCriteria=9999,fromTime=11.5)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()
outputFilePCRT = f"{output_video_name}.png"
pcrt.savePCRTPlot(outputFilePCRT)

