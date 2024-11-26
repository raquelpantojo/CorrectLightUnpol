import numpy as np
import sys
sys.path.append("C:/Users/RaquelPantojo/Documents/GitHub/pyCRT")
from src.pyCRT import PCRT  

# Creating some arbitrary arrays for illustration
timeScdsArr = np.linspace(0, 1, 100)
avgIntensArr = np.array([[n, n*2, n*3] for n in np.exp(-timeScdsArr)])  # Fixed parentheses

pcrt = PCRT(timeScdsArr, avgIntensArr)
pcrt.showAvgIntensPlot()
pcrt.showPCRTPlot()