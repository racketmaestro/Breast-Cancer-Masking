import dill
import pandas as pd
import json
import sys
import numpy as np


sys.path.append('Risk_Model_Internal')

import Data_Synth

with open("Risk_Model_External/Risk_Prediction.model","rb") as file:
    riskCalc = dill.load(file)

for i in range(500):
    Data_Synth.data_gen()

    with open('Risk_Model_External/BCRA_Data.json', 'r') as f:
        jsonData = json.load(f)
    
    data = pd.DataFrame([jsonData])

    print(data)
    
    personalRisk = riskCalc.RiskModel(data)
    riskDict = personalRisk.run_model()
    print(riskDict)
    






