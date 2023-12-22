import dill
import pandas as pd
import json
import sys
import numpy as np

sys.path.append('Risk_Model_Internal')

with open("Risk_Model_External/Risk_Prediction.model","rb") as file:
    riskCalc = dill.load(file)

data = pd.read_csv("Risk_Model_External\BCRA_Data.csv")

personalRisk = riskCalc.RiskModel(data)
riskDict = personalRisk.run_model()
print(riskDict)






