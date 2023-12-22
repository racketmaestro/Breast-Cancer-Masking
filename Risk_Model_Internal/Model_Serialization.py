import dill
import Gail_ModelV5

with open("Risk_Model_External/Risk_Prediction.model","wb") as file:
    dill.dump(Gail_ModelV5, file)

