import dill
import Gail_Model_Final

with open("Risk_Model_External/Risk_Prediction.model","wb") as file:
    dill.dump(Gail_Model_Final, file)

