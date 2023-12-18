import numpy as np 
import pandas as pd
data = pd.read_csv("C:\\Users\\camsa\BreastCancerProject\Breast-Cancer-Masking\Risk_Model\BCRA_Data.csv")
#print(data.head())

def recode(data):

    # Initialising variables
    T1 = data.at[0, 'T1']
    T2 = data.at[0, 'T2']
    biopCat = data.at[0, 'N_Biop']
    race = data.at[0, 'Race']
    ageMen = data.at[0, 'AgeMen']
    age1st = data.at[0, 'Age1st']
    nRels = data.at[0, 'N_Rels']
    hypPlas = data.at[0, 'HypPlas']
    menCat = np.nan
    birthCat = np.nan
    relativesCat = np.nan
    hypRiskScale = np.nan

    # Categorising menarchy age
    if (ageMen >= 14 and ageMen <= T1) or ageMen == 99: menCat = 0  
    elif ageMen >= 12 and ageMen < 14: menCat = 1
    elif ageMen > 7 and ageMen < 12: menCat = 2      

    if menCat == 2 and race == 2: menCat = 1 #  for African-Americans AgeMen code 2 (age <= 11) grouped with code 1(age == 12 or 13)

    # Categorising age of first birth
    if age1st < 20 or age1st == 99: birthCat = 0
    elif age1st >= 20 and age1st < 25: birthCat = 1
    elif (age1st >= 25 and age1st < 30) or age1st == 98: birthCat = 2
    elif age1st >= 30 and age1st < 98: birthCat = 3
    elif age1st > T1 and age1st < 98: birthCat = np.nan
    
    if race == 2: birthCat = 0 # for African-Americans Age1st is not a RR covariate and not in RR model, set to 0

    ## Categorising number of relatives with BRCA gene
    if nRels == 0 or nRels == 99: relativesCat = 0
    elif nRels == 1: relativesCat = 1
    elif nRels >= 2 and nRels < 99: relativesCat = 2
    
    if (race >= 6 and race <= 11) and relativesCat == 2: relativesCat = 1 # for Asians relativesCat = 2 is pooled with relativesCat = 1

    # Scaling by the risk of hyperplasia
    if biopCat == 0:
        hypRiskScale = 1.00
    elif biopCat > 0:
        if hypPlas == 0:
            hypRiskScale = 0.93
        elif hypPlas == 1:
            hypRiskScale = 1.82
        elif hypPlas == 99:
            hypRiskScale = 1.00

    # Recoding race data
    charRace = '??'
    raceDict = {1: "Wh", 2: "AA", 3: "HU", 4: "NA", 5: "HF", 6: "Ch", 
                 7: "Ja", 8: "Fi", 9: "Hw", 10: "oP", 11: "oA"}
    
    if race in raceDict:
        charRace = raceDict[race]
        
    recodedData = pd.DataFrame({
        'T1': [T1],
        'T2': [T2],
        'biopCat': [biopCat],
        'menCat': [menCat],
        'birthCat': [birthCat],
        'relativesCat': [relativesCat],
        'hypRiskScale': [hypRiskScale],
        'charRace': [charRace]
    })

    return recodedData 

recodedDataFrame = recode(data)
print(recodedDataFrame)


        




