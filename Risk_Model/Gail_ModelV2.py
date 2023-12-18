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
    
    ### hispanic RR model from San Francisco Bay Area Breast Cancer Study (SFBCS):
    ###         (1) groups N_Biop ge 2 with N_Biop eq 1
    ###         (2) eliminates  AgeMen from model for US Born hispanic women
    ###         (3) group Age1st=25-29 with Age1st=20-24 and code as 1
    ###             for   Age1st=30+, 98 (nulliparous)       code as 2
    ###         (4) groups N_Rels=2 with N_Rels=1;
    
    if race in [3,5] and biopCat in [0,99]: biopCat = 0
    elif race in [3,5] and biopCat == 2: biopCat = 1
    elif race == 3: menCat =0
    elif race in [3,5] and age1st != 98 and birthCat == 2: birthCat = 1
    elif race in [3,5] and birthCat == 3: birthCat = 2
    elif race in [3,5] and relativesCat == 2: relativesCat = 1


    # race == 1 : "Wh"      white SEER 1983:87 BrCa Rate
    # race == 2 : "AA"      african-american
    # race == 3 : "HU"      hispanic-american (US born)
    # race == 4 : "NA"      other (native american and unknown race)
    # race == 5 : "HF"      hispanic-american (foreign born)
    # race == 6 : "Ch"      chinese
    # race == 7 : "Ja"      japanese
    # race == 8 : "Fi"      filipino
    # race == 9 : "Hw"      hawaiian
    # race == 10 : "oP"     other pacific islander
    # race == 11 : "oA"     other asian
        
    recodedData = pd.DataFrame({
        'T1': [T1],
        'T2': [T2],
        'biopCat': [biopCat],
        'menCat': [menCat],
        'birthCat': [birthCat],
        'relativesCat': [relativesCat],
        'hypRiskScale': [hypRiskScale],
        'race': [race]
    })

    return recodedData 

recodedDataFrame = recode(data)
print(recodedDataFrame.head())

def relative_risk(data):
    # Define beta coefficients for different races
    Beta_coeffs = [
        [0.5292641686, 0.0940103059, 0.2186262218, 0.9583027845, -0.2880424830, -0.1908113865],  # White
        [0.1822121131, 0.2672530336, 0.0, 0.4757242578, -0.1119411682, 0.0],  # Black
        [0.0970783641, 0.0000000000, 0.2318368334, 0.166685441, 0.0000000000, 0.0000000000],  # Hispanic US born
        [0.4798624017, 0.2593922322, 0.4669246218, 0.9076679727, 0.0000000000, 0.0000000000],  # Hispanic Foreign born
        [0.5292641686, 0.0940103059, 0.2186262218, 0.9583027845, -0.2880424830, -0.1908113865],  # Other
        [0.55263612260619, 0.07499257592975, 0.27638268294593, 0.79185633720481, 0.0, 0.0]  # Asian
    ]

    # Obtain covariates using recode_check function
    check_cov = recode(data)

    # Extract covariates
    biopCat= check_cov.at[0,'biopCat']
    menCat = check_cov.at[0,'menCat']
    birthCat = check_cov.at[0,'birthCat']
    relativesCat = check_cov.at[0,'relativesCat']
    hypRiskScale = check_cov.at[0,'hypRiskScale']
    race = check_cov.at[0,'race']

    # Calculate PatternNumber
    if not np.isnan(birthCat):
        PatternNumber = biopCat * 36 + menCat * 12 + birthCat * 3 + relativesCat + 1
    else:
        PatternNumber = np.nan


    # Select the appropriate beta coefficients
    Beta = Beta_coeffs[race - 1]

    # Check if all covariates are available to calculate LP1 and LP2
    if not np.isnan(birthCat):
        LP1 = biopCat * Beta[0] + menCat * Beta[1] + birthCat * Beta[2] + relativesCat * Beta[3] + birthCat * relativesCat * Beta[5] + np.log(hypRiskScale)
        LP2 = LP1 + biopCat * Beta[4]
    else:
        LP1 = LP2 = np.nan


    # Calculate Relative Risks
    RR_Star1 = np.exp(LP1) if not np.isnan(LP1) else np.nan
    RR_Star2 = np.exp(LP2) if not np.isnan(LP2) else np.nan

    # Create a DataFrame for the result
    RR_Star = pd.DataFrame({'RR_Star1': [RR_Star1], 'RR_Star2': [RR_Star2], 'PatternNumber': [PatternNumber]})

    return RR_Star

# Example usage
# Assuming 'data' is a pandas Series or a one-row DataFrame representing the individual
rr = relative_risk(data)
print(rr)

