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
    ### set error indicator to default value of 0 for each subject
    ## if mean not 0, implies ERROR in file
   
def relative_risk(data, Raw_Ind=1):
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
    check_cov = recode_check(data, Raw_Ind)

    # Extract covariates
    NB_Cat= check_cov.NB_Cat
    AM_Cat = check_cov.AM_Cat
    AF_Cat = check_cov.AF_Cat
    NR_Cat = check_cov.NR_Cat
    R_Hyp = check_cov.R_Hyp
    CharRace = check_cov.CharRace

    # Set NB_Cat to NaN if it is -100 or -200
    if NB_Cat in [-100, -200]:
        NB_Cat = np.nan

    # Calculate PatternNumber
    if not np.isnan(NB_Cat) and not np.isnan(AM_Cat) and not np.isnan(AF_Cat) and not np.isnan(NR_Cat):
        PatternNumber = NB_Cat * 36 + AM_Cat * 12 + AF_Cat * 3 + NR_Cat + 1
    else:
        PatternNumber = np.nan

    LP1 = LP2 = np.nan

    # Select the appropriate beta coefficients
    if CharRace!= "??":
        race_index = int(data.Race.iloc[0])
        Beta = Beta_coeffs[race_index - 1]

        # Check if all covariates are available to calculate LP1 and LP2
        if not np.isnan(NB_Cat) and not np.isnan(AM_Cat) and not np.isnan(AF_Cat) and not np.isnan(NR_Cat) and not np.isnan(R_Hyp):
            LP1 = NB_Cat * Beta[0] + AM_Cat * Beta[1] + AF_Cat * Beta[2] + NR_Cat * Beta[3] + AF_Cat * NR_Cat * Beta[5] + np.log(R_Hyp)
            LP2 = LP1 + NB_Cat * Beta[4]


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

def absolute_risk(data):


