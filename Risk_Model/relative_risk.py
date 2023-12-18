import sys

import numpy as np
import pandas as pd

data = pd.read_csv(r"C:\Users\hanzi\PycharmProjects\Breast-Cancer-Masking\Risk_Model\BCRA_Data.csv")
data.head()


def recode_check(data, Raw_Ind=1):
    ### set error indicator to default value of 0 for each subject
    ## if mean not 0, implies ERROR in file
    Error_Ind = np.zeros(data.shape[0])
    ### test for consistency of T1 (initial age) and T2 (projection age)
    set_T1_missing = data.T1.values.copy()

    set_T2_missing = data.T2.values.copy()
    set_T1_missing[np.where((data.T1 < 20) | (data.T1 >= 90) | (data.T1 >= data.T2))] = np.nan
    set_T2_missing[(data.T2.values > 90) | (data.T1.values >= data.T2.values)] = np.nan
    Error_Ind[np.isnan(set_T1_missing)] = 1
    Error_Ind[np.isnan(set_T2_missing)] = 1

    ### RR covariates are in raw/original format
    if Raw_Ind == 1:
        ### test for consistency of NumBiop (#biopsies) and Hyperplasia
        ## set NB_Cat to default value of -1
        NB_Cat = np.repeat(-1., data.shape[0])

        ## REQUIREMENT (A)
        NB_Cat[((data.N_Biop == 0) | (data.N_Biop == 99)) & (data.HypPlas != 99)] = -100
        Error_Ind[np.where(NB_Cat == -100)] = 1

        ## REQUIREMENT (B)
        NB_Cat[((data.N_Biop > 0) & (data.N_Biop < 99)) & (
                (data.HypPlas != 0) & (data.HypPlas != 1) & (data.HypPlas != 99))] = -200
        Error_Ind[NB_Cat == -200] = 1

        ### editing and recoding for N_Biop
        NB_Cat[(NB_Cat == -1) & ((data.N_Biop == 0) | (data.N_Biop == 99))] = 0

        NB_Cat[(NB_Cat == -1) & (data.N_Biop == 1)] = 1
        NB_Cat[(NB_Cat == -1) & ((data.N_Biop >= 2) | (data.N_Biop != 99))] = 2
        NB_Cat[NB_Cat == -1] = np.nan

        ### editing and recoding for AgeMen
        AM_Cat = np.repeat(np.nan, data.shape[0])
        AM_Cat[((data.AgeMen >= 14) & (data.AgeMen <= data.T1)) | (data.AgeMen == 99)] = 0
        AM_Cat[(data.AgeMen >= 12) & (data.AgeMen < 14)] = 1
        AM_Cat[(data.AgeMen > 0) & (data.AgeMen < 12)] = 2
        AM_Cat[(data.AgeMen > data.T1) & (data.AgeMen != 99)] = np.nan
        ## for African-Americans AgeMen code 2 (age <= 11) grouped with code 1(age == 12 or 13)
        AM_Cat[(data.Race == 2) & (AM_Cat == 2)] = 1

        ### editing and recoding for Age1st
        AF_Cat = np.repeat(np.nan, data.shape[0])
        AF_Cat[(data.Age1st < 20) | (data.Age1st == 99)] = 0
        AF_Cat[(data.Age1st >= 20) & (data.Age1st < 25)] = 1
        AF_Cat[(((data.Age1st >= 25) & (data.Age1st < 30))) | (data.Age1st == 98)] = 2
        AF_Cat[(data.Age1st >= 30) & (data.Age1st < 98)] = 3
        AF_Cat[(data.Age1st < data.AgeMen) & (data.AgeMen != 99)] = np.nan
        AF_Cat[(data.Age1st > data.T1) & (data.Age1st < 98)] = np.nan
        ## for African-Americans Age1st is not a RR covariate and not in RR model, set to 0
        AF_Cat[data.Race == 2] = 0

        ### editing and recoding for N_Rels
        NR_Cat = np.repeat(np.nan, data.shape[0])
        NR_Cat[(data.N_Rels == 0) | (data.N_Rels == 99)] = 0
        NR_Cat[data.N_Rels == 1] = 1
        NR_Cat[(data.N_Rels >= 2) & (data.N_Rels < 99)] = 2
        ## for Asian-American NR_Cat=2 is pooled with NR_Cat=2
        NR_Cat[((data.Race >= 6) & (data.Race <= 11)) & (NR_Cat == 2)] = 1

    ### Raw_Ind=0 means RR covariates have already been re-coded to 0, 1, 2 or 3 (when necessary)
    ### edit/consistency checks for all relative four risk covariates not performed when Raw_Ind=0. (use this option at your own risk)
    if Raw_Ind == 0:
        NB_Cat = data.N_Biop
        AM_Cat = data.AgeMen
        AF_Cat = data.Age1st
        NR_Cat = data.N_Rels

    ### setting RR multiplicative factor for atypical hyperplasia
    R_Hyp = np.repeat(np.nan, data.shape[0])
    R_Hyp[NB_Cat == 0] = 1.00
    R_Hyp[((NB_Cat != -100) & (NB_Cat > 0)) & (data.HypPlas == 0)] = 0.93
    R_Hyp[((NB_Cat != -100) & (NB_Cat > 0)) & (data.HypPlas == 1)] = 1.82
    R_Hyp[((NB_Cat != -100) & (NB_Cat > 0)) & (data.HypPlas == 99)] = 1.00

    set_HyperP_missing = data.HypPlas.values
    set_R_Hyp_missing = R_Hyp.copy()
    set_HyperP_missing[NB_Cat == -100] = -100
    set_R_Hyp_missing[NB_Cat == -100] = -100
    set_HyperP_missing[NB_Cat == -200] = -200
    set_R_Hyp_missing[NB_Cat == -200] = -200

    set_Race_missing = data.Race.values
    Race_range = np.array(range(1, 12))
    set_Race_missing[-data.Race.isin(Race_range)] = -300

    Error_Ind[(np.isnan(NB_Cat)) | (np.isnan(AM_Cat)) | (np.isnan(AF_Cat)) | (np.isnan(NR_Cat)) | (
            set_Race_missing == -300)] = 1

    ### african-american RR model from CARE study:(1) eliminates Age1st from model;(2) groups AgeMen=2 with AgeMen=1;
    ## setting AF_Cat=0 eliminates Age1st and its interaction from RR model;
    AF_Cat[data.Race == 2] = 0
    ## group AgeMen RR level 2 with 1;
    AM_Cat[(data.Race == 2) & (AM_Cat == 2)] = 1

    ### hispanic RR model from San Francisco Bay Area Breast Cancer Study (SFBCS):
    ###         (1) groups N_Biop ge 2 with N_Biop eq 1
    ###         (2) eliminates  AgeMen from model for US Born hispanic women
    ###         (3) group Age1st=25-29 with Age1st=20-24 and code as 1
    ###             for   Age1st=30+, 98 (nulliparous)       code as 2
    ###         (4) groups N_Rels=2 with N_Rels=1;
    NB_Cat[(data.Race.isin([3, 5])) & (data.N_Biop.isin([0, 99]))] = 0
    NB_Cat[(data.Race.isin([3, 5])) & (NB_Cat == 2)] = 1
    AM_Cat[data.Race == 3] = 0

    AF_Cat[(data.Race.isin([3, 5])) & (data.Age1st != 98) & (AF_Cat == 2)] = 1
    AF_Cat[(data.Race.isin([3, 5])) & (AF_Cat == 3)] = 2
    NR_Cat[(data.Race.isin([3, 5])) & (NR_Cat == 2)] = 1

    ### for asian-americans NR_Cat=2 is pooled with NR_Cat=1;
    NR_Cat[(data.Race >= 6) & (data.Race <= 11) & (NR_Cat == 2)] = 1

    CharRace = np.repeat('??', data.shape[0])
    CharRace[data.Race == 1] = "Wh"  # white SEER 1983:87 BrCa Rate
    CharRace[data.Race == 2] = "AA"  # african-american
    CharRace[data.Race == 3] = "HU"  # hispanic-american (US born)
    CharRace[data.Race == 4] = "NA"  # other (native american and unknown race)
    CharRace[data.Race == 5] = "HF"  # hispanic-american (foreign born)
    CharRace[data.Race == 6] = "Ch"  # chinese
    CharRace[data.Race == 7] = "Ja"  # japanese
    CharRace[data.Race == 8] = "Fi"  # filipino
    CharRace[data.Race == 9] = "Hw"  # hawaiian
    CharRace[data.Race == 10] = "oP"  # other pacific islander
    CharRace[data.Race == 11] = "oA"  # other asian

    #     recode_check= cbind(Error_Ind, set_T1_missing, set_T2_missing, NB_Cat, AM_Cat, AF_Cat, NR_Cat, R_Hyp, set_HyperP_missing, set_R_Hyp_missing, set_Race_missing, CharRace)
    recode_check = pd.DataFrame({'Error_Ind': Error_Ind, 'set_T1_missing': set_T1_missing,
                                 'set_T2_missing': set_T2_missing, 'NB_Cat': NB_Cat,
                                 'AM_Cat': AM_Cat, 'AF_Cat': AF_Cat, 'NR_Cat': NR_Cat,
                                 'R_Hyp': R_Hyp, 'set_HyperP_missing': set_HyperP_missing,
                                 'set_R_Hyp_missing': set_R_Hyp_missing,
                                 'set_Race_missing': set_Race_missing, 'CharRace': CharRace})
    return (recode_check.iloc[0])

r_ch = recode_check(data, Raw_Ind=1)
print(r_ch)
# Everything above is from the OG code

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

