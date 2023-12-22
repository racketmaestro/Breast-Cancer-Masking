import numpy as np
import json

def data_synth_norm():

    # Define mean and standard deviation for T1, ageMen, and age1st
    T1_mean, T1_std = 60, 10
    ageMen_mean, ageMen_std = 12, 1.5
    age1st_mean, age1st_std = 24, 3

    # Generate normally distributed data and apply constraints
    T1 = min(max(int(np.random.normal(T1_mean, T1_std)), 35), 85)
    AgeMen = min(max(int(np.random.normal(ageMen_mean, ageMen_std)), 7), 15)
    
    # Ensure age1st is between 15 and T1 or 98
    has_children = np.random.choice([True, False])

    if has_children:
        # Generate normally distributed age between 15 and T1
        Age1st = min(max(int(np.random.normal(age1st_mean, age1st_std)), 15), T1)
    else:
        # Assign 98 to represent no children
        Age1st = 98

    # Categorical parameters are sampled from their respective categories
    N_Biop = np.random.choice([0, 1, 2, 99])
    Race = np.random.choice(range(1, 12))
    N_Rels = np.random.choice([0, 1, 2, 99])
    HypPlas = np.random.choice([0, 1, 99])
    BiRads = np.random.choice(range(1, 5))

    # Create and return the dictionary
    return {
        'T1': T1,
        'biopCat': N_Biop,
        'race': Race,
        'ageMen': AgeMen,
        'age1st': Age1st,
        'nRels': N_Rels,
        'hypPlas': HypPlas,
        'biRads': BiRads
    }

def convert_numpy_int_to_python_int(data):
    for key, value in data.items():
        if isinstance(value, np.integer):
            data[key] = int(value)
    return data

# Generate the data
data_dict = data_synth_norm()

# Convert NumPy integers to Python integers
data_dict = convert_numpy_int_to_python_int(data_dict)

with open("Risk_Model_External/BCRA_Data.json", 'w') as file:
    json.dump(data_dict, file, indent=4)
