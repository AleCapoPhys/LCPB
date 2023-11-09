import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import uproot
from tqdm import tqdm
import os
import numpy as np
from dataclasses import dataclass
from ROOT import TLorentzVector
import math


KEEP = [
        "nElectron",
        "nMuon",
        "Electron_mass",
        "Electron_charge",
        "Electron_pt",
        "Electron_cutBased",
        "Electron_eta",
        "Electron_phi",
        "Electron_pfRelIso03_all",
        "Muon_charge",
        "Muon_mass",
        "Muon_pt",
        "Muon_tightId",
        "Muon_pt",
        "Muon_mediumId",
        "Muon_mass",
        "Muon_eta",
        "Muon_phi",
        "Muon_pfRelIso03_all",
        "Muon_pfRelIso04_all",
        "nJet",
        "Jet_pt",
        "Jet_mass",
         "Jet_phi",
        "Jet_eta",
        "Muon_jetIdx",
        "eventWeightLumi",
        "Jet_btagCSVV2",
        "CaloMET_phi",
        "MET_covXX",
        "MET_covXY",
        "MET_covYY",
        "MET_phi",
        "MET_pt",
        "MET_significance",
        "MET_sumEt",
        "CaloMET_pt",
        "CaloMET_sumEt",
        "isSingleMuIsoTrigger", 
        "isSingleMuTrigger",
        "iSkim",
        "PuppiMET_sumEt",
        "nIsoTrack"
       ]


NBINS = 60
LOWER_LIM = 100
UPPER_LIM = 220
    


def filter_dataframe_iSkim1(df):
    # Filter rows where "iSkim" is equal to 1
    mask = df["iSkim"] == 1
    
    # Filter rows where "nMuon" is greater than or equal to 2
    mask &= df["nMuon"] >= 2
    
    # Filter rows where either "isSingleMuIsoTrigger" or "isSingleMuTrigger" or both are equal to 1
    mask &= (df["isSingleMuIsoTrigger"] == 1) | (df["isSingleMuTrigger"] == 1)
    
    # Filter rows where the first and second elements of "Muon_pt" are greater than 27 and 7, respectively
    mask &= (df["Muon_pt"].str[0] > 27) & (df["Muon_pt"].str[1] > 7)
    
    # Filter rows where the first and second elements of "Muon_mediumId" are both True
    mask &= df["Muon_mediumId"].str[0] & df["Muon_mediumId"].str[1]
    
    inv_mass = []
    for i,row in df.iterrows():
        muon_0 = TLorentzVector(row["Muon_pt"][0], row["Muon_eta"][0], row["Muon_phi"][0],np.sqrt(row["Muon_mass"][0]**2 + (row["Muon_pt"][0]**2)*np.cosh(row["Muon_eta"][0])**2))
        muon_1 = TLorentzVector(row["Muon_pt"][1], row["Muon_eta"][1], row["Muon_phi"][1],np.sqrt(row["Muon_mass"][1]**2 + (row["Muon_pt"][1]**2)*np.cosh(row["Muon_eta"][1])**2))
        inv_mass.append((muon_0 + muon_1).M())

    inv_mass = np.array(inv_mass)
    
    # Filter rows where "inv_mass" is greater than 15
    mask &= inv_mass > 15
    
    return df[mask]

def filter_dataframe_iSkim2(df):
    # Filter rows where "iSkim" is equal to 2
    df = df[df["iSkim"] == 2]
    
    # Filter rows where "nMuon" is greater than or equal to 1 and "nElectron" is greater than or equal to 1
    df = df[(df["nMuon"] >= 1) & (df["nElectron"] >= 1)]
    
    # Filter rows where either "isSingleMuIsoTrigger" or "isSingleMuTrigger" or both are equal to 1
    df = df[(df["isSingleMuIsoTrigger"] == 1) | (df["isSingleMuTrigger"] == 1)]
    
    # Filter rows where the first element of "Muon_pt" is greater than 27
    df = df[df["Muon_pt"].str[0] > 27]
    
    # Filter rows where the first element of "Electron_pt" is greater than 20
    df = df[df["Electron_pt"].str[0] > 20]
    
    # Filter rows where the first element of "Muon_mediumId" is True
    df = df[df["Muon_mediumId"].str[0]]
    
    # Filter rows where the first element of "Electron_cutBased" is greater than or equal to 2
    if 'Electron_cutBased' in df.columns:
        df = df[df["Electron_cutBased"].str[0] >= 2]
    
    # Calculate the invariant mass for each row
    df["inv_mass"] = np.nan
    
    for i, row in df.iterrows():
        if row["nMuon"] == 2:
            df.at[i, "inv_mass"] = np.sqrt(2 * row["Muon_pt"][0] * row["Muon_pt"][1] * \
                                (np.cosh(row["Muon_eta"][0] - row["Muon_eta"][1]) - np.cos(row["Muon_phi"][0] - row["Muon_phi"][1])) + \
                                row["Muon_mass"][0]**2 + row["Muon_mass"][1]**2)
        elif row["nElectron"] == 2:
            df.at[i, "inv_mass"] = np.sqrt(2 * row["Electron_pt"][0] * row["Electron_pt"][1] * \
                                (np.cosh(row["Electron_eta"][0] - row["Electron_eta"][1]) - np.cos(row["Electron_phi"][0] - row["Electron_phi"][1])) + \
                                row["Electron_mass"][0]**2 + row["Electron_mass"][1]**2)
        else:
            df.at[i, "inv_mass"] = 0.0  # If nMuon and nElectron are not both 2, set the invariant mass to 0

    # Filter rows where "inv_mass" is greater than 15
    df = df[df["inv_mass"] > 15]
    df = df.drop('inv_mass', axis = 1)
    
    return df

def filter_dataframe_iSkim3(df):
    # Filter rows where "nMuon" is greater than or equal to 3
    df = df[df["iSkim"] == 3]
    df = df[df["nMuon"] >= 3]
    
    # Filter rows where either "isSingleMuIsoTrigger" or "isSingleMuTrigger" or both are equal to 1
    mask = (df["isSingleMuIsoTrigger"] == 1) | (df["isSingleMuTrigger"] == 1)
    
    # Filter rows where the first, second, and third elements of "Muon_pt" are greater than 27, 15, and 15, respectively
    mask &= (df["Muon_pt"].str[0] > 27) & (df["Muon_pt"].str[1] > 15) & (df["Muon_pt"].str[2] > 15)
    
    # Filter rows where the first, second, and third elements of "Muon_mediumId" are True
    mask &= df["Muon_mediumId"].str[0] & df["Muon_mediumId"].str[1] & df["Muon_mediumId"].str[2]

    df = df[mask]

    # Calculate the invariant mass for all muons with opposite signs for each row
    inv_mass_values = []
    for _, row in df.iterrows():
        muon_charges = row["Muon_charge"]
        muon_pts = row["Muon_pt"]
        muon_etas = row["Muon_eta"]
        muon_phis = row["Muon_phi"]
        muon_masses = row["Muon_mass"]
        
        inv_mass_list = []
        for i, j in [(i, j) for i in range(len(muon_charges)) for j in range(i+1, len(muon_charges)) if muon_charges[i] * muon_charges[j] < 0]:
            inv_mass_value = np.sqrt(2 * muon_pts[i] * muon_pts[j] * \
                                (np.cosh(muon_etas[i] - muon_etas[j]) - np.cos(muon_phis[i] - muon_phis[j])) + \
                                muon_masses[i]**2 + muon_masses[j]**2)
            inv_mass_list.append(inv_mass_value)

        if inv_mass_list:
            closest_mass = min(inv_mass_list, key=lambda x: abs(x - 91.2))
            inv_mass_values.append(closest_mass)
        else:
            inv_mass_values.append(np.nan)
    
    
    # Add the calculated invariant mass as a new column in the DataFrame
    df["inv_mass"] = inv_mass_values

    # Filter rows where "inv_mass" is greater than 15
    df = df[df["inv_mass"] > 15]
    
    df = df.drop('inv_mass', axis = 1)
    
    return df



def filter_dataframe_iSkim4(df):
    # Filter rows where "iSkim" is 4
    df = df[df["iSkim"] == 4]
    
    # Filter rows where "nMuon" is greater than or equal to 2
    df = df[df["nMuon"] >= 2]
    
    # Filter rows where "nElectron" is greater than or equal to 1
    df = df[df["nElectron"] >= 1]
    
    # Filter rows where either "isSingleMuIsoTrigger" or "isSingleMuTrigger" or both are equal to 1
    mask = (df["isSingleMuIsoTrigger"] == 1) | (df["isSingleMuTrigger"] == 1)
    
    # Filter rows where the first and second elements of "Muon_pt" are greater than 27 and 15, respectively
    mask &= (df["Muon_pt"].str[0] > 27) & (df["Muon_pt"].str[1] > 15)
    
    # Filter rows where the first element of "Electron_pt" is greater than 15
    mask &= df["Electron_pt"].str[0] > 15
    
    # Filter rows where the first and second elements of "Muon_mediumId" are True
    mask &= df["Muon_mediumId"].str[0] & df["Muon_mediumId"].str[1]
    
    # Filter rows where the first element of "Electron_cutBased" is greater than or equal to 2
    mask &= df["Electron_cutBased"].str[0] >= 2
    
    mask &= df["Muon_charge"].str[0] * df["Muon_charge"].str[1] < 0

    df = df[mask]
    # Calculate the invariant mass for each row (muon[0] and muon[1])
    inv_mass = []
    for i,row in df.iterrows():
        muon_0 = TLorentzVector(row["Muon_pt"][0], row["Muon_eta"][0], row["Muon_phi"][0],np.sqrt(row["Muon_mass"][0]**2 + (row["Muon_pt"][0]**2)*np.cosh(row["Muon_eta"][0])**2))
        muon_1 = TLorentzVector(row["Muon_pt"][1], row["Muon_eta"][1], row["Muon_phi"][1],np.sqrt(row["Muon_mass"][1]**2 + (row["Muon_pt"][1]**2)*np.cosh(row["Muon_eta"][1])**2))
        inv_mass.append((muon_0 + muon_1).M())
    # Filter rows where "inv_mass" is greater than 15
    mask_inv_mass = np.array(inv_mass) > 15
    df = df[mask_inv_mass]
    
    return df



def btag_enough(df):
    return df[df['Jet_btagCSVV2'].apply(lambda x: 
               len(x) >= 2 and 
               (sorted(x)[-1] > 0.8) and 
               (sorted(x)[-2] > 0.8) and 
               all(val < 0.6 for val in x if val < 0.8))]





def check_muon_condition(row):
    for pt, eta, iso in zip(row['Muon_pt'], row['Muon_eta'], row['Muon_pfRelIso03_all']):
        if pt > 28 and np.abs(eta) <= 2.4 and iso <= 0.125:
            return True
    return False



def filter_function_jet_pt_actual(row):
    # Initialize counts for high-energy jets and high-energy jets within |eta| < 2.4
    high_pt_count = 0
    eta_condition = 0
    
    # Iterate through the jet energies and etas
    for pt, eta in zip(row['Jet_pt'], row['Jet_eta']):
        if pt > 30:
            high_pt_count += 1
            if np.abs(eta) <= 2.4:
                eta_condition += 1

    # Return True if there are at least 4 high-energy jets and at least 4 high-energy jets with |eta| < 2.4
    return high_pt_count >= 4 and eta_condition >= 4


def calculate_invariant_mass_HadronicDecay_4Vector_modified(df_ori):
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    df = df_ori.copy()

    # Calculate the invariant mass for each row
    WHadronic = []
    highest_btag_indices = []
    invariant_masses = []

    for i, row in df_ori.iterrows():
        pt_values = row["Jet_pt"]
        eta_values = row["Jet_eta"]
        phi_values = row["Jet_phi"]
        masses = row["Jet_mass"]
        btag_values = row["Jet_btagCSVV2"]

        # Determine the indices of the two jets with the highest b-tag values
        highest_btag_indices_row = np.argsort(btag_values)[-2:]
        highest_btag_indices.append(highest_btag_indices_row)

        # Exclude the highest b-tagged jets from the non-btag_indices
        non_btag_indices = [idx for idx in range(len(pt_values)) if idx not in highest_btag_indices_row]

        closest_invariant_mass = float("inf")
        closest_WHadronic = None

        # Calculate the invariant mass for all possible pairs of jets within non_btag_indices
        if len(non_btag_indices) >= 2:
            for j1_idx in range(len(non_btag_indices) - 1):
                for j2_idx in range(j1_idx + 1, len(non_btag_indices)):
                    j1 = non_btag_indices[j1_idx]
                    j2 = non_btag_indices[j2_idx]

                    energy_1 = np.sqrt(masses[j1] ** 2 + (pt_values[j1] ** 2) * np.cosh(eta_values[j1]) ** 2)
                    jet_vec_1 = TLorentzVector(pt_values[j1], eta_values[j1], phi_values[j1], energy_1)
                    energy_2 = np.sqrt(masses[j2] ** 2 + (pt_values[j2] ** 2) * np.cosh(eta_values[j2]) ** 2)
                    jet_vec_2 = TLorentzVector(pt_values[j2], eta_values[j2], phi_values[j2], energy_2)

                    # Calculate the invariant mass for the pair of jets
                    invariant_mass = (jet_vec_1 + jet_vec_2).M()
                    WHadronic_mass = (jet_vec_1 + jet_vec_2)

                    # Check if this invariant mass is closer to the target
                    if abs(invariant_mass - 80.38) < abs(closest_invariant_mass - 80.38):
                        closest_invariant_mass = invariant_mass
                        closest_WHadronic = WHadronic_mass

            invariant_masses.append(closest_invariant_mass)
            WHadronic.append(closest_WHadronic)

        else:
            invariant_masses.append(0)  # Default value when there are not enough jets
            WHadronic.append(TLorentzVector(0, 0, 0, 0))

    # Add the calculated invariant mass as a new column in the DataFrame
    df_ori["WHadronic"] = WHadronic
    df_ori['invariantmass_W'] = invariant_masses
    df_ori['highest_btag_indices'] = highest_btag_indices

    return df_ori[(df_ori['invariantmass_W'] >= 70) & (df_ori['invariantmass_W'] <= 90)]


def calculate_top_quark_masses(df):
    cand_1 = []
    cand_2 = []
    for i, row in df.iterrows():
        W_vec = row['WHadronic']
        # Identify 2 jets which are most likely to have originated from a bottom quark
        if len(row['highest_btag_indices']) >= 2:
            idx1, idx2 = row['highest_btag_indices'][0], row['highest_btag_indices'][1]
            
    
            
            # Calculate Jet energy
            energy_1 = np.sqrt(row["Jet_mass"][idx1]**2 + (row["Jet_pt"][idx1]**2) * np.cosh(row["Jet_eta"][idx1])**2)
            # Create 4 vector of first jet
            jet_vec_1 = TLorentzVector(row["Jet_pt"][idx1],row["Jet_eta"][idx1],row["Jet_phi"][idx1],energy_1)
            
            # Repeat for jet 2
            energy_2 = np.sqrt(row["Jet_mass"][idx2]**2 + (row["Jet_pt"][idx2]**2) * np.cosh(row["Jet_eta"][idx2])**2)
            jet_vec_2 = TLorentzVector(row["Jet_pt"][idx2],row["Jet_eta"][idx2],row["Jet_phi"][idx2],energy_2)
            
            
            # Create 2 candidate top quark jets
            cand_1.append((jet_vec_1 + W_vec).M())
            cand_2.append((jet_vec_2 + W_vec).M())
        else:
            cand_1.append(0)
            cand_2.append(0)
     
    df['top_quark_candidate_1'] = cand_1
    df['top_quark_candidate_2'] = cand_2
    
    return df


def calculate_closest_mass(df):
    # Calculate the absolute differences between the values and the target
    diff_1 = abs(df['top_quark_candidate_1'] - 172.76)
    diff_2 = abs(df['top_quark_candidate_2'] - 172.76)
    
    # Create a new column 'top_quark_mass' with the closest value
    df['top_quark_mass'] = df.apply(lambda row: row['top_quark_candidate_1'] if diff_1[row.name] < diff_2[row.name] else row['top_quark_candidate_2'], axis=1)
    
    return df





def list_folder_elements(folder_path):
    try:
        elements = os.listdir(folder_path)
        #elements = [os.path.join(folder_path, element) for element in elements]
        return elements
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")
        return []
    
    
def get_characters_after_third_slash_and_create_folder(input_string):
    parts = input_string.split('/', 3)
    if len(parts) >= 4:
        result = parts[3]
        folder_path = os.path.join(os.getcwd(), result)
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Error: Unable to create folder '{folder_path}'. {e}")
        return result
    else:
        print("Error: The input string does not have at least three '/' characters.")
        return None
    
 
            
            
def save_weighted_binned_histogram_to_txt(dataframe, column_name, nbins, output_file, weight_column):
    if column_name not in dataframe.columns:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return

    if weight_column not in dataframe.columns:
        print(f"Column '{weight_column}' not found in the DataFrame.")
        return

    column_values = dataframe[column_name].values
    weights = dataframe[weight_column].values
    bin_edges = np.linspace(LOWER_LIM, UPPER_LIM, nbins + 1)
    weighted_binned_counts, _ = np.histogram(column_values, bins=bin_edges, weights=weights)

    with open(output_file, 'w') as file:
        for count in weighted_binned_counts:
            file.write(str(count) + '\n')

def load_binned_histogram_from_txt(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]

    return data
