import torch
import numpy as np
import pandas as pd

dpps_scores = {
    'A': torch.Tensor([-1.02, -2.88, -0.56, 0.36, -6.15, -1.68, 0.04, -2.51, -1.94, -0.01]),
    'R': torch.Tensor([1.99, 4.13, -4.41, -1.02, 4.78, 3.04, -9.06, 6.71, 4.41, 0.07]),
    'N': torch.Tensor([-2.19, 1.86, 0.38, -0.13, -2.3, 1.41, -5.71, -1.11, 1.73, -0.19]),
    'D': torch.Tensor([-6.6, 3.32, 1.61, 0.36, -3.25, 1.95, -7.36, 0.14, 1.24, -0.15]),
    'C': torch.Tensor([0.21, 1.12, 3.42, -0.68, -2.27, -1.22, 3.11, -2.98, -1.7, 1.57]),
    'Q': torch.Tensor([-0.47, 1.16, -0.57, 0.69,0.39,1.93,-5.46,-0.84,1.93,0.85]),
    'E': torch.Tensor([-5.39,0.65,-0.98,1.39,-0.23,2.51,-6.84,-0.68,1.41,1.28]),
    'G': torch.Tensor([-2.86,-5,-2.97,0.53,-11.45,1.89,-2.11,-3.99,-2.16,-0.76]),
    'H': torch.Tensor([0.73, 2.68, -0.66, -1.89, 1.6, 1.13, -1.94, -0.11, 0.44, 0.15]),
    'I': torch.Tensor([1.91,-3.13,0.01, 1.14, 2.7, -4.55, 8.93, 0.18, -1.1, -0.76]),
    'L': torch.Tensor([1.64,-2.57,0,1.35,2.62,-2.65,7.72,0.05,-1.03,-1.81]),
    'K': torch.Tensor([2.47, 1.54, -4.28,-0.86, 2.77, 2.06, -6.18, 2.05, 2.19,-1.65]),
    'M': torch.Tensor([1.93, -0.01, 1.21, 0.99, 2.79, -0.56,5.33,-0.87, -0.99, -1.09]),
    'F': torch.Tensor([2.68, 0.84, 2.22, 0.71, 5.02, -0.3, 8.6, 1.13, -1.4, -0.28]),
    'P': torch.Tensor([0.45, -2.89, 1.77, -5.81, -3.79, -0.61, 0.7, 1.21, -1.67, 1.79]),
    'S': torch.Tensor([-1.76, -0.19, 1.06, -0.69, -5.72, 0.14, -4.14, -2.42, -0.13, 0.69]),
    'T': torch.Tensor([-0.55,-0.66,0.13,-0.31,-2.76,-1.56,-2.46,-2.12,0.17,0.08]),
    'W': torch.Tensor([3.88,1.78,1.68,2,9.31,0.89,7.53,4.27,-0.23,-1.42]),
    'Y': torch.Tensor([2.1,1.26,1.15,0.91,5.9,0.74,3.71,3.32,0.25,1.33]),
    'V': torch.Tensor([0.83,-3.02, -0.22, 0.97,0.05,-4.55, 5.61,-1.41,-1.44,0.3]),
    '-': torch.Tensor([0]*10)
}

hydropathy_dict = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}


# ===========================
# FEATURE BUILDERS
# ===========================
def make_position_matrix(peptides, amino_acids="ACDEFGHIKLMNPQRSTVWY"):
    L = len(peptides[0])
    pfm = pd.DataFrame(0, index=range(L), columns=list(amino_acids))
    for pos in range(L):
        residues = [pep[pos] for pep in peptides]
        counts = Counter(residues)
        for aa, c in counts.items():
            if aa in pfm.columns:
                pfm.at[pos, aa] = c
    return pfm

def make_logodds_matrix(peptides):
    pfm = make_position_matrix(peptides)
    ppm = pfm.div(pfm.sum(axis=1), axis=0) + 1e-6
    bg = 1.0 / len(pfm.columns)
    return (ppm / bg).applymap(lambda x: np.log2(x))

def build_dpps_profile(peptides):
    L = len(peptides[0])
    n_features = len(next(iter(dpps_scores.values())))
    profile = np.zeros((L, n_features))
    counts = np.zeros(L)
    for pep in peptides:
        for i, aa in enumerate(pep):
            if aa in dpps_scores:
                profile[i] += dpps_scores[aa].numpy()
                counts[i] += 1
    return profile / counts[:, None]

def build_hydro_profile(peptides):
    L = len(peptides[0])
    profile = np.zeros(L)
    counts = np.zeros(L)
    for pep in peptides:
        for i, aa in enumerate(pep):
            if aa in hydropathy_dict:
                profile[i] += hydropathy_dict[aa]
                counts[i] += 1
    return profile / counts

# ===========================
# ENCODERS
# ===========================
def encode_logodds_sum(peptides, logodds_b, logodds_nb):
    X = np.zeros((len(peptides), 2))
    for i, pep in enumerate(peptides):
        X[i,0] = sum(logodds_b.at[pos, aa] for pos, aa in enumerate(pep) if aa in logodds_b.columns)
        X[i,1] = sum(logodds_nb.at[pos, aa] for pos, aa in enumerate(pep) if aa in logodds_nb.columns)
    return X

def encode_dpps_sum(peptides, binder_prof, nonbinder_prof):
    X = np.zeros((len(peptides), 2))
    for i, pep in enumerate(peptides):
        b, nb = 0.0, 0.0
        for pos, aa in enumerate(pep):
            if aa in dpps_scores:
                vec = dpps_scores[aa].numpy()
                b += np.dot(vec, binder_prof[pos])
                nb += np.dot(vec, nonbinder_prof[pos])
        X[i] = [b, nb]
    return X

def encode_hydro_sum(peptides, binder_prof, nonbinder_prof):
    X = np.zeros((len(peptides), 2))
    for i, pep in enumerate(peptides):
        b, nb = 0.0, 0.0
        for pos, aa in enumerate(pep):
            val = hydropathy_dict.get(aa, 0.0)
            b += val * binder_prof[pos]
            nb += val * nonbinder_prof[pos]
        X[i] = [b, nb]
    return X
