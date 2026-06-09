import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import torch
import random
import joblib


# ===========================
# SEEDS
# ===========================
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# ===========================
# DPPS + HYDRO DEFINITIONS
# ===========================

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

def build_features(peptides, y_labels):
    binders = [p for p, l in zip(peptides, y_labels) if l == 1]
    nonbinders = [p for p, l in zip(peptides, y_labels) if l == 0]

    log_b = make_logodds_matrix(binders)
    log_nb = make_logodds_matrix(nonbinders)

    dpps_b = build_dpps_profile(binders)
    dpps_nb = build_dpps_profile(nonbinders)

    hydro_b = build_hydro_profile(binders)
    hydro_nb = build_hydro_profile(nonbinders)

    X = np.hstack([
        encode_logodds_sum(peptides, log_b, log_nb),
        encode_dpps_sum(peptides, dpps_b, dpps_nb),
        encode_hydro_sum(peptides, hydro_b, hydro_nb)
    ])

    return X, (log_b, log_nb, dpps_b, dpps_nb, hydro_b, hydro_nb)

# ===========================
# DATA
# ===========================
binders_file = pd.read_csv('Final_positive_training.txt',header=None, sep='\t')
binders_set = set(binders_file[0].tolist())
nonbinder_file = pd.read_csv('Final_negative_training.txt',sep='\t',header=None)
nonbinders_set = set(nonbinder_file[0].tolist())

overlap = binders_set & nonbinders_set
binders = list(binders_set - overlap)
nonbinders = list(nonbinders_set - overlap)

X_peptides = binders + nonbinders
y = np.array([1]*len(binders) + [0]*len(nonbinders))

# ===================================================
# STEP 1: NESTED CV — correct, no leakage
# ===================================================
LABEL = 'model_spy-C'

param_grid = {
    'svc__C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
    'svc__gamma': [0.001, 0.01, 0.05, 0.1, 1]
}

outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
outer_results = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_peptides, y), 1):
    X_train_pep = [X_peptides[i] for i in train_idx]
    X_test_pep  = [X_peptides[i] for i in test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Profiles built from TRAIN only
    X_train, profiles = build_features(X_train_pep, y_train)
    log_b_fold, log_nb_fold, dpps_b_fold, dpps_nb_fold, hydro_b_fold, hydro_nb_fold = profiles


    # Test encoded using TRAIN profiles — no test info used

    X_test = np.hstack([
        encode_logodds_sum(X_test_pep, log_b_fold,  log_nb_fold),
        encode_dpps_sum(X_test_pep,    dpps_b_fold, dpps_nb_fold),
        encode_hydro_sum(X_test_pep,   hydro_b_fold, hydro_nb_fold)
    ])

    # MinMaxScaler inside Pipeline — fit on X_train, applied to X_test
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svc', SVC(kernel='rbf', probability=True, class_weight='balanced'))
    ])

    grid = GridSearchCV(
        pipeline, param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1
    )
    grid.fit(X_train, y_train)  

    # Scaler from pipeline applied to X_test (transform only, not fit)
    y_pred = grid.best_estimator_.predict(X_test)
    y_prob = grid.best_estimator_.predict_proba(X_test)[:, 1]

    outer_results.append({
        'fold':       fold,
        'accuracy':   accuracy_score(y_test, y_pred),
        'f1':         f1_score(y_test, y_pred),
        'roc_auc':    roc_auc_score(y_test, y_prob),
        'best_C':     grid.best_params_['svc__C'],
        'best_gamma': grid.best_params_['svc__gamma']
    })
    print(f"Fold {fold}: ROC={outer_results[-1]['roc_auc']:.3f} | "
          f"C={grid.best_params_['svc__C']} | gamma={grid.best_params_['svc__gamma']}")


# Report performance
perf_df = pd.DataFrame(outer_results)
print(f"\nModel performance on {LABEL}:")
print(f"Accuracy : {perf_df['accuracy'].mean():.3f} ± {perf_df['accuracy'].std():.3f}")
print(f"F1       : {perf_df['f1'].mean():.3f} ± {perf_df['f1'].std():.3f}")
print(f"ROC-AUC  : {perf_df['roc_auc'].mean():.3f} ± {perf_df['roc_auc'].std():.3f}")

perf_df.to_csv(f'{LABEL}_nested_cv_performance.csv', index=False)
print(f"Saved → {LABEL}_nested_cv_performance.csv")

# ===================================================
# STEP 2: GRIDSEARCH ON FULL DATA × 10 RUNS
# find stable hyperparams for deployment
# ===================================================

# Build features once — same for all 10 runs
X_full, profiles_full = build_features(X_peptides, y)
log_b, log_nb, dpps_b, dpps_nb, hydro_b, hydro_nb = profiles_full

X_full_enc = np.hstack([
    encode_logodds_sum(X_peptides, log_b, log_nb),
    encode_dpps_sum(X_peptides, dpps_b, dpps_nb),
    encode_hydro_sum(X_peptides, hydro_b, hydro_nb)
])

N_RUNS = 10
run_results = []

for run_id in range(N_RUNS):
    print(f"\nRun {run_id+1}/{N_RUNS} ...", end=" ")

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svc', SVC(kernel='rbf', probability=True, class_weight='balanced'))
    ])

    grid = GridSearchCV(
        pipeline, param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=run_id),  # seed varies
        n_jobs=1
    )
    grid.fit(X_full_enc, y)

    cv_auc     = grid.best_score_
    best_C     = grid.best_params_['svc__C']
    best_gamma = grid.best_params_['svc__gamma']

    pkl_name = f'{LABEL}_run{run_id+1}_auc{cv_auc:.4f}_C{best_C}_g{best_gamma}.pkl'

    joblib.dump({
        'model':      grid.best_estimator_,
        'log_b':      log_b,
        'log_nb':     log_nb,
        'dpps_b':     dpps_b,
        'dpps_nb':    dpps_nb,
        'hydro_b':    hydro_b,
        'hydro_nb':   hydro_nb,
        'cv_auc':     cv_auc,
        'best_C':     best_C,
        'best_gamma': best_gamma,
        'run_id':     run_id + 1,
        'label':      LABEL,
    }, pkl_name)

    print(f"CV AUC={cv_auc:.4f} | C={best_C} | gamma={best_gamma} → {pkl_name}")

    run_results.append({
        'run_id':     run_id + 1,
        'pkl':        pkl_name,
        'cv_auc':     cv_auc,
        'best_C':     best_C,
        'best_gamma': best_gamma,
    })

# ===================================================
# ANALYZE RUNS — pick best deployment model
# ===================================================
runs_df = pd.DataFrame(run_results).sort_values('cv_auc', ascending=False)

print("\n=== All runs ranked by CV AUC ===")
print(runs_df.to_string(index=False))

print(f"\nCV AUC range : {runs_df['cv_auc'].min():.4f} – {runs_df['cv_auc'].max():.4f}")
print(f"CV AUC std   : {runs_df['cv_auc'].std():.4f}")

print(f"\nHyperparameter frequency (majority vote):")
hp_counts = runs_df[['best_C', 'best_gamma']].value_counts()
print(hp_counts)

majority_C     = runs_df['best_C'].mode()[0]
majority_gamma = runs_df['best_gamma'].mode()[0]

stable_runs = runs_df[
    (runs_df['best_C']     == majority_C) &
    (runs_df['best_gamma'] == majority_gamma)
].sort_values('cv_auc', ascending=False)

best_run  = stable_runs.iloc[0]
best_pkl  = best_run['pkl']

print(f"\n✓ Majority hyperparams : C={majority_C}, gamma={majority_gamma}")
print(f"  Runs agreeing        : {len(stable_runs)}/{N_RUNS}")
print(f"  Best deployment pkl  : {best_pkl}")
print(f"  CV AUC               : {best_run['cv_auc']:.4f}")

model_selection_file = f'{LABEL}_run_selection_log.csv'
runs_df.to_csv(model_selection_file, index=False)
print(f"\nRun selection log saved → {model_selection_file}")


