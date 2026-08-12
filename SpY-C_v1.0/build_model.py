import argparse
import random
from collections import Counter
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

SVC_RANDOM_STATE = 42  # fixed everywhere -- only CV partitioning varies across runs

# ---------------------------------------------------------------------
# DPPS + hydropathy scales
# ---------------------------------------------------------------------
dpps_scores = {
    'A': torch.Tensor([-1.02, -2.88, -0.56, 0.36, -6.15, -1.68, 0.04, -2.51, -1.94, -0.01]),
    'R': torch.Tensor([1.99, 4.13, -4.41, -1.02, 4.78, 3.04, -9.06, 6.71, 4.41, 0.07]),
    'N': torch.Tensor([-2.19, 1.86, 0.38, -0.13, -2.3, 1.41, -5.71, -1.11, 1.73, -0.19]),
    'D': torch.Tensor([-6.6, 3.32, 1.61, 0.36, -3.25, 1.95, -7.36, 0.14, 1.24, -0.15]),
    'C': torch.Tensor([0.21, 1.12, 3.42, -0.68, -2.27, -1.22, 3.11, -2.98, -1.7, 1.57]),
    'Q': torch.Tensor([-0.47, 1.16, -0.57, 0.69, 0.39, 1.93, -5.46, -0.84, 1.93, 0.85]),
    'E': torch.Tensor([-5.39, 0.65, -0.98, 1.39, -0.23, 2.51, -6.84, -0.68, 1.41, 1.28]),
    'G': torch.Tensor([-2.86, -5, -2.97, 0.53, -11.45, 1.89, -2.11, -3.99, -2.16, -0.76]),
    'H': torch.Tensor([0.73, 2.68, -0.66, -1.89, 1.6, 1.13, -1.94, -0.11, 0.44, 0.15]),
    'I': torch.Tensor([1.91, -3.13, 0.01, 1.14, 2.7, -4.55, 8.93, 0.18, -1.1, -0.76]),
    'L': torch.Tensor([1.64, -2.57, 0, 1.35, 2.62, -2.65, 7.72, 0.05, -1.03, -1.81]),
    'K': torch.Tensor([2.47, 1.54, -4.28, -0.86, 2.77, 2.06, -6.18, 2.05, 2.19, -1.65]),
    'M': torch.Tensor([1.93, -0.01, 1.21, 0.99, 2.79, -0.56, 5.33, -0.87, -0.99, -1.09]),
    'F': torch.Tensor([2.68, 0.84, 2.22, 0.71, 5.02, -0.3, 8.6, 1.13, -1.4, -0.28]),
    'P': torch.Tensor([0.45, -2.89, 1.77, -5.81, -3.79, -0.61, 0.7, 1.21, -1.67, 1.79]),
    'S': torch.Tensor([-1.76, -0.19, 1.06, -0.69, -5.72, 0.14, -4.14, -2.42, -0.13, 0.69]),
    'T': torch.Tensor([-0.55, -0.66, 0.13, -0.31, -2.76, -1.56, -2.46, -2.12, 0.17, 0.08]),
    'W': torch.Tensor([3.88, 1.78, 1.68, 2, 9.31, 0.89, 7.53, 4.27, -0.23, -1.42]),
    'Y': torch.Tensor([2.1, 1.26, 1.15, 0.91, 5.9, 0.74, 3.71, 3.32, 0.25, 1.33]),
    'V': torch.Tensor([0.83, -3.02, -0.22, 0.97, 0.05, -4.55, 5.61, -1.41, -1.44, 0.3]),
    '-': torch.Tensor([0] * 10),
}

hydropathy_dict = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


# ---------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------
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


def make_logodds_matrix(peptides, alpha=0.5):
    """Position-specific log-odds relative to background, with a small pseudocount added to avoid near-zero probabilities for residues not observed in a training subset. Probabilities are normalized to sum to 1 at each position."""
    pfm = make_position_matrix(peptides)
    n_categories = pfm.shape[1]
    ppm = (pfm + alpha).div(pfm.sum(axis=1) + alpha * n_categories, axis=0)
    bg = 1.0 / n_categories
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


# ---------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------
def encode_logodds_sum(peptides, logodds_b, logodds_nb):
    X = np.zeros((len(peptides), 2))
    for i, pep in enumerate(peptides):
        X[i, 0] = sum(logodds_b.at[pos, aa] for pos, aa in enumerate(pep) if aa in logodds_b.columns)
        X[i, 1] = sum(logodds_nb.at[pos, aa] for pos, aa in enumerate(pep) if aa in logodds_nb.columns)
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
    """Build profiles from these peptides only, then encode these same
    peptides with them. Pass this only peptides that are allowed to
    contribute to the profile (e.g. a fold's training set)."""
    binders_ = [p for p, l in zip(peptides, y_labels) if l == 1]
    nonbinders_ = [p for p, l in zip(peptides, y_labels) if l == 0]

    log_b = make_logodds_matrix(binders_)
    log_nb = make_logodds_matrix(nonbinders_)
    dpps_b = build_dpps_profile(binders_)
    dpps_nb = build_dpps_profile(nonbinders_)
    hydro_b = build_hydro_profile(binders_)
    hydro_nb = build_hydro_profile(nonbinders_)

    X = np.hstack([
        encode_logodds_sum(peptides, log_b, log_nb),
        encode_dpps_sum(peptides, dpps_b, dpps_nb),
        encode_hydro_sum(peptides, hydro_b, hydro_nb),
    ])
    return X, (log_b, log_nb, dpps_b, dpps_nb, hydro_b, hydro_nb)


def encode_with_profiles(peptides, profiles):
    """Encode peptides using profiles that were built elsewhere. Use this
    for validation/test peptides so they never contribute to their own
    profile."""
    log_b, log_nb, dpps_b, dpps_nb, hydro_b, hydro_nb = profiles
    return np.hstack([
        encode_logodds_sum(peptides, log_b, log_nb),
        encode_dpps_sum(peptides, dpps_b, dpps_nb),
        encode_hydro_sum(peptides, hydro_b, hydro_nb),
    ])


# ---------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Nested-CV SVM training pipeline for pY peptide classification."
    )
    parser.add_argument(
        '--binders-file', required=True,
        help="Path to the binder (positive) peptide file. Tab-separated, no header; "
             "the first column is treated as the peptide sequence list."
    )
    parser.add_argument(
        '--nonbinders-file', required=True,
        help="Path to the nonbinder (negative) peptide file. Same format as --binders-file."
    )
    parser.add_argument(
        '--label', required=True,
        help="Run label used to name every output file this script produces "
             "(nested CV results, candidate log, fold scores, final .pkl)."
    )
    return parser.parse_args()


args = parse_args()
LABEL = args.label

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
binders_file = pd.read_csv(args.binders_file, header=None, sep='\t')
binders_set = set(binders_file[0].tolist())
nonbinder_file = pd.read_csv(args.nonbinders_file, header=None, sep='\t')
nonbinders_set = set(nonbinder_file[0].tolist())

overlap = binders_set & nonbinders_set
binders = sorted(binders_set - overlap)
nonbinders = sorted(nonbinders_set - overlap)

X_peptides = binders + nonbinders
y = np.array([1] * len(binders) + [0] * len(nonbinders))

assert len(X_peptides) == len(set(X_peptides)), "Duplicate peptides in X_peptides"

param_grid = {
    'svc__C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
    'svc__gamma': [0.001, 0.01, 0.05, 0.1, 1],
}


# =======================================================================
# STEP 1 — Nested CV
# =======================================================================

def inner_hp_search(train_peptides, y_train, grid_params, n_splits=5, split_seed=42):
    """Inner CV: pick (C, gamma) using only outer-training
    peptides, rebuilding profiles per inner fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    peptides_arr = np.array(train_peptides, dtype=object)

    fold_data = []
    for tr_idx, val_idx in skf.split(peptides_arr, y_train):
        tr_pep = peptides_arr[tr_idx].tolist()
        val_pep = peptides_arr[val_idx].tolist()
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        X_tr, fold_profiles = build_features(tr_pep, y_tr)
        X_val = encode_with_profiles(val_pep, fold_profiles)
        fold_data.append((X_tr, y_tr, X_val, y_val))

    best_C, best_gamma, best_mean_auc = None, None, -np.inf

    for C in grid_params['svc__C']:
        for gamma in grid_params['svc__gamma']:
            fold_aucs = []
            for X_tr, y_tr, X_val, y_val in fold_data:
                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('svc', SVC(kernel='rbf', C=C, gamma=gamma,
                                probability=True, class_weight='balanced',
                                random_state=SVC_RANDOM_STATE)),
                ])
                pipeline.fit(X_tr, y_tr)
                y_prob = pipeline.predict_proba(X_val)[:, 1]
                fold_aucs.append(roc_auc_score(y_val, y_prob))

            mean_auc = float(np.mean(fold_aucs))
            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_C, best_gamma = C, gamma

    return best_C, best_gamma, best_mean_auc


outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
outer_results = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_peptides, y), 1):
    X_train_pep = [X_peptides[i] for i in train_idx]
    X_test_pep = [X_peptides[i] for i in test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # inner search: uses outer-training peptides only
    best_C, best_gamma, inner_auc = inner_hp_search(X_train_pep, y_train, param_grid)

    # refit on the FULL outer-training fold with the chosen hyperparameters
    X_train, profiles = build_features(X_train_pep, y_train)
    log_b_fold, log_nb_fold, dpps_b_fold, dpps_nb_fold, hydro_b_fold, hydro_nb_fold = profiles

    X_test = np.hstack([
        encode_logodds_sum(X_test_pep, log_b_fold, log_nb_fold),
        encode_dpps_sum(X_test_pep, dpps_b_fold, dpps_nb_fold),
        encode_hydro_sum(X_test_pep, hydro_b_fold, hydro_nb_fold),
    ])

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svc', SVC(kernel='rbf', C=best_C, gamma=best_gamma,
                    probability=True, class_weight='balanced',
                    random_state=SVC_RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    outer_results.append({
        'fold': fold,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'best_C': best_C,
        'best_gamma': best_gamma,
        'inner_mean_auc': inner_auc,
    })
    print(f"Fold {fold}: outer ROC={outer_results[-1]['roc_auc']:.3f} | "
          f"C={best_C} | gamma={best_gamma} | inner CV AUC={inner_auc:.3f}")

perf_df = pd.DataFrame(outer_results)

print(f"\nModel performance on {LABEL} (nested CV, {len(outer_results)} outer folds):")
print(f"Accuracy : {perf_df['accuracy'].mean():.3f} ± {perf_df['accuracy'].std():.3f}")
print(f"F1       : {perf_df['f1'].mean():.3f} ± {perf_df['f1'].std():.3f}")
print(f"ROC-AUC  : {perf_df['roc_auc'].mean():.3f} ± {perf_df['roc_auc'].std():.3f}")

perf_df.to_csv(f'{LABEL}_nested_cv_performance.csv', index=False)
print(f"Saved -> {LABEL}_nested_cv_performance.csv")


# =======================================================================
# STEP 2 — select stable hyperparameters, then train one final model
# =======================================================================

N_SPLITS = 5
N_RUNS = 10


def run_single_cv_pass(run_id, peptides, y_labels, grid_params, n_splits=N_SPLITS):
    """One seeded 5-fold CV pass. Returns every (C, gamma) candidate's
    AUC on every fold of this run. Only the CV split seed varies by
    run_id -- the SVC's own random_state stays fixed."""
    print(f"\nRun {run_id + 1} ...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run_id)
    peptides_arr = np.array(peptides, dtype=object)

    fold_data = []
    for train_idx, val_idx in skf.split(peptides_arr, y_labels):
        train_pep = peptides_arr[train_idx].tolist()
        val_pep = peptides_arr[val_idx].tolist()
        y_train_fold, y_val_fold = y_labels[train_idx], y_labels[val_idx]

        X_train_fold, fold_profiles = build_features(train_pep, y_train_fold)
        X_val_fold = encode_with_profiles(val_pep, fold_profiles)

        fold_data.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))

    records = []
    for C in grid_params['svc__C']:
        for gamma in grid_params['svc__gamma']:
            for fold_id, (X_train_fold, y_train_fold, X_val_fold, y_val_fold) in enumerate(fold_data, 1):
                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('svc', SVC(kernel='rbf', C=C, gamma=gamma,
                                probability=True, class_weight='balanced',
                                random_state=SVC_RANDOM_STATE)),
                ])
                pipeline.fit(X_train_fold, y_train_fold)
                y_prob = pipeline.predict_proba(X_val_fold)[:, 1]
                auc = roc_auc_score(y_val_fold, y_prob)

                records.append({
                    'run_id': run_id + 1,
                    'fold_id': fold_id,
                    'C': C,
                    'gamma': gamma,
                    'auc': auc,
                })

    return records


print(f"\n--- Running {N_RUNS} seeded 5-fold CV passes ---")
all_records = []
for run_id in range(N_RUNS):
    all_records.extend(run_single_cv_pass(run_id, X_peptides, y, param_grid))

records_df = pd.DataFrame(all_records)
#records_df.to_csv(f'{LABEL}_cv_fold_scores.csv', index=False)
#print(f"\nRaw per-fold scores saved -> {LABEL}_cv_fold_scores.csv")

# stage 1: collapse each run's 5 folds into one mean AUC per (run, C, gamma)
run_candidate_summary = (
    records_df.groupby(['run_id', 'C', 'gamma'])['auc']
    .mean()
    .reset_index()
)

# stage 2: mean and std ACROSS the 10 run-level means, per candidate
candidate_summary = (
    run_candidate_summary.groupby(['C', 'gamma'])['auc']
    .agg(mean_auc='mean', std_across_runs='std', n_runs='count')
    .reset_index()
)

# explicit tie-break: highest mean AUC first; ties (including near-
# identical floating point AUCs, not just exact equality) favor lower
# C, then lower gamma. AUC_TIE_TOLERANCE is deliberately tight -- this
# is meant to catch cases like 0.78123 vs 0.78124 that are numerically
# distinct but not meaningfully different, NOT to override a candidate
# that is genuinely ahead by a real margin.
AUC_TIE_TOLERANCE = 1e-4

candidate_summary = candidate_summary.sort_values(
    ['mean_auc', 'C', 'gamma'],
    ascending=[False, True, True],
).reset_index(drop=True)

print(f"\nCandidate ranking ({N_RUNS} runs, run-level means aggregated across runs):")
print(candidate_summary.to_string(index=False))

top_auc = candidate_summary.iloc[0]['mean_auc']
tied = candidate_summary[candidate_summary['mean_auc'] >= top_auc - AUC_TIE_TOLERANCE]
tied = tied.sort_values(['C', 'gamma'], ascending=[True, True]).reset_index(drop=True)

best_row = tied.iloc[0]
best_C, best_gamma = best_row['C'], best_row['gamma']
model_selection_auc = best_row['mean_auc']
model_selection_auc_std = best_row['std_across_runs']

if len(tied) > 1:
    print(f"\n{len(tied)} candidates tied within {AUC_TIE_TOLERANCE} AUC of the top score "
          f"({top_auc:.4f}) -- selecting the simplest (lowest C, then lowest gamma) among them.")

print(f"\nSelected hyperparameters      : C={best_C}, gamma={best_gamma}")
print(f"Model-selection AUC (across runs) : {model_selection_auc:.4f}")
print(f"Std across runs                : {model_selection_auc_std:.4f}")

candidate_summary.to_csv(f'{LABEL}_candidate_selection_log.csv', index=False)
print(f"Candidate ranking saved -> {LABEL}_candidate_selection_log.csv")

# ---------------------------------------------------------------------
# Final model: fit once on 100% of the data with the selected
# hyperparameters. Deterministic and reproducible end to end.
# ---------------------------------------------------------------------
X_full_enc, full_profiles = build_features(X_peptides, y)
log_b, log_nb, dpps_b, dpps_nb, hydro_b, hydro_nb = full_profiles

final_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('svc', SVC(kernel='rbf', C=best_C, gamma=best_gamma,
                probability=True, class_weight='balanced',
                random_state=SVC_RANDOM_STATE)),
])
final_pipeline.fit(X_full_enc, y)

final_pkl_name = f'{LABEL}_FinalDeploy_C{best_C}_g{best_gamma}_selAUC{model_selection_auc:.4f}.pkl'

joblib.dump({
    'model': final_pipeline,
    'log_b': log_b,
    'log_nb': log_nb,
    'dpps_b': dpps_b,
    'dpps_nb': dpps_nb,
    'hydro_b': hydro_b,
    'hydro_nb': hydro_nb,
    'best_C': best_C,
    'best_gamma': best_gamma,
    'model_selection_auc': model_selection_auc,
    'model_selection_auc_std_across_runs': model_selection_auc_std,
    'nested_cv_roc_auc_mean': perf_df['roc_auc'].mean(),
    'nested_cv_roc_auc_std': perf_df['roc_auc'].std(),
    'n_runs': int(best_row['n_runs']),
    'n_splits': N_SPLITS,
    'svc_random_state': SVC_RANDOM_STATE,
    'label': LABEL,
}, final_pkl_name)

print(f"\nFinal deployment model saved -> {final_pkl_name}")

