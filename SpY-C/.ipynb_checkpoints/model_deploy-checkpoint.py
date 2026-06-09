import argparse
import numpy as np
import pandas as pd
import joblib
import os
from collections import Counter

# ===========================
# ARGUMENTS
# ===========================
parser = argparse.ArgumentParser(
    description="Deploy trained binder model on one or multiple datasets.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("--model",      required=True,  help="Path to .pkl model file")
parser.add_argument("--binders",    required=True,  help="Path to final training binders .txt file")
parser.add_argument("--nonbinders", required=True,  help="Path to final training nonbinders .txt file")
parser.add_argument("--outdir",     default=".",    help="Directory to save outputs (default: current dir)")
parser.add_argument("--n_iter",     type=int,   default=10000, help="Bootstrap iterations (default: 10000)")
parser.add_argument("--frac",       type=float, default=0.95,  help="Fraction of peptides per bootstrap sample (default: 0.95)")

# --- Single dataset ---
parser.add_argument("--dataset",
                    default=None,
                    metavar="FILE",
                    help="Path to a single dataset file (CSV or TXT)")
parser.add_argument("--name",
                    default=None,
                    metavar="LABEL",
                    help="Label for the single dataset (default: filename stem)")

# --- Multiple datasets ---
parser.add_argument("--datasets",
                    nargs="+",
                    default=None,
                    metavar="FILE",
                    help=(
                        "Paths to one or more dataset files, space-separated.\n"
                        "Example:\n"
                        "  --datasets data1.csv data2.csv data3.txt\n"
                        "Pair with --names to assign a label to each (must match order)."
                    ))
parser.add_argument("--names",
                    nargs="+",
                    default=None,
                    metavar="LABEL",
                    help=(
                        "Labels for datasets passed via --datasets, space-separated.\n"
                        "Example:\n"
                        "  --names SetA SetB Control\n"
                        "Count must match --datasets. If omitted, filename stem is used."
                    ))

args = parser.parse_args()

# ===========================
# VALIDATE & BUILD DATASET LIST
# ===========================
dataset_list = []

if args.datasets:
    
    if args.names is not None and len(args.names) != len(args.datasets):
        parser.error(
            f"--names has {len(args.names)} label(s) but --datasets has "
            f"{len(args.datasets)} file(s). Counts must match."
        )
    for i, fpath in enumerate(args.datasets):
        if not os.path.isfile(fpath):
            parser.error(f"Dataset file not found: '{fpath}'")
        label = args.names[i] if args.names else os.path.splitext(os.path.basename(fpath))[0]
        dataset_list.append((fpath, label))

elif args.dataset:
    if not os.path.isfile(args.dataset):
        parser.error(f"Dataset file not found: '{args.dataset}'")
    label = args.name if args.name else os.path.splitext(os.path.basename(args.dataset))[0]
    dataset_list.append((args.dataset, label))

else:
    parser.error(
        "No dataset provided. Use:\n"
        "  --dataset FILE              for a single dataset\n"
        "  --datasets FILE1 FILE2 ...  for multiple datasets"
    )

os.makedirs(args.outdir, exist_ok=True)

# ===========================
# SCORING TABLES
# ===========================
import torch

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
    '-': torch.Tensor([0]*10)
}

hydropathy_dict = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}

# ===========================
# ENCODING FUNCTIONS
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

def encode_logodds_sum(peptides, logodds_b, logodds_nb):
    X = np.zeros((len(peptides), 2))
    for i, pep in enumerate(peptides):
        X[i, 0] = sum(logodds_b.at[pos, aa]  for pos, aa in enumerate(pep) if aa in logodds_b.columns)
        X[i, 1] = sum(logodds_nb.at[pos, aa] for pos, aa in enumerate(pep) if aa in logodds_nb.columns)
    return X

def encode_dpps_sum(peptides, binder_prof, nonbinder_prof):
    X = np.zeros((len(peptides), 2))
    for i, pep in enumerate(peptides):
        b, nb = 0.0, 0.0
        for pos, aa in enumerate(pep):
            if aa in dpps_scores:
                vec = dpps_scores[aa].numpy()
                b  += np.dot(vec, binder_prof[pos])
                nb += np.dot(vec, nonbinder_prof[pos])
        X[i] = [b, nb]
    return X

def encode_hydro_sum(peptides, binder_prof, nonbinder_prof):
    X = np.zeros((len(peptides), 2))
    for i, pep in enumerate(peptides):
        b, nb = 0.0, 0.0
        for pos, aa in enumerate(pep):
            val = hydropathy_dict.get(aa, 0.0)
            b  += val * binder_prof[pos]
            nb += val * nonbinder_prof[pos]
        X[i] = [b, nb]
    return X

# ===========================
# LOAD MODEL
# ===========================
print(f"\nLoading model: {args.model}")
artifacts    = joblib.load(args.model)
model        = artifacts['model']
log_b        = artifacts['log_b']
log_nb       = artifacts['log_nb']
dpps_b       = artifacts['dpps_b']
dpps_nb      = artifacts['dpps_nb']
hydro_b      = artifacts['hydro_b']
hydro_nb     = artifacts['hydro_nb']
EXPECTED_LEN = log_b.shape[0]

print(f"CV AUC        : {artifacts['cv_auc']:.4f}")
print(f"C             : {artifacts['best_C']}  |  gamma: {artifacts['best_gamma']}")
print(f"Peptide length: {EXPECTED_LEN}")

# ===========================
# LOAD BINDERS / NONBINDERS
# ===========================
binders_file    = pd.read_csv(args.binders,    header=None, sep='\t')
nonbinders_file = pd.read_csv(args.nonbinders, header=None, sep='\t')
binders_set     = set(binders_file[0].tolist())
nonbinders_set  = set(nonbinders_file[0].tolist())
overlap         = binders_set & nonbinders_set
binders         = list(binders_set - overlap)
nonbinders      = list(nonbinders_set - overlap)
print(f"\nBinders loaded   : {len(binders)}")
print(f"Nonbinders loaded: {len(nonbinders)}")

# ===========================
# LOAD DATASET HELPER
# ===========================
def load_peptides(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        ds  = pd.read_csv(path)
        col = 'peptide' if 'peptide' in ds.columns else ds.columns[0]
        return ds[col].tolist()
    else:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]

# ===========================
# PREDICT
# ===========================
def predict_peptides(peptides, name="Dataset"):
    total = len(peptides)
    valid = [
        pep.strip() for pep in peptides
        if isinstance(pep, str)
        and '-' not in pep
        and len(pep.strip()) == EXPECTED_LEN
    ]
    n_invalid = total - len(valid)
    if n_invalid:
        print(f"{name}: {n_invalid} peptides rejected (wrong length or contains gap)")
    if not valid:
        print(f"{name}: No valid peptides to predict.")
        return None

    X = np.hstack([
        encode_logodds_sum(valid, log_b,  log_nb),
        encode_dpps_sum(valid,   dpps_b, dpps_nb),
        encode_hydro_sum(valid,  hydro_b, hydro_nb)
    ])

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    print(f"{name}: {preds.sum()} / {len(valid)} predicted binders ({preds.mean()*100:.2f}%)")

    return pd.DataFrame({
        'peptide':         valid,
        'predicted_class': preds,
        'binder_prob':     probs
    }).sort_values('binder_prob', ascending=False).reset_index(drop=True)

# ===========================
# BOOTSTRAP
# ===========================
def bootstrap_evaluate(peptides, name="Dataset", n_iter=10000, frac=0.95):
    filtered = [
        pep.strip() for pep in peptides
        if isinstance(pep, str)
        and '-' not in pep
        and len(pep.strip()) == EXPECTED_LEN
    ]
    eval_peptides = list(set(filtered) - set(binders) - set(nonbinders))
    print(f"\n{name}: {len(eval_peptides)} peptides after filtering training set")

    if len(eval_peptides) == 0:
        print(f"{name}: No valid peptides after filtering.")
        return None

    pred_df = predict_peptides(eval_peptides, name=name)
    if pred_df is None:
        return None

    y_pred      = pred_df["predicted_class"].values
    N           = len(y_pred)
    sample_size = int(N * frac)

    if sample_size < 1:
        print(f"{name}: Not enough peptides for bootstrap sampling.")
        return None

    fractions = []
    for _ in range(n_iter):
        idx = np.random.choice(N, size=sample_size, replace=False)
        fractions.append(y_pred[idx].mean())

    mean_frac = np.mean(fractions)
    sd_frac   = np.std(fractions)

    print(f"\n{'='*45}")
    print(f"  {name} Bootstrap Result")
    print(f"  Binder fraction : {mean_frac:.3f} +/- {sd_frac:.3f}")
    print(f"  N peptides      : {N}")
    print(f"  Bootstrap iters : {n_iter}")
    print(f"{'='*45}\n")

    return {
        "name":           name,
        "fractions":      fractions,
        "mean":           mean_frac,
        "sd":             sd_frac,
        "N":              N,
        "predictions_df": pred_df
    }

# ===========================
# RUN ALL DATASETS
# ===========================
all_summaries = []

for dataset_path, dataset_name in dataset_list:
    print(f"\n{'='*50}")
    print(f"  Processing : {dataset_name}  ({dataset_path})")
    print(f"{'='*50}")

    raw_peptides = load_peptides(dataset_path)
    print(f"Loaded {len(raw_peptides)} peptides")

    result = bootstrap_evaluate(
        raw_peptides,
        name=dataset_name,
        n_iter=args.n_iter,
        frac=args.frac
    )

    if result is None:
        continue

    # Per-peptide predictions
    pred_out = os.path.join(args.outdir, f"{dataset_name}_predictions.csv")
    result["predictions_df"].to_csv(pred_out, index=False)
    print(f"Saved: {pred_out}")

    # All bootstrap fractions
    #frac_out = os.path.join(args.outdir, f"{dataset_name}_bootstrap_fractions.csv")
    #pd.DataFrame({"bootstrap_fraction": result["fractions"]}).to_csv(frac_out, index=False)
    #print(f"Saved: {frac_out}")

    all_summaries.append({
        "dataset":         result["name"],
        "n_peptides":      result["N"],
        "binder_fraction": result["mean"],
        "sd":              result["sd"],
        "n_iter":          args.n_iter,
        "sample_frac":     args.frac
    })

# ===========================
# COMBINED SUMMARY
# ===========================
if all_summaries:
    summary_df  = pd.DataFrame(all_summaries)
    summary_out = os.path.join(args.outdir, "bootstrap_summary_all.csv")
    summary_df.to_csv(summary_out, index=False)
    print(f"\n{'='*50}")
    print(f"  COMBINED SUMMARY -> {summary_out}")
    print(f"{'='*50}")
    print(summary_df.to_string(index=False))
