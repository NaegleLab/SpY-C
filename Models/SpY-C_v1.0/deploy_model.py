import argparse
import numpy as np
import pandas as pd
import joblib
import os
from collections import Counter
import re
import torch

# ===========================
# ARGUMENTS
# ===========================
parser = argparse.ArgumentParser(
    description="Deploy trained binder model on one or multiple datasets.",
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--model",      required=True,  help="Path to .pkl model file")
parser.add_argument("--binders",    required=True,  help="Path to final training binders .txt file")
parser.add_argument("--nonbinders", required=True,  help="Path to final training nonbinders .txt file")
parser.add_argument("--outdir",     default=".",    help="Directory to save outputs (default: current dir)")
parser.add_argument("--n_iter",     type=int,   default=10000, help="Bootstrap iterations (default: 10000)")
parser.add_argument("--frac",       type=float, default=0.95,  help="Fraction of peptides per bootstrap sample (default: 0.95)")
parser.add_argument("--summary_name",
                    default="bootstrap_summary",
                    metavar="NAME",
                    help=(
                        "Base name (without extension) for the combined summary CSV.\n"
                        "Default: bootstrap_summary_all\n"
                        "Example: --summary_name run1_mysample"))

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
                        "Pair with --names to assign a label to each (must match order)."))
parser.add_argument("--names",
                    nargs="+",
                    default=None,
                    metavar="LABEL",
                    help=(
                        "Labels for datasets passed via --datasets, space-separated.\n"
                        "Example:\n"
                        "  --names SetA SetB Control\n"
                        "Count must match --datasets. If omitted, filename stem is used."))

parser.add_argument("--use_labels",
                    nargs="+",
                    default=None,
                    metavar="true|false",
                    help=(
                        "Per-file flag: whether to stratify bootstrap by a 'label' column.\n"
                        "Provide one value per file in the same order as --datasets.\n"
                        "Accepted values (case-insensitive): true / false / yes / no / 1 / 0\n\n"
                        "Examples:\n"
                        "  --dataset file.csv --use_labels true\n\n"
                        "  --datasets a.csv b.csv c.txt --use_labels true false false\n\n"
                        "Single value with --datasets broadcasts to all files.\n"
                        "Omit entirely to disable label-stratification for all files."))

args = parser.parse_args()

# ===========================
# VALIDATE & BUILD DATASET LIST
# ===========================

def _parse_bool(val, argname):
    """Convert 'true'/'false'/'1'/'0' string to Python bool."""
    if val.lower() in ('true', 'yes', '1'):
        return True
    if val.lower() in ('false', 'no', '0'):
        return False
    parser.error(f"{argname}: unrecognised boolean value '{val}'. "
                 f"Use true/false, yes/no, or 1/0.")

dataset_list = []   # list of (path, name, use_labels_bool)

if args.datasets:
    n_files = len(args.datasets)

    if args.names is not None and len(args.names) != n_files:
        parser.error(
            f"--names has {len(args.names)} value(s) but --datasets has "
            f"{n_files} file(s). Counts must match.")

    if args.use_labels is None:
        ul_flags = [False] * n_files
    elif len(args.use_labels) == 1:
        ul_flags = [_parse_bool(args.use_labels[0], '--use_labels')] * n_files
    elif len(args.use_labels) == n_files:
        ul_flags = [_parse_bool(v, '--use_labels') for v in args.use_labels]
    else:
        parser.error(
            f"--use_labels has {len(args.use_labels)} value(s) but --datasets has "
            f"{n_files} file(s). Provide either 1 value (applied to all) or one per file.")

    for i, fpath in enumerate(args.datasets):
        if not os.path.isfile(fpath):
            parser.error(f"Dataset file not found: '{fpath}'")
        name = args.names[i] if args.names else os.path.splitext(os.path.basename(fpath))[0]
        dataset_list.append((fpath, name, ul_flags[i]))

elif args.dataset:
    if not os.path.isfile(args.dataset):
        parser.error(f"Dataset file not found: '{args.dataset}'")
    name = args.name if args.name else os.path.splitext(os.path.basename(args.dataset))[0]

    if args.use_labels is None:
        ul = False
    elif len(args.use_labels) == 1:
        ul = _parse_bool(args.use_labels[0], '--use_labels')
    else:
        parser.error("--use_labels: only one value expected when using --dataset (single file).")

    dataset_list.append((args.dataset, name, ul))

else:
    parser.error(
        "No dataset provided. Use:\n"
        "  --dataset FILE              for a single dataset\n"
        "  --datasets FILE1 FILE2 ...  for multiple datasets")

os.makedirs(args.outdir, exist_ok=True)

# ===========================
# SCORING TABLES
# ===========================

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
    '-': torch.Tensor([0]*10)}

hydropathy_dict = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}

# encoding helpers
def make_position_matrix(peptides, amino_acids="ACDEFGHIKLMNPQRSTVWY"):
    L = len(peptides[0])
    pfm = pd.DataFrame(0, index=range(L), columns=list(amino_acids))
    for pos in range(L):
        counts = Counter(pep[pos] for pep in peptides)
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

def encode_dpps_sum(peptides, bp, nbp):
    X = np.zeros((len(peptides), 2))
    for i, pep in enumerate(peptides):
        b = nb = 0.0
        for pos, aa in enumerate(pep):
            if aa in dpps_scores:
                vec = dpps_scores[aa].numpy()
                b  += np.dot(vec, bp[pos])
                nb += np.dot(vec, nbp[pos])
        X[i] = [b, nb]
    return X

def encode_hydro_sum(peptides, bp, nbp):
    X = np.zeros((len(peptides), 2))
    for i, pep in enumerate(peptides):
        b = nb = 0.0
        for pos, aa in enumerate(pep):
            val = hydropathy_dict.get(aa, 0.0)
            b  += val * bp[pos]
            nb += val * nbp[pos]
        X[i] = [b, nb]
    return X

# load model + training data
print(f"\nloading model from {args.model}")
artifacts = joblib.load(args.model)
model     = artifacts['model']
log_b, log_nb   = artifacts['log_b'],   artifacts['log_nb']
dpps_b, dpps_nb = artifacts['dpps_b'],  artifacts['dpps_nb']
hydro_b, hydro_nb = artifacts['hydro_b'], artifacts['hydro_nb']
EXPECTED_LEN = log_b.shape[0]

print(f"  model-selection auc={artifacts['model_selection_auc']:.4f}  "
      f"nested-cv auc={artifacts['nested_cv_roc_auc_mean']:.4f}  "
      f"C={artifacts['best_C']}  gamma={artifacts['best_gamma']}  pep_len={EXPECTED_LEN}")

#print(f"  cv auc={artifacts['cv_auc']:.4f}  C={artifacts['best_C']}  gamma={artifacts['best_gamma']}  pep_len={EXPECTED_LEN}")

# binders / nonbinders - remove anything that appear in both
binders_file    = pd.read_csv(args.binders,    header=None, sep='\t')
nonbinders_file = pd.read_csv(args.nonbinders, header=None, sep='\t')
binders_set    = set(binders_file[0].tolist())
nonbinders_set = set(nonbinders_file[0].tolist())
overlap        = binders_set & nonbinders_set
binders    = list(binders_set - overlap)
nonbinders = list(nonbinders_set - overlap)
print(f"  binders: {len(binders)}  nonbinders: {len(nonbinders)}  overlap removed: {len(overlap)}")

def assign_confidence(prob):
    # >=0.6 binder, <=0.4 nonbinder, else low-confidence
    if prob >= 0.6:
        return 'confident binder'
    if prob <= 0.4:
        return 'confident nonbinder'
    return 'low-confidence'

# ===========================
# LOAD DATASET HELPER
# Returns full DataFrame + the name of the peptide column
# ===========================

def load_peptides(path, use_labels=False):
    # reads csv or txt, returns (dataframe, peptide_col_name)
    # for csv: looks for 6mer_peptide col, else uses first col
    # for txt: checks header for 6mer_peptide, else treats first field as pep
    ext = os.path.splitext(path)[1].lower()

    if ext == '.csv':
        ds = pd.read_csv(path)
        if '6mer_peptide' in ds.columns:
            pep_col = '6mer_peptide'
        else:
            ds = pd.read_csv(path, header=None)
            n_cols = ds.shape[1]
            cols_names = (['6mer_peptide'] + [f'col{i}' for i in range(1, n_cols)])
            ds.columns = cols_names
            pep_col = '6mer_peptide' 
            print(f"  no '6mer_peptide' col in csv, using first col: '{pep_col}'")
            

        if use_labels and 'label' not in ds.columns:
            raise ValueError(f"use_labels=True but no 'label' column in {path}\n"
                             f"columns found: {list(ds.columns)}")
        return ds, pep_col

    else:
        with open(path) as f:
            lines = [ln.rstrip('\n') for ln in f if ln.strip()]

        if not lines:
            raise ValueError(f"file is empty: {path}")

        first_fields = lines[0].split('\t')
        if '6mer_peptide' in first_fields:
            from io import StringIO
            ds = pd.read_csv(StringIO('\n'.join(lines)), sep='\t')
            pep_col = '6mer_peptide'
            print("  txt header found, using '6mer_peptide' column")
        else:
            # no header, grab first tab field from each line
            cols = [line.split('\t') for line in lines]
            n_extra_cols = len(cols[0]) - 1
            cols_names = ['6mer_peptide'] + [f'col{i}' for i in range(1, n_extra_cols + 1)]
            ds = pd.DataFrame(cols, columns=cols_names)

            # peps = [line.split('\t')[0].strip() for line in lines]
            # ds = pd.DataFrame({'6mer_peptide': peps})
            pep_col = '6mer_peptide'
            print("  no header in txt, treating first field per line as peptide")

        if use_labels and 'label' not in ds.columns:
            raise ValueError(f"use_labels=True but no 'label' column in {path}\n"
                             f"columns found: {list(ds.columns)}")
        return ds, pep_col
# predict on a de-duplicated list of peptides, return a lookup df
def predict_unique_peptides(peps, name="Dataset"):
    X = np.hstack([
        encode_logodds_sum(peps, log_b,  log_nb),
        encode_dpps_sum(peps,   dpps_b, dpps_nb),
        encode_hydro_sum(peps,  hydro_b, hydro_nb)])

    probs = model.predict_proba(X)[:, 1]
    # Derive predicted_class FROM the probability, don't call model.predict()
    # separately. SVC(probability=True) computes .predict() from the raw
    # decision_function sign and .predict_proba() from a separately fit
    # Platt-scaling sigmoid -- these can disagree near the boundary, which
    # would make a peptide show up as "confident nonbinder" in
    # confidence_category while still counting as a binder (y=1) in the
    # bootstrap binder_fraction below. Thresholding the probability directly
    # guarantees predicted_class, binder_prob, and confidence_category can
    # never contradict each other.
    preds = (probs >= 0.5).astype(int)

    n_bind = int(preds.sum())
    print(f"  {name}: {n_bind}/{len(peps)} unique novel predicted binders ({preds.mean()*100:.1f}%)")

    res = pd.DataFrame({
        'predicted_class':     preds,
        'binder_prob':         probs,
        'confidence_category': [assign_confidence(p) for p in probs]
    }, index=peps)
    res.index.name = '_pep_key'
    return res

# ===========================
# BOOTSTRAP
# ===========================
def bootstrap_evaluate(input_df, pep_col, name="Dataset",
                        n_iter=10000, frac=0.95, use_labels=False):
    # predict on novel peps, bootstrap binder fraction
    # use_labels=True -> stratify by 'label' column, also run overall
    df = input_df.copy()
    df[pep_col] = df[pep_col].astype(str).str.strip()

    # classify each row: novel / training / short / invalid
    # novel   = right length, not in training set -> gets model prediction
    # training = in binders or nonbinders -> gets ground truth label, NaN probs
    # short   = length < EXPECTED_LEN -> both NaN
    # invalid = gap char, 'nan' string, or too long -> both NaN
    training_set = set(binders) | set(nonbinders)

    df['_clean_peptide'] = (
    df[pep_col]
      .fillna('')
      .astype(str)
      .str.strip()
      .str.upper()
      .str.replace(r'[^A-Z]', '', regex=True))
    def categorize(pep):

        if pep == '':
            return 'invalid'
    
        n = len(pep)
    
        if n < EXPECTED_LEN:
            return 'short'
    
        if n > EXPECTED_LEN:
            return 'invalid'
    
        if pep in training_set:
            return 'training'
    
        return 'novel'

    cats = df['_clean_peptide'].map(categorize)

    novel_mask    = cats == 'novel'
    training_mask = cats == 'training'
    short_mask    = cats == 'short'

    n_novel    = novel_mask.sum()
    n_training = training_mask.sum()
    n_short    = short_mask.sum()
    n_invalid  = (cats == 'invalid').sum()

    print(f"\n{name}: {n_novel} novel rows | {n_training} training | "
          f"{n_short} short (<{EXPECTED_LEN}aa) | {n_invalid} invalid  (total {len(df)})")

    # unique novel peptides -> these go to the model
    # unique_novel = df.loc[novel_mask, pep_col].unique().tolist()
    unique_novel = (df.loc[novel_mask, '_clean_peptide'].unique().tolist())
    n_dupes = n_novel - len(unique_novel)
    if n_dupes:
        print(f"  {n_dupes} duplicate novel rows, model runs on {len(unique_novel)}")

    if not unique_novel:
        print(f"{name}: no novel peptides to predict, skipping.")
        return None

    novel_lookup = predict_unique_peptides(unique_novel, name=name)

    # training peptides get ground-truth label, NaN for probs
    # unique_tr = df.loc[training_mask, pep_col].unique().tolist()
    unique_tr = (df.loc[training_mask, '_clean_peptide'].unique().tolist())

    def tr_label(pep):
        if pep in binders_set - overlap:
            return 1
        elif pep in nonbinders_set - overlap:
            return 0
        return np.nan

    tr_lookup = pd.DataFrame({
        'predicted_class':     np.nan,
        'binder_prob':         np.nan,
        'confidence_category': [tr_label(p) for p in unique_tr]
    }, index=unique_tr)
    tr_lookup.index.name = '_pep_key'

    if unique_tr:
        print(f"  {len(unique_tr)} unique training peptides -> ground-truth label, NaN probs")

    # merge everything back; short/invalid rows get NaN automatically (not in any lookup)
    combined = pd.concat([novel_lookup, tr_lookup])
    df['predicted_class']     = df['_clean_peptide'].map(combined['predicted_class'])
    df['binder_prob']         = df['_clean_peptide'].map(combined['binder_prob'])
    df['confidence_category'] = df['_clean_peptide'].map(combined['confidence_category'])

    # if n_short:
    #     print(f"  {n_short} short peptides -> predicted_class and binder_prob set to NaN")

    # explicitly force NaN for short and invalid rows - do not rely on map() missing them
    # if the same sequence string exists in the lookup for any reason, map() would fill it in
    exclude_mask = short_mask | (cats == 'invalid')
    if exclude_mask.any():
        df.loc[exclude_mask, 'predicted_class']     = np.nan
        df.loc[exclude_mask, 'binder_prob']         = np.nan
        df.loc[exclude_mask, 'confidence_category'] = np.nan
    
    if n_short:
        print(f"  {n_short} short peptides -> all prediction columns forced to NaN")
    if n_invalid:
        print(f"  {n_invalid} invalid/gap peptides -> all prediction columns forced to NaN")


    # bootstrap helper
    # mask selects which rows to include (e.g. a label group)
    # unique novel peptides within that group are counted correctly
    def run_bootstrap(row_mask, group_name):
        # get unique novel peps within this group
        grp_novel_mask = novel_mask & row_mask
        # uniq_peps_grp = df.loc[grp_novel_mask, pep_col].unique()
        uniq_peps_grp = (df.loc[grp_novel_mask, '_clean_peptide'].unique())
        N = len(uniq_peps_grp)

        if N < 1:
            print(f"  {group_name}: no novel peptides, skipping bootstrap.")
            return None, None, N

        # pull predicted classes for those unique peps from the lookup
        y = novel_lookup.loc[novel_lookup.index.isin(uniq_peps_grp), 'predicted_class'].values
        sample_sz = max(1, int(N * frac))

        fracs = []
        for _ in range(n_iter):
            idx = np.random.choice(N, size=sample_sz, replace=False)
            fracs.append(y[idx].mean())

        mu = float(np.mean(fracs))
        sd = float(np.std(fracs))

        print(f"  {group_name}: binder fraction {mu:.3f} +/- {sd:.3f}  (N unique novel = {N})")
        return mu, sd, N

    summaries = []

    if use_labels:
        label_col = 'label'
        labels = df[label_col].unique()
        print(f"\n{name}: label-stratified bootstrap for {len(labels)} group(s): {list(labels)}")

        df['label_bootstrap_mean'] = np.nan
        df['label_bootstrap_sd']   = np.nan

        for lbl in labels:
            lbl_mask = df[label_col] == lbl
            mu, sd, N = run_bootstrap(lbl_mask, f"label={lbl}")
            if mu is not None:
                df.loc[lbl_mask, 'label_bootstrap_mean'] = mu
                df.loc[lbl_mask, 'label_bootstrap_sd']   = sd

            summaries.append({
                "dataset":               name,
                "label":                 lbl,
                "n_total_in_group":      int(lbl_mask.sum()),
                "n_unique_novel_in_group": N,
                "binder_fraction":       mu if mu is not None else np.nan,
                "sd":                    sd if sd is not None else np.nan,
                "n_iter":                n_iter,
                "sample_frac":           frac})

        # whole-file row (label = __ALL__)
        mu_all, sd_all, N_all = run_bootstrap(
            pd.Series(True, index=df.index), "ALL GROUPS")
        summaries.insert(0, {
            "dataset":               name,
            "label":                 "__ALL__",
            "n_total_in_group":      len(df),
            "n_unique_novel_in_group": N_all,
            "binder_fraction":       mu_all if mu_all is not None else np.nan,
            "sd":                    sd_all if sd_all is not None else np.nan,
            "n_iter":                n_iter,
            "sample_frac":           frac})

        N_total = N_all if N_all else len(unique_novel)

    else:
        mu, sd, N = run_bootstrap(pd.Series(True, index=df.index), name)
        if mu is None:
            return None

        summaries.append({
            "dataset":               name,
            "label":                 np.nan,
            "n_total_in_group":      len(df),
            "n_unique_novel_in_group": N,
            "binder_fraction":       mu,
            "sd":                    sd,
            "n_iter":                n_iter,
            "sample_frac":           frac})
        N_total = N

    return {
        "name":        name,
        "summaries":   summaries,
        "N":           N_total,
        "n_total":     len(input_df),
        "predictions_df": df,
        "use_labels":  use_labels}

# run everything
all_summaries = []

for path, dname, ul_flag in dataset_list:
    print(f"\n--- {dname} ({path})  label-stratified: {ul_flag} ---")

    data, pep_col = load_peptides(path, use_labels=ul_flag)
    print(f"  loaded {len(data)} rows, peptide col: {pep_col}")

    res = bootstrap_evaluate(data, pep_col, name=dname,
                             n_iter=args.n_iter, frac=args.frac,
                             use_labels=ul_flag)
    if res is None:
        continue

    final_df = res["predictions_df"]
    final_df_to_write = final_df.drop(columns=['_clean_peptide'], errors='ignore')
    pred_out = os.path.join(args.outdir, f"{dname}_predictions.csv")
    final_df_to_write.to_csv(pred_out, index=False) 
    print(f"  saved predictions -> {pred_out}")

    all_summaries.extend(res["summaries"])

if all_summaries:
    summary_df  = pd.DataFrame(all_summaries)
    # make sure column order is consistent regardless of use_labels mix
    col_order = ["dataset", "label", "n_total_in_group", "n_unique_novel_in_group",
                 "binder_fraction", "sd", "n_iter", "sample_frac"]
    summary_df = summary_df.reindex(columns=col_order)

    out = os.path.join(args.outdir, f"{args.summary_name}.csv")
    summary_df.to_csv(out, index=False)
    print(f"\nsummary -> {out}")
    print(summary_df.to_string(index=False))
