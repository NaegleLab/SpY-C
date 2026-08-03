#!/usr/bin/env python3
"""
Evaluate a saved model artifact (.pkl) on a combined positive + negative
peptide test set, excluding peptides that were used in training, and
after removing duplicate peptides in the evaluation datasets.

This script imports peptide_encodings.py ONLY to reuse its pure encoding functions
(encode_logodds_sum, encode_dpps_sum, encode_hydro_sum).

peptide_encodings.py must define:
    - dpps_scores         (dict)
    - hydropathy_dict     (dict)
    - encode_logodds_sum(peptides, log_b, log_nb)
    - encode_dpps_sum(peptides, dpps_b, dpps_nb)
    - encode_hydro_sum(peptides, hydro_b, hydro_nb)

Positive and negative peptides are combined into a single labeled set
before scoring. This is required for Accuracy, F1, and ROC-AUC to be
meaningful -- these metrics need both classes present in y_true.

Usage
-----
    python evaluate_model.py \
        --best_pkl final_artifact_files/SetA+ITK_run5_auc0.9624_C5_g1.pkl \
        --binder Final_positive_training.txt \
        --nonbinder Final_negative_training.txt \
        --test_pos evaluation_Sets/comb_positive_evaluation1.txt \
        --test_neg evaluation_Sets/comb_negative_evaluation1.txt

Both --test_pos and --test_neg are required.
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

# ---------------------------------------------------------------------
# Import encoding functions / dictionaries from peptide_encodings.py
# ---------------------------------------------------------------------
from peptide_encodings import (
    encode_logodds_sum,
    encode_dpps_sum,
    encode_hydro_sum)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_training_peptides(binder_file, nonbinder_file):
    """Load the set of all peptides seen during training (binders + nonbinders)."""
    binders = pd.read_csv(binder_file, header=None, sep="\t")
    nonbinders = pd.read_csv(nonbinder_file, header=None, sep="\t")

    training_peptides = set(binders[0].astype(str).str.strip().tolist()) | set(
        nonbinders[0].astype(str).str.strip().tolist()
    )
    return training_peptides


def load_eval_peptides(test_file, training_peptides):
    """
    Load a raw evaluation peptide list, strip whitespace, drop any peptide
    seen in training, then de-duplicate.
    """
    test_df = pd.read_csv(test_file, header=None)
    raw_peptides = test_df[0].astype(str).str.strip().tolist()

    filtered = [p for p in raw_peptides if p not in training_peptides]
    # de-duplicate while keeping order (stable, unlike set() which is unordered)
    unique_peptides = list(dict.fromkeys(filtered))

    return unique_peptides


def encode_peptides(peptides, saved):
    """Build the feature matrix for a list of peptides using the artifact's dicts."""
    X = np.hstack(
        [
            encode_logodds_sum(peptides, saved["log_b"], saved["log_nb"]),
            encode_dpps_sum(peptides, saved["dpps_b"], saved["dpps_nb"]),
            encode_hydro_sum(peptides, saved["hydro_b"], saved["hydro_nb"]),
        ]
    )
    return X


def evaluate_combined(model, saved, test_pos_file, test_neg_file, training_peptides):
    """
    Evaluate the model on positive + negative peptides together.
    """
    pos_peptides = load_eval_peptides(test_pos_file, training_peptides)
    neg_peptides = load_eval_peptides(test_neg_file, training_peptides)

    all_peptides = pos_peptides + neg_peptides
    y_true = np.array(
        [1] * len(pos_peptides) + [0] * len(neg_peptides), dtype=int
    )

    X = encode_peptides(all_peptides, saved)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    fpr = fp / (tn + fp) if (tn + fp) else float("nan")

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print("=" * 50)
    print("COMBINED SET RESULTS")
    print("=" * 50)
    print(f"Positive peptides evaluated : {len(pos_peptides)}")
    print(f"Negative peptides evaluated : {len(neg_peptides)}")
    print(f"Total peptides evaluated    : {len(all_peptides)}")
    print()
    print(f"TP : {tp}    FN : {fn}")
    print(f"TN : {tn}    FP : {fp}")
    print()
    print(f"Sensitivity          : {sensitivity:.3f}")
    print(f"Specificity          : {specificity:.3f}")
    print(f"False positive rate  : {fpr:.3f}")
    print()
    print(f"Accuracy    : {accuracy:.3f}")
    print(f"F1-score    : {f1:.3f}")
    print(f"ROC-AUC     : {roc_auc:.3f}")
    print()

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TP": tp,
        "FN": fn,
        "TN": tn,
        "FP": fp,
        "Positive_peptides": len(pos_peptides),
        "Negative_peptides": len(neg_peptides),
        "Total_peptides": len(all_peptides),
        "Accuracy": round(accuracy, 3),
        "F1-score": round(f1, 3),
        "ROC-AUC": round(roc_auc, 3),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model artifact on positive/negative peptide sets."
    )
    parser.add_argument(
        "--best_pkl",
        required=True,
        help="Path to the already-trained, best-selected model artifact (.pkl)."
        " This is loaded as-is; nothing is retrained.",
    )
    parser.add_argument(
        "--binder", required=True, help="Path to positive/binder training file"
    )
    parser.add_argument(
        "--nonbinder", required=True, help="Path to negative/nonbinder training file"
    )
    parser.add_argument(
        "--test_pos", required=True, help="Path to positive evaluation set"
    )
    parser.add_argument(
        "--test_neg", required=True, help="Path to negative evaluation set"
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Load the already-trained artifact (no retraining happens here)
    # -------------------------------------------------------------
    saved = joblib.load(args.best_pkl)
    model = saved["model"]

    # -------------------------------------------------------------
    # Load training peptides (binders + nonbinders)
    # -------------------------------------------------------------
    training_peptides = load_training_peptides(args.binder, args.nonbinder)
    print(f"Training peptides loaded: {len(training_peptides)}")
    print()

    results = evaluate_combined(
        model, saved, args.test_pos, args.test_neg, training_peptides
    )

    return results


if __name__ == "__main__":
    main()