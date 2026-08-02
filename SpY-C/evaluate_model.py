#!/usr/bin/env python3
"""
evaluate_model.py

Evaluate a saved model artifact (.pkl) on positive and/or negative
peptide test sets, excluding peptides that were used in training,
and after removing duplicate peptides.

This script does NOT train or re-fit anything. The model is already
trained and saved inside --best_pkl; this script only loads it with
joblib and calls model.predict(). peptide_encodings.py is imported ONLY to reuse
its pure encoding functions (encode_logodds_sum, encode_dpps_sum,
encode_hydro_sum)

Usage
-----
    python evaluate_model.py \
        --best_pkl Data/Prototype/Final_selected_artifactFiles/SetA+ITK_run5_auc0.9624_C5_g1.pkl \
        --binder Data/Training_peptides/Final_positive_training.txt \
        --nonbinder Data/Training_peptides/Final_negative_training.txt \
        --test_neg Data/Prototype/Evaluation/Positive_evaluation_Set.txt \
        --test_pos Data/Prototype/Evaluation/Negative_evaluation_Set.txt

At least one of --test_pos / --test_neg must be given. Both can be given
together.
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------
# Import encoding functions / dictionaries from peptide_encodings.py
# ---------------------------------------------------------------------
from peptide_encodings import (
    encode_logodds_sum,
    encode_dpps_sum,
    encode_hydro_sum,
)

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


def evaluate_negative_set(model, saved, test_neg_file, training_peptides):
    peptides = load_eval_peptides(test_neg_file, training_peptides)

    X = encode_peptides(peptides, saved)
    y_pred = model.predict(X)
    y_true = np.zeros(len(peptides), dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    fpr = fp / (tn + fp) if (tn + fp) else float("nan")

    print("=" * 50)
    print("NEGATIVE SET RESULTS")
    print("=" * 50)
    print(f"Negative peptides evaluated : {len(peptides)}")
    print(f"TN : {tn}")
    print(f"FP : {fp}")
    print(f"Specificity         : {specificity:.3f}")
    print(f"False positive rate : {fpr:.3f}")
    print()

    return {
        "Specificity": specificity,
        "TN": tn,
        "FP": fp,
        "Total": tn + fp,
        "Unique_peptides": len(peptides),
    }


def evaluate_positive_set(model, saved, test_pos_file, training_peptides):
    peptides = load_eval_peptides(test_pos_file, training_peptides)

    X = encode_peptides(peptides, saved)
    y_pred = model.predict(X)
    y_true = np.ones(len(peptides), dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")

    print("=" * 50)
    print("POSITIVE SET RESULTS")
    print("=" * 50)
    print(f"Positive peptides evaluated : {len(peptides)}")
    print(f"TP : {tp}")
    print(f"FN : {fn}")
    print(f"Sensitivity          : {sensitivity:.3f}")
    print()

    return {
        "Sensitivity": sensitivity,
        "TP": tp,
        "FN": fn,
        "Total": tp + fn,
        "Unique_peptides": len(peptides),
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
        "--test_pos", default=None, help="Path to positive evaluation set (optional)"
    )
    parser.add_argument(
        "--test_neg", default=None, help="Path to negative evaluation set (optional)"
    )

    args = parser.parse_args()

    if args.test_pos is None and args.test_neg is None:
        parser.error("Provide at least one of --test_pos or --test_neg")

    # -------------------------------------------------------------
    # Load the already-trained artifact 
    # -------------------------------------------------------------
    saved = joblib.load(args.best_pkl)
    model = saved["model"]

    # -------------------------------------------------------------
    # Load training peptides (binders + nonbinders)
    # -------------------------------------------------------------
    training_peptides = load_training_peptides(args.binder, args.nonbinder)
    print(f"Training peptides loaded: {len(training_peptides)}")
    print()

    results = {}

    if args.test_pos:
        results["positive"] = evaluate_positive_set(
            model, saved, args.test_pos, training_peptides
        )

    if args.test_neg:
        results["negative"] = evaluate_negative_set(
            model, saved, args.test_neg, training_peptides
        )

    return results


if __name__ == "__main__":
    main()
