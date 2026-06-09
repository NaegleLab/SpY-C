import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===========================
# ARGUMENTS
# ===========================
parser = argparse.ArgumentParser(description="Analyze nested CV performance and hyperparameter selection.")
parser.add_argument("--perf",    required=True, help="Path to nested CV performance CSV")
parser.add_argument("--runlog",  required=True, help="Path to run selection log CSV")
parser.add_argument("--outdir",  default=".",   help="Directory to save output plots and CSVs (default: current dir)")
args = parser.parse_args()

import os
os.makedirs(args.outdir, exist_ok=True)

# ===========================
# LOAD FILES
# ===========================
perf   = pd.read_csv(args.perf)
runlog = pd.read_csv(args.runlog)

# ===========================
# 1. PERFORMANCE BOXPLOT + SEM
# ===========================
metrics = ['accuracy', 'f1', 'roc_auc']

plt.figure(figsize=(5, 5))
data = [perf[m] for m in metrics]
plt.boxplot(data, tick_labels=metrics)

means = [np.mean(perf[m]) for m in metrics]
sems  = [np.std(perf[m], ddof=1) / np.sqrt(len(perf[m])) for m in metrics]

for i, (mean, sem) in enumerate(zip(means, sems), start=1):
    plt.errorbar(i, mean, yerr=sem, fmt='o', color='black')

plt.ylabel("Score", fontsize=17)
plt.title("Nested CV Performance")
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "performance_boxplot.png"), dpi=300)
plt.close()
print("Saved: performance_boxplot.png")

# ===========================
# 2. HYPERPARAM FREQUENCY (NESTED CV)
# ===========================
hp_counts = (
    perf.groupby(['best_C', 'best_gamma'])
    .size()
    .reset_index(name='count')
    .sort_values(by='count', ascending=False)
)
hp_counts['percentage'] = hp_counts['count'] / hp_counts['count'].sum() * 100

plt.figure(figsize=(7, 5))
plt.bar(range(len(hp_counts)), hp_counts['percentage'])
plt.xticks(
    range(len(hp_counts)),
    [f"C={c}, g={g}" for c, g in zip(hp_counts['best_C'], hp_counts['best_gamma'])],
    rotation=45, ha='right', fontsize=12
)
plt.ylabel("Selection (%)", fontsize=14)
plt.title("Hyperparameter Selection (Nested CV)", fontsize=14)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "nested_hyperparams.png"), dpi=300)
plt.close()
hp_counts.to_csv(os.path.join(args.outdir, "nested_hyperparam_summary.csv"), index=False)
print("Saved: nested_hyperparams.png, nested_hyperparam_summary.csv")

# ===========================
# 3. FINAL MODEL HYPERPARAMS (RUN LOG)
# ===========================
hp_final = (
    runlog.groupby(['best_C', 'best_gamma'])
    .size()
    .reset_index(name='count')
    .sort_values(by='count', ascending=False)
)
hp_final['percentage'] = hp_final['count'] / hp_final['count'].sum() * 100

plt.figure(figsize=(7, 5))
plt.bar(range(len(hp_final)), hp_final['percentage'])
plt.xticks(
    range(len(hp_final)),
    [f"C={c}, g={g}" for c, g in zip(hp_final['best_C'], hp_final['best_gamma'])],
    rotation=45, ha='right', fontsize=12
)
plt.ylabel("Selection (%)", fontsize=14)
plt.title("Final Model Hyperparameter Selection", fontsize=14)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "final_hyperparams.png"), dpi=300)
plt.close()
hp_final.to_csv(os.path.join(args.outdir, "final_hyperparam_summary.csv"), index=False)
print("Saved: final_hyperparams.png, final_hyperparam_summary.csv")

# ===========================
# 4. SUMMARY
# ===========================
best_nested = hp_counts.iloc[0]
best_final  = hp_final.iloc[0]

print("\n=== NESTED CV BEST HYPERPARAMETERS ===")
print(f"  C={best_nested['best_C']}, gamma={best_nested['best_gamma']} "
      f"— selected {best_nested['percentage']:.1f}% of the time")

print("\n=== FINAL MODEL BEST HYPERPARAMETERS ===")
print(f"  C={best_final['best_C']}, gamma={best_final['best_gamma']} "
      f"— selected {best_final['percentage']:.1f}% of the time")

print("\n=== NESTED CV PERFORMANCE SUMMARY ===")
for m in metrics:
    print(f"  {m}: mean={np.mean(perf[m]):.3f}, SEM={np.std(perf[m], ddof=1)/np.sqrt(len(perf[m])):.3f}")
