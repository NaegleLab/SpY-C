# Model Evaluation Peptide Datasets

Here we summarize the peptide datasets used for SpY-C model evaluation. The evaluation sets are grouped into **positive** and **negative** peptide groups based on experimental evidence of SH2-domain binding.

---

# Positive Evaluation Datasets

## 1. Martyn Dataset (`35613471`)
Positive evaluation peptides collected from experiments performed:
- Across multiple experimental conditions (With and without pervanadate treatment)
- Identified those bound to WT SH2 domains (Src, CRKL, Fes, P85A_N, Abl1, Nck1, Grb2, Lck, PTN11_N, P85B_N)

---
## 2. PDB Structural Dataset
- Peptides extracted from experimentally resolved PDB structures.

---
## 3. Nash Paper Positive Evaluation Dataset (`20627867`)
- Control peptides used in their SPOT array experiments are used for positive evaluation here.

---
## 4. FP Polarization Dataset (`32540967`)
- Peptides classified as binders for individual SH2 domains
- Binding threshold: **Kd < 10 μM**

---
## 5. Tinti Dataset
- Top 5% peptides were selected for each domain
- Only peptides binding to at least 5% of the tested domains were retained

---

# Negative Evaluation Datasets

## 1. Chang Dataset (`36711935`)
Peptides identified as:
- Bound only to antibody or IMAC
- Not bound to sSH2 (superbinder) domain in any of their experiments

## 2. FP Polarization Dataset (`32540967`)
- Peptides that did not bind any of the SH2 domains are identified as non-binders and are used as negative examples.

---
# Notes

- Positive datasets consist of experimentally validated SH2-binding peptides.
- Negative datasets consist of peptides lacking SH2-binding evidence under the tested experimental conditions.
- Dataset identifiers in parentheses correspond to associated publication identifiers or references.
