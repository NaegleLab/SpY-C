# Candidate positive training Datasets nomenclature 


* FinalSet - Final optimized training set used for SpY-C (LYN(AP-MS), and the rest CRK,SYK-N, SYK-C, ITK from PepspotDB))
* SetA - LYN (AP-MS), and CRK, SYK-N, SYK-C (SPOT arrays) bound peptides from mixed experimental sources. 
* SetB - LYN, CRK, SYK-N, SYK-C all from SPOT arrays (PepspotDB)
* SetC - Strongest binders from PepspotDB (top5% and binding >= 4% SH2 domains) 
* SetD - 'SH2_CBLB', 'SH2_TENC1', 'SH2_ABL1', 'SH2_ITK', 'SH2_FER', 'SH2_BLK', 'SH2_GRB2','SH2_SRC', 'SH2_SOCS5', 'SH2_PLCG2', 'SH2_VAV1', 'SH2_PTPN11' , 'SH2_CRK', 'SH2_SykNSH2' - top 30 peptides for each of these 14 domains from PepspotDB
* SetE - 'SH2_CBLB', 'SH2_SHD', 'SH2_ABL1', 'SH2_GRB14', 'SH2_FES', 'SH2_APS', 'SH2_GRB2','SH2_LYN', 'SH2_SOCS5', 'SH2_ZAP70', 'SH2_VAV2', 'SH2_PTPN6' , 'SH2_CRKL', 'SH2_SykCSH2' - top 30 peptides for each of these 14 domains from PepspotDB
* SetF1 - SetA+ITK+NCK2
* SetF2 - F1+HCK
* SetF3 - F2+FRK
* SetF4 - F3+BRK
* SetF5 - F4+GRB14
* SetF6 - F5+FES
* SetF7 - F6+SH3BP2
*SetF8 - F7+TENC1

---

# Evaluation peptides 

Here we summarize the peptide datasets used for SpY-C model evaluation. The evaluation sets are grouped into **positive** and **negative** peptide groups based on experimental evidence of SH2-domain binding.


# Positive Evaluation Datasets (Evaluation/Positive_evaluation_set.txt)

## 1. K562 Dataset (`35613471`)
Positive evaluation peptides collected from experiments performed:
- Across multiple experimental conditions (With and without pervanadate treatment)
- Identified those bound to WT SH2 domains (Src, CRKL, Fes, P85A_N, Abl1, Nck1, Grb2, Lck, PTN11_N, P85B_N)

---
## 2. PDB Structural Dataset
- Peptides extracted from experimentally resolved PDB structures.

---
## 3. Nash Paper Positive Evaluation Dataset (`20627867`)
- Control peptides used in their SPOT array experiments are used for positive evaluation here. 
	'DDAVPP', 'DGDVPK', 'DVDVPP','DEDEVP','PSVNVQ','NDIIPL','TIAQVQ','VESTVV','PQEEIP','DDDDVD' 

---
## 4. FP Dataset (`32540967`)
- Peptides classified as binders for individual SH2 domains (`functional_binding_call` is 'binder')
- Binding threshold: **Kd < 10 μM**

---
## 5. PepspotDB Dataset
- Top 5% peptides were selected for each domain
- Only peptides binding to at least 5% of the tested domains were retained

---
# Negative Evaluation Datasets (Evaluation/Negative_evaluation_set.txt)

## 1. Chang Dataset (`36711935`)
Peptides identified as:
- Bound only to antibody or IMAC
- Not bound to sSH2 (superbinder) domain in any of their experiments

## 2. FP Polarization Dataset (`32540967`)
- Peptides that did not bind any of the SH2 domains are identified as non-binders and are used as negative examples. `funtional_binding_call` is no-binding for all SH2 domains)

---
#### Input training peptide files can be found in Input_files folder.

---

## Notes

- Positive datasets consist of experimentally validated SH2-binding peptides.
- Negative datasets consist of peptides lacking SH2-binding evidence under the tested experimental conditions.
- Dataset identifiers in parentheses correspond to associated publication identifiers or references.
- Step3_output folder contains example files on which one can use their trained model and make predictions on new datasets. 

