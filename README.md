# SpY-C
SH2-pY global classification SVM model
<p align="center">
  <img src="SPYC_graphic.png" width="600" height="400">
</p>

---
# Requirements

## Python Version
- Python **≥ 3.9**

## Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```
---
### Overview of SpY-C pipeline

![Pipeline Overview](SPYC_workflow.png) 


### Step 1 - To build an SVM-based SH2-pY classifier using curated binder and non-binder peptide datasets.
```bash 
python build_model.py
```
1. Inputs
- Positive training set (Final_positive_training.txt)
- Negative training set (Final_negative_training.txt)

2. Outputs
- Trained model artifact (.pkl)
- test_nested_cv_performance.csv
- _candidate_selection_log.csv

### Step 2 - To evaluate model performance.
- Useful to compare across candidate training models to pick the best training dataset
```bash
python evaluate_model.py \
  --best_pkl FinalSet_FinalDeploy_C0.5_g1.0_selAUC0.9304.pkl \
  --binder Final_positive_training.txt \
  --nonbinder Final_negative_training.txt \
  --test_pos comb_positive_evaluation.txt \
  --test_neg comb_negative_evaluation.txt \
  --label FinalSet \
  --output evaluation_results_FinalSet.csv
```

Outputs
- files


### Step 4 - To run predictions using the selected trained model on new peptide datasets.
```bash

python deploy_model.py \
  --model examples/test.pkl \
  --binders Data/Training_peptides/Final_positive_training.txt \
  --nonbinders Data/Training_peptides/Final_negative_training.txt \
  --datasets examples/set1.txt examples/set2.txt \
  --outdir results/ \
  --names set1 set2
```

Inside results/:

1. *_predictions.csv → peptide wise predicted class + probabilities
2. Bootstrap based Summary statistics across each Dataset 

