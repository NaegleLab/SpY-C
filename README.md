# SpY-C
SH2-pY global classification SVM model
<img src="SPYC_graphic.png.png" width="600" height="400">


# Requirements

## Python Version
- Python **≥ 3.9**

## Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```
### To build an SVM-based SH2-pY classifier using curated binder and non-binder peptide datasets.
```bash 
python model.py
```
1. Inputs
- Positive training set (Final_positive_training.txt)
- Negative training set (Final_negative_training.txt)

2. Outputs
- Trained model artifacts (.pkl)
- test_nested_cv_performance.csv
- test_run_Selection_log.csv

### To identify best hyperparameters for Final model development.
```bash
python analyze_hyperparams.py \
  --perf test_nested_cv_performance.csv \
  --runlog test_run_Selection_log.csv 
```

Outputs
- summary csv files and png images to chose the best hyperparameters across runs
- The complete training pipeline, including feature construction, hyperparameter optimization, and classification, is evaluated using nested cross-validation. Mean Accuracy, F1-score, and ROC-AUC across the outer folds are used to estimate the generalization performance of the training workflow.

### To access model performance
- Useful to compare candidate training models to pick the best training dataset
```bash
python evaluate_model.py \
        --best_pkl Data/Prototype/Final_selected_artifactFiles/SetA+ITK_run5_auc0.9624_C5_g1.pkl \
        --binder Data/Training_peptides/Final_positive_training.txt \
        --nonbinder Data/Training_peptides/Final_negative_training.txt \
        --test_neg Data/Prototype/Evaluation/Positive_evaluation_Set.txt \
        --test_pos Data/Prototype/Evaluation/Negative_evaluation_Set.txt
```
Outputs
- Accuracy, F1-score, ROC-AUC, Sensitivity, and Specificity are calculated by applying the final selected model (the chosen `.pkl` file together with its corresponding training dataset) to independent positive and negative evaluation datasets. These metrics estimate how well the deployed model generalizes to previously unseen data.
performance on unseen data.  

   
### To run predictions using the trained model on new peptide datasets.
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

### Overview of SpY-C pipeline

![Pipeline Overview](SPYC_workflow.png) 
