# Peptide Input Format Test Cases

## Test Case 1: Input Files with Only `6mer_peptide` Header
**Description:**  
Input `.txt` and `.csv` files contain only the `6mer_peptide` column header.  
Predictions are generated without label-based grouping.
```python deploy_binder_model_final2.py --model SetA+ITK_run5_auc0.9624_C5_g1.pkl --binders Final_positive_training.txt --nonbinders Final_negative_training.txt --datasets case1.txt case1.csv --outdir ./ --names case1_txt case1_csv --summary_name testcase_1
```
---

## Test Case 2: Input Files with `6mer_peptide` and `label` Headers
**Description:**  
Input files contain both `6mer_peptide` and `label` columns.  
Used for group-wise binder fraction calculations using `--use_labels`.
```python deploy_binder_model_final2.py --model SetA+ITK_run5_auc0.9624_C5_g1.pkl --binders Final_positive_training.txt --nonbinders Final_negative_training.txt --datasets case2.txt case2.csv --outdir ./ --names case2_txt case2_csv --summary_name testcase_2 --use_labels true true
```
---

## Test Case 3: Input Files with `6mer_peptide` and Additional Annotation Columns
**Description:**  
Input `.txt` and `.csv` files contain peptide sequences along with additional annotation columns.  
Files include headers, but no label grouping is performed.
```python deploy_binder_model_final2.py --model SetA+ITK_run5_auc0.9624_C5_g1.pkl --binders Final_positive_training.txt --nonbinders Final_negative_training.txt --datasets case3.txt case3.csv --outdir ./ --names case3_txt case3_csv --summary_name testcase_3 
```
---

## Test Case 4: Annotated Input Files with Label-Based Group Analysis
**Description:**  
Same format as Test Case 3, but includes a `label` column.  
Runs group-wise binder fraction analysis using `--use_labels`.
```python deploy_binder_model_final2.py --model SetA+ITK_run5_auc0.9624_C5_g1.pkl --binders Final_positive_training.txt --nonbinders Final_negative_training.txt --datasets case3.txt case3.csv --outdir ./ --names case4_txt case4_csv --summary_name testcase_4 --use_labels true true
```
---

## Test Case 5: Headerless Input Files Containing Only Peptide Sequences
**Description:**  
Input files do not contain headers and contain only peptide sequences.  
The first column is automatically assigned as `6mer_peptide`.
```python deploy_binder_model_final2.py --model SetA+ITK_run5_auc0.9624_C5_g1.pkl --binders Final_positive_training.txt --nonbinders Final_negative_training.txt --datasets case5.txt case5.csv --outdir ./ --names case5_txt case5_csv --summary_name testcase_5 
```
---

## Test Case 6: Headerless Input Files with Peptides and Additional Annotation Columns
**Description:**  
Input files do not contain headers but contain peptide sequences plus annotation columns.  
The first column is assumed to be `6mer_peptide`, and remaining columns are renamed automatically.
```python deploy_binder_model_final2.py --model SetA+ITK_run5_auc0.9624_C5_g1.pkl --binders Final_positive_training.txt --nonbinders Final_negative_training.txt --datasets case6.txt case6.csv --outdir ./ --names case6_txt case6_csv --summary_name testcase_6 
```
---

## Test Case 7: Input Files with Reordered Columns
**Description:**  
Tests whether the script correctly identifies `6mer_peptide` and `label` columns when columns are not in the expected order.
```python deploy_binder_model_final2.py --model SetA+ITK_run5_auc0.9624_C5_g1.pkl --binders Final_positive_training.txt --nonbinders Final_negative_training.txt --datasets case7.txt case7.csv --outdir ./ --names case7_txt case7_csv --summary_name testcase_7 
```
---

## Important Note

For files without headers:

- The script assumes the **first column contains peptide sequences**.
- Columns are automatically renamed internally.
- `--use_labels` cannot be used for group-wise binder fraction calculations because the script cannot determine which column represents the label without a header.
