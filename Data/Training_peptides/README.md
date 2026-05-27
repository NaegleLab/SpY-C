# Model Training Datasets

# Positive Training Sets

## Tinti dataset
- Syk-N
- Syk-C
- Itk

## Crk (`20627867`)

## Lyn (`25587033`)

---

# Negative Training Sets

## Chang dataset (`36711935`)

### Media 2
- ab_HeLaImg = CST antibody [pTyr1000 & pTyr1000] enrichment of 1 mg of HeLa digest
- IMAC_HeLa250 = Fe3+ IMAC enrichment of 250 μg of HeLa peptides
- HeLaImg = Enrichment of pY-peptides from 1 mg of HeLa digest using Src conjugated to magnetic maleimide
- HeLaImg = Enrichment of pY-peptides from 1 mg of HeLa digest using Src conjugated to magnetic maleimide
- IMAC only or IMAC and src superbinder but different conjugation

### Selection Criteria
They used antibodies, IMAC only and src superbinder with different conjugation chemistries.

From this experiment, identified those bound to only IMAC and retained those that were bound only to IMAC but not to any SH2 superbinder from across all their other experiments and used those as negatives for training.

Used the confidently localized phosphosites.

---

## Martyn (`35613471`)

From two conditions:
- pervanadate treatment
- no pervanadate treatment

### Selection Criteria
All the peptides bound to only IMAC but not to any WT or superbinder SH2 domains 
