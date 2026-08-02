# Candidate positive training Datasets nomenclature 

SetA - LYN (AP-MS), and CRK, SYK-N, SYK-C (SPOT arrays) bound peptides from mixed experimental sources. 
SetB - LYN, CRK, SYK-N, SYK-C all from SPOT arrays (PepspotDB)
SetC - Strongest binders from PepspotDB (top5% and binding >= 4% SH2 domains) 
SetD - 'SH2_CBLB', 'SH2_TENC1', 'SH2_ABL1', 'SH2_ITK', 'SH2_FER', 'SH2_BLK', 'SH2_GRB2','SH2_SRC', 'SH2_SOCS5', 'SH2_PLCG2', 'SH2_VAV1', 'SH2_PTPN11' , 'SH2_CRK', 'SH2_SykNSH2' - top 30 peptides for each of these 14 domains from PepspotDB
SetE - 'SH2_CBLB', 'SH2_SHD', 'SH2_ABL1', 'SH2_GRB14', 'SH2_FES', 'SH2_APS', 'SH2_GRB2','SH2_LYN', 'SH2_SOCS5', 'SH2_ZAP70', 'SH2_VAV2', 'SH2_PTPN6' , 'SH2_CRKL', 'SH2_SykCSH2' - top 30 peptides for each of these 14 domains from PepspotDB
SetF1 - SetA+ITK+NCK2
SetF2 - F1+HCK
SetF3 - F2+FRK
SetF4 - F3+BRK
SetF5 - F4+GRB14
SetF6 - F5+FES
SetF7 - F6+SH3BP2
SetF8 - F7+TENC1

# Evaluation peptides 

## Positive set 
The following are the different sources used to generate a positive evaluation set of peptides that are known to bind native SH2 domains and a negative evaluation set that are confident nonbinders. 

1. PDB structures 
2. PepspotDB - top5% of each SH2 domain and peptides that have bound atleast 5% SH2 domains tested 
3. K562 - Peptides bound to WT SH2 across all their experiments
4. FP - peptides whose `functional_binding_call` is 'binder' 
5. controls - 'DDAVPP', 'DGDVPK', 'DVDVPP','DEDEVP','PSVNVQ','NDIIPL','TIAQVQ','VESTVV','PQEEIP','DDDDVD' - obtained from PUBMED:20627867

## Negative set

1. Hela - Peptides bound to IMAC or Antibodies but not to any sSH2 within their experiments
2. FP - a small set of peptides were found to not bind any of the SH2 domains within the set of SH2 domains tested in their experiment; `funtional_binding_call` is no-binding
