# RNA-Virus-Host-Prediction
The repository to the paper "Virus host prediction using machine learning and short sequence k-mers: effect of taxonomic, host-dependent features and sample bias"

## Data

 - VHDB_genomes.fasta - RNA virus genomes in fasta-format

 - VHDB_cds.fasta - RNA virus protein sequences in fasta-format

 - Table_S1.tsv - RNA virus genomes annotation, including host


## Code

 - remove_similar.py - Removes similar sequences by identity
   
 - extract_genome_fragmnets.py - function for fragments extraction from virus genomes

 - count_features.py - Calculate k-mer frequencies

 - train_val_test_split.py - functions for "Closely related" and "Non-overlapping taxa" strategies dataset splits 

 - hyperparameter_tuning.py - Tuning ML hyperparameters on the first iteration of dataset split

 - models_tuned.py - Training classifiers on the remaining nine dataset splits

 - HTP_baseline.ipynb and tBLASTx_baseline.ipynb - ML-based and homology-based baseline methods
