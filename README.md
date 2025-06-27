# RNA-Virus-Host-Prediction
Code material for publication "Virus host prediction using machine learning and short sequence k-mers: effect of taxonomic, host-dependent features and sample bias"

HTP_baseline.ipynb and tBLASTx_baseline.ipynb - ML-based and homology-based baseline methods

hyperparameter_tuning.py and models_tuned.py - Tuning ML hyperparameters on the first iteration of dataset split (hyperparameter_tuning.py) and training classifiers on the remaining nine dataset splits (models_tuned.py)

count_features.py - Calculate k-mer frequencies

VHDB_genomes_.fasta - RNA virus genomes in fasta-format

meta_df_.tsv - RNA virus genomes annotation, including host
