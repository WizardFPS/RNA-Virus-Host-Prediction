def extract_sequences(X_indices, dataframe, fasta_file, outfile_name, MIN_SEQ_LENGTH = None, replace_N_channel=True):

	# X_indices - genome indices, obtained via train_val_test_split.py
	# dataframe - table with meta data (e.g. virus host column); Table_S1.tsv
	# fasta_file - virus genomes in .fasta format; VHDB_genomes.fasta
    # outfile_name - output file name
	# MIN_SEQ_LENGTH - desired fragment length
	# replace_N_channel - replace ambiguous nucleotides 
	
    with open(fasta_file, "r") as handle:
        if not MIN_SEQ_LENGTH:
            MIN_SEQ_LENGTH = []
            for record in SeqIO.parse(handle, "fasta"):
                if record.id in X_indices:
                    MIN_SEQ_LENGTH.append(len(record.seq))
            MIN_SEQ_LENGTH = min(MIN_SEQ_LENGTH)

        print("Sequnece length equals", MIN_SEQ_LENGTH)
        sequences = []
    
        for record in SeqIO.parse(handle, "fasta"):
            if (record.id in X_indices) & (len(record.seq) >= MIN_SEQ_LENGTH):
                for i in range(0, len(record.seq)//MIN_SEQ_LENGTH):
                    tmp_record = Bio.SeqRecord.SeqRecord(seq="")
                    
                    start = i*MIN_SEQ_LENGTH

                    if replace_N_channel:
                        tmp_record.seq = record.seq[start:(start+MIN_SEQ_LENGTH)].replace("h", random.choice(["a", "t", "c"])).replace("b", random.choice(["g", "t", "c"])).replace("d", random.choice(["a", "t", "g"])).replace("v", random.choice(["a", "g", "c"])).replace("y", random.choice(["t", "c"])).replace("s", random.choice(["g", "c"])).replace("r", random.choice(["a", "g"])).replace("w", random.choice(["a", "t"])).replace("m", random.choice(["a", "c"])).replace("k", random.choice(["g", "t"])).replace("n", random.choice(["a", "t", "g", "c"]))
                    else:
                        tmp_record.seq = record.seq[start:(start+MIN_SEQ_LENGTH)].replace("h", "n").replace("b", "n").replace("d", "n").replace("v", "n").replace("y", "n").replace("s", "n").replace("r", "n").replace("w", "n").replace("m", "n").replace("k", "n")
                    
                    
                    if len(tmp_record.seq) > 1.0*MIN_SEQ_LENGTH: # 1.0 value can be varied in case the researcher allows shorter fragments (e.g. 0.85)
                        tmp_record.description = f"host-{dataframe.loc[record.id]["virus host"]}"
                        tmp_record.id = record.id+f"|{i}"
                        sequences.append(tmp_record)
                    else:
                        print("short fragment detected")
    with open(outfile_name, "w") as handle:
        for record in sequences:
            SeqIO.write(record, handle, "fasta")
            
    print("Done")        
    return
"""    
usage: 

extract_sequences(X_indices, dataframe, fasta_file, outfile_name, MIN_SEQ_LENGTH = None, replace_N_channel=True)


X_indices - genome indices, obtained via train_val_test_split.py | numpy array or python list
dataframe - table with meta data (e.g. virus host column); Table_S1.tsv
fasta_file - virus genomes in .fasta format; VHDB_genomes.fasta
outfile_name - output file name
MIN_SEQ_LENGTH - desired fragment length
replace_N_channel - replace ambiguous nucleotides
"""