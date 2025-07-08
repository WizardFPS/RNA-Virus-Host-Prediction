"""
tBLASTx analysis

First step:
Perform tBLASTx search (test sample against train sample)
tblastx -max_hsps 1 -max_target_seqs 5 -word_size 4 -query TEST_SAMPLE_FASTA -db TRAIN_SAMPLE_BLASTDB\
		-evalue 0.001 -out TBLASTX_OUTPUT_TABLE_TSV\
		-outfmt '6 qseqid sseqid pident qcovs evalue bitscore length qlen slen qstart qend sstart send'
"""


"""
Second step (optional):
Filter tBLASTx output table.tsv via python, e.g.
tblastx_output = pd.read_csv("tblastx_output_table.tsv", sep="\t", index_col="qseqid")
tblastx_output = tblastx_output[(tblastx_output.pident >= 50.0) & (tblastx_output.length >= 100)]
"""

"""
Third step:
Calculate weights and assign hosts for objects from test sample
"""

PATH_SAMPLE_IDS = 'CHANGE_ME' # PATH to files with sequences indices (AC) for train and test samples
PATH_DATA = 'CHANGE_ME' # PATH to fasta-files with sequences and meta data table
PATH_BLAST = "CHANGE_ME" # Directory with tBLASTx results 


def find_and_weight_hosts(finding, df, func_type):
    
    # weights calculation functions
    if func_type:
        function = finding.pident*finding.qcovs
    else:
        function = finding.pident
    
    tmp = pd.DataFrame()
    tmp["findings"] = finding.sseqid.values # virus findings AC
    tmp["host"] = df.loc[finding.sseqid.values].host.values # virus findings hosts
    tmp["weight"] = (function/function.sum()).values # virus findings weights
    
    tmp.fillna(0, inplace=True)
    tmp["index"] = finding.index
    
    # in case host weight equals zero "unclassified" value is assigned
    tmp.loc[tmp.weight == 0, "host"] = "unclassified" 
    tmp.set_index("index", drop=True, inplace=True)

    
    return(tmp)
	
def tblastx_analysis(tblastx_out, df_query, df_db, ids_query, weight_type, save_name):

    # tblastx_out - tblastx ouput table | table.tsv
    # df_query - meta data table for test sample
    # df_db - meta data table for train sample
    # ids_query - indices (AC) for test sample objects (genoms/fragments)
    # weight_type - weights calculation function type; 0 - pident, 1 - pident*qcovs
    # save_name - output filename
    
    y_pred, y_true, weights = [], df_query.loc[ids_query].host.values, []


    for seq_id in ids_query:

        try:
            match = tblastx_out.loc[seq_id] 
            
            if type(match) == type(pd.DataFrame()): 
                tmp = find_and_weight_hosts(match, df_db, weight_type).groupby("host").weight.sum() 
                predicted_host = max(zip(tmp.values, tmp.index))[1]
                weight = max(tmp.values)
                
            if type(match) == type(pd.Series(0)): 
                tmp = find_and_weight_hosts(pd.DataFrame(match).T, df_db, weight_type).groupby("host").weight.sum()
                predicted_host = max(zip(tmp.values, tmp.index))[1]
                weight = max(tmp.values)
            
        except KeyError:
            predicted_host = "unclassified"
            weight = 1.0
        
   
        y_pred.append(predicted_host), weights.append(weight)
        
    print(classification_report(y_true, y_pred, zero_division=0))
    out = pd.DataFrame(zip(ids_query, y_pred, weights), columns=["AC", "host", "weights"])
    
    out.to_csv(save_name, sep="\t", index=0)
    return