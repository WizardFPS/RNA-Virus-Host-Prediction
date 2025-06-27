from argparse import ArgumentParser
from Bio import SeqIO
import pandas as pd
import numpy as np
from itertools import product

### Functions


def check_nonamb(seq, alphabet):
    return all(s in alphabet for s in seq)

def create_feature_set(FEATURE_TYPE, k, alph):

    set_name = f'{FEATURE_TYPE}_{k}'
    feature_set =[''.join(c) for c in product(alph, repeat=k)]
    return set_name, feature_set

def count_features(FEATURE_TYPE, FASTA_FILE, k, shift = None):
    if FEATURE_TYPE == "RNA":
        alph = "AUGC"
    if FEATURE_TYPE == "DNA":
        alph = "ATGC"
    if FEATURE_TYPE == "AA":
        alph = "ARNDCEQGHILKMFPSTWYV"
    set_name, k_mers = create_feature_set(FEATURE_TYPE, k, alph)
    output = pd.DataFrame(columns = k_mers)
    indicies, hosts = [], []

    with open(FASTA_FILE, "r") as file:
        for seq in SeqIO.parse(file, "fasta"):
            seq.seq = seq.seq.upper()
            hosts.append(seq.description.split("-")[1])
            indicies.append(seq.id)
            feature_dict = {a:0 for a in k_mers}

            if shift != None:
                for i in range(0 + shift, len(seq.seq)-k+1, k):
                    w = seq.seq[i:i+k]
                    if check_nonamb(w, alph):
                        feature_dict[w] += 1

            else:
                for i in range(len(seq.seq)-k+1):
                    w =  seq.seq[i:i+k]
                    if check_nonamb(w, alph):
                        feature_dict[w] += 1

            feature_dict = {k: [v/len(seq.seq)] for k, v in feature_dict.items()}
            output = pd.concat([output, pd.DataFrame(feature_dict)])
        output["extra_index"] = indicies
        output["host"] = hosts
        output.set_index(output["extra_index"].values, inplace=True)
        output.drop("extra_index", axis=1, inplace=True)

        return(set_name, output)



### Body


if __name__ == "__main__":

    parser = ArgumentParser(description="Hi")
    parser.add_argument("-i", help="Input PATH + filename", type=str)
    parser.add_argument("-o", help="Output PATH + filename", type=str)
    parser.add_argument("-ft", help="Feature type, RNA/AA", type=str)
    parser.add_argument("-k", help="K-mers length", type=int)
    #parser.add_argument("-s", help="Separator in output table", type=str)
    
    args = parser.parse_args()
    PATH_INPUT_FILE, PATH_OUTPUT_FILE, FEATURE_TYPE, k, SEPARATOR = args.i, args.o, args.ft, args.k, "\t"

    """
    EXAMPLE

    FEATURE_TYPE, k = "RNA", 2
    PATH_INPUT_FILE = "silva_v4_annotated.fasta"
    PATH_OUTPUT_FILE = "D://DATA//bacteria_geo//RNA_1_v4.tsv"
    SEPARATOR = "\t"
    """
	
set_name, output = count_features(FEATURE_TYPE, PATH_INPUT_FILE, k)
output.to_csv(PATH_OUTPUT_FILE, sep=SEPARATOR)
