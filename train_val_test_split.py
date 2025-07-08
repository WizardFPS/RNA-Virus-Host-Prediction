"""
Train-test function for "Non-overlapping taxa" strategy dataset split,
originally used in the paper "Virus host prediction using machine learning and short sequence k-mers: effect of taxonomic, host-dependent features and sample bias".
"""

def taxa_splitter(df, list_taxa, taxa_level_input = "genus", test_taxon = None):
   
    taxa_level = "virus "+taxa_level_input # column name in df; column contains taxon names
    
    if test_taxon:
        future_sample_id, t_genus = [], [test_taxon]
        future_sample_id += list(df[df[taxa_level] == test_taxon].index)
        return(future_sample_id, t_genus)
    
    else:
        z = random.choice(list_taxa)
        list_taxa.remove(z)
        l = len(df[df[taxa_level] == z].index)
        a = round((l)/len(df), 3)
        t_genus = [z]

        print('Start', z, taxa_level_input+'_%: ' + str(round(len(df[df[taxa_level] == z].index)*100/len(df), 3)), 'total_%: ' + str(round(a, 3)*100))
        mark = 0
        
        
        left_border = 0.2  # min proportion of test sample | can be varied
        right_border = 0.3 # max proportion of test sample | can be varied
        
        
        if a < left_border:
            while a < left_border or a>=right_border:

                z = random.choice(list_taxa)

                mark+=1
                print('Cycle ' + str(mark), z, taxa_level_input+'_%: ' + str(round(len(df[df[taxa_level] == z].index)*100/len(df), 3)), 'total_%: ' + str(round((a+round(len(df[df[taxa_level] == z].index)/len(df), 3))*100, 3)))
                list_taxa.remove(z)
                a+=len(df[df[taxa_level] == z].index)/len(df)
                a=round(a, 3)
                if a<left_border:
                    t_genus.append(z)
                if a>=left_border and a<right_border:
                    t_genus.append(z)
                    break
                if a>=right_border:
                    a-=round(len(df[df[taxa_level] == z].index)/len(df), 3)
        print('End' ,a, t_genus)

    future_sample_id =[]
    for genus in t_genus:
        future_sample_id += list(df[df[taxa_level] == genus].index)

    if abs(round(len(future_sample_id)/len(df), 2) - round(a, 2)) <=0.1:   
        print('Result: True')
    else:
        print('Error', round(len(future_sample_id)/len(df), 2), round(a, 2))

    return(future_sample_id, t_genus)
    
def distribution_check(a, b, c):
    
    bad_split = False
    
    if a.keys() != b.keys():
        bad_split = True
        #print('Not enough classes')
        return(bad_split)
    
    if a['Insecta'] < 0.07:
            bad_split = True
            return(bad_split)
    if b['Insecta'] < 0.07:
            bad_split = True
            return(bad_split)

    for key in a.keys():
        
        if a[key] < c[key]-0.05: 
            bad_split = True
            return(bad_split)
        if b[key] < c[key]-0.05: 
            bad_split = True
            return(bad_split)       

        if abs(a[key] - b[key]) > 0.05:
            #print(key)
            bad_split = True
            return(bad_split)

    return(bad_split)
"""
Options:
df - meta dataframe table | includes sequence indices
list_taxa - all unique taxa | can be replace in function with df[f"virus {taxa_level_input}"].unique()
taxa_level_input - taxon level for non-overlapping taxa split
test_taxon - if not None script creates test sample from test_taxon objects (genomes/fragments)


Usage:
from collections import Counter
import pandas as pd
import numpy as np

taxa_level = "virus genus"
meta_df = pd.read_csv("meta_df.tsv", index_col=0)
bad_split = True




while bad_split:
    list_genus = sorted(meta_df[taxa_level].unique())
    X_test_indices, t_genus = taxa_splitter(meta_df, list_genus)
    X_train_indices = meta_df[~meta_df.index.isin(X_test_indices)]
    y_train, y_test = meta_df.loc[X_train_indices]["virus host"], meta_df.loc[X_test_indices]["virus host"]
    
    a = dict(zip(list(Counter(y_test).keys()), list(map(lambda x: round(x/len(y_test), 4), list(Counter(y_test).values())))))
    b = dict(zip(list(Counter(y_train).keys()), list(map(lambda x: round(x/len(y_train), 4), list(Counter(y_train).values())))))
    c = dict(zip(list(Counter(meta_df['host']).keys()), list(map(lambda x: round(x/len(meta_df), 3), list(Counter(meta_df['host']).values())))))
    bad_split = distribution_check(a, b, c)
    
print('Hosts in test:', dict(zip(list(Counter(y_test).keys()), list(map(lambda x: str(round(x/len(y_test)*100, 3))+'%', list(Counter(y_test).values()))))))
print('Hosts in train:', dict(zip(list(Counter(y_train).keys()), list(map(lambda x: str(round(x/len(y_train)*100, 3))+'%', list(Counter(y_train).values()))))))
"""

"""
Further developed similar train-val-test function for "Non-overlapping taxa" strategy dataset split.
"""
def stratified_split_data(df, target_col, feature1_col, feature2_col, test_size=0.2, val_size=0.2, random_state=None):

    df = df.copy()
    original_count = len(df)
    df['_stratify'] = df[target_col].astype(str) + "_" + df[feature1_col].astype(str)
    
    grouped = df.groupby(feature2_col)
    groups = [group for _, group in grouped]
    
    np.random.seed(random_state)
    np.random.shuffle(groups)
    
    n = len(groups)
    n_test = max(1, int(n * test_size))
    n_val = max(1, int(n * val_size))
    n_train = max(1, n - n_test - n_val)
    
    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train+n_val]
    test_groups = groups[n_train+n_val:]
    
    train_df = pd.concat(train_groups) if train_groups else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_groups) if val_groups else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_groups) if test_groups else pd.DataFrame(columns=df.columns)
    
    def adjust_distribution(data, strat_col):
        if len(data) == 0:
            return data
        
        current_dist = data[strat_col].value_counts(normalize=True)
        if current_dist.max() > 0.5:  
            try:
                _, adjusted = train_test_split(
                    data,
                    test_size=0.8,
                    stratify=data[strat_col],
                    random_state=random_state
                )
                return adjusted
            except:
                return data
        return data
    
    train_df = adjust_distribution(train_df, '_stratify')
    val_df = adjust_distribution(val_df, '_stratify')
    test_df = adjust_distribution(test_df, '_stratify')
    
    def get_unique_features(df_part):
        return set(df_part[feature2_col].unique()) if len(df_part) > 0 else set()
    
    train_features = get_unique_features(train_df)
    val_features = get_unique_features(val_df)
    test_features = get_unique_features(test_df)
   
    assert len(train_features & val_features) == 0, "Пересечения между train и val"
    assert len(train_features & test_features) == 0, "Пересечения между train и test"
    assert len(val_features & test_features) == 0, "Пересечения между val и test"
    
    total_after_split = len(train_df) + len(val_df) + len(test_df)
    assert total_after_split == original_count, f"Потеряны объекты: было {original_count}, стало {total_after_split}"
    
    for df_part in [train_df, val_df, test_df]:
        if '_stratify' in df_part.columns:
            df_part.drop('_stratify', axis=1, inplace=True)
        df_part = df_part.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return train_df, val_df, test_df


"""
Train-test function for "Closely related" strategy dataset split.
"""
def train_val_test_split(X, y, stratify=None, val_size=0.20, test_size=0.20, random_state=13):
    from sklearn.model_selection import train_test_split

    data = pd.DataFrame()
    data["X"], data["y"], data["stratify_group"] = X, y, stratify
    unique_indices_train = data.groupby("stratify_group").filter(lambda x: len(x) == 1).index.values # taxa with 1 object are places to th train sample 
    
    X_train, X_step, y_train, y_step = train_test_split(data[~data.index.isin(unique_indices_train)].X.values,
                                                        data[~data.index.isin(unique_indices_train)].y.values,
                                                        stratify=data[~data.index.isin(unique_indices_train)].stratify_group.values,
                                                        test_size=val_size+test_size,
                                                        random_state=random_state)

    X_train = np.hstack((X_train, data.loc[unique_indices_train].X.values))
    y_train = np.hstack((y_train, data.loc[unique_indices_train].y.values))

    
    ### Step 2
    data = data[~data.X.isin(X_train)]
    unique_indices_val = data.groupby("stratify_group").filter(lambda x: len(x) == 1).index.values

    X_val, X_test, y_val, y_test = train_test_split(data[~data.index.isin(unique_indices_val)].X.values,
                                                    data[~data.index.isin(unique_indices_val)].y.values,
                                                    stratify=data[~data.index.isin(unique_indices_val)].stratify_group.values,
                                                    test_size=test_size/(val_size+test_size),
                                                    random_state=random_state)
    
    X_val = np.hstack((X_val, data.loc[unique_indices_val].X.values))
    y_val = np.hstack((y_val, data.loc[unique_indices_val].y.values))
    
    return X_train, X_val, X_test, y_train, y_val, y_test
"""
Options:
X - genomes/fragments indices | meta_df.index
y - virus hosts | meta_df["virus host"]
stratify - taxon level column for non-overlapping taxa split | meta_df["virus family"]
val_size - validation sample proportion
test_size - test sample proportion
random_state - random state


Usage:
X_train_indices, X_val_indices, X_test_indices, y_train, y_val, y_test = train_val_test_split(meta_df.index, 
                                                                                              meta_df["virus host"].values, 
                                                                                              stratify=meta_df["virus family"].values, 
                                                                                              val_size=0.20, 
                                                                                              test_size=0.20, 
                                                                                              random_state=42)

With |stratify = None| option function 
"""
