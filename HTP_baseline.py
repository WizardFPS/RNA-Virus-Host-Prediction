from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.svm import SVC

import pandas as pd
import numpy as np
import pickle

meta_df = pf.read_csv("META DATA TABLE", sep="\t", index_col=0)

def get_X_y(name, meta_df):
    iteration = name.split("|")[1]
	name = name.split("|")[0]
    df = pd.read_csv(PATH_FEATURES+name+f"_features_it{iteration}.tsv", sep="\t", index_col=0) ### Change filename of features dataset
    y = meta_df.loc[df.index].host.values
    X = StandardScaler().fit_transform(df.values)
    
    return X, y

def find_best_model(X, y, clf, param_grid):
   
    model = GridSearchCV(estimator = clf, param_grid = param_grid, cv=5, verbose=1, scoring="f1_weighted", n_jobs = -1)
    model.fit(X,y)
    print('Best weighted f1-score of Grid search: {}'.format(model.best_score_))
    print("Best model parameters of Grid search: {}".format(model.best_params_))
    return model.best_estimator_
	
# Logistic Regression
param_grid = {
                "penalty":["l1"], 
                "C": [1, 10, 100, 1000],
                "solver": ["liblinear", "saga"],
                "class_weight" : ["balanced"],
                "max_iter": [300]
             }
for iteration in range(10):
	X_train, y_train = get_X_y(f"train_genomes|{iteration}", meta_df)
	best_model = find_best_model(X_train, y_train, LogisticRegression(), param_grid)
	
	X_test, y_test = get_X_y(f"test_genomes|{iteration}", meta_df)
	print(classification_report(y_test, best_model.predict(X_test)))
	
"""	
Support Vector Machine
Regularization values (C) were extracted from Host Taxon Predictor research data
"""

host_c_svc = {"Insecta": 0.03125, "Mammalia": 0.03125, "Viridiplantae": 0.25}

def SVC_multiclassification(models, X_test, y_test, print_binary = False):
    
    y_proba = np.zeros(shape = y_test.shape)
    y_dict_svc = dict(zip(["Insecta", "Mammalia", "Viridiplantae"], LabelBinarizer().fit(y_test).transform(y_test).T))
    
    for host in y_dict_svc.keys():
        y_proba = np.vstack((y_proba, models[host].predict_proba(X_test)[:,1]))
        if print_binary:
            print(classification_report(y_dict_svc[host], models[host].predict(X_test), target_names = ['Others', host]))

    y_pred  = pd.Series(np.argmax((y_proba[1:]/y_proba[1:].sum(axis=0)), axis=0)).map({0: "Insecta", 1: "Mammalia", 2: "Viridiplantae"}).values
    print(classification_report(y_test, y_pred, target_names = ["Insecta", "Mammalia", "Viridiplantae"], zero_division=1))
    
    return

for iteration in range(10):
	X_train, y_train = get_X_y(f"train_genomes|{iteration}", meta_df_genomes)

	svc_genome_models = {}
	y_dict_svc = dict(zip(["Insecta", "Mammalia", "Viridiplantae"], LabelBinarizer().fit(y_train).transform(y_train).T))

	for host in host_c_svc.keys():
		
		svc_genome_models[host] = \
					SVC(kernel = 'linear', probability = True, C = host_c_svc[host]).fit(X_train, y_dict_svc[host])
					
	X_test, y_test = get_X_y(f"test_genomes|{iteration}", meta_df_genomes)

	SVC_multiclassification(svc_genome_models, X_test, y_test)
