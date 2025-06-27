import warnings
warnings.filterwarnings('ignore')

import multiprocessing
n_jobs = multiprocessing.cpu_count()-10

import gc
#import cupy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import product
from collections import Counter, defaultdict
import scipy
import pickle

from lightgbm import LGBMClassifier
import xgboost

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report, accuracy_score, f1_score
from sklearn import preprocessing
from sklearn.preprocessing import scale, StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, GroupKFold, StratifiedKFold

taxa_level = "genera"
genomes_800_400 = "400"

PATH_FEATURES = '/home/pereligin/host_prediction/Features/'
PATH_SAMPLE_IDS = '/home/pereligin/host_prediction/sample_ids/genomes_fragments_connected/'
PATH_DATA = '/home/pereligin/host_prediction/data/'
PATH_REPORTS = "/home/pereligin/host_prediction/reports/genomes_fragments_connected/"+taxa_level+"/"
PATH_CLASSIFICATORS = "/home/pereligin/host_prediction/classificators/genomes_fragments_connected/"+taxa_level+"/"

scoring="f1_weighted"
train_val_test_ids = pickle.load(open(PATH_SAMPLE_IDS + taxa_level + f"/train_val_test_{genomes_800_400}_{taxa_level}.pkl", "rb"))[0]

def find_best_model(X, y, clf, param_grid):
    model = GridSearchCV(estimator = clf, param_grid = param_grid, cv=5, verbose=10, scoring=scoring, n_jobs=n_jobs)
    model.fit(X,y)
    print('Best weighted f1-score of Grid search: {}'.format(model.best_score_))
    print("Best model parameters of Grid search: {}".format(model.best_params_))
    return model.best_estimator_

def svc_prep(data, y_class_h):

    classes, list_y = sorted(data[y_class_h].unique()), []

    y = np.array(data[y_class_h])
    pre = preprocessing.LabelEncoder()
    pre.fit(classes)
    y_int = pre.transform(y)

    for i in range(len(data[y_class_h].unique())):
        list_y.append(np.array(y_int == i).astype(int))

    return(list_y, classes, y_int)

# Достают признаки из файлов

def get_data_pd(feature_set):
    X = pd.read_csv(PATH_FEATURES+feature_set[0]+'.csv', index_col=0)
    for feature in feature_set[1:]:
        X = X.join(pd.read_csv(PATH_FEATURES+feature+'.csv', index_col=0))
    return X

def create_feature_set(features, kmer_lists):
    feature_set = [f'{f}_{k}' for i,f in enumerate(features) for k in kmer_lists[i] ]
    return feature_set

def get_X_y(df_table, df_feature, train_ids, test_ids, y_class_h):

    df_feature['indices'] = range(len(df_feature))

    X_train, X_test = df_feature.loc[train_ids].iloc[:, :-1].values, df_feature.loc[test_ids].iloc[:, :-1].values
    y_train, y_test = df_table.loc[train_ids][y_class_h].values, df_table.loc[test_ids][y_class_h].values

    indices_train, indices_test = df_feature.loc[train_ids].iloc[:,-1].values, df_feature.loc[test_ids].iloc[:,-1].values

    print("Train size:", len(y_train), "Test size:", len(y_test))
    return(X_train, X_test, y_train, y_test, indices_train, indices_test)


# Отвевает за обучение классификаторов. С помощью переменной models можно задать как одну модель, так и несколько сразу
# Сохраняются в словарь в порядке, в котором модели указаны в переменной models

def calc_models(models, X_train, y_train, X_test, y_test, indices_test, indices_train, y_class_h, list_y, classes, y_int):

    classificators = []
    svc_classes = []
    cl_reps = []
    for el in models:

        if el == 'rf':

            param_grid = {
                "n_estimators": [100, 250, 500],
                "max_features": ['auto', 'sqrt', 'log2'],
                "max_depth" : [3,5,10,12],
                "n_jobs":[1],
                "criterion" :['gini', 'entropy']

            }


            model = RandomForestClassifier(random_state=42)
            gridCV_model = find_best_model(X_train, y_train, model, param_grid)
            classificators.append(gridCV_model)
            y_pred = gridCV_model.predict(X_test)
            cl_reps.append(classification_report(y_test, y_pred, output_dict = True, zero_division=1))
            print(el, classification_report(y_test, y_pred))

        if el == 'lgbm':

            param_grid = {
                "n_estimators": [250, 500, 1000],
                "num_leaves": [8, 12, 16, 24, 32],
                "max_depth":[-1],#,6,12],
                "reg_lambda":[0.1, 1, 10],
                "reg_alpha": [0, 0.01, 0.1],
                "device":["cpu"],
                "n_jobs":[1],
                "learning_rate":[0.1, 0.01]

            }



            model = LGBMClassifier(random_state=42, verbose=-1)
            gridCV_model = find_best_model(X_train, y_train, model, param_grid)
            classificators.append(gridCV_model)
            y_pred = gridCV_model.predict(X_test)
            cl_reps.append(classification_report(y_test, y_pred, output_dict = True, zero_division=1))
            print(el, classification_report(y_test, y_pred))

        if el == 'xgb':
            le = LabelEncoder()
            y_train, y_test = le.fit_transform(y_train), le.fit_transform(y_test)

            param_grid = {
                "max_depth":[6,12,18],
                "lambda":[0.1, 1, 10],
                "alpha": [0, 0.01, 0.1],
                "n_estimators":[50, 100, 250, 500],
                #"tree_method":["hist"],
                "device":["cpu"],
                "n_jobs":[1],
                "min_child_weight":[3, 7, 12]
            }

            model = xgboost.XGBClassifier(random_state=42, objective='multi:softprob', eval_metric='auc')
            gridCV_model = find_best_model(X_train, y_train, model, param_grid)
            classificators.append(gridCV_model)
            y_pred = gridCV_model.predict(X_test)
            cl_reps.append(classification_report(y_test, y_pred, output_dict = True, target_names = classes, zero_division=1))
            print(el, classification_report(y_test, y_pred, target_names = classes, zero_division=1))

        if el == 'svc':

            param_grid = {
                "C":[1,10,100,1000],
                "gamma":[1, 0.1, 0.001, 0.0001, 'auto'],
                "kernel":['linear','rbf']}


            svc_classificators = []
            scale = True
            y_proba = np.zeros(shape=indices_test.shape)

            for y_class, class_name in tqdm(zip(list_y, classes)):
                y_train = y_class[indices_train]
                y_test = y_class[indices_test]

                gridCV_model = make_pipeline(StandardScaler(),
                                             GridSearchCV(estimator = SVC(probability=True),
                                                          param_grid = param_grid,
                                                          cv=5,
                                                          verbose=10,
                                                          scoring=scoring,
                                                          refit=True,
                                                          n_jobs=n_jobs,
                                                         )
                                            )

                gridCV_model.fit(X_train, y_train)
                svc_classificators.append(gridCV_model)
                y_pred = gridCV_model.predict(X_test)
                y_proba = np.vstack((y_proba, gridCV_model.predict_proba(X_test)[:,1]))
                print(class_name)

                print(classification_report(y_test, y_pred, target_names = ['Others', class_name], zero_division=1))
                svc_classes.append(classification_report(y_test, y_pred, output_dict=True, target_names = ['Others', class_name], zero_division=1))

            classificators.append(svc_classificators)
            y_proba = y_proba[1:]
            y_proba = (y_proba/y_proba.sum(axis=0))


            y_pred_all = np.argmax(y_proba, axis=0)
            cl_reps.append(classification_report(y_int[indices_test], y_pred_all, output_dict = True, target_names = classes, zero_division=1))
            print('F1-score (weighted):', round(f1_score(y_int[indices_test], y_pred_all, average='weighted'), 2))
            print('F1-score (macro):', round(f1_score(y_int[indices_test], y_pred_all, average='macro'), 2))
    return(cl_reps, svc_classes, classificators)


def training(models, features_sets, y_class_h, full_or_ranged, subseq_len = ""):

    # Изменение путей исключительно внутри функции. Подготовка индексов для SVC

    fun_PATH_FEATURES = PATH_FEATURES
    fun_PATH_CLASSIFICATORS = PATH_CLASSIFICATORS
    fun_PATH_REPORTS = PATH_REPORTS


    if full_or_ranged == "full":
        filename_word = "genomes"
        meta_df = pd.read_csv(PATH_DATA+"data_table.tsv", sep="\t", index_col = 0)
        fun_PATH_FEATURES = PATH_FEATURES[:-1]
        list_y, classes, y_int = svc_prep(meta_df, y_class_h)
        fun_all_sets_names = all_sets_names
        fun_sole_sets_names = sole_sets_names

    if full_or_ranged == "ranged":
        filename_word = subseq_len
        fun_PATH_FEATURES = PATH_FEATURES
        meta_df = pd.read_csv(PATH_DATA+"ranged_genus/data_table_" + subseq_len + ".tsv", sep="\t", index_col = 0)
        list_y, classes, y_int = svc_prep(meta_df, y_class_h)
        fun_all_sets_names = all_sets_names
        fun_sole_sets_names = sole_sets_names

    all_classificators = {}
    all_clfs = {}
    svc_classes_clfs = {}


    if len(features_sets[0]) > 1: # если длина 1 призака > 1, то это all_sets (комбинации признаков)

        for fset in features_sets: # берём комбинацию признаков (fset)

            print(fset, 'Collecting data...')
            feature_df = pd.read_csv(fun_PATH_FEATURES+subseq_len+'/'+fset[0]+'.csv', index_col=0)
            X_train, X_test, y_train, y_test, indices_train, indices_test = get_X_y(meta_df, feature_df, train_val_test_ids[0], train_val_test_ids[2], y_class_h) # Сделали начальные X и у по 1 признаку (fset[0]) из нашей комбинации 

            for feature in fset[1:]: # далее добавляем остальные признаки из комбинации (fset[1:])
                feature_df = pd.read_csv(fun_PATH_FEATURES+subseq_len+'/'+feature+'.csv', index_col=0)
                X_train_0, X_test_0, y_train_0, y_test_0, indices_train_0, indices_test_0 = get_X_y(meta_df, feature_df, train_val_test_ids[0], train_val_test_ids[2], y_class_h)
                X_train, X_test = np.hstack((X_train, X_train_0)), np.hstack((X_test, X_test_0))

            print('Data obtained. Calculating models')
            out = calc_models(models, X_train, y_train, X_test, y_test, indices_test, indices_train, y_class_h, list_y, classes, y_int) # Обучаем модели
            all_clfs[fun_all_sets_names[features_sets.index(fset)]], \
            svc_classes_clfs[fun_all_sets_names[features_sets.index(fset)]], \
            all_classificators[fun_all_sets_names[features_sets.index(fset)]] = out[0], out[1], out[2]

        pickle.dump(all_clfs, open(fun_PATH_REPORTS + "reports_comb_" + filename_word + "_part3.pkl", "wb"))
        pickle.dump(svc_classes_clfs, open(fun_PATH_REPORTS + "reports_svc_classes_comb_" + filename_word + "_part3.pkl", "wb"))
        pickle.dump(all_classificators, open(fun_PATH_CLASSIFICATORS + "classificators_comb_" + filename_word + "_part3.pkl", "wb"))

    if len(features_sets[0]) == 1: # если длина первого призака = 1, то это sole_sets (одиночные признаки)

        for fset in features_sets:

            feature_df = pd.read_csv(fun_PATH_FEATURES + subseq_len + "/" + fset[0] + ".csv", index_col=0)
            print(fset, 'Collecting data...')
            X_train, X_test, y_train, y_test, indices_train, indices_test = get_X_y(meta_df, feature_df, train_val_test_ids[0], train_val_test_ids[2], y_class_h) # Создаём X и y

            print('Data obtained. Calculating models')
            out = calc_models(models, X_train, y_train, X_test, y_test, indices_test, indices_train, y_class_h, list_y, classes, y_int) # Обучаем модели
            all_clfs[fun_sole_sets_names[features_sets.index(fset)]], \
            svc_classes_clfs[fun_sole_sets_names[features_sets.index(fset)]], \
            all_classificators[fun_sole_sets_names[features_sets.index(fset)]] = out[0], out[1], out[2]

            gc.collect()
            print('gc collect ', len(gc.get_objects()))

        pickle.dump(all_clfs, open(fun_PATH_REPORTS + "reports_sole_" + filename_word + "_RNA_4.pkl", "wb"))
        pickle.dump(svc_classes_clfs, open(fun_PATH_REPORTS + "reports_svc_classes_sole_" + filename_word + "_RNA_4.pkl", "wb"))
        pickle.dump(all_classificators, open(fun_PATH_CLASSIFICATORS + "classificators_sole_" + filename_word + "_RNA_4.pkl", "wb"))

    return

# Features
all_sets = [
    #create_feature_set(['AA'], [list(range(1,3,1))]),
    #create_feature_set(['AA'], [list(range(1,4,1))]),
    #create_feature_set(['DNA'], [list(range(1,3,1))]),
    #create_feature_set(['DNA'], [list(range(1,4,1))]),
    #create_feature_set(['DNA'], [list(range(1,5,1))]),
    #create_feature_set(['DNA'], [list(range(1,6,1))]),
    #create_feature_set(['DNA'], [list(range(1,7,1))]),
    #create_feature_set(['DNA'], [list(range(1,8,1))]),
    #create_feature_set(['AA'], [[1,3]]),
    #create_feature_set(['DNA'], [[1,3,5]]),
    create_feature_set(['DNA'], [[3,5]]),
    create_feature_set(['DNA'], [[2,4,6]]),
    create_feature_set(['DNA'], [[3,5,7]]),
    #create_feature_set(['DNA','AA'], [list(range(1,4,1)),list(range(1,3,1))]),
    #create_feature_set(['DNA','AA'], [list(range(2,5,1)),list(range(1,3,1))])
    ]

all_sets_names = [
                 #'AA1-2',
                 #'AA1-3',
                 #'RNA1-2',
                 #'RNA1-3',
                 #'RNA1-4',
                 #'RNA1-5',
                 #'RNA1-6',
                 #'RNA1-7',
                 #'AA1,3',
                 #'RNA1,3,5',
                 'RNA3,5',
                 'RNA2,4,6',
                 'RNA3,5,7',
                 #'RNA1-3,AA1-2',
                 #'RNA2-4,AA1-2'
                ]

sole_sets = [
             #['DNA_1'],
             #['DNA_2'],
             #['DNA_3'],
             ['DNA_4'],
             #['DNA_5'],
             #['DNA_6'],
             #['DNA_7'],
             #['AA_1'],
             #['AA_2'],
             #['AA_3'],
            ]

sole_sets_names = [
                  #'RNA_1',
                  #'RNA_2',
                  #'RNA_3',
                  'RNA_4',
                  #'RNA_5',
                  #'RNA_6',
                  #'RNA_7',
                  #'AA_1',
                  #'AA_2',
                  #'AA_3',
                  ]


models = ['rf', 'lgbm', 'xgb', 'svc']

training(models, sole_sets, "host", "ranged", genomes_800_400)
#training(models, sole_sets, "host", "full")
