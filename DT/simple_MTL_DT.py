import pandas as pd
import copy
import numpy as np
import math
import sys, os, time
import random
import multiprocessing as mp
import itertools
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, accuracy_score, roc_curve, auc, mean_squared_error
import xgboost as xgb

xgb.set_config(verbosity=0)
import os, shutil
from sklearn.metrics import log_loss,roc_auc_score,average_precision_score,accuracy_score,roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import ast
import tqdm

from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


USE_GPU = False
if USE_GPU:
    device_idx = int(sys.argv[2])
    # device_idx=0
    gpus = tf.config.list_physical_devices('GPU')
    gpu_device = gpus[device_idx]
    core_config = tf.config.experimental.set_visible_devices(gpu_device, 'GPU')
    tf.config.experimental.set_memory_growth(gpu_device, True)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=core_config))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def readData_landmine(task_group_list):
    data_param_dictionary = {}
    for landmine in task_group_list:
        csv = (f"{DataPath}LandmineData_{landmine}_for_MTL.csv")
        df = pd.read_csv(csv, low_memory=False)

        DataSet = np.array(df, dtype=float)
        Number_of_Records = np.shape(DataSet)[0]
        Number_of_Features = np.shape(DataSet)[1]
        # print(f'Number_of_Records = {Number_of_Records}, Number_of_Features = {Number_of_Features}')

        Input_Features = df.columns[:Number_of_Features - 1]
        Sample_Inputs = np.zeros((Number_of_Records, len(Input_Features)))
        for t in range(Number_of_Records):
            Sample_Inputs[t] = DataSet[t, :len(Input_Features)]

        Y = df.pop('Labels')
        Sample_Label = np.array(Y)

        data_param_dictionary.update({f'{dataset}_{landmine}_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'{dataset}_{landmine}_Labels': Sample_Label})

        '''*********************************'''

    return data_param_dictionary


def readData_chemical(molecule_list):
    data_param_dictionary = {}
    for molecule in molecule_list:
        csv = (f"{DataPath}{molecule}_Chemical_Data_for_MTL.csv")
        if 'SVM' in ModelName:
            df = pd.read_csv(csv, nrows=800, low_memory=False)
        else:
            df = pd.read_csv(csv, low_memory=False)
        df.loc[df['181'] < 0, '181'] = 0

        DataSet = np.array(df, dtype=float)
        Number_of_Records = np.shape(DataSet)[0]
        Number_of_Features = np.shape(DataSet)[1]

        Input_Features = df.columns[:Number_of_Features - 1]

        Sample_Inputs = np.zeros((Number_of_Records, len(Input_Features)))
        for t in range(Number_of_Records):
            Sample_Inputs[t] = DataSet[t, :len(Input_Features)]

        Y = df.pop('181')
        Sample_Label = np.array(Y)

        data_param_dictionary.update({f'{dataset}_{molecule}_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'{dataset}_{molecule}_Labels': Sample_Label})

        '''*********************************'''

    return data_param_dictionary


def readData_school(school_list):
    data_param_dictionary = {}
    for sch_id in school_list:
        # csv = (f"{DataPath}/{sch_id}_School_Data.csv")
        csv = (f"{DataPath}{sch_id}_School_Data_for_MTL.csv")
        df = pd.read_csv(csv, low_memory=False)

        df = df[['1985', '1986', '1987',
                 'ESWI', 'African', 'Arab', 'Bangladeshi', 'Caribbean', 'Greek', 'Indian', 'Pakistani', 'SE_Asian',
                 'Turkish', 'Other',
                 'VR_Band', 'Gender',
                 'FSM', 'VR_BAND_Student', 'School_Gender', 'Maintained', 'Church', 'Roman_Cath',
                 'ExamScore',
                 ]]

        DataSet = np.array(df, dtype=float)
        Number_of_Records = np.shape(DataSet)[0]
        Number_of_Features = np.shape(DataSet)[1]

        Input_Features = df.columns[:Number_of_Features - 1]

        Y = df.pop('ExamScore')

        Sample_Inputs = np.zeros((Number_of_Records, len(Input_Features)))
        for t in range(Number_of_Records):
            Sample_Inputs[t] = DataSet[t, :len(Input_Features)]

        Sample_Label = np.array(Y)

        data_param_dictionary.update({f'{dataset}_{sch_id}_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'{dataset}_{sch_id}_Labels': Sample_Label})
        '''*********************************'''
    return data_param_dictionary


def readData_parkinsons(TASKS):
    data_param_dictionary = {}
    for subj_id in TASKS:

        csv = (f"{DataPath}parkinsons_subject_{subj_id}_for_MTL.csv")

        df = pd.read_csv(csv, low_memory=False)

        df = df[[
            # 'age', 'sex', 'test_time',
            'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer',
            'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
            'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE',
            'motor_UPDRS', 'total_UPDRS']]

        target1 = list(df.pop('motor_UPDRS'))
        target2 = list(df.pop('total_UPDRS'))

        # df = (df - df.min()) / (df.max() - df.min())
        Sample_Label = []
        for t1, t2 in zip(target1, target2):
            Sample_Label.append([t1, t2])

        Sample_Label = np.array(Sample_Label)

        Sample_Inputs = np.array(df)

        data_param_dictionary.update({f'{dataset}_{subj_id}_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'{dataset}_{subj_id}_Labels': Sample_Label})
        # data_param_dictionary.update({f'{dataset}_{subj_id}_Label_1': Sample_Label_target_1})
        # data_param_dictionary.update({f'{dataset}_{subj_id}_Label_2': Sample_Label_target_2})
        '''*********************************'''
    return data_param_dictionary

def readData_parkinsons_different(TASKS):
    data_param_dictionary = {}
    LIST = []
    for subj_id in TASKS:
        DataPath = f'../Dataset/{dataset.upper()}/'
        csv = (f"{DataPath}parkinsons_subject_{subj_id}.csv")
        df = pd.read_csv(csv, low_memory=False)

        LIST.append(df)
        sep_df = df[[
            # 'age', 'sex', 'test_time',
            'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer',
            'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
            'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE',
            'motor_UPDRS', 'total_UPDRS']]
        target1 = list(sep_df.pop('motor_UPDRS'))
        target2 = list(sep_df.pop('total_UPDRS'))

        Sample_Label = []
        for t1, t2 in zip(target1, target2):
            Sample_Label.append([t1, t2])

        Sample_Label = np.array(Sample_Label)
        Sample_Inputs = np.array(sep_df)

        data_param_dictionary.update({f'{dataset}_{subj_id}_Inputs': Sample_Inputs})
        data_param_dictionary.update({f'{dataset}_{subj_id}_Labels': Sample_Label})

    df = pd.concat(LIST)
    df = df[[
        # 'age', 'sex', 'test_time',
        'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer',
        'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11',
        'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE',
        'motor_UPDRS', 'total_UPDRS']]

    target1 = list(df.pop('motor_UPDRS'))
    target2 = list(df.pop('total_UPDRS'))

    Sample_Label = []
    for t1, t2 in zip(target1, target2):
        Sample_Label.append([t1, t2])

    Sample_Label = np.array(Sample_Label)
    Sample_Inputs = np.array(df)

    data_param_dictionary.update({f'{dataset}_Inputs': Sample_Inputs})
    data_param_dictionary.update({f'{dataset}_Labels': Sample_Label})
    '''*********************************'''
    return Sample_Inputs,Sample_Label,data_param_dictionary
def convert_to_LibSVM_format(task_group_list,fold,data_param_dict_for_specific_task, dataset, data_folder):


    with open(f'{data_folder}{dataset}_train_data_fold_{fold}.data', 'w') as f:
        for task_id in task_group_list:
            X_train = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_train']
            X_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_test']
            y_train = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_train']
            y_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_test']

            # print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')

            #save train data
            for i in range(X_train.shape[0]):
                non_zero_indices = np.nonzero(X_train[i, :])[0]
                # print(f"{y_train[i]}")
                f.write(f"{y_train[i]} ")
                f.write(f"{0}:{int(task_id)} ")
                for j in non_zero_indices:
                    f.write(f"{j + 1}:{X_train[i, j]} ")
                f.write("\n")

            # save test and val data
            with open(f'{data_folder}{dataset}_val_data_{task_id}_fold_{fold}.data', 'w') as valf:
                for i in range(X_test.shape[0]):
                    non_zero_indices = np.nonzero(X_test[i, :])[0]
                    valf.write(f"{y_test[i]} ")
                    valf.write(f"{0}:{int(task_id)} ")
                    for j in non_zero_indices:
                        valf.write(f"{j + 1}:{X_test[i, j]} ")
                    valf.write("\n")

    with open(f'{data_folder}{dataset}_test_data_fold_{fold}.data', 'w') as f:
        for task_id in task_group_list:
            X_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_test']
            y_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_test']

            #save train data
            for i in range(X_test.shape[0]):
                non_zero_indices = np.nonzero(X_test[i, :])[0]
                f.write(f"{y_test[i]} ")
                f.write(f"{0}:{int(task_id)} ")
                for j in non_zero_indices:
                    f.write(f"{j + 1}:{X_test[i, j]} ")
                f.write("\n")

def kFold_validation(task_group_list, random_seed, task_type, dataset):
    data_param_dict_for_specific_task = {}

    args = []

    if dataset == 'Parkinsons':
        Sample_Inputs,Sample_Label,data_param_dictionary = readData_parkinsons_different(task_group_list)

        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

        for t in range(len(task_group_list)):
            task_id = task_group_list[t]
            X = data_param_dictionary[f'{dataset}_{task_id}_Inputs']
            Y = data_param_dictionary[f'{dataset}_{task_id}_Labels']
            # print(f'Sample_Inputs.shape: {Sample_Inputs.shape}, Sample_Label.shape: {Sample_Label.shape}')
            kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

            fold = 0
            for train_ix, test_ix in kfold.split(X, Y):
                X_train, X_test = X[train_ix], X[test_ix]
                y_train, y_test = Y[train_ix], Y[test_ix]
                data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_X_train': X_train})
                data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_X_test': X_test})
                data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_y_train': y_train})
                data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_y_test': y_test})

                fold += 1

        fold = 0
        args_park = []
        for train, test in kfold.split(Sample_Inputs):
            X_train = Sample_Inputs[train]
            X_test = Sample_Inputs[test]
            y_train = Sample_Label[train]
            y_test = Sample_Label[test]
            tmp = (X_train, X_test, y_train, y_test, data_param_dict_for_specific_task, task_group_list, fold)

            args_park.append(tmp)

            fold += 1
    else:
        if dataset == 'Landmine':
            data_param_dictionary = readData_landmine(task_group_list)

        if dataset == 'Chemical':
            data_param_dictionary = readData_chemical(task_group_list)

        if dataset == 'School':
            data_param_dictionary = readData_school(task_group_list)

        for t in range(len(task_group_list)):
            task_id = task_group_list[t]
            Sample_Inputs = data_param_dictionary[f'{dataset}_{task_id}_Inputs']
            Sample_Label = data_param_dictionary[f'{dataset}_{task_id}_Labels']

            if dataset == 'Landmine':
                kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
            else:
                kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

            fold = 0
            for train_ix, test_ix in kfold.split(Sample_Inputs, Sample_Label):
                X_train, X_test = Sample_Inputs[train_ix], Sample_Inputs[test_ix]
                y_train, y_test = Sample_Label[train_ix], Sample_Label[test_ix]


                data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_X_train': X_train})
                data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_X_test': X_test})
                data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_y_train': y_train})
                data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_y_test': y_test})

                fold += 1

        data_param = {}
        args = []

        data_folder = f'data/{dataset}/'

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        for fold in range(0, 10):
            convert_to_LibSVM_format(task_group_list, fold, data_param_dict_for_specific_task,dataset, data_folder)

        args = [(task_group_list,fold, dataset, data_folder) for fold in range(0,10)]
    number_of_pools = num_folds
    time_start = time.time()
    pool = mp.Pool(number_of_pools)
    if dataset != 'Parkinsons':
        if task_type == 'classification':
            all_scores = pool.starmap(MTL_XGBoost, args)
            pool.close()

        else:
            all_scores = pool.starmap(MTL_XGBoost_regression, args)
            pool.close()

        folder = data_folder
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    if dataset == 'Parkinsons':
        all_scores = pool.starmap(xgBoost_Park, args_park)
        pool.close()

    # print(f'Time taken for {number_of_pools} pools = {time.time() - time_start}')

    scores = []
    auc_scores = []
    error_rates = []
    explained_variance = []
    r2_scores = []
    final_param = []
    for i in range(len(all_scores)):
        scores.append(np.sum(all_scores[i][0]))
        if dataset == 'Landmine':
            auc_scores.append(all_scores[i][2])
        if dataset == 'Chemical':
            error_rates.append(1 - all_scores[i][2])
        if dataset == 'Parkinsons':
            explained_variance.append(all_scores[i][2])
            r2_scores.append(all_scores[i][4])

    score_param_per_task_group_per_fold = {}
    Auc_per_task_per_fold = {}
    ErrorRate_per_task_per_fold = {}

    ExplainedVariance_per_task_per_fold = {}
    R2_per_task_per_fold = {}
    for t in range(len(task_group_list)):
        score_param_per_task_group_per_fold.update({f'task_id_{task_group_list[t]}': []})
        if dataset == 'Landmine':
            Auc_per_task_per_fold.update({f'task_id_{task_group_list[t]}': []})
        if dataset == 'Chemical':
            ErrorRate_per_task_per_fold.update({f'task_id_{task_group_list[t]}': []})
        if dataset == 'Parkinsons':
            ExplainedVariance_per_task_per_fold.update({f'task_id_{task_group_list[t]}': []})
            R2_per_task_per_fold.update({f'task_id_{task_group_list[t]}': []})

    for c in range(len(all_scores)):
        for t in range(len(task_group_list)):
            score_param_per_task_group_per_fold[f'task_id_{task_group_list[t]}'].append(all_scores[c][1][t])
            if dataset == 'Landmine':
                Auc_per_task_per_fold[f'task_id_{task_group_list[t]}'].append(all_scores[c][3][t])
            if dataset == 'Chemical':
                ErrorRate_per_task_per_fold[f'task_id_{task_group_list[t]}'].append(1 - all_scores[c][3][t])
            if dataset == 'Parkinsons':
                ExplainedVariance_per_task_per_fold[f'task_id_{task_group_list[t]}'].append(all_scores[c][3][t])
                R2_per_task_per_fold[f'task_id_{task_group_list[t]}'].append(all_scores[c][5][t])

        total_loss_per_task_group_per_fold = 0
        for t, loss_val in score_param_per_task_group_per_fold.items():
            total_loss_per_task_group_per_fold += np.mean(loss_val)

    if dataset == 'Chemical':
        return total_loss_per_task_group_per_fold, score_param_per_task_group_per_fold, np.mean(
            error_rates), ErrorRate_per_task_per_fold
    if dataset == 'Landmine':
        return total_loss_per_task_group_per_fold, score_param_per_task_group_per_fold, np.mean(
            auc_scores), Auc_per_task_per_fold
    if dataset == 'School':
        return total_loss_per_task_group_per_fold, score_param_per_task_group_per_fold
    if dataset == 'Parkinsons':
        return total_loss_per_task_group_per_fold, score_param_per_task_group_per_fold, np.mean(
            explained_variance), ExplainedVariance_per_task_per_fold, np.mean(r2_scores), R2_per_task_per_fold

    # return scores, (auc1, auc2), trainable_count


def MTL_XGBoost(task_list,fold, dataset, data_folder):

    param = {
        'silent': 1,
        "early_stopping_rounds": 50,
        "learning_rate": 0.2,

        "min_child_weight": 1,
        "n_estimators": 600,
        "subsample": 1,
        "reg_lambda": 12,
        "reg_alpha": 0.0005,
        "objective": 'binary:logistic',
        "max_depth": 9,
        "gamma": 0.45,

        "nthread": 8,
        'eval_metric': 'logloss',

        'tree_method': 'exact',
        'debug': 0,
        # 'use_task_gain_self': 0,
        # 'when_task_split': 1,
        # 'how_task_split': 0,
        # 'min_task_gain': 0.0,
        # 'task_gain_margin': 0.0,
        # 'max_neg_sample_ratio': 0.5,
        # 'which_task_value': 2,
        # 'baseline_alpha': 1.0,
        # 'baseline_lambda': 1.0,
        'tasks_list_': tuple(task_list),
        # 'task_num_for_init_vec': 5,
        # 'task_num_for_OLF': 4,
    }

    # load data
    traindata = xgb.DMatrix(data_folder + f'{dataset}_train_data_fold_{fold}.data')
    testdata = xgb.DMatrix(data_folder + f'{dataset}_test_data_fold_{fold}.data')
    validationdata = xgb.DMatrix(data_folder + f'{dataset}_test_data_fold_{fold}.data')

    if dataset == 'School':
        vals = [None] * 240
    else:
        vals = [None] * 100
    for task in param['tasks_list_']:
        # print(f'task {task}')
        vals[task] = xgb.DMatrix(data_folder + f'{dataset}_val_data_{task}_fold_{fold}.data')

    # train
    evallist = [(traindata, 'train'), (validationdata, 'eval')]
    bst = xgb.train(param,traindata, param['n_estimators'], early_stopping_rounds=param['early_stopping_rounds'], evals=evallist)

    y_real = testdata.get_label()
    y_score = bst.predict(testdata, ntree_limit=bst.best_ntree_limit)

    # print(f"Fold {fold} Accuracy: {accuracy_score(y_real, y_score.round())}")

    # save model
    # with open('mt-gbdt.model', 'wb') as model:
    #     pickle.dump(bst, model)
    # load model
    # with open('mt-gbdt.model', 'rb') as model:
    #     bst = pickle.load(model)

    # compute ROC
    fpr, tpr, thresholds = roc_curve(y_real, y_score, pos_label=1)
    # all_roc_auc = auc(fpr, tpr)
    # all_logloss = log_loss(y_real, y_score)


    log_loss_val = []
    auc_val = []
    accuracy_val = []
    indi_auc = []
    for task in param['tasks_list_']:
        best_auc = 0.5
        best_logloss = 0
        best_acc = 0.5
        y_real = vals[task].get_label()
        tree_num = 0
        for tree in range(2, bst.best_ntree_limit):
            y_score = bst.predict(vals[task], ntree_limit=tree)
            fpr, tpr, thresholds = roc_curve(y_real, y_score, pos_label=1)
            roc_auc = auc(fpr, tpr)
            logloss = log_loss(y_real, y_score)
            acc = accuracy_score(y_real, y_score.round())

            if roc_auc > best_auc:
                best_auc = roc_auc
                best_logloss = logloss
                tree_num = tree
                best_acc = acc
        # acc = accuracy_score(y_real, y_score)

        print("task {} 's AUC={} logloss={} at {} tree".format(task, best_auc, best_logloss, tree_num))

        log_loss_val.append(best_logloss)
        auc_val.append(best_auc)
        accuracy_val.append(best_acc)
        indi_auc.append(best_auc)



    if dataset=='Landmine':
        return np.sum(log_loss_val), log_loss_val, np.mean(indi_auc),  indi_auc
    if dataset == 'Chemical':
        return np.sum(log_loss_val), log_loss_val, np.mean(accuracy_val), accuracy_val

def MTL_XGBoost_regression(task_list,fold, dataset, data_folder):
    param = {
        'silent': 1,
        "early_stopping_rounds": 50,
        "learning_rate": 0.2,

        "min_child_weight": 1,
        "n_estimators": 120,
        "subsample": 1,
        "reg_lambda": 12,
        "reg_alpha": 0.0005,
        "objective": 'reg:squarederror',
        "max_depth": 9,
        "gamma": 0.45,

        "nthread": 8,
        # 'eval_metric': 'auc',

        'tree_method': 'exact',
        'debug': 0,
        # 'use_task_gain_self': 0,
        # 'when_task_split': 1,
        # 'how_task_split': 0,
        # 'min_task_gain': 0.0,
        # 'task_gain_margin': 0.0,
        # 'max_neg_sample_ratio': 0.5,
        # 'which_task_value': 2,
        # 'baseline_alpha': 1.0,
        # 'baseline_lambda': 1.0,
        'tasks_list_': tuple(task_list),
        # 'task_num_for_init_vec': 5,
        # 'task_num_for_OLF': 4,
    }

    # load data
    traindata = xgb.DMatrix(data_folder + f'{dataset}_train_data_fold_{fold}.data')
    testdata = xgb.DMatrix(data_folder + f'{dataset}_test_data_fold_{fold}.data')
    validationdata = xgb.DMatrix(data_folder + f'{dataset}_test_data_fold_{fold}.data')


    if dataset == 'School':
        vals = [None] * 240
    else:
        vals = [None] * 50
    for task in param['tasks_list_']:
        # print(f'task {task}')
        vals[task] = xgb.DMatrix(data_folder + f'{dataset}_val_data_{task}_fold_{fold}.data')

    # train
    evallist = [(traindata, 'train'), (validationdata, 'eval')]
    bst = xgb.train(param, traindata, param['n_estimators'], early_stopping_rounds=param['early_stopping_rounds'],
                    evals=evallist)

    y_real = testdata.get_label()
    y_score = bst.predict(testdata, ntree_limit=bst.best_ntree_limit)

    # print(f"Fold {fold} MSE: {mean_squared_error(y_real, y_score)}")

    indi_mse = []
    ev_score = []
    r_square_score = []
    for task in param['tasks_list_']:
        y_real = vals[task].get_label()
        tree_num = 0
        best_mse = math.inf
        best_ev = -math.inf
        best_r_square = -math.inf
        for tree in range(2, bst.best_ntree_limit):
            y_score = bst.predict(vals[task], ntree_limit=tree)
            mse = mean_squared_error(y_real, y_score)

            if mse < best_mse:
                best_mse = mse
                tree_num = tree
        # acc = accuracy_score(y_real, y_score)

        # print(f'task {task} MSE: {best_mse} at {tree_num} trees')
        # fout.write("{},{},{}\n".format(task, best_auc, best_logloss))
        indi_mse.append(best_mse)
        ev_score.append(best_ev)
        r_square_score.append(best_r_square)
        # auc_val.append(best_auc)
        # accuracy_val.append(best_acc)

    # fout.write("all,{},{},\n".format(all_roc_auc, all_logloss))
    # fout.close()

    return np.sum(indi_mse), indi_mse


def MTL_XGBoost_regression_Park(task_group_list, data_param_dict_for_specific_task, fold, dataset):

    obj = ['reg:squarederror' for i in range(len(task_group_list))]
    eval_metric = ['rmse' for i in range(len(task_group_list))]
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for t in range(len(task_group_list)):
        task_id = task_group_list[t]
        X_train = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_train']
        y_train = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_train']
        X_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_test']
        y_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_test']

        if t == 0:
            train_data = X_train
            train_label = y_train
            test_data = X_test
            test_label = y_test
        else:
            train_data = np.concatenate((train_data, X_train), axis=0)
            train_label = np.concatenate((train_label, y_train), axis=0)
            test_data = np.concatenate((test_data, X_test), axis=0)
            test_label = np.concatenate((test_label, y_test), axis=0)



    # train_data = [data_param_dict_for_specific_task[i]['train_data'] for i in task_group_list]

    # Define the model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        num_round=150
    )
    # print(f'train_data.shape = {train_data.shape}')
    # print(f'train_label.shape = {train_label.shape}')


    # Train the model
    model.fit(
        train_data,
        train_label,
        eval_set=[(test_data, test_label)], verbose=True,
    )


    # Evaluate the model
    eval_results = model.evals_result()
    # print(f'eval_results = {eval_results}')

    # Make predictions
    predictions = model.predict(test_data)
    # print(f'shape of predictions = {predictions.shape}')

    mse = mean_squared_error(test_label, predictions)
    explained_variance = explained_variance_score(test_label, predictions)
    r_square = r2_score(test_label, predictions)
    print(f'fold {fold} MSE = {mse}, explained_variance = {explained_variance}, r_square = {r_square}')

    # return mse, explained_variance, r_square
    indi_mse = []
    ev_score = []
    r_square_score = []
    for task_id in task_group_list:
        X_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_test']
        y_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_test']

        tree_num = 0
        best_mse = math.inf
        best_ev = -math.inf
        best_r_square = -math.inf
        for tree in range(2,15):
            y_score = model.predict(X_test, ntree_limit=tree)
            mse = mean_squared_error(y_test, y_score)

            rsq = r2_score(y_test, y_score)
            evs = explained_variance_score(y_test, y_score)
            if rsq > best_r_square:
                best_r_square = rsq
                best_ev = evs
                best_mse = mse
                tree_num = tree


        print(f'task {task_id} MSE: {best_mse} at {tree_num} trees')
        # fout.write("{},{},{}\n".format(task, best_auc, best_logloss))
        indi_mse.append(best_mse)
        ev_score.append(best_ev)
        r_square_score.append(best_r_square)
        # auc_val.append(best_auc)
        # accuracy_val.append(best_acc)

    # fout.write("all,{},{},\n".format(all_roc_auc, all_logloss))
    # fout.close()

    return np.sum(indi_mse), indi_mse, best_ev, ev_score, best_r_square, r_square_score

def xgBoost_Park(X_train, X_test, y_train, y_test,data_param_dict_for_specific_task,task_group_list,fold):
    dataset = 'Parkinsons'
    reg = xgb.XGBRegressor(tree_method='auto', n_estimators=400)
    reg.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    y_predt = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_predt)
    explained_var = explained_variance_score(y_test, y_predt, multioutput='uniform_average')
    R_Square = r2_score(y_test, y_predt, multioutput='uniform_average')
    # print(f'ALL FOLD {fold} XGBoost MSE = {mse}, explained_var = {explained_var}, R_Square = {R_Square}')
    indi_mse = []
    ev_score = []
    r_square_score = []
    for task_id in task_group_list:
        X_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_test']
        y_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_test']

        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred, multioutput='uniform_average')
        R_Square = r2_score(y_test, y_pred, multioutput='uniform_average')
        indi_mse.append(mse)
        ev_score.append(explained_var)
        r_square_score.append(R_Square)

    # print(f'Sep FOLD = {fold} XGBoost indi MSE = {indi_mse}, explained_var = {ev_score}, R_Square = {r_square_score}')

    return np.sum(indi_mse), indi_mse, np.mean(ev_score), ev_score, np.mean(r_square_score), r_square_score

if __name__ == "__main__":
    dataset = sys.argv[1]
    # dataset = 'Parkinsons'
    ModelName = 'xgBoost'

    print(f'dataset = {dataset}, ModelName = {ModelName}')

    DataPath = f'../Dataset/{dataset.upper()}/DATA/'

    ResultPath = '../Results/partition_sample/'

    num_folds = 10
    number_of_epochs = 100
    max_iter = 3000

    Variance = []
    Random_Seed = []
    StdDev = []
    Accuracy = []
    Error_Rate = []

    task_indi_score = {}
    Single_res_dict = {}
    STL_error = {}
    STL_AUC = {}
    STL_EV = {}
    STL_r_square = {}


    ChemicalData = pd.read_csv(f'../Dataset/CHEMICAL/ChemicalData_All.csv', low_memory=False)

    tasks_list_dict = {
        'School': [i for i in range(1, 140)],
        'Landmine': [i for i in range(0, 29)],
        'Chemical': list(ChemicalData['180'].unique()),
        'Parkinsons': [i for i in range(1, 43)]
    }
    group_range_dict = {
        'School': [10, 30],
        'Landmine': [3, 7],
        'Chemical': [3, 7],
        'Parkinsons': [4, 10]
    }

    if dataset == 'School' or dataset == 'Parkinsons':
        task_type = 'regression'
    else:
        task_type = 'classification'

    TASKS = tasks_list_dict[dataset]

    single_results = pd.read_csv(f'../Results/STL/STL_{dataset}_{ModelName}.csv')
    pair_results = pd.read_csv(f'../Results/Pairwise/{ModelName}/{dataset}_Results_from_Pairwise_Training_ALL_{ModelName}.csv')
    print(f'single_results = {single_results.columns}')


    for Selected_Task in TASKS:
        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.LOSS[0]})
        if dataset == 'Chemical':
            STL_error.update({Selected_Task: single_res.Error_Rate[0]})
        if dataset == 'Landmine':
            STL_AUC.update({Selected_Task: single_res.AUC[0]})
        if dataset == 'Parkinsons':
            STL_EV.update({Selected_Task: single_res.Explained_Variance[0]})
            STL_r_square.update({Selected_Task: single_res.R_Square[0]})

    Pairwise_res_dict = {}
    Pairwise_err_dict = {}
    Pairwise_auc_dict = {}
    Pairwise_EV_dict = {}
    Pairwise_r_square_dict = {}
    Task1 = list(pair_results.Task_1)
    Task2 = list(pair_results.Task_2)
    Pairs = [(Task1[i], Task2[i]) for i in range(len(Task1))]
    for p in Pairs:
        task1 = p[0]
        task2 = p[1]
        pair_res = pair_results[(pair_results.Task_1 == task1) & (pair_results.Task_2 == task2)].reset_index()
        Pairwise_res_dict.update({p: pair_res.Total_Loss[0]})
        if dataset == 'Chemical':
            Pairwise_err_dict.update({p: pair_res.Avg_Error[0]})
        if dataset == 'Landmine':
            Pairwise_auc_dict.update({p: pair_res.AUC[0]})
        if dataset == 'Parkinsons':
            Pairwise_EV_dict.update({p: pair_res.Explained_Var[0]})
            Pairwise_r_square_dict.update({p: pair_res.R_Square[0]})

    print(len(Single_res_dict), len(Pairwise_res_dict))


    Task_group = []
    Total_Loss = []
    Individual_Group_Score = []
    Individual_Error_Rate = []
    Individual_AP = []
    Number_of_Groups = []
    Individual_AUC = []
    Individual_EV = []
    Individual_r_square = []

    Prev_Groups = {}

    task_group = {0:TASKS}
    print(f'task_group = {task_group}')

    random_seed = random.randint(0, 100)

    group_score = {}
    group_avg_err = {}
    # group_avg_AP = {}
    group_avg_AUC = {}
    group_r_square = {}
    group_EV = {}
    tot_loss = 0
    group_no = 0


    if dataset == 'Chemical':
        loss, task_score, avg_error, task_error_rate = kFold_validation(task_group[group_no],
                                                                        random_seed, task_type,
                                                                        dataset)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'loss = {loss}\n'
              f'task_score = {task_score}\n'
              f'error = {avg_error}\n'
              f'task_error_rate = {task_error_rate}')

    if dataset == 'Landmine':
        loss, task_score, mean_auc, task_auc = kFold_validation(task_group[group_no], random_seed,
                                                                task_type, dataset)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'loss = {loss}\n'
              f'task_score = {task_score}\n'
              f'AUC = {mean_auc}\n'
              f'task_AUC = {task_auc}')

    if dataset == 'School':
        loss, task_score = kFold_validation(task_group[group_no], random_seed, task_type, dataset)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'loss = {loss}\n'
              f'task_score = {task_score}')

    if dataset == 'Parkinsons':
        loss, task_score, mean_EV, task_EV, mean_r_square, task_r_square = kFold_validation(
            task_group[group_no],
            random_seed,
            task_type,
            dataset)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'loss = {loss}\n'
              f'task_score = {task_score}\n'
              f'EV = {mean_EV}\n'
              f'task_EV = {task_EV}\n'
              f'r_square = {mean_r_square}\n'
              f'task_r_square = {task_r_square}'
              )

    print(f'Simple MTL loss = {loss}')



