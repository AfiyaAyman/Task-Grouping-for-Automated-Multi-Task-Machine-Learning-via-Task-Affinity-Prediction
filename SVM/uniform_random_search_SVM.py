import pandas as pd
import copy
import numpy as np
import math
import sys, os, time
import random
import multiprocessing as mp
import itertools
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, accuracy_score, roc_curve, auc, mean_squared_error

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
from tensorflow.keras.layers import *
# print(f'version = {tf.__version__}')
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model, backend

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
        csv = (f"{DataPath}LandmineData_{landmine}.csv")
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
        csv = (f"{DataPath}{molecule}_Molecule_Data.csv")
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
        csv = (f"{DataPath}{sch_id}_School_Data.csv")
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

        csv = (f"{DataPath}parkinsons_subject_{subj_id}.csv")

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


def kFold_validation(task_group_list, random_seed, hp_dictionary, dataset):
    data_param_dict_for_specific_task = {}

    args = []

    if dataset == 'Landmine':
        data_param_dictionary = readData_landmine(task_group_list)

    if dataset == 'Chemical':
        data_param_dictionary = readData_chemical(task_group_list)

    if dataset == 'School':
        data_param_dictionary = readData_school(task_group_list)

    if dataset == 'Parkinsons':
        data_param_dictionary = readData_parkinsons(task_group_list)

    if dataset == 'Parkinsons':
        max_size = 170
    if dataset == 'School':
        max_size = 200
    if dataset == 'Chemical':
        max_size = 700
    if dataset == 'Landmine':
        max_size = 700
    train_set_size = math.floor(max_size * (1 - num_folds / 100))
    test_set_size = math.ceil(max_size * (num_folds / 100))

    for t in range(len(task_group_list)):
        task_id = task_group_list[t]
        Sample_Inputs = data_param_dictionary[f'{dataset}_{task_id}_Inputs']
        Sample_Label = data_param_dictionary[f'{dataset}_{task_id}_Labels']

        if dataset == 'Landmine' or dataset == 'Chemical':
            kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
        else:
            kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

        fold = 0
        for train_ix, test_ix in kfold.split(Sample_Inputs, Sample_Label):
            X_train, X_test = Sample_Inputs[train_ix], Sample_Inputs[test_ix]
            y_train, y_test = Sample_Label[train_ix], Sample_Label[test_ix]

            samples_to_be_repeated = train_set_size - len(X_train)
            # print(f'train samples_to_be_repeated = {samples_to_be_repeated}')
            if samples_to_be_repeated > 0:
                random_indices = np.random.choice(X_train.shape[0], samples_to_be_repeated)
                X_train = np.concatenate((X_train, X_train[random_indices]), axis=0)
                y_train = np.concatenate((y_train, y_train[random_indices]), axis=0)
            else:
                random_indices = np.random.choice(X_train.shape[0], train_set_size)
                X_train = X_train[random_indices]
                y_train = y_train[random_indices]

            samples_to_be_repeated = test_set_size - len(X_test)
            # print(f'test samples_to_be_repeated = {samples_to_be_repeated}')
            if samples_to_be_repeated > 0:
                random_indices = np.random.choice(X_test.shape[0], samples_to_be_repeated)
                X_test = np.concatenate((X_test, X_test[random_indices]), axis=0)
                y_test = np.concatenate((y_test, y_test[random_indices]), axis=0)

            else:
                random_indices = np.random.choice(X_test.shape[0], test_set_size)
                X_test = X_test[random_indices]
                y_test = y_test[random_indices]

            if ModelName == 'SVM':
                '''Appending task id to the end of the input'''
                X_train = np.c_[X_train, [t for _ in range(X_train.shape[0])]]
                X_test = np.c_[X_test, [t for _ in range(X_test.shape[0])]]

            data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_X_train': X_train})
            data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_X_test': X_test})
            data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_y_train': y_train})
            data_param_dict_for_specific_task.update({f'{dataset}_{task_id}_fold_{fold}_y_test': y_test})

            fold += 1

    data_param = {}
    args = []
    for fold in range(num_folds):
        X_list, y_list = [], []
        X_test_list, y_test_list = [], []
        for t in range(len(task_group_list)):
            task_id = task_group_list[t]
            X_train = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_train']
            y_train = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_train']
            X_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_test']
            y_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_test']

            if ModelName == 'SVM':
                if t == 0:
                    X_list = X_train
                    y_list = y_train
                    X_test_list = X_test
                    y_test_list = y_test
                else:
                    X_list = np.concatenate((X_list, X_train), axis=0)
                    y_list = np.concatenate((y_list, y_train), axis=0)
                    X_test_list = np.concatenate((X_test_list, X_test), axis=0)
                    y_test_list = np.concatenate((y_test_list, y_test), axis=0)

            else:
                X_list.append(X_train)
                y_list.append(y_train)
                X_test_list.append(X_test)
                y_test_list.append(y_test)

        tmp = (dataset, len(task_group_list), X_list, y_list, X_test_list, y_test_list, hp_dictionary)
        args.append(tmp)

        data_param[f'fold_{fold}_X_list'] = X_list
        data_param[f'fold_{fold}_y_list'] = y_list
        data_param[f'fold_{fold}_X_test_list'] = X_test_list
        data_param[f'fold_{fold}_y_test_list'] = y_test_list

    num_tasks = len(task_group_list)

    # print(f'Keys for data_param_dict_for_specific_task: {len(data_param_dict_for_specific_task.keys())}')
    # print(f'Keys for data_param: {len(data_param.keys())}')
    # print(f'len(TASKS) {len(TASKS)} * num_folds {num_folds} * 4 (x,y,xt,yt) = {len(TASKS)*num_folds*4}')
    # print(f'len(args) = {len(args)}')

    number_of_pools = len(args) + 10
    pool = mp.Pool(number_of_pools)

    time_start = time.time()
    all_scores = pool.starmap(regularized_mtl_NonLinearSVM, args)
    pool.close()
    print(f'Time taken for {number_of_pools} pools = {time.time() - time_start}')

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
        # final_param.append(all_scores[i][3])

    # print(f'error_rates  = {error_rates}')

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

    # print(f'total_loss_per_task_group_per_fold = {total_loss_per_task_group_per_fold}')
    # print(f'score_param_per_task_group_per_fold = {score_param_per_task_group_per_fold}')
    # print(f'Auc_per_task_per_fold = {Auc_per_task_per_fold}')
    # print(f'ErrorRate_per_task_per_fold = {ErrorRate_per_task_per_fold}')

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

def regularized_mtl_NonLinearSVM(dataset, T, X_list, y_list, X_test_list, y_test_list, hp_dictionary):
    lambda_1 = hp_dictionary['lambda1']
    lambda_2 = hp_dictionary['lambda2']
    # alpha = hp_dictionary['alpha']
    sigma = hp_dictionary['sigma']
    tol = hp_dictionary['tol']
    max_iter = hp_dictionary['max_iter']

    C = T / (2 * lambda_1)
    mu = T * lambda_2 / lambda_1  # mu

    def G_fast(X1, X2):
        tasks1 = X1[:, -1]
        tasks2 = X2[:, -1]
        features1 = X1[:, :-1]
        features2 = X2[:, :-1]
        features_sqsum1 = np.sum(X1[:, :-1] ** 2, axis=1)
        features_sqsum2 = np.sum(X2[:, :-1] ** 2, axis=1)

        A = np.array([[
            np.exp(-(1 / sigma ** 2) * math.sqrt(
                sum((features1[i1] - features2[i2]) ** 2) * (1 + 1 / mu) if tasks1[i1] == tasks2[i2]
                else sum((features1[i1] - features2[i2]) ** 2) / mu + features_sqsum1[i1] + features_sqsum2[i2]
            ))
            for i1 in np.arange(X1.shape[0])]
            for i2 in np.arange(X2.shape[0])])

        return np.transpose(A)

    if dataset == 'Landmine' or dataset == 'Chemical':
        reg = SVC(kernel=G_fast, C=C, max_iter=max_iter, probability=True, tol=tol)
        reg.fit(X_list, y_list)
        y_pred = reg.predict(X_test_list)
        y_pred_prob = reg.predict_proba(X_test_list)

    if dataset == 'School':
        reg = SVR(kernel=G_fast, C=C, max_iter=max_iter, tol=tol)
        reg.fit(X_list, y_list)
        y_pred = reg.predict(X_test_list)

    if dataset == 'Parkinsons':
        reg = SVR(kernel=G_fast, C=C, max_iter=max_iter, tol=tol)
        reg = MultiOutputRegressor(reg)
        reg.fit(X_list, y_list)
        y_pred = reg.predict(X_test_list)

    indi_accu = []
    indi_logloss = []
    indi_auc = []
    indi_mse = []
    indi_r_square = []
    indi_EV = []

    t = 0
    while t < len(X_test_list):
        y_t = y_test_list[t:len(y_test_list) // T + t]
        y_pred_t = y_pred[t:len(y_pred) // T + t]

        if dataset == 'Landmine' or dataset == 'Chemical':
            y_pred_prob_t = y_pred_prob[t:len(y_pred_prob) // T + t]
            indi_accu.append(accuracy_score(y_t, y_pred_t))
            indi_logloss.append(log_loss(y_t, y_pred_prob_t))
            indi_auc.append(roc_auc_score(y_t, y_pred_t))

        if dataset == 'School':
            indi_mse.append(mean_squared_error(y_t, y_pred_t))

        if dataset == 'Parkinsons':
            mse_one = mean_squared_error(y_t[:, 0], y_pred_t[:, 0])
            mse_two = mean_squared_error(y_t[:, 1], y_pred_t[:, 1])
            MSE = (mse_one + mse_two) / 2
            indi_mse.append(MSE)
            indi_r_square.append(r2_score(y_t, y_pred_t, multioutput='uniform_average'))
            indi_EV.append(explained_variance_score(y_t, y_pred_t, multioutput='uniform_average'))

        t += len(y_test_list) // T

    if dataset == 'Landmine':
        return np.sum(indi_logloss), indi_logloss, np.mean(indi_auc), indi_auc
    if dataset == 'Chemical':
        return np.sum(indi_logloss), indi_logloss, np.mean(indi_accu), indi_accu
    if dataset == 'School':
        return np.sum(indi_mse), indi_mse
    if dataset == 'Parkinsons':
        return np.sum(indi_mse), indi_mse, np.mean(indi_EV), indi_EV, np.mean(indi_r_square), indi_r_square






if __name__ == "__main__":
    dataset = sys.argv[1]
    v = int(sys.argv[2])
    # dataset = 'Parkinsons'
    ModelName = 'SVM'

    print(f'dataset = {dataset}, ModelName = {ModelName}')

    DataPath = f'../Dataset/{dataset.upper()}/'

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

    hyperparameters_RMTL_dict = {
        'School': {'lambda1': 4.881691462997538e-06,
                   'lambda2': 2.1626904185443982e-08,
                   'sigma': 7.627160536925234e-14,
                   'tol': 0.0013373317742556123,
                   'max_iter': 41023},
        'Chemical': {'lambda1': 3.6080687884790464e-06, 'lambda2': 0.010362019533397326,
                     'sigma': 1, 'tol': 9.71296238978123e-05, 'max_iter': -1},
        'Landmine': {'lambda1': 0.20086653637444066,
                     'lambda2': 2.6715547107541463e-06, 'sigma': 0.027596024102822553,
                     'tol': 0.001, 'max_iter': -1},
        'Parkinsons': {'lambda1': 2.1303981245590451e-10,
                       'lambda2': 4.008371743829368, 'sigma': 1,
                       'tol': 0.005779782206924083, 'max_iter': 41517},
    }

    hyperparameters_RMTL = hyperparameters_RMTL_dict[dataset]

    TASKS = tasks_list_dict[dataset]

    single_results = pd.read_csv(f'../Results/STL/STL_{dataset}_{ModelName}.csv')
    # pair_results = pd.read_csv(f'../Results/Pairwise/{ModelName}/{dataset}_Results_from_Pairwise_Training_ALL_{ModelName}.csv')
    pair_results = pd.read_csv(f'../Results/new_runs/SVM/{dataset}_Results_from_Pairwise_Training_ALL_{ModelName}.csv')
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
    # c = 0
    # for k, v in Pairwise_res_dict.items():
    #     print(f'{k} = {v}')
    #     c+=1
    #     if c==3:
    #         break


    URS_dataset = pd.read_csv(f'{ResultPath}{dataset}_OnlyGroups_URS.csv')
    # Uniform_RS(URS_dataset, dataset, hyperparameters_RMTL, TASKS)

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
    URS_Task_Groups = list(URS_dataset['Task_group'])
    mid_point = int(len(URS_Task_Groups)/2)

    for count in range(v,v+50):
        task_group = URS_Task_Groups[count]
        task_group = ast.literal_eval(task_group)
        print(f'task_group = {task_group}')

        random_seed = random.randint(0, 100)

        group_score = {}
        group_avg_err = {}
        # group_avg_AP = {}
        group_avg_AUC = {}
        group_r_square = {}
        group_EV = {}
        tot_loss = 0
        for group_no, task in task_group.items():
            group_score.update({group_no: 0})
            group_avg_err.update({group_no: 0})
            # group_avg_AP.update({group_no: 0})
            group_avg_AUC.update({group_no: 0})
            group_r_square.update({group_no: 0})
            group_EV.update({group_no: 0})
            grouping_sorted = tuple(sorted(task))
            if grouping_sorted not in Prev_Groups.keys():
                if len(task) == 1:
                    loss = Single_res_dict[task[0]]
                    if dataset == 'Chemical':
                        avg_error = STL_error[task[0]]
                    if dataset == 'Landmine':
                        mean_auc = STL_AUC[task[0]]
                    if dataset == 'Parkinsons':
                        mean_EV = STL_r_square[task[0]]
                        mean_r_square = STL_EV[task[0]]

                elif len(task) == 2:
                    task1 = task[0]
                    task2 = task[1]

                    if (task1, task2) in Pairwise_res_dict.keys():
                        loss = Pairwise_res_dict[(task1, task2)]
                        if dataset == 'Chemical':
                            avg_error = Pairwise_err_dict[(task1, task2)]
                        if dataset == 'Landmine':
                            mean_auc = Pairwise_auc_dict[(task1, task2)]
                        if dataset == 'Parkinsons':
                            mean_EV = Pairwise_EV_dict[(task1, task2)]
                            mean_r_square = Pairwise_r_square_dict[(task1, task2)]
                    else:
                        loss = Pairwise_res_dict[(task2, task1)]
                        if dataset == 'Chemical':
                            avg_error = Pairwise_err_dict[(task2, task1)]
                        if dataset == 'Landmine':
                            mean_auc = Pairwise_auc_dict[(task2, task1)]
                        if dataset == 'Parkinsons':
                            mean_EV = Pairwise_EV_dict[(task2, task1)]
                            mean_r_square = Pairwise_r_square_dict[(task2, task1)]

                else:
                    if dataset == 'Chemical':
                        loss, task_score, avg_error, task_error_rate = kFold_validation(task_group[group_no],
                                                                                        random_seed, hyperparameters_RMTL,
                                                                                        dataset)
                        # print(f'\n\n\n******{dataset}*****\n')
                        # print(f'loss = {loss}\n'
                        #       f'task_score = {task_score}\n'
                        #       f'error = {avg_error}\n'
                        #       f'task_error_rate = {task_error_rate}')

                    if dataset == 'Landmine':
                        loss, task_score, mean_auc, task_auc = kFold_validation(task_group[group_no], random_seed,
                                                                                hyperparameters_RMTL, dataset)
                        # print(f'\n\n\n******{dataset}*****\n')
                        # print(f'loss = {loss}\n'
                        #       f'task_score = {task_score}\n'
                        #       f'AUC = {mean_auc}\n'
                        #       f'task_AUC = {task_auc}')

                    if dataset == 'School':
                        loss, task_score = kFold_validation(task_group[group_no], random_seed, hyperparameters_RMTL, dataset)
                        # print(f'\n\n\n******{dataset}*****\n')
                        # print(f'loss = {loss}\n'
                        #       f'task_score = {task_score}')

                    if dataset == 'Parkinsons':
                        loss, task_score, mean_EV, task_EV, mean_r_square, task_r_square = kFold_validation(
                            task_group[group_no],
                            random_seed,
                            hyperparameters_RMTL,
                            dataset)
                        # print(f'\n\n\n******{dataset}*****\n')
                        # print(f'loss = {loss}\n'
                        #       f'task_score = {task_score}\n'
                        #       f'EV = {mean_EV}\n'
                        #       f'task_EV = {task_EV}\n'
                        #       f'r_square = {mean_r_square}\n'
                        #       f'task_r_square = {task_r_square}')

                    for k, val in task_score.items():
                        task_score[k] = np.mean(val)

                tot_loss += loss
                group_score[group_no] = loss

                if dataset == 'School':
                    Prev_Groups[grouping_sorted] = loss
                if dataset == 'Chemical':
                    group_avg_err[group_no] = avg_error
                    Prev_Groups[grouping_sorted] = (loss, avg_error)
                if dataset == 'Landmine':
                    group_avg_AUC[group_no] = mean_auc
                    Prev_Groups[grouping_sorted] = (loss, mean_auc)

                if dataset == 'Parkinsons':
                    group_r_square[group_no] = mean_r_square
                    group_EV[group_no] = mean_EV
                    Prev_Groups[grouping_sorted] = (loss, mean_EV, mean_r_square)

            else:
                if dataset == 'School':
                    loss = Prev_Groups[grouping_sorted]
                    if group_no not in group_score.keys():
                        group_score.update({group_no: loss})
                    else:
                        group_score[group_no] = loss
                if dataset == 'Chemical':
                    loss, avg_error = Prev_Groups[grouping_sorted]
                    if group_no not in group_score.keys():
                        group_score.update({group_no: loss})
                        group_avg_err.update({group_no: avg_error})
                        # group_avg_AP.update({group_no: AP})
                    else:
                        group_score[group_no] = loss
                        group_avg_err[group_no] = avg_error
                        # group_avg_AP[group_no] = AP
                if dataset == 'Landmine':
                    loss, mean_auc = Prev_Groups[grouping_sorted]
                    if group_no not in group_score.keys():
                        group_score.update({group_no: loss})
                    else:
                        group_score[group_no] = loss
                if dataset == 'Parkinsons':
                    loss, mean_EV, mean_r_square = Prev_Groups[grouping_sorted]
                    if group_no not in group_score.keys():
                        group_score.update({group_no: loss})
                        group_r_square.update({group_no: mean_r_square})
                        group_EV.update({group_no: mean_EV})
                    else:
                        group_score[group_no] = loss
                        group_r_square[group_no] = mean_r_square
                        group_EV[group_no] = mean_EV

                tot_loss += loss

        print(f'tot_loss = {tot_loss}')
        print(f'group_score = {group_score}')
        Task_group.append(task_group)
        Number_of_Groups.append(len(task_group))
        Total_Loss.append(tot_loss)
        Individual_Group_Score.append(group_score.copy())
        if dataset == 'Chemical':
            Individual_Error_Rate.append(group_avg_err.copy())
            # Individual_AP.append(group_avg_AP.copy())
        if dataset == 'Landmine':
            Individual_AUC.append(group_avg_AUC.copy())
        if dataset == 'Parkinsons':
            Individual_EV.append(group_EV.copy())
            Individual_r_square.append(group_r_square.copy())
        # print(Individual_Group_Score)

        print(len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))

        tail_pointer = 5
        length = len(Total_Loss)
        if length>0 and length % tail_pointer == 0:
            initial_results = pd.DataFrame({'Total_Loss': Total_Loss,
                                            'Number_of_Groups': Number_of_Groups,
                                            'Task_group': Task_group,
                                            'Individual_Group_Score': Individual_Group_Score,
                                            })
            if dataset == 'Chemical':
                initial_results['Individual_Error_Rate'] = Individual_Error_Rate
            if dataset == 'Landmine':
                initial_results['Individual_AUC'] = Individual_AUC
            if dataset == 'Parkinsons':
                initial_results['Individual_EV'] = Individual_EV
                initial_results['Individual_r_square'] = Individual_r_square

            initial_results.to_csv(f'{ResultPath}/{dataset}_URS_MTL_{ModelName}_{v+length}_v_{v}.csv',
                                   index=False)


            old_file = f'{ResultPath}/{dataset}_URS_MTL_{ModelName}_{v+length-tail_pointer}_v_{v}.csv'
            if os.path.exists(old_file):
                os.remove(os.path.join(old_file))

    initial_results = pd.DataFrame({'Total_Loss': Total_Loss,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score,
                                    })
    if dataset == 'Chemical':
        initial_results['Individual_Error_Rate'] = Individual_Error_Rate
    if dataset == 'Landmine':
        initial_results['Individual_AUC'] = Individual_AUC
    if dataset == 'Parkinsons':
        initial_results['Individual_EV'] = Individual_EV
        initial_results['Individual_r_square'] = Individual_r_square

    initial_results.to_csv(f'{ResultPath}/{dataset}_URS_MTL_{ModelName}_{length}_v_{v}.csv',
                           index=False)

