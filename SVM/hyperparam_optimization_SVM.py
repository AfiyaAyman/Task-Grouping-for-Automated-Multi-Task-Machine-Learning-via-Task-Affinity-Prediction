import pandas as pd
import copy
import numpy as np
import math
import sys, os, time
import random
import multiprocessing as mp
import itertools
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, accuracy_score, roc_curve, auc, \
    mean_squared_error

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
    device_idx = sys.argv[2]
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

        '''balance the dataset'''
        # print(df.head())
        if 'Balanced' in ModelName:
            lm = df[df.Labels == 1]
            clutter = df[df.Labels == 0]
            print(len(lm), len(clutter))
            clutter = clutter.sample(n=len(lm) * 4, random_state=None)
            df = pd.concat([lm, lm, clutter, lm, lm])
            # print(len(df))
            df = df.sample(frac=1, random_state=None).reset_index(drop=True)

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
        if 'SVM_NL' in ModelName:
            df = pd.read_csv(csv, nrows=500, low_memory=False)
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
        csv = (f"{DataPath}/{sch_id}_School_Data_for_MTL.csv")
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
        # print(df.columns)
        # exit(0)

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

        Sample_Label_target_1 = np.array(target1)
        Sample_Label_target_2 = np.array(target2)

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

            if ModelName == 'RMTL_SVM_NL':
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

        tmp = (dataset, len(task_group_list), X_list, y_list, X_test_list, y_test_list,hp_dictionary)
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
    # print(f'all_scores = {all_scores}')

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
    if dataset == 'Parkinsons':
        sigma = 1
    else:
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

    def feature_map(x, task_id):
        phi = np.zeros(np.shape(x)[0])# dimension of features
        return np.array([x / math.sqrt(mu)] + [phi if i != task_id else x for i in range(T)])

    # activating the input data for a particular task and everything else is zero

    # kernel function
    def G(x1, x2):
        # Following is RBF kernel with feature_mapping
        return np.array(
            [[np.exp(-(1 / sigma ** 2)* np.linalg.norm(feature_map(x_1[:-1], x_1[-1]) - feature_map(x_2[:-1], x_2[-1]))) for
              x_2 in x2] for x_1 in x1])



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

            # mse_one = mean_squared_error(y_t[:, 0], y_pred_t[:, 0])
            # mse_two = mean_squared_error(y_t[:, 1], y_pred_t[:, 1])
            # MSE = (mse_one + mse_two) / 2
            # print(f'Avg y_t = {np.mean(y_t[:, 0])}, Avg y_pred={np.mean(y_pred_t[:, 0])}, MSE = {MSE}')
            MSE = mean_squared_error(y_t, y_pred_t)
            # print(f'mean_squared_error(y_t, y_pred_t) = {MSE}')
            indi_mse.append(MSE)
            indi_r_square.append(r2_score(y_t, y_pred_t, multioutput='variance_weighted'))
            indi_EV.append(explained_variance_score(y_t, y_pred_t, multioutput='variance_weighted'))

        t += len(y_test_list) // T

    if dataset == 'Landmine':
        return np.sum(indi_logloss), indi_logloss, np.mean(indi_auc), indi_auc
    if dataset == 'Chemical':
        return np.sum(indi_logloss), indi_logloss, np.mean(indi_accu), indi_accu
    if dataset == 'School':
        return np.sum(indi_mse), indi_mse
    if dataset == 'Parkinsons':
        return np.sum(indi_mse), indi_mse, np.mean(indi_EV), indi_EV, np.mean(indi_r_square), indi_r_square

def random_local_search(curr_hp, probability):
    CHANGE_PERC = np.random.uniform(0.0001, 1)
    hp_list = list(curr_hp.keys())
    random_hp = np.random.choice(hp_list, 1, p=probability)[0]
    change = np.random.choice(['+', '-'], 1)


    if random_hp == 'lambda1':
        CHANGE_PERC = np.random.uniform(0.01, 1)
        if change == '+':
            curr_hp['lambda1'] = curr_hp['lambda1'] + (curr_hp['lambda1'] * CHANGE_PERC)#np.random.uniform(0.00001, 10)
        else:
            curr_hp['lambda1'] = curr_hp['lambda1'] - (curr_hp['lambda1'] * CHANGE_PERC) #np.random.uniform(0.000001, curr_hp['lambda1'])
            if curr_hp['lambda1'] < 0:
                # curr_hp['lambda1'] = 0.000001
                curr_hp['lambda1'] = curr_hp['lambda1'] + (curr_hp['lambda1'] * CHANGE_PERC)

    if random_hp == 'lambda2':
        CHANGE_PERC = np.random.uniform(0.01, 1)
        if change == '+':
            curr_hp['lambda2'] = curr_hp['lambda2'] + (
                        curr_hp['lambda2'] * CHANGE_PERC)  # np.random.uniform(0.00001, 10)
        else:
            curr_hp['lambda2'] = curr_hp['lambda2'] - (
                        curr_hp['lambda2'] * CHANGE_PERC)  # np.random.uniform(0.000001, curr_hp['lambda1'])
            if curr_hp['lambda2'] < 0:
                # curr_hp['lambda1'] = 0.000001
                curr_hp['lambda2'] = curr_hp['lambda2'] + (curr_hp['lambda2'] * CHANGE_PERC)

    if random_hp == 'alpha':
        if change == '+':
            curr_hp['alpha'] = curr_hp['alpha'] + (curr_hp['alpha'] * CHANGE_PERC)
        else:
            curr_hp['alpha'] = curr_hp['alpha'] - (curr_hp['alpha'] * CHANGE_PERC)
            if curr_hp['alpha'] < 0:
                curr_hp['sigma'] = random.choice(np.logspace(-6, -1, 20))

    if random_hp == 'sigma':
        if change == '+':
            curr_hp['sigma'] = curr_hp['sigma'] + (curr_hp['sigma'] * CHANGE_PERC)
        else:
            curr_hp['sigma'] = curr_hp['sigma'] - (curr_hp['sigma'] * CHANGE_PERC)
            if curr_hp['sigma'] < 0:
                curr_hp['sigma'] = random.choice(np.logspace(-100, 100, 100))

    if random_hp == 'tol':
        if change == '+':
            curr_hp['tol'] = curr_hp['tol'] + (curr_hp['tol'] * CHANGE_PERC)
        else:
            curr_hp['tol'] = curr_hp['tol'] - (curr_hp['tol'] * CHANGE_PERC)
            if curr_hp['tol'] < 0:
                curr_hp['tol'] = random.choice(np.logspace(-100, 100, 100))

    if random_hp == 'max_iter':
        all_iters = random.sample(range(10, 50000), 1000)
        all_iters.append(-1)
        curr_hp['max_iter'] = random.choice(all_iters)


    return curr_hp



def hyperparameter_tuning(dataset, best_hp, probability_distribution, initial_temperature,max_iter):

    Current_HP = []
    Prev_Loss = []
    Random_Seed = []
    Current_Loss = []
    Iteration = []
    Switch = []
    Temperature = []
    Prev_Iteration = []
    Task_Score = []
    Task_group = []

    Error_Rate = []
    AUC = []
    R_Square = []
    EV = []
    Diff = []
    Acceptance_Rate = []

    temperature = initial_temperature
    random_seed = random.randint(0, 100)
    if dataset == 'Chemical':
        loss, task_score, error, task_error_rate = kFold_validation(task_group, random_seed, best_hp,dataset)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'STL Results = {STL_loss}, Error = {STL_accu / len(task_group)}')
        print(f'loss = {loss}\n'
              f'task_score = {task_score}\n'
              f'error = {error}\n'
              f'task_error_rate = {task_error_rate}')

    if dataset == 'Landmine':
        loss, task_score, mean_auc, task_auc = kFold_validation(task_group, random_seed, best_hp, dataset)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'STL Results = {STL_loss}, AUC = {STL_auc / len(task_group)}')
        print(f'loss = {loss}\n'
              f'task_score = {task_score}\n'
              f'AUC = {mean_auc}\n'
              f'task_AUC = {task_auc}')

    if dataset == 'School':
        loss, task_score = kFold_validation(task_group, random_seed, best_hp, dataset)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'STL Results = {STL_loss}')
        print(f'loss = {loss}\n'
              f'task_score = {task_score}')

    if dataset == 'Parkinsons':
        loss, task_score, mean_EV, task_EV, mean_r_square, task_r_square = kFold_validation(task_group, random_seed, best_hp, dataset)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'STL Results = {STL_loss}, EV = {STL_EV / len(task_group)}')
        print(f'loss = {loss}\n'
              f'EV = {mean_EV}\n'
              f'r_square = {mean_r_square}\n')


    for k,v in task_score.items():
        task_score[k] = np.mean(v)

    prev_solution = loss
    Prev_Loss.append(prev_solution)
    Current_HP.append(best_hp)
    Random_Seed.append(random_seed)
    Current_Loss.append(loss)
    Iteration.append(0)
    Switch.append(0)
    Temperature.append(temperature)
    Prev_Iteration.append(0)
    Diff.append(0)
    Acceptance_Rate.append("None")
    Task_Score.append(task_score.copy())
    Task_group.append(task_group.copy())

    if dataset == 'Chemical':
        Error_Rate.append(error)
    if dataset == 'Landmine':
        AUC.append(mean_auc)
    if dataset == 'Parkinsons':
        R_Square.append(mean_r_square)
        EV.append(mean_EV)

    for iter in range(1,max_iter):
        curr_hp = best_hp.copy()
        curr_hp = random_local_search(curr_hp, probability_distribution)
        random_seed = random.randint(0, 100)

        if dataset == 'Chemical':
            loss, task_score, error, task_error_rate = kFold_validation(task_group, random_seed, curr_hp, dataset)

        if dataset == 'Landmine':
            loss, task_score, mean_auc, task_auc = kFold_validation(task_group, random_seed, curr_hp, dataset)

        if dataset == 'School':
            loss, task_score = kFold_validation(task_group, random_seed, curr_hp, dataset)

        if dataset == 'Parkinsons':
            loss, task_score, mean_EV, task_EV, mean_r_square, task_r_square = kFold_validation(task_group, random_seed, curr_hp, dataset)
            print(f'loss = {loss}\n'
                  f'EV = {mean_EV}\n'
                  f'r_square = {mean_r_square}\n')

        current_solution = loss

        diff = (current_solution - prev_solution) / prev_solution
        # delta = current_solution - prev_solution

        # if iter == 1:
        #     delta_avg = delta

        if current_solution - prev_solution<0:
            Accept_Probability = 1
        else:
            # Accept_Probability = min(1, math.exp(-(current_solution - prev_solution)/(delta_avg*temperature)))
            # Accept_Probability = min(1, math.exp(prev_solution - current_solution / (delta_avg*temperature)))
            if diff<0.05:
                Accept_Probability = 0.01
            else:
                Accept_Probability = 0

        Diff.append(diff)
        Acceptance_Rate.append(Accept_Probability)
        # print(f'delta = {delta}, loss = {current_solution}, prev = {prev_solution} temperature = {temperature}, Accept_Probability = {Accept_Probability}')
        # try:
        #     Accept_Probability = min(1, math.exp(-delta/temperature))
        # except OverflowError:
        #     Accept_Probability = min(1, math.inf)

        PROBABILITIES_SWITCH = [Accept_Probability, max(0, 1 - Accept_Probability)]
        switch_parameter = np.random.choice(['y','n'], 1, p=PROBABILITIES_SWITCH)
        # print(f'\nLoss = {round(loss,3)}, Prev_loss = {round(prev_solution,3)}, '
        #       f'Diff = {diff}, Delta = {delta}, '
        #       f'Temperature = {temperature}, Accept_Probability = {Accept_Probability}, Probability Switch = {switch_parameter}')
        # math.exp(-delta / temperature)
        if switch_parameter == 'y':
            # print('here')
            prev_solution = current_solution
            Switch.append('yes')
            best_hp = curr_hp
            Prev_Iteration.append(iter)

        else:
            Switch.append('no')
            Prev_Iteration.append(Iteration[-1])

        Prev_Loss.append(prev_solution)
        Current_HP.append(curr_hp)
        Random_Seed.append(random_seed)
        Current_Loss.append(loss)
        Iteration.append(iter)
        Task_group.append(task_group.copy())


        # Temperature.append(temperature)

        for k, v in task_score.items():
            task_score[k] = np.mean(v)
        Task_Score.append(task_score.copy())
        # delta_avg = delta_avg + (delta - delta_avg) / len(Current_Loss)

        # temperature = temperature * cooling_rate

        if dataset == 'Chemical':
            Error_Rate.append(error)
        if dataset == 'Landmine':
            AUC.append(mean_auc)
        if dataset == 'Parkinsons':
            R_Square.append(mean_r_square)
            EV.append(mean_EV)



        length = len(Prev_Loss)
        windowsize = 5
        if length%windowsize == 0:
            print(f'Iteration = {len(Iteration)}, Loss = {len(Current_Loss)}, Temperature = {len(Temperature)}, Switch = {len(Switch)}, Prev_Iteration = {len(Prev_Iteration)}'
                  f'Prev_Loss = {len(Prev_Loss)}, Current_HP = {len(Current_HP)}, Random_Seed = {len(Random_Seed)}, Error_Rate = {len(Error_Rate)}, AUC = {len(AUC)}, R_Square = {len(R_Square)}, EV = {len(EV)}')
            res = pd.DataFrame({'Iteration': Iteration,
                                'Switch': Switch,
                                'Current_Loss': Current_Loss,
                                'Prev_Loss': Prev_Loss,
                                'Current_HP': Current_HP,
                                'Random_Seed': Random_Seed,
                                'Prev_Iteration': Prev_Iteration,
                                'Task_Score': Task_Score,
                                'Diff': Diff,
                                'Acceptance_Rate': Acceptance_Rate,
                                'Task_Group': Task_group,
                                # 'Temperature': Temperature
                                })
            if dataset == 'Chemical':
                res['Error_Rate'] = Error_Rate
            if dataset == 'Landmine':
                res['AUC'] = AUC
            if dataset == 'Parkinsons':
                res['R_Square'] = R_Square
                res['EV'] = EV

            res.to_csv(f'HP_Tuning_SVM_{dataset}_{length}_group_{group_no}.csv', index=False)

            if length > windowsize:
                length = length - windowsize
                os.remove(os.path.join(f'HP_Tuning_SVM_{dataset}_{length}_group_{group_no}.csv'))

        # if temperature< 0.000001:
        #     break
    res = pd.DataFrame({'Iteration': Iteration,
                        'Switch': Switch,
                        'Current_Loss': Current_Loss,
                        'Prev_Loss': Prev_Loss,
                        'Current_HP': Current_HP,
                        'Random_Seed': Random_Seed,
                        'Prev_Iteration': Prev_Iteration,
                        'Task_Score': Task_Score,
                        'Diff': Diff,
                        'Acceptance_Rate': Acceptance_Rate,
                        'Task_Group': Task_group,
                        # 'Temperature': Temperature
                        })
    if dataset == 'Chemical':
        res['Error_Rate'] = Error_Rate
    if dataset == 'Landmine':
        res['AUC'] = AUC
    if dataset == 'Parkinsons':
        res['R_Square'] = R_Square
        res['EV'] = EV
    res.to_csv(f'HP_Tuning_SVM_{dataset}_group_{group_no}.csv', index=False)



if __name__ == "__main__":
    dataset = sys.argv[1]
    group_no = sys.argv[2]
    # ModelName = sys.argv[2]

    # dataset = 'Chemical'
    ModelName = 'SVM'

    print(f'dataset = {dataset}, ModelName = {ModelName}')
    if dataset == 'Chemical':
        DataPath = f'../Dataset/{dataset.upper()}/SVM_DATA/'
    else:
        DataPath = f'../Dataset/{dataset.upper()}/DATA/'

    ResultPath = '../Results'
    num_folds = 10
    number_of_epochs = 100
    if dataset == 'Chemical':
        max_iter = 500
        task_number = random.randint(4, 6)
    else:
        max_iter = 800
        task_number = random.randint(5, 10)

    Variance = []
    Random_Seed = []
    StdDev = []
    Accuracy = []
    Error_Rate = []

    task_indi_score = {}
    single_results = pd.read_csv(f'{ResultPath}/STL/STL_{dataset}_SVM.csv')
    ChemicalData = pd.read_csv(f'../Dataset/CHEMICAL/ChemicalData_All.csv', low_memory=False)

    tasks_list_dict = {
        'School': [i for i in range(1, 140)],
        'Landmine': [i for i in range(0, 29)],
        'Chemical': list(ChemicalData['180'].unique()),
        'Parkinsons': [i for i in range(1, 43)]
    }

    TASKS = tasks_list_dict[dataset]
    task_group = random.sample(TASKS, task_number)

    STL_loss = 0
    STL_accu = 0
    STL_auc = 0
    STL_mse = 0
    STL_r_square = 0
    STL_EV = 0
    for task in task_group:
        task_stl = single_results.loc[(single_results['Task'] == task)]
        STL_loss += task_stl['LOSS'].values[0]
        # print(f'task = {task}, loss = {task_stl["LOSS"].values[0]}')
        if dataset == 'Chemical':
            STL_accu += task_stl['Accuracy'].values[0]
        if dataset == 'Landmine':
            STL_auc += task_stl['AUC'].values[0]
        if dataset == 'Parkinsons':
            # STL_r_square += task_stl['R_Square'].values[0]
            STL_EV += task_stl['Explained_Variance'].values[0]

    print(f'dataset = {dataset}, task_group = {task_group}, len(task_group) = {len(task_group)}')

    hyperparameters_RMTL= {
        'lambda1': 0.0001,
        'lambda2': 100,
        'sigma': 2e-03,
        'tol': 0.001,
        'max_iter': -1,
    }

    probability_distribution = [0.3, 0.3, 0.3, 0.05, 0.05]
    print(f'probability_distribution = {probability_distribution}, sum = {np.sum(probability_distribution)}')

    initial_temperature = 10
    cooling_rate = 0.99
    stopping_temperature = 0

    hyperparameter_tuning(dataset, hyperparameters_RMTL, probability_distribution, initial_temperature, max_iter)
