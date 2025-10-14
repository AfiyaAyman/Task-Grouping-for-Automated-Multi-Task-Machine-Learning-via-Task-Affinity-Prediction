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
from sklearn.svm import SVC
import ast
import tqdm
import xgboost as xgb

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

USE_GPU = False
if USE_GPU:
    device_idx = sys.argv[2]
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


def kFold_validation(task_group_list, random_seed, hp_dictionary):
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
    if ModelName == 'xgBoost':
        data_folder = f'data/{dataset}/'

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        for fold in range(0, 10):
            convert_to_LibSVM_format(task_group_list, fold, data_param_dict_for_specific_task, dataset, data_folder)

        args = [(task_group_list, fold, dataset, data_folder) for fold in range(0, 10)]
    else:
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

                if ModelName == 'RMTL_SVM_NL':
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

    number_of_pools = len(args)

    time_start = time.time()
    if 'SVM_NL' in ModelName:
        pool = mp.Pool(number_of_pools)
        all_scores = pool.starmap(regularized_mtl_NonLinearSVM, args)
        pool.close()

    if 'xgBoost' in ModelName:
        pool = mp.Pool(number_of_pools)
        if dataset == 'Landmine' or dataset == 'Chemical':
            all_scores = pool.starmap(MTL_XGBoost, args)
        else:
            all_scores = pool.starmap(MTL_XGBoost_regression, args)
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


def SVM_MTL(task_group_list, fold, data_param_dict_for_specific_task):
    hp = [1.55587223, 0.20295698]
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, auc

    # Sample input data and target variables

    # Initialize the pipelines for each task
    mtl_svc = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='linear'))])
    # Train the regressor
    data_param = copy.deepcopy(data_param_dict_for_specific_task)
    MSE_all_tasks = []
    auc_all_tasks = []
    for subj_id in task_group_list:
        X_train = data_param[f'Landmine_{subj_id}_fold_{fold}_X_train']
        y_train = data_param[f'Landmine_{subj_id}_fold_{fold}_y_train']

        mtl_svc.fit(X_train, y_train)
        print('***************')
        # for subj_id in task_group_list:
        X_test = data_param[f'Landmine_{subj_id}_fold_{fold}_X_test']
        y_test = data_param[f'Landmine_{subj_id}_fold_{fold}_y_test']

        y_pred = mtl_svc.predict(X_test)

        # Evaluate the regressor
        # print(mean_squared_error(y_test, y_pred))
        MSE = log_loss(y_test, y_pred)

        MSE_all_tasks.append(MSE)
        # print(f'fold = {fold}, subj = {subj_id}, MSE = {MSE}')
        auc_all_tasks.append(auc(y_test, y_pred))

    MSE = np.mean(MSE_all_tasks)
    auc = np.mean(auc_all_tasks)
    # print(f'fold = {fold}, MSE = {MSE}, {len(MSE_all_tasks)}, {MSE_all_tasks}')

    return MSE, auc, MSE_all_tasks, auc_all_tasks


def mtl_SVM(task_group_list, data_param_dict_for_specific_task, fold, dataset):
    data_param = copy.deepcopy(data_param_dict_for_specific_task)

    for i in range(len(task_group_list)):
        task = task_group_list[i]
        X_train = data_param[f'{dataset}_{task}_fold_{fold}_X_train']
        y_train = data_param[f'{dataset}_{task}_fold_{fold}_y_train']
        X_test = data_param[f'{dataset}_{task}_fold_{fold}_X_test']
        y_test = data_param[f'{dataset}_{task}_fold_{fold}_y_test']

        if i == 0:
            X_train_all_tasks = X_train
            y_train_all_tasks = y_train
            X_test_all_tasks = X_test
            y_test_all_tasks = y_test
        else:
            X_train_all_tasks = np.concatenate((X_train_all_tasks, X_train), axis=0)
            y_train_all_tasks = np.concatenate((y_train_all_tasks, y_train), axis=0)
            X_test_all_tasks = np.concatenate((X_test_all_tasks, X_test), axis=0)
            y_test_all_tasks = np.concatenate((y_test_all_tasks, y_test), axis=0)

        # print(f'X_train_all_tasks.shape = {X_train_all_tasks.shape}, y_train_all_tasks.shape = {y_train_all_tasks.shape}, X_test_all_tasks.shape = {X_test_all_tasks.shape}, y_test_all_tasks.shape = {y_test_all_tasks.shape}')

    # Combine the datasets for different tasks into a single array
    X_train = X_train_all_tasks
    y_train = y_train_all_tasks

    # print(f'X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}')

    # Define the parameter grid to search over
    '''
    timeStart = time.time()
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001]}

    # Initialize the SVC model
    svc = SVC(kernel='rbf')
    # Initialize the GridSearchCV object
    grid = GridSearchCV(svc, param_grid, cv=10)
    # Fit the GridSearchCV object to the data
    grid.fit(X_train, y_train)

    # Print the best parameters and the best score
    # print("Best parameters: ", grid.best_params_)
    # print("Best score: ", grid.best_score_)
    # print(f'fold = {fold}, time = {time.time() - timeStart}')
    # {'C': 10, 'gamma': 0.1}
    final_param = [grid.best_params_['C'], grid.best_params_['gamma']]
    '''
    final_param = [10, 0.1]

    # Train the SVM model

    clf = make_pipeline(
        SVC(kernel='rbf', C=final_param[0], gamma=final_param[1], max_iter=-1, probability=True, tol=0.01))
    # clf = SVC(,kernel='rbf', C=final_param[0], gamma=final_param[1],max_iter=500)
    clf.fit(X_train, y_train)

    LOSS = []
    AUC_SCORES = []
    ERROR_RATES = []
    for task in task_group_list:
        X_test = data_param[f'{dataset}_{task}_fold_{fold}_X_test']
        y_test = data_param[f'{dataset}_{task}_fold_{fold}_y_test']

        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)
        print(f'fold = {fold}, task = {task}, Pos y_pred = {list(y_pred).count(1)}, y_test = {list(y_test).count(1)}')
        # print(f'fold = {fold}, task = {task}\n y_pred = {y_pred}\n y_test = {y_test}')

        # compute ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        # print(f'fpr = {fpr}, tpr = {tpr}, thresholds = {thresholds}')
        # exit(0)
        all_roc_auc = auc(fpr, tpr)
        all_logloss = log_loss(y_test, y_pred_prob)
        all_error_rate = 1 - accuracy_score(y_test, y_pred)
        LOSS.append(all_logloss)
        AUC_SCORES.append(all_roc_auc)
        ERROR_RATES.append(all_error_rate)

    return LOSS, AUC_SCORES, ERROR_RATES, final_param


def regularized_mtl_LinearSVM(X_list, Y_list, X_test_list, Y_test_list):
    """
    Parameters:
    X_list (list of numpy arrays): list of input feature matrices for each task, where each input feature matrix has dimensions (n_i, d_i)
    Y_list (list of numpy arrays): list of output vectors for each task, where each output vector has dimensions (n_i,)
    lambdas (list of floats): list of regularization parameters for each task
    lambda2 (float): regularization parameter for shared parameters

    Returns:
    W (numpy array): weight matrix for shared parameters, with dimensions (d,)
    V_list (list of numpy arrays): list of weight matrices for task-specific parameters, where each weight matrix has dimensions (d_i,)
    """

    # define the number of tasks and number of features
    # print(f'X_list = {len(X_list)}')
    num_tasks = len(X_list)
    num_features = X_list[0].shape[1]
    # print(f'num_features = {num_features}, num_tasks = {num_tasks}')

    best_loss = math.inf
    accu_best = -math.inf
    old_loss = math.inf
    best_params = []

    # lambda_1 = [0.001, 0.01, 0.1, 10, 100]
    # lambda_2 = [0.001, 0.01, 0.1, 1, 10]

    lambda_1 = [0.001]
    lambda_2 = [0.001]

    for lam_1 in tqdm.tqdm(lambda_1):  # regularization parameter for each task
        for lam_2 in lambda_2:  # regularization param for shared params

            # Initialize weight matrices
            W = np.zeros(num_features)
            V_list = [np.zeros(num_features) for i in range(num_tasks)]
            b_list = []

            # lambdas = [lam_1 for t in range(len(X_list))]
            # Loop over tasks
            for i in range(len(X_list)):
                # Fit task-specific classifier
                X, Y, V_i, lam_i = X_list[i], Y_list[i], V_list[i], lam_1  # using same lambda for all tasks

                C = num_tasks / (2 * lam_i)
                # print(f'C = {C}, lam_2 = {lam_2}, lam_1 = {lam_1}')
                # clf = LinearSVC(C = C, max_iter=1000, probability=True)
                clf = SVC(kernel='linear', C=C, probability=True)
                clf.fit(X, Y)
                V_i = clf.coef_.ravel()  # get the weight vector
                b_i = clf.intercept_  # get bias

                # Update task-specific weight matrix and bias
                V_list[i] = V_i
                b_list.append(b_i[0])

            # print(f'Updated Task-Specific Weight Matrix\n Contains NAN?')
            for t_k in range(len(V_list)):
                w_k = V_list[t_k]
                # print(f'task {t_k}: {np.isnan(w_k.any())}')

            # Loop over iterations for shared parameters
            X = np.concatenate(X_list, axis=0)
            Y = np.concatenate(Y_list, axis=0)
            mu = (num_tasks * lam_2) / lam_1
            # print(f'mu = {mu}, lam_2 = {lam_2}, lam_1 = {lam_1}')

            alpha = 0.01

            for iter in range(number_of_epochs):
                for j in range(X.shape[0]):
                    # get task-specific parameters - programmed for two tasks
                    if j == 0 and j < len(X_list[0]):
                        V_i = V_list[0]
                        bias = b_list[0]
                    if j == len(X_list[0]):
                        V_i = V_list[1]
                        bias = b_list[1]
                    # print(f'res = {(Y[j] * (np.dot(W, X[j]) + np.dot(V_i, X[j]))) + bias}')
                    # if np.isnan((Y[j] * (np.dot(W, X[j]) + np.dot(V_i, X[j]))) + bias):
                    #     exit(0)

                    if (Y[j] * (np.dot(W, X[j]) + np.dot(V_i, X[j]))) + bias < 1:
                        # print(f'(Y[j] * X[j]) = {(Y[j] * X[j])}')
                        V_i = V_i - (2 * alpha * (V_i - np.dot(Y[j], X[j])))
                        # V_i = V_i - (2 * alpha * (V_i - (Y[j] * X[j])))
                        W = W - (mu * 2 * alpha * lam_2) * (W - np.dot(Y[j], X[j]))
                        # W = W - (mu * 2 * alpha * lam_2) * (W - (Y[j] * X[j]))

                    else:
                        V_i = V_i - (2 * lam_1 * alpha * V_i)
                        W = W - mu * (2 * lam_2 * alpha * W)
                        # print(f'mu * (2 * lam_2 * alpha) = {mu * (2 * lam_2 * alpha)}')
                        # print(mu * (2 * lam_2 * alpha * W))
                        # print(f'\n W = {W}')

            # print(f'Updated Shared Weight Matrix, '
            #       f'Contains NAN in shared weights: {np.isnan(W).any()}')
            # print(f'W = {np.shape(W)}, V_list = {np.shape(V_list)}, b_list = {np.shape(b_list)}')
            # print(f'Output sample: \nW = {W[:5]}\nV_list for task 0 = {V_list[0][:5]}\nV_list for task 1 = {V_list[1][:5]}')
            # exit(0)

            # print('''\n****************PREDICTION*****************\n''')
            LogLoss = []
            indi_accu = []

            for i in range(len(X_test_list)):
                X_test, y_test, w_k = X_test_list[i], Y_test_list[i], V_list[i]
                weighted_sum = np.dot(X_test, W) + np.dot(X_test, w_k) + b_list[i]

                y_pred = np.sign(weighted_sum)
                # print(f'Contains NAN in weights: {np.isnan(W).any()}, {np.isnan(w_k).any()}, {np.isnan(b_list[i]).any()}, {np.isnan(weighted_sum).any()}')
                y_pred = [1 if i > 0 else 0 for i in y_pred]

                predict_probabilities = 1 / (1 + np.exp(-weighted_sum))

                # print(y_test.shape, predict_probabilities.shape)
                loss = log_loss(y_test, predict_probabilities)
                # print(f'Log loss for task {i} = {loss}')
                LogLoss.append(loss)
                indi_accu.append(accuracy_score(y_test, y_pred))

            accuracy = np.mean(indi_accu)
            totLoss = np.sum(LogLoss)
            # print(f'Accuracy = {accuracy}, Log loss = {totLoss}')

            best_loss = min(best_loss, totLoss)
            accu_best = max(accu_best, accuracy)

            if best_loss < old_loss:
                old_loss = best_loss
                best_indi_accu = indi_accu
                best_indi_logloss = LogLoss
                best_params = [lam_1, lam_2]

    return best_loss, accu_best, best_indi_logloss, best_indi_accu, best_params


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
        # print(A.shape)

        # file = open(f"../weight_dict_{dataset}.txt", "w")
        # file.write("weight_dict = " + repr(np.transpose(A)) + "\n")
        # file.close()

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

    # file = open(f"../weight_dict_{dataset}.txt", "w")
    # file.write("weight_dict = " + repr(weight_dict) + "\n")
    # file.close()

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
            MSE = mean_squared_error(y_t, y_pred_t)
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


def convert_to_LibSVM_format(task_group_list, fold, data_param_dict_for_specific_task, dataset, data_folder):
    with open(f'{data_folder}{dataset}_train_data_fold_{fold}.data', 'w') as f:
        for task_id in task_group_list:
            X_train = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_train']
            X_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_test']
            y_train = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_train']
            y_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_test']

            # print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')

            # save train data
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

            # save train data
            for i in range(X_test.shape[0]):
                non_zero_indices = np.nonzero(X_test[i, :])[0]
                f.write(f"{y_test[i]} ")
                f.write(f"{0}:{int(task_id)} ")
                for j in non_zero_indices:
                    f.write(f"{j + 1}:{X_test[i, j]} ")
                f.write("\n")


def MTL_XGBoost(task_list, fold, dataset, data_folder):
    param = {
        'silent': 1,
        "early_stopping_rounds": 50,
        "learning_rate": 0.2,

        "min_child_weight": 1,
        "n_estimators": 1000,
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

    fout = open('result.csv', 'a')
    vals = [None] * 100
    for task in param['tasks_list_']:
        vals[task] = xgb.DMatrix(data_folder + f'{dataset}_val_data_{task}_fold_{fold}.data')

    # train
    evallist = [(traindata, 'train'), (validationdata, 'eval')]
    bst = xgb.train(param, traindata, param['n_estimators'], early_stopping_rounds=param['early_stopping_rounds'],
                    evals=evallist)

    y_real = testdata.get_label()
    y_score = bst.predict(testdata, ntree_limit=bst.best_ntree_limit)

    # save model
    # with open('mt-gbdt.model', 'wb') as model:
    #     pickle.dump(bst, model)
    # load model
    # with open('mt-gbdt.model', 'rb') as model:
    #     bst = pickle.load(model)

    # compute ROC
    fpr, tpr, thresholds = roc_curve(y_real, y_score, pos_label=1)
    all_roc_auc = auc(fpr, tpr)
    all_logloss = log_loss(y_real, y_score)
    print(f'shape of y_real: {y_real.shape}, shape of y_score: {y_score.shape}')
    print(f"Fold {fold} together all_logloss = {all_logloss}")

    # output
    fout.write('\n')
    for key in param:
        fout.write(str(key))
        fout.write(',{},'.format(param[key]))
    fout.write('\n')
    fout.write('task,auc,\n')
    log_loss_val = []
    auc_val = []
    accuracy_val = []
    # print(f"param['tasks_list_'] = {param['tasks_list_']}")
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
            # print(f'y_score: {np.shape(y_score)}, y_real: {np.shape(y_real)}')
            logloss = log_loss(y_real, y_score)
            acc = accuracy_score(y_real, y_score.round())
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_logloss = logloss
                tree_num = tree
                best_acc = acc
        # acc = accuracy_score(y_real, y_score)

        # print("task {} 's AUC={} logloss={} at {} tree".format(task, best_auc, best_logloss, tree_num))
        fout.write("{},{},{}\n".format(task, best_auc, best_logloss))
        log_loss_val.append(best_logloss)
        auc_val.append(best_auc)
        accuracy_val.append(best_acc)
        print(f"Fold {fold} task = {task}, best_logloss = {best_logloss}")

    fout.write("all,{},{},\n".format(all_roc_auc, all_logloss))
    fout.close()
    # exit(0)

    if dataset == 'Landmine':
        return np.sum(log_loss_val), log_loss_val, np.mean(auc_val), auc_val
    if dataset == 'Chemical':
        return np.sum(log_loss_val), log_loss_val, np.mean(accuracy_val), accuracy_val


def MTL_XGBoost_regression(task_list, fold, dataset, data_folder):
    param = {
        'silent': 1,
        "early_stopping_rounds": 50,
        "learning_rate": 0.2,

        "min_child_weight": 1,
        "n_estimators": 500,
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
        print(f'task {task}')
        vals[task] = xgb.DMatrix(data_folder + f'{dataset}_val_data_{task}_fold_{fold}.data')

    # train
    evallist = [(traindata, 'train'), (validationdata, 'eval')]
    bst = xgb.train(param, traindata, param['n_estimators'], early_stopping_rounds=param['early_stopping_rounds'],
                    evals=evallist)

    y_real = testdata.get_label()
    y_score = bst.predict(testdata, ntree_limit=bst.best_ntree_limit)

    print(f"Fold {fold} MSE: {mean_squared_error(y_real, y_score)}")

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
            if dataset == 'Parkinsons':
                rsq = r2_score(y_real, y_score)
                evs = explained_variance_score(y_real, y_score)
                if rsq > best_r_square:
                    best_r_square = rsq
                    best_ev = evs
                    best_mse = mse
                    tree_num = tree
            # acc = accuracy_score(y_real, y_score.round())
            if dataset == 'School':
                if mse < best_mse:
                    best_mse = mse
                    tree_num = tree
        # acc = accuracy_score(y_real, y_score)

        print(f'task {task} MSE: {best_mse} at {tree_num} trees')
        # fout.write("{},{},{}\n".format(task, best_auc, best_logloss))
        indi_mse.append(best_mse)
        ev_score.append(best_ev)
        r_square_score.append(best_r_square)
        # auc_val.append(best_auc)
        # accuracy_val.append(best_acc)

    # fout.write("all,{},{},\n".format(all_roc_auc, all_logloss))
    # fout.close()

    if dataset == 'School':
        return np.sum(indi_mse), indi_mse
    if dataset == 'Parkinsons':
        return np.sum(indi_mse), indi_mse, best_ev, ev_score, best_r_square, r_square_score


if __name__ == "__main__":
    dataset = sys.argv[1]

    ModelName = 'SVM'
    print(f'dataset = {dataset}, ModelName = {ModelName}')

    if dataset == 'Chemical' and ModelName == 'SVM':
        DataPath = f'../Dataset/{dataset.upper()}/SVM_DATA/'
    else:
        DataPath = f'../Dataset/{dataset.upper()}/DATA/'

    ResultPath = '../Results'
    num_folds = 10
    number_of_epochs = 100

    Variance = []
    Random_Seed = []
    StdDev = []
    Accuracy = []
    Error_Rate = []

    task_indi_score = {}
    if 'SVM' in ModelName:
        single_results = pd.read_csv(f'{ResultPath}/SingleTaskTraining_{dataset}_SVM.csv')
    if 'xgBoost' in ModelName:
        single_results = pd.read_csv(f'{ResultPath}/SingleTaskTraining_{dataset}_DT.csv')

    ChemicalData = pd.read_csv(f'../Dataset/CHEMICAL/ChemicalData_All.csv', low_memory=False)

    tasks_list_dict = {
        'School': [i for i in range(1, 140)],
        'Landmine': [i for i in range(0, 29)],
        'Chemical': list(ChemicalData['180'].unique()),
        'Parkinsons': [i for i in range(1, 43)]
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

    TASKS = tasks_list_dict[dataset]
    # group_range = group_range_dict[dataset]
    hyperparameters_RMTL = hyperparameters_RMTL_dict[dataset]

    TASKS = tasks_list_dict[dataset]
    # TASKS = TASKS[:5]
    STL_loss = 0
    STL_accu = 0
    STL_auc = 0
    STL_mse = 0
    STL_r_square = 0
    STL_EV = 0
    for task in TASKS:
        task_stl = single_results.loc[(single_results['Task'] == task)]
        STL_loss += task_stl['LOSS'].values[0]
        if dataset == 'Chemical':
            STL_accu += task_stl['Accuracy'].values[0]
        if dataset == 'Landmine':
            STL_auc += task_stl['AUC'].values[0]
        if dataset == 'Parkinsons':
            # STL_r_square += task_stl['R_Square'].values[0]
            STL_EV += task_stl['Explained_Variance'].values[0]

    print(f'dataset = {dataset}, TASKS = {TASKS}, len(TASKS) = {len(TASKS)}')

    random_seed = random.randint(0, 100)

    if dataset == 'Chemical':
        loss, task_score, error, task_error_rate = kFold_validation(TASKS, random_seed, hyperparameters_RMTL)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'STL Results = {STL_loss}, Accuracy = {STL_accu / len(TASKS)}')
        print(f'loss = {loss}\n'
              # f'task_score = {task_score}\n'
              f'error = {error}\n'
              f'task_error_rate = {task_error_rate}')

    if dataset == 'Landmine':
        loss, task_score, mean_auc, task_auc = kFold_validation(TASKS, random_seed, hyperparameters_RMTL)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'STL Results = {STL_loss}, AUC = {STL_auc / len(TASKS)}')
        print(f'loss = {loss}\n'
              f'task_score = {task_score}\n'
              f'AUC = {mean_auc}\n'
              f'task_AUC = {task_auc}')

    if dataset == 'School':
        loss, task_score = kFold_validation(TASKS, random_seed, hyperparameters_RMTL)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'STL Results = {STL_loss}')
        print(f'loss = {loss}\n')
        # f'task_score = {task_score}')

    if dataset == 'Parkinsons':
        loss, task_score, mean_EV, task_EV, mean_r_square, task_r_square = kFold_validation(TASKS, random_seed,
                                                                                            hyperparameters_RMTL)
        print(f'\n\n\n******{dataset}*****\n')
        print(f'STL Results = {STL_loss}, EV = {STL_EV / len(TASKS)}')
        print(f'loss = {loss}\n'
              f'task_score = {task_score}\n'
              f'EV = {mean_EV}\n'
              f'task_EV = {task_EV}\n'
              f'r_square = {mean_r_square}\n'
              f'task_r_square = {task_r_square}')


