import pandas as pd
import copy
import numpy as np
import math
import sys, os, time
import random
import multiprocessing as mp
import itertools
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.svm import SVC
import ast
import tqdm
import xgboost as xgb
import os, shutil
from sklearn.metrics import log_loss,roc_auc_score,average_precision_score,accuracy_score,roc_curve, auc

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
import keras
from keras.callbacks import ModelCheckpoint
from keras import Model
# from Predictor_Class import *
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

USE_GPU = False
if USE_GPU:
    device_idx = int(sys.argv[2])
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
                # print('non_zero_indices = ', len(non_zero_indices), non_zero_indices)
                # print(f"{y_train[i]}")
                f.write(f"{y_train[i]} ")
                f.write(f"{0}:{int(task_id)} ")
                if dataset=='Chemical':
                    for j in non_zero_indices:
                        f.write(f"{j}:{X_train[i, j]} ")
                    f.write("\n")
                else:
                    for j in non_zero_indices:
                        f.write(f"{j + 1}:{X_train[i, j]} ")
                    f.write("\n")

            # save test and val data
            with open(f'{data_folder}{dataset}_val_data_{task_id}_fold_{fold}.data', 'w') as valf:
                for i in range(X_test.shape[0]):
                    non_zero_indices = np.nonzero(X_test[i, :])[0]
                    valf.write(f"{y_test[i]} ")
                    valf.write(f"{0}:{int(task_id)} ")
                    if dataset == 'Chemical':
                        for j in non_zero_indices:
                            valf.write(f"{j}:{X_train[i, j]} ")
                    else:
                        for j in non_zero_indices:
                            valf.write(f"{j + 1}:{X_train[i, j]} ")
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
                if dataset == 'Chemical':
                    for j in non_zero_indices:
                        f.write(f"{j}:{X_test[i, j]} ")
                else:
                    for j in non_zero_indices:
                        f.write(f"{j + 1}:{X_test[i, j]} ")
                f.write("\n")

def convert_to_LibSVM_format_chem(task_group_list,fold,data_param_dict_for_specific_task, dataset, data_folder):
    # print('\n\n*****************fold = ', fold, '*****************')
    # print(f'task_group_list = {task_group_list}')
    max_train_feat = -100
    max_val_feat = -100

    with open(f'{data_folder}{dataset}_train_data_fold_{fold}.data', 'w') as f:
        save_train_instance = [[]]
        save_val_instance = [[]]

        flag_idx_train = False
        flag_idx_test = False
        flag_idx_val = False
        for task_id in task_group_list:

            X_train = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_train']
            X_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_test']
            y_train = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_train']
            y_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_test']

            print(f'unique values in y_train = {np.unique(y_train)}, unique values in y_test = {np.unique(y_test)}')

            # print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')

            for i in range(X_train.shape[0]):
                non_zero_indices = np.nonzero(X_train[i, :])[0]
                f.write(f"{y_train[i]} ")
                f.write(f"{0}:{int(task_id)} ")
                old_max_train_feat = max_train_feat
                max_train_feat = max(max_train_feat, max(non_zero_indices))
                if old_max_train_feat != max_train_feat:
                    save_train_instance[0] = non_zero_indices
                for j in non_zero_indices:
                    f.write(f"{j + 1}:{X_train[i, j]} ")
                f.write("\n")

            # print(f'task id ={task_id}, flag_idx_train = {flag_idx_train}')

            # flag_dict[task_id]=flag_idx_train

            # save test and val data
            with open(f'{data_folder}{dataset}_val_data_{task_id}_fold_{fold}.data', 'w') as valf:
                for i in range(X_test.shape[0]):
                    non_zero_indices = np.nonzero(X_test[i, :])[0]
                    valf.write(f"{y_test[i]} ")
                    valf.write(f"{0}:{int(task_id)} ")

                    old_max_val_feat = max_val_feat
                    max_val_feat = max(max_val_feat, max(non_zero_indices))
                    if max_val_feat != old_max_val_feat:
                        save_val_instance[0] = non_zero_indices
                    for j in non_zero_indices:
                        valf.write(f"{j + 1}:{X_train[i, j]} ")
                    valf.write("\n")

                # print(f'task id ={task_id}, flag_idx_train = {flag_idx_train}, flag_idx_val = {flag_idx_val}')

    # print(f'task id ={task_id}, flag_idx_train = {flag_idx_train}, flag_idx_val = {flag_idx_val}')
    save_test_instance = [[]]
    max_test_feat = -100
    with open(f'{data_folder}{dataset}_test_data_fold_{fold}.data', 'w') as f:
        for task_id in task_group_list:

            X_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_X_test']
            y_test = data_param_dict_for_specific_task[f'{dataset}_{task_id}_fold_{fold}_y_test']

            # save train data
            for i in range(X_test.shape[0]):
                non_zero_indices = np.nonzero(X_test[i, :])[0]
                f.write(f"{y_test[i]} ")
                f.write(f"{0}:{int(task_id)} ")
                old_max_test_feat = max_test_feat
                max_test_feat = max(max_test_feat, max(non_zero_indices))
                if max_test_feat != old_max_test_feat:
                    save_test_instance[0] = non_zero_indices
                for j in non_zero_indices:
                    f.write(f"{j + 1}:{X_test[i, j]} ")
                f.write("\n")

    traindata = xgb.DMatrix(data_folder + f'{dataset}_train_data_fold_{fold}.data')
    testdata = xgb.DMatrix(data_folder + f'{dataset}_test_data_fold_{fold}.data')

    data_split = ['train', 'test']
    num_feat = [traindata.num_col(), testdata.num_col()]
    if traindata.num_col() != testdata.num_col():
        print(f'fold = {fold}, feature dimension mismatch')
        print(f'num_feat = {num_feat},data_split = {data_split}')

        get_min = min(num_feat)
        get_split = data_split[num_feat.index(get_min)]
        get_diff = max(num_feat) - get_min
        print(f'get_split = {get_split}, get_diff = {get_diff}')

        if get_split == 'train':
            with open(f'{data_folder}{dataset}_train_data_fold_{fold}.data', 'a') as f:
                for count in range(get_diff):
                    task_id = random.choice(task_group_list)
                    f.write(f'{0} ')
                    f.write(f"{0}:{int(task_id)} ")
                    non_zero_indices = save_test_instance[0]
                    for j in non_zero_indices:
                        f.write(f"{j + 1}:{0} ")
                    f.write("\n")

        if get_split == 'test':

            with open(f'{data_folder}{dataset}_test_data_fold_{fold}.data', 'a') as f:
                for count in range(get_diff):
                    task_id = random.choice(task_group_list)
                    f.write(f'{0} ')
                    f.write(f"{0}:{int(task_id)} ")
                    non_zero_indices = save_train_instance[0]
                    # print(f'non_zero_indices = {non_zero_indices}')
                    for j in non_zero_indices:
                        f.write(f"{j + 1}:{0} ")
                    f.write("\n")

def kFold_validation(task_group_list, random_seed, task_type):
    data_param_dict_for_specific_task = {}

    args = []
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

    if dataset == 'Landmine':
        data_param_dictionary = readData_landmine(task_group_list)

    if dataset == 'Chemical':
        data_param_dictionary = readData_chemical(task_group_list)

    if dataset == 'School':
        data_param_dictionary = readData_school(task_group_list)

    if dataset == 'Parkinsons':
        Sample_Inputs, Sample_Label, data_param_dictionary = readData_parkinsons_different(task_group_list)

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
            if dataset == 'Chemical':
                convert_to_LibSVM_format_chem(task_group_list, fold, data_param_dict_for_specific_task, dataset,
                                              data_folder)
            else:
                convert_to_LibSVM_format(task_group_list, fold, data_param_dict_for_specific_task, dataset, data_folder)

        args = [(task_group_list, fold, dataset, data_folder) for fold in range(0, 10)]


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

    # print(f'all_scores = {all_scores}')

    print(f'Time taken for {number_of_pools} pools = {time.time() - time_start}')

    scores = []
    auc_scores = []
    error_rates = []
    explained_variance = []
    r2_scores = []
    final_param = []
    for i in range(len(all_scores)):
        if 'inf' not in str(all_scores[i][0]):
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
            if 'inf' not in str(all_scores[c][0]):
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
            if dataset == 'Parkinsons':
                rsq = r2_score(y_real, y_score)
                evs = explained_variance_score(y_real, y_score)
                if rsq > best_r_square:
                    best_r_square = rsq
                    best_ev = evs
                    best_mse = mse
                    tree_num = tree
            # acc = accuracy_score(y_real, y_score.round())
            if dataset=='School':
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



def xgBoost_Park(X_train, X_test, y_train, y_test,data_param_dict_for_specific_task,task_group_list,fold):
    dataset = 'Parkinsons'
    reg = xgb.XGBRegressor(tree_method='auto', n_estimators=40)
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

def mutate_groups(new_task_group, new_group_score):
    task_rand = random.sample(TASKS, 1)
    task_rand = task_rand[0]
    changed_group = []

    # find out old group
    for key, task_list in new_task_group.items():
        if task_rand in task_list:
            g_old = key

    # check if old group is empty->delete the old group and assign task to new group
    print(f'task_rand = {task_rand}\tg_old = {g_old}\tTask-Group = {new_task_group}')

    if len(new_task_group[g_old]) == 1:
        del new_task_group[g_old]
        del new_group_score[g_old]
        g_new = random.choice(list(new_task_group.keys()))
        new_task_group[g_new].append(task_rand)
        changed_group.append(g_new)

    else:
        new_task_group[g_old].remove(task_rand)
        g_new = random.choice([g for g in range(len(new_task_group) + 1) if g != g_old])

        if g_new not in new_task_group.keys():
            new_task_group.update({g_new: list([task_rand])})
        else:
            new_task_group[g_new].append(task_rand)
        changed_group = [g_old, g_new]

    return changed_group,task_rand
def predictor_network(x_train, y_train, network_architecture):
    number_of_epoch = 100
    filepath = f'{run_results}/SavedModels/{dataset}_TG_predictor_Best_test.h5'

    number_of_features = np.shape(x_train)[1]

    Input_FF = tf.keras.layers.Input(shape=(number_of_features,))
    hidden_FF = tf.keras.layers.Dense(network_architecture['FF_Neurons'][0],
                                      activation='sigmoid')(Input_FF)
    for h in range(1, network_architecture['FF_Layers']):
        hidden_FF = tf.keras.layers.Dense(network_architecture['FF_Neurons'][h],
                                              activation='sigmoid')(hidden_FF)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_FF)
    # output = tf.keras.layers.Dense(1, activation='linear')(hidden_FF)

    finalModel = Model(inputs=Input_FF, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=network_architecture['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')
    # print(finalModel.summary())

    checkpoint = ModelCheckpoint(filepath, verbose=0, monitor='loss', save_best_only=True, mode='auto')

    history = finalModel.fit(x=x_train,
                 y=y_train,
                 shuffle=True,
                 epochs=number_of_epoch,
                 batch_size=16,
                 callbacks=checkpoint,
                 verbose=0)

def predict_performance_of_new_group(tasks, affinity_pred_arch):
    if len(tasks) > 1:
        number_of_tasks = []
        pairwise_improvement_average = []
        pairwise_improvement_variance = []
        pairwise_improvement_stddev = []
        group_variance = []
        group_stddev = []
        group_distance = []
        group_dataset_size = []

        pairwise_ITA_average = []
        pairwise_Weight_average = []
        number_of_tasks.append(len(tasks))
        sample_size = 0
        avg_var = []
        avg_stddev = []
        avg_dist = []
        task_combo = list(itertools.combinations(sorted(tasks), 2))
        for task_pair in task_combo:
            task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[0]) & (
                    task_distance_info.Task_2 == task_pair[1])].reset_index()
            if len(task_dist_data) == 0:
                task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[1]) & (
                        task_distance_info.Task_2 == task_pair[0])].reset_index()

            if dataset == 'Chemical':
                avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
            else:
                avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])

        group_distance.append(np.mean(avg_dist))
        paired_improvement = []
        paired_ITA = []
        paired_weight = []
        for pair in task_combo:
            stl_loss = 0
            stl_loss += Single_res_dict[pair[0]]
            stl_loss += Single_res_dict[pair[1]]
            pair_specific = pair_results[
                (pair_results.Task_1 == pair[0]) & (pair_results.Task_2 == pair[1])].reset_index()
            if len(pair_specific) == 0:
                pair_specific = pair_results[
                    (pair_results.Task_1 == pair[1]) & (pair_results.Task_2 == pair[0])].reset_index()
            paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)
            # p = tuple(sorted([pair[0], pair[1]]))
            # pair_idx = list(ITA_data['Pairs']).index(str(p))
            # paired_ITA.append(list(ITA_data.Pairwise_ITA)[pair_idx])
            #
            # pair_idx = list(Weight_Matrix['Pairs']).index(str(p))
            # paired_weight.append(list(Weight_Matrix.Weight)[pair_idx])

        pairwise_improvement_average.append(np.mean(paired_improvement))
        pairwise_improvement_variance.append(np.var(paired_improvement))
        pairwise_improvement_stddev.append(np.std(paired_improvement))
        # pairwise_ITA_average.append(np.mean(paired_ITA))
        # pairwise_Weight_average.append(np.mean(paired_weight))

        single_task_total_loss = 0
        for t in tasks:
            single_task_total_loss += Single_res_dict[t]
            sample_size += task_len[t]
            avg_var.append(variance_dict[t])
            avg_stddev.append(std_dev_dict[t])

        group_variance.append(np.mean(avg_var))
        group_stddev.append(np.mean(avg_stddev))
        group_dataset_size.append(sample_size / len(TASKS))

        new_groups = pd.DataFrame({
            'group_dataset_size': group_dataset_size,
            'group_variance': group_variance,
            'group_stddev': group_stddev,
            'group_distance': group_distance,
            'number_of_tasks': number_of_tasks,
            'pairwise_improvement_average': pairwise_improvement_average,
            'pairwise_improvement_variance': pairwise_improvement_variance,
            'pairwise_improvement_stddev': pairwise_improvement_stddev,
            # 'pairwise_ITA_average': pairwise_ITA_average,
            # 'pairwise_Weight_average': pairwise_Weight_average
        })

        pred_features = affinity_pred_arch['Features']
        new_groups = new_groups[pred_features]
        # print(f'new_groups.columns = {new_groups.columns}')
        x = np.array(new_groups, dtype='float32')
        filepath = f'{run_results}/SavedModels/{dataset}_TG_predictor_Best_test.h5'
        finalModel = tf.keras.models.load_model(filepath)
        final_score = finalModel.predict(x, verbose=0)
        return final_score[0][0], single_task_total_loss
    else:
        return 0, Single_res_dict[tasks[0]]

def retrain_predictor(dataset,affinity_pred_arch):
    predictor_data = pd.read_csv(f'{run_results}/Data_for_Predictor_{dataset}_updated_{ModelName}.csv')
    print(f'\n\n******* Training Samples = {len(predictor_data)} *******\n\n')

    pred_features = affinity_pred_arch['Features']
    print(f'\n\n******* Training Samples = {len(predictor_data)} *******\n\n')
    predictor_data.dropna(inplace=True)
    y_train_pred = np.array(list(predictor_data.change), dtype=float)
    predictor_data = predictor_data[pred_features]
    # print(f'predictor_data.columns = {predictor_data.columns}')

    # DataSet = np.array(predictor_data, dtype=float)
    x_train_pred = np.array(predictor_data, dtype=float)
    predictor_network(x_train_pred, y_train_pred,affinity_pred_arch)


def predictor_data_prep(task_grouping_results, counter, run, dataset_name):
    DataPath = f'../Dataset/{dataset.upper()}/'
    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    single_res_dict = {}

    if dataset_name == 'School':
        TASKS = [i for i in range(1, 140)]
    if dataset_name == 'Landmine':
        TASKS = [i for i in range(0, 29)]
    if dataset_name == 'Parkinsons':
        TASKS = [i for i in range(1, 43)]
    if dataset_name == 'Chemical':
        chemical_data = pd.read_csv(f'{DataPath}ChemicalData_All.csv', low_memory=False)
        TASKS = list(chemical_data['180'].unique())


    task_info = pd.read_csv(f'{DataPath}Task_Information_{dataset_name}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{dataset_name}.csv')
    single_results = pd.read_csv(f'../Results/STL/STL_{dataset_name}_{ModelName}.csv')

    pair_results = pd.read_csv(
        f'../Results/Pairwise/{ModelName}/{dataset_name}_Results_from_Pairwise_Training_ALL_{ModelName}.csv')

    for selected_task in TASKS:
        if dataset_name == 'Chemical':
            task_data = task_info[task_info.Molecule == selected_task].reset_index()
        else:
            task_data = task_info[task_info.Task_Name == selected_task].reset_index()
        task_len.update({selected_task: task_data.Dataset_Size[0]})
        variance_dict.update({selected_task: task_data.Variance[0]})
        std_dev_dict.update({selected_task: task_data.Std_Dev[0]})
        dist_dict.update({selected_task: task_data.Average_Euclidian_Distance_within_Task[0]})
        single_res = single_results[single_results.Task == selected_task].reset_index()
        single_res_dict.update({selected_task: single_res.LOSS[0]})
    single_task_loss = []
    group_info = []
    group_loss = []
    group_dataset_size = []
    group_variance = []
    group_stddev = []
    group_distance = []
    number_of_tasks = []

    pairwise_improvement_average = []
    pairwise_improvement_variance = []
    pairwise_improvement_stddev = []

    # pairwise_ITA_average = []
    # pairwise_Weight_average = []


    change = []



    # count = 0
    for group in tqdm.tqdm(range(len(task_grouping_results))):
        Task_Group = ast.literal_eval(task_grouping_results.Task_group[group])

        Individual_Group_Score = ast.literal_eval(task_grouping_results.Individual_Group_Score[group])
        if counter!=0:

            Changed_Groups = ast.literal_eval(task_grouping_results.Changed_Groups[group])

            if Changed_Groups != None:
                for gr in Changed_Groups:
                    sample_size = 0
                    avg_var = []
                    avg_stddev = []
                    avg_dist = []
                    sum_loss_single_task = 0

                    if len(Task_Group[gr])<=1:
                        continue

                    task_combo = list(itertools.combinations(sorted(Task_Group[gr]), 2))
                    for task_pair in task_combo:
                        task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[0]) & (
                                task_distance_info.Task_2 == task_pair[1])].reset_index()

                        if len(task_dist_data) == 0:
                            task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[1]) & (
                                    task_distance_info.Task_2 == task_pair[0])].reset_index()

                        if dataset_name == 'Chemical':
                            avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
                        else:
                            avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])
                    # group_distance.append(np.mean(avg_dist))

                    paired_improvement = []
                    # paired_ITA = []
                    # paired_weight = []
                    for pair in task_combo:
                        stl_loss = 0
                        stl_loss += single_res_dict[pair[0]]
                        stl_loss += single_res_dict[pair[1]]
                        pair_specific = pair_results[
                            (pair_results.Task_1 == pair[0]) & (pair_results.Task_2 == pair[1])].reset_index()
                        if len(pair_specific) == 0:
                            pair_specific = pair_results[
                                (pair_results.Task_1 == pair[1]) & (pair_results.Task_2 == pair[0])].reset_index()
                        paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)


                    pairwise_improvement_average.append(np.mean(paired_improvement))
                    pairwise_improvement_variance.append(np.var(paired_improvement))
                    pairwise_improvement_stddev.append(np.std(paired_improvement))





                    # pairwise_ITA_average.append(np.mean(paired_ITA))
                    # pairwise_Weight_average.append(np.mean(paired_weight))

                    for t in Task_Group[gr]:
                        sample_size += task_len[t]
                        avg_var.append(variance_dict[t])
                        avg_stddev.append(std_dev_dict[t])
                        sum_loss_single_task += single_res_dict[t]
                    group_info.append(Task_Group[gr])
                    group_loss.append(Individual_Group_Score[gr])
                    number_of_tasks.append(len(Task_Group[gr]))
                    group_dataset_size.append(sample_size / len(Task_Group[gr]))  # individual length

                    group_distance.append(np.mean(avg_dist))
                    group_variance.append(np.mean(avg_var))
                    group_stddev.append(np.mean(avg_stddev))

                    # group_distance.append(np.mean(avg_dist))
                    single_task_loss.append(sum_loss_single_task)

                    change.append((sum_loss_single_task - Individual_Group_Score[gr]) / sum_loss_single_task)
        else:
            for group_no, tasks in Task_Group.items():
                sample_size = 0
                avg_var = []
                avg_stddev = []
                avg_dist = []
                sum_loss_single_task = 0
                if len(tasks)<=1:
                    continue
                task_combo = list(itertools.combinations(sorted(tasks), 2))
                for task_pair in task_combo:
                    task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[0]) & (
                            task_distance_info.Task_2 == task_pair[1])].reset_index()
                    if len(task_dist_data) == 0:
                        task_dist_data = task_distance_info[(task_distance_info.Task_1 == task_pair[1]) & (
                                task_distance_info.Task_2 == task_pair[0])].reset_index()

                    if dataset_name == 'Chemical':
                        avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
                    else:
                        avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])


                paired_improvement = []
                paired_ITA = []
                paired_weight = []
                for pair in task_combo:
                    stl_loss = 0
                    stl_loss += single_res_dict[pair[0]]
                    stl_loss += single_res_dict[pair[1]]
                    pair_specific = pair_results[
                        (pair_results.Task_1 == pair[0]) & (pair_results.Task_2 == pair[1])].reset_index()
                    if len(pair_specific) == 0:
                        pair_specific = pair_results[
                            (pair_results.Task_1 == pair[1]) & (pair_results.Task_2 == pair[0])].reset_index()
                    paired_improvement.append((stl_loss - pair_specific.Total_Loss[0]) / stl_loss)

                    p = tuple(sorted([pair[0], pair[1]]))
                    # pair_idx = list(ITA_data['Pairs']).index(str(p))
                    # paired_ITA.append(list(ITA_data.Pairwise_ITA)[pair_idx])

                    # pair_idx = list(Weight_Matrix['Pairs']).index(str(p))
                    # paired_weight.append(list(Weight_Matrix.Weight)[pair_idx])

                # pairwise_ITA_average.append(np.mean(paired_ITA))
                # pairwise_Weight_average.append(np.mean(paired_weight))


                pairwise_improvement_average.append(np.mean(paired_improvement))
                pairwise_improvement_variance.append(np.var(paired_improvement))
                pairwise_improvement_stddev.append(np.std(paired_improvement))
                for t in tasks:
                    sample_size += task_len[t]
                    avg_var.append(variance_dict[t])
                    avg_stddev.append(std_dev_dict[t])
                    sum_loss_single_task += single_res_dict[t]
                group_info.append(tasks)
                group_loss.append(Individual_Group_Score[group_no])
                number_of_tasks.append(len(tasks))
                group_dataset_size.append(sample_size / len(tasks))  # individual length

                group_distance.append(np.mean(avg_dist))
                group_variance.append(np.mean(avg_var))
                group_stddev.append(np.mean(avg_stddev))

                change.append((sum_loss_single_task - Individual_Group_Score[group_no]) / sum_loss_single_task)

    predictor_data = pd.DataFrame({
        'group_variance': group_variance,
        'group_stddev': group_stddev,
        'group_distance': group_distance,
        'number_of_tasks': number_of_tasks,
        'group_dataset_size': group_dataset_size,
        'pairwise_improvement_average': pairwise_improvement_average,
        'pairwise_improvement_variance': pairwise_improvement_variance,
        'pairwise_improvement_stddev': pairwise_improvement_stddev,
        # 'pairwise_ITA_average': pairwise_ITA_average,
        # 'pairwise_Weight_average': pairwise_Weight_average,
        'change': change
    })

    predictor_data = predictor_data[predictor_data.number_of_tasks > 2]

    predictor_data.to_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_new_updated_{ModelName}.csv', index=False)

    if counter == 0:
        predictor_data.to_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_first_{ModelName}.csv', index=False)
        predictor_data.to_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_updated_{ModelName}.csv', index=False)

    elif counter == 'rerun':
        old_file = f'{run_results}/Data_for_Predictor_{dataset_name}_first_{ModelName}.csv'
        if os.path.exists(old_file):
            df_1 = pd.read_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_first_{ModelName}.csv')
            df_2 = pd.read_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_new_updated_{ModelName}.csv')
            frames = [df_1, df_2]

            result = pd.concat(frames)
            result.to_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_updated_{ModelName}.csv', index=False)

    else:
        old_file = f'{run_results}/Data_for_Predictor_{dataset_name}_updated_{ModelName}.csv'
        if os.path.exists(old_file):
            df_1 = pd.read_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_updated_{ModelName}.csv')
            df_2 = pd.read_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_new_updated_{ModelName}.csv')
            frames = [df_1, df_2]

            result = pd.concat(frames)
            result.to_csv(f'{run_results}/Data_for_Predictor_{dataset_name}_updated_{ModelName}.csv', index=False)




if __name__ == "__main__":
    dataset = sys.argv[1]
    run = int(sys.argv[2])

    # dataset = 'Parkinsons'
    # run = 1
    ModelName = 'xgBoost'

    print(f'dataset = {dataset}, ModelName = {ModelName}')
    if dataset == 'Chemical' or dataset == 'Landmine':
        task_type = 'classification'
    else:
        task_type = 'regression'

    ResultPath = '../Results'
    run_results = f'../Results/Run_{run}'
    if not os.path.exists(run_results):
        os.mkdir(run_results)

    ChemicalData = pd.read_csv(f'../Dataset/CHEMICAL/ChemicalData_All.csv', low_memory=False)

    tasks_list_dict = {
        'School': [i for i in range(1, 140)],
        'Landmine': [i for i in range(0, 29)],
        'Chemical': list(ChemicalData['180'].unique()),
        'Parkinsons': [i for i in range(1, 43)]
    }
    K_dict = {'Chemical': 120, 'Landmine': 25, 'Parkinsons': 0.99, 'School': 20}
    TASKS = tasks_list_dict[dataset]


    num_folds = 10
    number_of_epochs = 100


    task_indi_score = {}
    
    

    Variance = []
    Random_Seed = []
    StdDev = []
    Accuracy = []
    Error_Rate = []


    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}
    STL_error = {}
    STL_AUC = {}
    STL_EV = {}
    STL_r_square = {}
    DataPath = f'../Dataset/{dataset.upper()}/'
    task_info = pd.read_csv(f'{DataPath}Task_Information_{dataset}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{dataset}.csv')
    single_results = pd.read_csv(f'{ResultPath}/STL/STL_{dataset}_{ModelName}.csv')
    pair_results = pd.read_csv(f'{ResultPath}/Pairwise/{ModelName}/{dataset}_Results_from_Pairwise_Training_ALL_{ModelName}.csv')
    print(f'pair_results = {pair_results.columns}')


    DataPath = f'../Dataset/{dataset.upper()}/'
    
    for Selected_Task in TASKS:
        if dataset == 'Chemical':
            task_data = task_info[task_info.Molecule == Selected_Task].reset_index()
            dist_dict.update({Selected_Task: task_data.Average_Hamming_Distance_within_Task[0]})
        else:
            task_data = task_info[task_info.Task_Name == Selected_Task].reset_index()
            dist_dict.update({Selected_Task: task_data.Average_Euclidian_Distance_within_Task[0]})

        task_len.update({Selected_Task: task_data.Dataset_Size[0]})
        variance_dict.update({Selected_Task: task_data.Variance[0]})
        std_dev_dict.update({Selected_Task: task_data.Std_Dev[0]})

        single_res = single_results[single_results.Task == Selected_Task].reset_index()
        Single_res_dict.update({Selected_Task: single_res.LOSS[0]})
        if dataset == 'Chemical':
            STL_error.update({Selected_Task: single_res.Error_Rate[0]})
        if dataset == 'Landmine':
            STL_AUC.update({Selected_Task: single_res.AUC[0]})
        if dataset == 'Parkinsons':
            STL_EV.update({Selected_Task: single_res.Explained_Variance[0]})
            STL_r_square.update({Selected_Task: single_res.R_Square[0]})

        # STL_AP.update({Selected_Task: single_res.Avg_Precision[0]})

    Pairwise_res_dict = {}
    Pairwise_err_dict = {}
    Pairwise_AUC_dict = {}
    Pairwise_EV_dict = {}
    Pairwise_rsq_dict = {}
    Task1 = list(pair_results.Task_1)
    Task2 = list(pair_results.Task_2)
    Pairs = [(Task1[i], Task2[i]) for i in range(len(Task1))]
    for p in Pairs:
        task1 = p[0]
        task2 = p[1]
        pair_res = pair_results[(pair_results.Task_1 == task1) & (pair_results.Task_2 == task2)].reset_index()
        if len(pair_res) == 0:
            pair_res = pair_results[(pair_results.Task_1 == task2) & (pair_results.Task_2 == task1)].reset_index()
        Pairwise_res_dict.update({p: pair_res.Total_Loss[0]})
        if dataset == 'Chemical':
            Pairwise_err_dict.update({p: 1 - pair_res.Avg_Error[0]})
        if dataset == 'Landmine':
            Pairwise_AUC_dict.update({p: pair_res.AUC[0]})
        if dataset == 'Parkinsons':
            Pairwise_EV_dict.update({p: pair_res.Explained_Var[0]})
            Pairwise_rsq_dict.update({p: pair_res.R_Square[0]})


    print(f'Pairwise_res_dict = {len(Pairwise_res_dict)}, '
          f'Pairwise_err_dict = {len(Pairwise_err_dict)},'
          f'Pairwise_AUC_dict = {len(Pairwise_AUC_dict)}, '
          f'Pairwise_EV_dict = {len(Pairwise_EV_dict)}, '
          f'Pairwise_rsq_dict = {len(Pairwise_rsq_dict)}')

    affinity_pred_architecture = {
        'School':{'FF_Layers': 2, 'FF_Neurons': [143, 25], 'learning_rate': 0.0013850679916489607,'activation_function': 'relu', 'output_activation': 'linear',
                'Features': ['pairwise_improvement_average', 'pairwise_improvement_variance', 'number_of_tasks','group_stddev']},
        'Chemical': {'FF_Layers': 6, 'FF_Neurons': [13, 10, 8, 29, 19, 15], 'learning_rate': 0.0011219050732356583,'activation_function': 'relu', 'output_activation': 'sigmoid',
                    'Features': ['pairwise_improvement_average', 'pairwise_improvement_stddev', 'number_of_tasks',
                                 'pairwise_improvement_variance']},
        'Landmine': {'FF_Layers': 2, 'FF_Neurons': [35, 12], 'learning_rate': 0.0013850679916489607,'activation_function': 'relu', 'output_activation': 'linear',
                    'Features': ['pairwise_improvement_average', 'group_variance', 'number_of_tasks',
                    'pairwise_improvement_variance', 'pairwise_improvement_stddev', 'group_stddev']},
        'Parkinsons': {'FF_Layers': 1, 'FF_Neurons': [150], 'learning_rate': 0.0016759322698952422, 'activation_function': 'relu', 'output_activation': 'linear',
                       'Features': ['pairwise_improvement_average', 'pairwise_improvement_stddev', 'group_stddev']}} # This is the architecture of the affinity prediction model

    affinity_pred_arch = affinity_pred_architecture[dataset]

    task_set = []
    data_param_dict_for_specific_task = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    Variance = []
    Improvement = []
    Variance_Target = []
    Random_Seed = []
    StdDev = []
    Accuracy = []
    Error_Rate = []
    Training_Samples = []

    Total_Hidden_layer = []
    Total_Neurons = []
    Total_Trainable_param = []
    Total_Features = []

    args_task = []
    changed_group = []
    # task_indi_score = {}
    prev_solution = 0
    group_score = {}
    trainable_param = {}

    number_of_epoch = 400
    min_task_groups = 5
    num_folds = 10

    switch_architecture = ['ACCEPT', 'REJECT']

    Iteration_MTL = []
    Task_group = []

    Total_Loss = []
    Group_No = []
    Trainable_Param = []
    Individual_Group_Score = []
    Individual_Task_Score = []
    Group_Score = []
    Prev_Solution = []
    Prev_iter = []
    Switch = []
    Changed_Group = []
    Number_of_Groups = []
    Random_Task = []
    Predicted_Score = []

    '''dataset specific parameters'''
    Classification_Error = []
    AUC = []
    Explained_Variance = []
    R_Squared = []
    '''end of dataset specific parameters'''

    # Average_Precision = []
    STL_Score = []
    TRAIN_Model = []
    Fast_Reject = []
    Training_Time = []
    Number_of_MTL_Training = []
    Prev_Groups = {}



    '''Predictor Functions'''

    # initial_results = pd.read_csv(f'../Results/partition_sample/{ModelName}/{dataset}_group_sample_{ModelName}.csv',
    #                               low_memory=False)
    initial_results = pd.read_csv(f'../Results/partition_sample/{ModelName}/{dataset}_URS_MTL_{ModelName}.csv',
                                  low_memory=False)
    old_file = f'{run_results}/Data_for_Predictor_{dataset}_first_{ModelName}.csv'
    new_file = f'{run_results}/Data_for_Predictor_{dataset}_updated_{ModelName}.csv'
    if os.path.exists(old_file):
        os.remove(os.path.join(old_file))
    if os.path.exists(new_file):
        os.remove(os.path.join(new_file))

    predictor_data_prep(initial_results, 0, run, dataset)
    time_pred_Start = time.time()
    retrain_predictor(dataset, affinity_pred_arch)
    print(f'Time to train predictor = {(time.time() - time_pred_Start) / 60} min')


    initial_results = initial_results.sort_values(by=['Total_Loss'], ascending=True).reset_index(drop=True)
    best_group_index = int(run) - 1
    task_group = ast.literal_eval(initial_results.Task_group[best_group_index])
    group_score = ast.literal_eval(initial_results.Individual_Group_Score[best_group_index])
    if dataset == 'Chemical':
        er = initial_results.Individual_Error_Rate[best_group_index]
        er = er.replace('nan', '30')
        error_rate = ast.literal_eval(er)
        for group_no, score in group_score.items():
            t = tuple(sorted(task_group[group_no]))
            Prev_Groups[t] = (group_score[group_no], error_rate[group_no])
    if dataset == 'Landmine':
        auc_score = ast.literal_eval(initial_results.Individual_AUC[best_group_index])
        for group_no, score in group_score.items():
            t = tuple(sorted(task_group[group_no]))
            Prev_Groups[t] = (group_score[group_no], auc_score[group_no])

    if dataset == 'School':
        for group_no, task in task_group.items():
            grouping_sorted = tuple(sorted(task))
            if grouping_sorted not in Prev_Groups.keys():
                Prev_Groups.update({grouping_sorted: 0})

        for group_no, score in group_score.items():
            t = tuple(sorted(task_group[group_no]))
            Prev_Groups[t] = score

    if dataset == 'Parkinsons':
        r_square = ast.literal_eval(initial_results.Individual_r_square[best_group_index])
        ev = ast.literal_eval(initial_results.Individual_EV[best_group_index])
        for group_no, score in group_score.items():
            t = tuple(sorted(task_group[group_no]))
            Prev_Groups[t] = (group_score[group_no], r_square[group_no], ev[group_no])

    prev_solution = sum(list(group_score.values()))
    first_sol = prev_solution

    K = K_dict[dataset]

    print(f'prev_solution = {prev_solution}')
    print(f'group_score = {group_score}')
    print(sum(list(group_score.values())))

    Switch.append('None')
    Changed_Group.append('None')
    Prev_Solution.append(prev_solution)
    Prev_iter.append(0)
    Training_Time.append('None')
    Iteration_MTL.append(0)
    Random_Task.append('None')
    Total_Loss.append(prev_solution)
    Task_group.append(task_group)
    Number_of_Groups.append(len(task_group))
    Predicted_Score.append('None')
    Fast_Reject.append('None')
    Individual_Group_Score.append(group_score)
    Number_of_MTL_Training.append(len(group_score))

    if dataset == 'Chemical':
        Classification_Error.append(sum(list(error_rate.values())))
    if dataset == 'Landmine':
        AUC.append(sum(list(auc_score.values())))
    if dataset == 'Parkinsons':
        R_Squared.append(sum(list(r_square.values())))
        Explained_Variance.append(sum(list(ev.values())))

    iter_max = 10000000
    last_iter = 0

    current_best_group = copy.deepcopy(task_group)
    current_best_group_score = copy.deepcopy(group_score)
    current_best_prev_solution = prev_solution
    timeStamp = time.time()
    x = 0
    old_length = 0
    for iteration_MTL in range(1, iter_max):
        predicted_performance = {}
        single_loss_dict = {}

        new_task_group = copy.deepcopy(task_group)
        new_group_score = copy.deepcopy(group_score)
        predicted_group_score = copy.deepcopy(group_score)

        changed_group, task_rand = mutate_groups(new_task_group, new_group_score)

        if len(changed_group) == 0:
            x += 1
            continue

        changed_group = sorted(changed_group)
        time_query_Start = time.time()
        for group_no in changed_group:
            prediction_of_performance = predict_performance_of_new_group(new_task_group[group_no],affinity_pred_arch)
            predicted_performance[group_no] = prediction_of_performance[0]
            single_loss_dict[group_no] = prediction_of_performance[1]

        for group_no, task in new_task_group.items():
            if group_no in changed_group:
                total_loss_per_task_group_per_fold = single_loss_dict[group_no] * (
                        1 - predicted_performance[group_no])
                predicted_group_score[group_no] = total_loss_per_task_group_per_fold
        print(f'Time to query predictor = {(time.time() - time_query_Start) / 60} min')
        predicted_solution = sum(list(predicted_group_score.values()))


        if predicted_solution < prev_solution:
            PROBABILITIES_ACCEPT = [1, 0]
        else:
            PROBABILITIES_ACCEPT = [0, 1]
        fast_reject = np.random.choice(switch_architecture, 1, p=PROBABILITIES_ACCEPT)
        if fast_reject[0] == 'ACCEPT':
            train_model = 1
        else:
            if len(Switch) < 100:
                train_model = np.random.choice([1, 0], 1, p=[0.15, 0.85])
            elif len(Switch) < 500:
                train_model = np.random.choice([1, 0], 1, p=[0.05, 0.95])
            else:
                train_model = np.random.choice([1, 0], 1, p=[0.01, 0.99])

            train_model = train_model[0]

        if train_model == 1:
            print(f'predicted_solution = {predicted_solution}')
            Train_MTL_Model = True
            random_seed = random.randint(0, 100)
            timeStart = time.time()
            count = 0
            for group_no, task in new_task_group.items():
                if group_no in changed_group:
                    grouping_sorted = tuple(sorted(task))
                    if grouping_sorted not in Prev_Groups.keys():
                        if len(task) == 1:
                            total_loss_per_task_group_per_fold = Single_res_dict[task[0]]
                            if dataset == 'Chemical':
                                avg_error = STL_error[task[0]]
                            if dataset == 'Landmine':
                                mean_auc = STL_AUC[task[0]]
                            if dataset == 'Parkinsons':
                                mean_EV = STL_r_square[task[0]]
                                mean_r_square = STL_EV[task[0]]

                        elif len(task) == 2:
                            if grouping_sorted in Pairwise_res_dict.keys():
                                total_loss_per_task_group_per_fold = Pairwise_res_dict[grouping_sorted]
                            else:
                                total_loss_per_task_group_per_fold = Pairwise_res_dict[(grouping_sorted[1],grouping_sorted[0])]

                            if dataset == 'Chemical':
                                if grouping_sorted in Pairwise_res_dict.keys():
                                    avg_error = Pairwise_err_dict[grouping_sorted]
                                else:
                                    avg_error = Pairwise_err_dict[(grouping_sorted[1],grouping_sorted[0])]

                            if dataset == 'Landmine':
                                mean_auc = Pairwise_AUC_dict[grouping_sorted]

                            if dataset == 'Parkinsons':
                                mean_EV = Pairwise_rsq_dict[grouping_sorted]
                                mean_r_square = Pairwise_EV_dict[grouping_sorted]

                        else:
                            count+=1
                            if dataset == 'Chemical':
                                total_loss_per_task_group_per_fold, task_score, avg_error, task_error_rate = kFold_validation(task_group[group_no],
                                                                                                random_seed,
                                                                                                task_type)

                            if dataset == 'Landmine':
                                total_loss_per_task_group_per_fold, task_score, mean_auc, task_auc = kFold_validation(task_group[group_no],
                                                                                                random_seed,
                                                                                                task_type)

                            if dataset == 'School':
                                total_loss_per_task_group_per_fold, task_score = kFold_validation(task_group[group_no],random_seed,
                                                                                                task_type)


                            if dataset == 'Parkinsons':
                                total_loss_per_task_group_per_fold, task_score, mean_EV, task_EV, mean_r_square, task_r_square = kFold_validation(task_group[group_no],
                                                                                                random_seed,
                                                                                                task_type)


                            for key, val in task_score.items():
                                task_score[key] = np.mean(val)

                            # total_loss_per_task_group_per_fold = loss

                        # if group_no not in new_group_score.keys():
                        #     new_group_score.update({group_no: total_loss_per_task_group_per_fold})
                        # else:
                        #     new_group_score[group_no] = total_loss_per_task_group_per_fold

                        if dataset == 'Chemical':
                            Prev_Groups[grouping_sorted] = (total_loss_per_task_group_per_fold, avg_error)
                        if dataset == 'Landmine':
                            Prev_Groups[grouping_sorted] = (total_loss_per_task_group_per_fold, mean_auc)

                        if dataset == 'Parkinsons':
                            Prev_Groups[grouping_sorted] = (total_loss_per_task_group_per_fold, mean_EV, mean_r_square)

                    else:
                        print(f'Prev_Groups[grouping_sorted] = {Prev_Groups[grouping_sorted]}')
                        if dataset == 'School':
                            total_loss_per_task_group_per_fold = Prev_Groups[grouping_sorted]
                        else:
                            total_loss_per_task_group_per_fold = Prev_Groups[grouping_sorted][0]

                    if group_no not in new_group_score.keys():
                        new_group_score.update({group_no: total_loss_per_task_group_per_fold})
                    else:
                        new_group_score[group_no] = total_loss_per_task_group_per_fold

            final_time = time.time() - timeStart
            mtl_solution = sum(list(new_group_score.values()))

            perc = (first_sol - prev_solution) / first_sol
            if perc > 0.10:
                K += K * 0.25
                first_sol = prev_solution
            Accept_Probability = min(1, math.exp((prev_solution - mtl_solution) * K))
            PROBABILITIES_ACCEPT = [Accept_Probability, max(0, 1 - Accept_Probability)]
            final_accept = np.random.choice(switch_architecture, 1, p=PROBABILITIES_ACCEPT)
            print(f'mtl_solution = {mtl_solution}\tfinal_time = {final_time}\tfinal_accept = {final_accept[0]}\t')

            for g, s in new_group_score.items():
                if s == 0:
                    print(f'new_group_score = {new_group_score}')
                    print(f'Exiting because of error')
                    exit(0)
                    break

            if len(new_task_group.keys()) != len(new_group_score.keys()):
                print(f'new_group_score = {new_group_score}\nnew_task_group = {new_task_group}')
                print(f'Exiting because of key-mismatch error')
                exit(0)
                break

            if final_accept[0] == 'ACCEPT':
                group_score = copy.deepcopy(new_group_score)
                task_group = copy.deepcopy(new_task_group)

                Switch.append('yes')
                prev_solution = mtl_solution
                Prev_Solution.append(prev_solution)
                Prev_iter.append(iteration_MTL)
                last_iter = iteration_MTL

            else:
                Switch.append('no')
                Prev_Solution.append(prev_solution)
                Prev_iter.append(last_iter)

            Training_Time.append(final_time)
            Iteration_MTL.append(iteration_MTL)
            Random_Task.append(task_rand)
            Total_Loss.append(mtl_solution)
            Task_group.append(new_task_group)
            Number_of_Groups.append(len(new_task_group))
            Changed_Group.append(changed_group)
            Predicted_Score.append(predicted_solution)
            Fast_Reject.append(fast_reject[0])
            Individual_Group_Score.append(new_group_score)
            Number_of_MTL_Training.append(count)
            if dataset == 'Chemical':
                Classification_Error.append(avg_error)
            if dataset == 'Landmine':
                AUC.append(mean_auc)
            if dataset == 'Parkinsons':
                R_Squared.append(r_square)
                Explained_Variance.append(ev)
            # Average_Precision.append(ap)

            length = len(Switch)
            print(f'Length = {length}')

        length = len(Switch)
        tail_pointer = 5
        if length > old_length and length % tail_pointer == 0:
            print(len(Switch), len(Iteration_MTL), len(Prev_Solution), len(Changed_Group), len(Total_Loss),
                  len(Number_of_Groups), len(Prev_iter), len(Task_group), len(Individual_Group_Score), len(Random_Seed),
                  len(Individual_Task_Score), len(Predicted_Score), len(TRAIN_Model), len(Fast_Reject),
                  len(Training_Time))
            results = pd.DataFrame({'Iteration_MTL': Iteration_MTL,
                                    'Random_Task': Random_Task,
                                    'ACCEPT': Switch,
                                    'Changed_Groups': Changed_Group,
                                    'Total_Loss': Total_Loss,
                                    'Prev_Solution': Prev_Solution,
                                    'Training_Time': Training_Time,
                                    'Number_of_Groups': Number_of_Groups,
                                    'Number_of_MTL_Training': Number_of_MTL_Training,
                                    'Fast_Reject': Fast_Reject,
                                    'Task_group': Task_group,
                                    'Individual_Group_Score': Individual_Group_Score,
                                    # 'Classification_Error': Classification_Error,
                                    # 'Average_Precision': Average_Precision,
                                    'Predicted_Score': Predicted_Score,
                                    "Last_switch_at_iter": Prev_iter})
            if dataset == 'Chemical':
                results['Individual_Error_Rate'] = Classification_Error
            if dataset == 'Landmine':
                results['Individual_AUC'] = AUC
            if dataset == 'Parkinsons':
                results['Individual_EV'] = Explained_Variance
                results['Individual_r_square'] = R_Squared
            results.to_csv(f'{run_results}/{dataset}_Task_Grouping_Results_{length}_run_{run}_{ModelName}.csv',
                           index=False)

            old_length = len(Switch)

            new_results = pd.read_csv(f'{run_results}/{dataset}_Task_Grouping_Results_{length}_run_{run}_{ModelName}.csv',
                                      low_memory=False)
            new_results = new_results.tail(tail_pointer).reset_index(drop=True)

            predictor_data_prep(new_results, 1, run, dataset)
            retrain_predictor(dataset,affinity_pred_arch)

            # if length > 5:
            length = length - tail_pointer
            old_file = f'{run_results}/{dataset}_Task_Grouping_Results_{length}_run_{run}_{ModelName}.csv'
            if os.path.exists(old_file):
                os.remove(os.path.join(f'{run_results}/{dataset}_Task_Grouping_Results_{length}_run_{run}_{ModelName}.csv'))

        if np.sum(Number_of_MTL_Training) >= 800:
            break
    print(len(Switch), len(Random_Task), len(Iteration_MTL), len(Prev_Solution), len(Changed_Group), len(Total_Loss),
          len(Number_of_Groups), len(Prev_iter), len(Task_group), len(Individual_Group_Score), len(Random_Seed),
          len(Individual_Task_Score))
    results = pd.DataFrame({'Iteration_MTL': Iteration_MTL,
                            'Random_Task': Random_Task,
                            'ACCEPT': Switch,
                            'Changed_Groups': Changed_Group,
                            'Total_Loss': Total_Loss,
                            'Prev_Solution': Prev_Solution,
                            'Training_Time': Training_Time,
                            'Number_of_Groups': Number_of_Groups,
                            'Number_of_MTL_Training': Number_of_MTL_Training,
                            'Fast_Reject': Fast_Reject,
                            'Task_group': Task_group,
                            'Individual_Group_Score': Individual_Group_Score,
                            # 'Classification_Error': Classification_Error,
                            # 'Average_Precision': Average_Precision,
                            'Predicted_Score': Predicted_Score,
                            "Last_switch_at_iter": Prev_iter})

    if dataset == 'Chemical':
        results['Individual_Error_Rate'] = Classification_Error
    if dataset == 'Landmine':
        results['Individual_AUC'] = AUC
    if dataset == 'Parkinsons':
        results['Individual_EV'] = Explained_Variance
        results['Individual_r_square'] = R_Squared
    results.to_csv(f'{run_results}/{dataset}_Task_Grouping_Results_run_{run}_{ModelName}.csv',
                   index=False)

