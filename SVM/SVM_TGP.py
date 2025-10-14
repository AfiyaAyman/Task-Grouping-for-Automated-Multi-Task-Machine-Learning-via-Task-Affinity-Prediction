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
from multiprocessing.pool import ThreadPool

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
# from Predictor_Class import TaskGroupingPredictor

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
    # pool = mp.Pool(number_of_pools)
    # all_scores = pool.starmap(regularized_mtl_NonLinearSVM, args)
    # pool.close()
    time_start = time.time()
    with ThreadPool(number_of_pools) as tp:
        all_scores = tp.starmap(regularized_mtl_NonLinearSVM, args)
    tp.join()

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


def random_task_grouping(task_Set, min_task_groups):
    task_group = {}
    total_dataset = len(task_Set)
    group = 0
    while total_dataset > 0 and min_task_groups > 0:
        team = random.sample(task_Set, int(total_dataset / min_task_groups))
        task_group.update({group: team})
        group += 1
        for x in team:
            task_Set.remove(x)
        total_dataset -= int(total_dataset / min_task_groups)
        min_task_groups -= 1

    return task_group


def mutate_groups(new_task_group, new_group_score):
    task_rand = random.sample(TASKS, 1)
    task_rand = task_rand[0]
    changed_group = []

    # find out old group
    for key, task_list in new_task_group.items():
        if task_rand in task_list:
            g_old = key

    # check if old group is empty->delete the old group and assign task to new group
    # print(f'task_rand = {task_rand}\tg_old = {g_old}\tTask-Group = {new_task_group}')

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

    return changed_group, task_rand


def predictor_network(x_train, y_train,network_architecture):
    filepath = f'{run_results}/SavedModels/{dataset}_TG_predictor_Best_{ModelName}.h5'

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

    if np.shape(x_train)[0] < 1500:
        batch_size = 264
        number_of_epoch = 300
    else:
        batch_size = 64
        number_of_epoch = 200

    history = finalModel.fit(x=x_train,
                             y=y_train,
                             shuffle=True,
                             epochs=number_of_epoch,
                             batch_size=batch_size,
                             callbacks=checkpoint,
                             verbose=0)


def predict_performance_of_new_group(tasks,datasetName, affinity_pred_arch):
    if len(tasks) > 1:
        number_of_tasks = []
        pairwise_improvement_average = []
        pairwise_improvement_variance = []
        pairwise_improvement_stddev = []
        group_variance = []
        group_stddev = []
        group_distance = []
        group_dataset_size = []

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

            if datasetName == 'Chemical':
                avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
            else:
                avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])

        group_distance.append(np.mean(avg_dist))
        paired_improvement = []

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
        })
        pred_features = affinity_pred_arch['Features']
        new_groups = new_groups[pred_features]
        x = np.array(new_groups, dtype='float32')
        filepath = f'{run_results}/SavedModels/{datasetName}_TG_predictor_Best_{ModelName}.h5'
        finalModel = tf.keras.models.load_model(filepath)
        final_score = finalModel.predict(x, verbose=0)
        return final_score[0][0], single_task_total_loss
    else:
        return 0, Single_res_dict[tasks[0]]


def retrain_predictor(datasetName, affinity_pred_arch):
    predictor_data = pd.read_csv(f'{run_results}/Data_for_Predictor_{datasetName}_updated_{ModelName}.csv')
    print(f'\n\n******* Training Samples = {len(predictor_data)} *******\n\n')
    predictor_data.dropna(inplace=True)

    pred_features = affinity_pred_arch['Features']

    y_train = np.array(list(predictor_data.change), dtype=float)
    predictor_data = predictor_data[pred_features]

    x_train = np.array(predictor_data, dtype=float)

    predictor_network(x_train, y_train, affinity_pred_arch)


def Initial_Training(k, dataset, group_range, best_hp, TASKS):
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
    for count in range(0, k):
        min_task_groups = random.randint(group_range[0], group_range[1])
        print(f'Initial Training for {dataset}-partition {count}')
        task_Set = copy.deepcopy(TASKS)
        print(f'Number of tasks = {len(task_Set)}\n task_Set = {task_Set}')
        task_group = random_task_grouping(task_Set, min_task_groups)
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
                    avg_error = STL_error[task[0]]

                else:
                    if dataset == 'Chemical':
                        loss, task_score, avg_error, task_error_rate = kFold_validation(task_group[group_no],
                                                                                        random_seed, best_hp,
                                                                                        dataset)
                        print(f'\n\n\n******{dataset}*****\n')
                        print(f'loss = {loss}\n'
                              f'task_score = {task_score}\n'
                              f'error = {error}\n'
                              f'task_error_rate = {task_error_rate}')

                    if dataset == 'Landmine':
                        loss, task_score, mean_auc, task_auc = kFold_validation(task_group[group_no], random_seed,
                                                                                best_hp, dataset)
                        print(f'\n\n\n******{dataset}*****\n')
                        print(f'loss = {loss}\n'
                              f'task_score = {task_score}\n'
                              f'AUC = {mean_auc}\n'
                              f'task_AUC = {task_auc}')

                    if dataset == 'School':
                        loss, task_score = kFold_validation(task_group[group_no], random_seed, best_hp, dataset)
                        print(f'\n\n\n******{dataset}*****\n')
                        print(f'loss = {loss}\n'
                              f'task_score = {task_score}')

                    if dataset == 'Parkinsons':
                        loss, task_score, mean_EV, task_EV, mean_r_square, task_r_square = kFold_validation(
                            task_group[group_no],
                            random_seed,
                            best_hp,
                            dataset)
                        print(f'\n\n\n******{dataset}*****\n')
                        print(f'loss = {loss}\n'
                              f'task_score = {task_score}\n'
                              f'EV = {mean_EV}\n'
                              f'task_EV = {task_EV}\n'
                              f'r_square = {mean_r_square}\n'
                              f'task_r_square = {task_r_square}')

                    for k, v in task_score.items():
                        task_score[k] = np.mean(v)

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
        print(Individual_Group_Score)

    print(len(Total_Loss), len(Number_of_Groups), len(Task_group), len(Individual_Group_Score))
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

    initial_results.to_csv(f'{ResultPath}/{dataset}_Initial_Task_Grouping_Results_{ModelName}.csv',
                           index=False)
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
    if dataset =='Chemical':
        task_info = pd.read_csv(f'{DataPath}Task_Information_{dataset_name}_SVM.csv')
    else:
        task_info = pd.read_csv(f'{DataPath}Task_Information_{dataset_name}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{dataset_name}.csv')
    single_results = pd.read_csv(f'../Results/STL/STL_{dataset_name}_{ModelName}.csv')

    pair_results = pd.read_csv(
        f'../Results/Pairwise/{ModelName}/{dataset_name}_Results_from_Pairwise_Training_ALL_{ModelName}.csv')

    for selected_task in TASKS:
        if dataset == 'Chemical':
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


    change = []

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

                        if dataset == 'Chemical':
                            avg_dist.append(task_dist_data.Average_Hamming_Distance[0])
                        else:
                            avg_dist.append(task_dist_data.Average_Euclidean_Distance[0])
                    # group_distance.append(np.mean(avg_dist))

                    paired_improvement = []
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
    run = sys.argv[2]

    # dataset = 'Chemical'
    ModelName = 'SVM'
    # run = 1

    print(f'dataset = {dataset}, ModelName = {ModelName}')

    DataPath = f'../Dataset/{dataset.upper()}/'

    ResultPath = '../Results'
    run_results = f'../Results/Run_{run}'
    if not os.path.exists(run_results):
        os.mkdir(run_results)

    num_folds = 10
    number_of_epochs = 100
    max_iter = 3000

    Variance = []
    Random_Seed = []
    StdDev = []
    Accuracy = []
    Error_Rate = []

    task_indi_score = {}
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
    hyperparameters_RMTL = hyperparameters_RMTL_dict[dataset]

    task_len = {}
    variance_dict = {}
    std_dev_dict = {}
    dist_dict = {}
    Single_res_dict = {}
    STL_error = {}
    STL_AUC = {}
    STL_EV = {}
    STL_r_square = {}
    if dataset == 'Chemical':
        task_info = pd.read_csv(f'{DataPath}Task_Information_{dataset}_SVM.csv')
    else:
        task_info = pd.read_csv(f'{DataPath}Task_Information_{dataset}.csv')
    task_distance_info = pd.read_csv(f'{DataPath}Task_Distance_{dataset}.csv')

    single_results = pd.read_csv(f'{ResultPath}/STL/STL_{dataset}_SVM.csv')
    # pair_results = pd.read_csv(f'{ResultPath}/Pairwise/{ModelName}/{dataset}_Results_from_Pairwise_Training_ALL_{ModelName}.csv')
    pair_results = pd.read_csv(f'../Results/new_runs/SVM/{dataset}_Results_from_Pairwise_Training_ALL_{ModelName}.csv')
    print(f'single_results = {single_results.columns}')

    print(single_results.columns)
    print(pair_results.columns)

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
        'School': {'FF_Layers': 2, 'FF_Neurons': [10, 17], 'learning_rate': 0.0015235747908138567, 'activation_function': 'relu', 'output_activation': 'linear',
                   'Features': ['pairwise_improvement_average', 'pairwise_improvement_stddev']},
        'Chemical': {'FF_Layers': 2, 'FF_Neurons': [58, 109], 'learning_rate': 0.0015083390429057183, 'activation_function': 'relu', 'output_activation': 'linear',
                      'Features': ['pairwise_improvement_average', 'pairwise_improvement_stddev']},
        'Landmine': {'FF_Layers': 2, 'FF_Neurons': [26, 30], 'learning_rate': 0.0015235747908138567, 'activation_function': 'relu', 'output_activation': 'linear',
                     'Features': ['pairwise_improvement_average', 'pairwise_improvement_variance']},
        'Parkinsons': {'FF_Layers': 2, 'FF_Neurons': [123, 67], 'learning_rate': 0.001234095580559224, 'activation_function': 'relu', 'output_activation': 'linear',
                       'Features': ['pairwise_improvement_average', 'pairwise_improvement_variance']}}  # This is the architecture of the affinity prediction model

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
        error_rate = ast.literal_eval(initial_results.Individual_Error_Rate[best_group_index])
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
    # K_dict = {'Chemical': 120, 'Landmine': 25, 'Parkinsons': 0.99, 'School': 20}
    K_dict = {'Chemical': 26, 'Landmine': 85, 'Parkinsons': 1.5, 'School': 5}
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
            prediction_of_performance = predict_performance_of_new_group(new_task_group[group_no], dataset,affinity_pred_arch)
            predicted_performance[group_no] = prediction_of_performance[0]
            single_loss_dict[group_no] = prediction_of_performance[1]

        for group_no, task in new_task_group.items():
            if group_no in changed_group:
                total_loss_per_task_group_per_fold = single_loss_dict[group_no] * (
                        1 - predicted_performance[group_no])
                predicted_group_score[group_no] = total_loss_per_task_group_per_fold
        # print(f'Time to query predictor = {(time.time() - time_query_Start) / 60} min')
        predicted_solution = sum(list(predicted_group_score.values()))

        # K = 70
        # Accept_Probability = min(1, math.exp((prev_solution - predicted_solution) * K))
        # PROBABILITIES_ACCEPT = [Accept_Probability, max(0, 1 - Accept_Probability)]
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
                                error = STL_error[task[0]]
                            if dataset == 'Landmine':
                                auc = STL_AUC[task[0]]
                            if dataset == 'Parkinsons':
                                r2 = STL_r_square[task[0]]
                                ev = STL_EV[task[0]]
                            # ap = STL_AP[task[0]]
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
                            count += 1
                            if dataset == 'Chemical':
                                total_loss_per_task_group_per_fold, task_score, avg_error, task_error_rate = kFold_validation(task_group[group_no],
                                                                                                random_seed,
                                                                                                hyperparameters_RMTL,
                                                                                                dataset)
                                # print(f'\n\n\n******{dataset}*****\n')
                                # print(f'loss = {loss}\n'
                                #       f'task_score = {task_score}\n'
                                #       f'error = {avg_error}\n'
                                #       f'task_error_rate = {task_error_rate}')

                            if dataset == 'Landmine':
                                total_loss_per_task_group_per_fold, task_score, mean_auc, task_auc = kFold_validation(task_group[group_no],
                                                                                        random_seed,
                                                                                        hyperparameters_RMTL, dataset)
                                # print(f'\n\n\n******{dataset}*****\n')
                                # print(f'loss = {loss}\n'
                                #       f'task_score = {task_score}\n'
                                #       f'AUC = {mean_auc}\n'
                                #       f'task_AUC = {task_auc}')

                            if dataset == 'School':
                                total_loss_per_task_group_per_fold, task_score = kFold_validation(task_group[group_no], random_seed,
                                                                    hyperparameters_RMTL, dataset)
                                # print(f'\n\n\n******{dataset}*****\n')
                                # print(f'loss = {loss}\n'
                                #       f'task_score = {task_score}')

                            if dataset == 'Parkinsons':
                                total_loss_per_task_group_per_fold, task_score, mean_EV, task_EV, mean_r_square, task_r_square = kFold_validation(
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

                            for k, v in task_score.items():
                                task_score[k] = np.mean(v)

                            # total_loss_per_task_group_per_fold = loss

                        if group_no not in new_group_score.keys():
                            new_group_score.update({group_no: total_loss_per_task_group_per_fold})
                        else:
                            new_group_score[group_no] = total_loss_per_task_group_per_fold

                        if dataset == 'Chemical':
                            Prev_Groups[grouping_sorted] = (total_loss_per_task_group_per_fold, avg_error)
                        if dataset == 'Landmine':
                            Prev_Groups[grouping_sorted] = (total_loss_per_task_group_per_fold, mean_auc)

                        if dataset == 'Parkinsons':
                            Prev_Groups[grouping_sorted] = (total_loss_per_task_group_per_fold, mean_EV, mean_r_square)

                    else:
                        print(f'Prev_Groups[grouping_sorted] = {Prev_Groups[grouping_sorted]}')
                        if dataset == 'School':
                            loss = Prev_Groups[grouping_sorted]
                        else:
                            loss = Prev_Groups[grouping_sorted][0]
                        if group_no not in new_group_score.keys():
                            new_group_score.update({group_no: loss})
                        else:
                            new_group_score[group_no] = loss

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
                Classification_Error.append(error)
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


            predictor_data_prep(new_results,1,run,dataset)
            retrain_predictor(dataset, affinity_pred_arch)

            # if length > 5:
            length = length - tail_pointer
            old_file = f'{run_results}/{dataset}_Task_Grouping_Results_{length}_run_{run}_{ModelName}.csv'
            if os.path.exists(old_file):
                os.remove(os.path.join(f'{run_results}/{dataset}_Task_Grouping_Results_{length}_run_{run}_{ModelName}.csv'))

        if dataset == 'School':
            if np.sum(Number_of_MTL_Training) >= 1500:
                break
        else:
            if np.sum(Number_of_MTL_Training) >= 1000:
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