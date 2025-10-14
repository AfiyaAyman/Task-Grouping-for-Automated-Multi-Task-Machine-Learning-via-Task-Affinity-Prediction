import sys
import pandas as pd
import numpy as np
import os
import time
import ast
import tqdm
import itertools
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import multiprocessing as mp
from sklearn.model_selection import KFold
from multiprocessing.pool import ThreadPool
import random
import copy
import math
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from sklearn.metrics import mean_squared_error,r2_score

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

def predictor_NAS(network_architecture, x_train, x_test, y_train, y_test, fold):

    filepath = f'SavedModels/{dataset_name}_Group_predictor_ALL_{fold}_{modelName}.h5'
    number_of_features = np.shape(x_train)[1]

    Input_FF = tf.keras.layers.Input(shape=(number_of_features,))
    hidden_FF = tf.keras.layers.Dense(network_architecture['FF_Neurons'][0],
                                      activation=network_architecture['activation_function'])(Input_FF)
    for h in range(1, network_architecture['FF_Layers']):
        hidden_FF = tf.keras.layers.Dense(network_architecture['FF_Neurons'][h],
                                          activation=network_architecture['activation_function'])(hidden_FF)

    output = tf.keras.layers.Dense(1, activation=network_architecture['output_activation'])(hidden_FF)

    finalModel = Model(inputs=Input_FF, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=network_architecture['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')

    from keras.utils.layer_utils import count_params
    trainable_count = count_params(finalModel.trainable_weights)

    checkpoint = ModelCheckpoint(filepath, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')

    batch_size = 128
    number_of_epoch = 200
    finalModel.fit(x=x_train,
                   y=y_train,
                   shuffle=True,
                   epochs=number_of_epoch,
                   batch_size=batch_size,
                   validation_data=(x_test,
                                    y_test),
                   callbacks=checkpoint,
                   verbose=0)

    finalModel = tf.keras.models.load_model(filepath)
    y_pred = finalModel.predict(x_test, verbose=0)
    r_square = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    if os.path.exists(filepath):
        os.remove(os.path.join(filepath))

    return r_square, trainable_count, mse


def kFold_validation(Sample_Inputs, Sample_Label, best_nn_architecture, random_seed):
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    args = []
    fold = 0
    for train, test in kfold.split(Sample_Inputs):
        FF_targets_train = Sample_Label[train]
        FF_targets_test = Sample_Label[test]
        FF_inputs_train = Sample_Inputs[train]
        FF_inputs_test = Sample_Inputs[test]
        tmp = (best_nn_architecture, FF_inputs_train, FF_inputs_test, FF_targets_train, FF_targets_test,
               fold)
        args.append(tmp)

        fold += 1

    # print(len(ALL_FOLDS))
    # if mp.get_start_method() != 'fork':
    #     mp.set_start_method('fork', force=True)
    #
    number_of_pools = len(args)+10
    pool = mp.Pool(number_of_pools)
    SCORES = pool.starmap(predictor_NAS, args)
    pool.close()

    # with ThreadPool(10) as tp:
    #     SCORES = tp.starmap(predictor_NAS, args)
    # tp.join()
    # print(f' SCORES = {SCORES}')

    Scores = []
    mse_score =  []
    for s in SCORES:
        Scores.append(s[0])
        mse_score.append(s[2])
    trainable_param = SCORES[0][1]

    # print(f'Scores = {Scores}')
    # print(f'trainable_param = {trainable_param}')

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print(f'Average R-sq: {np.mean(Scores)}')
    print(f'Average MSE: {np.mean(mse_score)}')

    # '''USEFULNESS SCORE'''
    # print(f'NAS SCORE: {np.mean(Scores) - (ALPHA * trainable_param)}')
    # print('-----------------------------------------------------------')
    Scores_all_Folds.append(Scores)

    return np.mean(Scores), np.var(Scores), trainable_param, np.mean(mse_score)


def random_neighbour(current_nn_architecture):
    # NEURON_CHANGE_PERC = random.randint(-30, +30)
    shared_hp = list(current_nn_architecture.keys())

    random_hp = np.random.choice(shared_hp, 1, p=PROBABILITIES_RANDOM_NEIGHBOUR)

    current_layers = current_nn_architecture['FF_Layers']
    layers_FF = [item for item in range(0, current_layers)]
    random_hp = random_hp[0]


    if random_hp == 'Features':
        current_featureset = current_nn_architecture['Features']
        remaining_features = list(set(entire_featureSet) - set(current_featureset))

        add_remove = np.random.choice(['add', 'remove'], 1, p=[0.5, 0.5])
        if len(current_featureset)==0:
            add_remove = 'add'
        if len(remaining_features)==0:
            add_remove = 'remove'
        if add_remove == 'add':
            add_feature = np.random.choice(remaining_features)
            current_featureset.append(add_feature)
        else:
            remove_feature = np.random.choice(current_featureset)
            current_featureset.remove(remove_feature)

        # print(f'feature {add_remove}ed, current_featureset = {current_featureset}')

        current_nn_architecture['Features'] = current_featureset


    elif random_hp == 'FF_Neurons':
        pick_hidden_layer = np.random.choice(layers_FF)
        neurons = current_nn_architecture['FF_Neurons'][pick_hidden_layer]
        current_nn_architecture['FF_Neurons'][pick_hidden_layer] = neurons + math.ceil(
            neurons * NEURON_CHANGE_PERC / 100)

    elif random_hp == 'FF_Layers':
        Highest_neurons = 2 * math.ceil(np.mean(current_nn_architecture['FF_Neurons']))
        neurons = random.randint(1, Highest_neurons)
        if current_nn_architecture['FF_Layers'] == 1:
            current_nn_architecture['FF_Layers'] = current_layers + 1
            current_nn_architecture['FF_Neurons'].extend([neurons])
        else:
            change_layers = np.random.choice(INCREASE_DECREASE_HP_VALUE)
            if change_layers == 'DECREASE':
                current_nn_architecture['FF_Layers'] = current_layers - 1
                pick_hidden_layer = np.random.choice(layers_FF)
                neurons_to_remove = current_nn_architecture['FF_Neurons'][pick_hidden_layer]
                current_nn_architecture['FF_Neurons'].remove(neurons_to_remove)
            else:
                current_nn_architecture['FF_Layers'] = current_layers + 1
                current_nn_architecture['FF_Neurons'].extend([neurons])

    elif random_hp == 'activation_function':
        get_current_activation_function = current_nn_architecture['activation_function']
        tmp_activation = candidate_activation_hidden.copy()
        tmp_activation.remove(get_current_activation_function)
        new_activation_function = np.random.choice(tmp_activation)
        current_nn_architecture['activation_function'] = new_activation_function

    elif random_hp == 'output_activation':
        get_current_activation_function = current_nn_architecture['output_activation']
        tmp_activation = candidate_activation_output.copy()
        tmp_activation.remove(get_current_activation_function)
        new_activation_function = np.random.choice(tmp_activation)
        current_nn_architecture['output_activation'] = new_activation_function

    else:
        #learning_rate
        change = np.random.choice(INCREASE_DECREASE_HP_VALUE)
        if change == 'INCREASE':
            current_nn_architecture['learning_rate'] = current_nn_architecture['learning_rate'] + (
                    current_nn_architecture['learning_rate'] * LEARNING_RATE_CHANGE_PERC)

        else:  # 'DECREASE'
            current_nn_architecture['learning_rate'] = current_nn_architecture['learning_rate'] - (
                    current_nn_architecture['learning_rate'] * LEARNING_RATE_CHANGE_PERC)

    print(f'Final = {current_nn_architecture}')
    return current_nn_architecture, random_hp


def random_search():
    ALPHA = 0.0001
    Prev_Arch = []
    current_best_nn_architecture = network_architecture
    current_best_featureset = network_architecture['Features']

    predictor_data = entire_predictor_data[current_best_featureset]
    Sample_Inputs = np.array(predictor_data)
    print(
        f'initial_featureset = {initial_featureset}, shape of Sample_Inputs = {np.shape(Sample_Inputs)}, shape of Sample_Label = {np.shape(Sample_Label)}')

    print(f'START_ARCHITECTURE = {network_architecture}')
    print(f'Initial feature Set = {initial_featureset}')

    random_seed = random.randint(0, 100)
    Prev_Arch.append(current_best_nn_architecture.copy())
    current_solution, variance, trainable_param, mse = kFold_validation(Sample_Inputs, Sample_Label,
                                                                   current_best_nn_architecture, random_seed)
    # print(f'current_solution = {current_solution}')
    # print(f'Trainable Parameter = {trainable_param}')

    best_solution = current_solution

    Iteration.append(0)
    Usefulness_Score.append(current_solution)
    current_solution = current_solution# - ALPHA * trainable_param
    print(f'current_solution = {current_solution}')
    kfold_MSE.append(mse)
    NAS_SCORE.append(current_solution)
    Variance.append(variance)
    ARCHITECTURE.append(current_best_nn_architecture.copy())
    prev_solution = current_solution
    Prev_Solution.append(prev_solution)
    Random_Seed.append(random_seed)
    Prev_iter.append(0)
    Switch.append('None')
    Tweaked_HP.append('None')
    Features.append(current_best_featureset)
    Number_of_Features.append(len(current_best_featureset))

    # Keep searching until reach max evaluations
    for iteration in range(1, MAX_EVALS):
        current_nn_architecture = copy.deepcopy(current_best_nn_architecture)
        # current_featureset = copy.deepcopy(current_best_featureset)
        current_nn_architecture,random_hp = random_neighbour(current_nn_architecture)
        count_arch = 0
        while current_nn_architecture in Prev_Arch:
            current_nn_architecture,random_hp = random_neighbour(current_nn_architecture)
            count_arch += 1
            if count_arch >= 20:
                break
        Prev_Arch.append(current_nn_architecture.copy())
        Tweaked_HP.append(random_hp)
        current_featureset = current_nn_architecture['Features']

        predictor_data = entire_predictor_data[current_featureset]
        Sample_Inputs = np.array(predictor_data)
        print(
            f'new_number of features = {len(current_featureset)}, shape of Sample_Inputs = {np.shape(Sample_Inputs)}, shape of Sample_Label = {np.shape(Sample_Label)}')

        random_seed = random.randint(0, 100)
        timeStart = time.time()
        current_solution, variance, trainable_param,mse = kFold_validation(Sample_Inputs, Sample_Label,
                                                                       current_nn_architecture, random_seed)
        print(f'total time = {time.time() - timeStart}')

        Iteration.append(iteration)
        Usefulness_Score.append(current_solution)
        '''USEFULNESS SCORE'''
        kfold_MSE.append(mse)
        current_solution = current_solution# - ALPHA * trainable_param
        NAS_SCORE.append(current_solution)
        Variance.append(variance)
        ARCHITECTURE.append(current_nn_architecture.copy())
        Random_Seed.append(random_seed)
        Features.append(current_featureset)
        Number_of_Features.append(len(current_featureset))

        # K = 30
        # Accept_Probability = min(1, math.exp((prev_solution - current_solution) * K))
        # PROBABILITIES_ACCEPT = [Accept_Probability, max(0, 1 - Accept_Probability)]
        if current_solution > prev_solution:
            PROBABILITIES_ACCEPT = [1, 0]
        else:
            diff = abs((prev_solution - current_solution) / prev_solution)
            if diff>0.01 and diff<0.02:
                PROBABILITIES_ACCEPT = [0.02, 0.98]
            else:
                PROBABILITIES_ACCEPT = [0, 1]

        switch_parameter = np.random.choice(switch_architecture, 1, p=PROBABILITIES_ACCEPT)

        if switch_parameter == 'SWITCH':
            if current_solution < best_solution:
                best_solution = current_solution
                current_best_nn_architecture = copy.deepcopy(current_nn_architecture)
            else:
                current_best_nn_architecture = copy.deepcopy(current_nn_architecture)

            Switch.append('yes')
            prev_solution = current_solution
            Prev_Solution.append(prev_solution)
            Prev_iter.append(iteration)
            print('\n\n******SWITCH******')

        else:
            Switch.append('no')
            Prev_Solution.append(prev_solution)
            Prev_iter.append(Prev_iter[-1])
            print('\n\n******NOT SWITCH******')

        length = len(Usefulness_Score)
        tail_pointer = 5
        if length % tail_pointer == 0:
            print(len(Switch), len(Iteration), len(Scores_all_Folds), len(Tweaked_HP), len(Prev_Solution),
                  len(Iteration), len(Usefulness_Score), len(NAS_SCORE), len(ARCHITECTURE), len(Prev_iter), len(Variance),
                  len(Random_Seed))
            # NAS_SCORE = Usefulness_Score + ALPHA* Total_Trainable_param
            results = pd.DataFrame({'Iteration': Iteration,
                                    'Switch': Switch,
                                    'Tweaked_HP': Tweaked_HP,
                                    'Usefulness_Score': Usefulness_Score,
                                    'NAS_SCORE': NAS_SCORE,
                                    'Prev_Solution': Prev_Solution,
                                    'Kfold_MSE': kfold_MSE,
                                    'Features': Features,
                                    'Number_of_Features': Number_of_Features,
                                    'ARCHITECTURE': ARCHITECTURE,
                                    "Last_switch_at_iter": Prev_iter,
                                    "Scores_all_Folds": Scores_all_Folds,
                                    'Variance': Variance,
                                    "Random_Seed": Random_Seed})
            results.to_csv(f'{ResultPath}/{dataset_name}_Groupwise_NAS_{length}_ALL_{modelName}.csv',
                           index=False)

            if length > tail_pointer:
                length = length - tail_pointer
                if os.path.exists(os.path.join(f'{ResultPath}/{dataset_name}_Groupwise_NAS_{length}_ALL_{modelName}.csv')):
                    os.remove(
                        os.path.join(f'{ResultPath}/{dataset_name}_Groupwise_NAS_{length}_ALL_{modelName}.csv'))

        else:
            continue

    print(len(Switch), len(Iteration), len(Scores_all_Folds), len(Tweaked_HP), len(Prev_Solution), len(Iteration),
          len(Usefulness_Score), len(NAS_SCORE), len(ARCHITECTURE), len(Prev_iter), len(Variance), len(Random_Seed))

    results = pd.DataFrame({'Iteration': Iteration,
                            'Switch': Switch,
                            'Tweaked_HP': Tweaked_HP,
                            'Usefulness_Score': Usefulness_Score,
                            'NAS_SCORE': NAS_SCORE,
                            'Prev_Solution': Prev_Solution,
                            'Kfold_MSE': kfold_MSE,  # 'NAS_SCORE': NAS_SCORE,
                            'Features': Features,
                            'Number_of_Features': Number_of_Features,
                            'ARCHITECTURE': ARCHITECTURE,

                            "Last_switch_at_iter": Prev_iter,
                            "Scores_all_Folds": Scores_all_Folds,
                            'Variance': Variance,
                            "Random_Seed": Random_Seed})

    results.to_csv(f'{ResultPath}/{dataset_name}_Groupwise_NAS_ALL_{modelName}.csv',
                   index=False)


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    modelName = sys.argv[2]

    # dataset_name = 'School'
    # modelName = 'NN'

    ResultPath = '../Results/NAS_Results/'

    if dataset_name == 'School':
        initial_featureset = ['pairwise_improvement_average', 'group_dataset_size',
                              'pairwise_improvement_variance',
                                'group_variance']
    if dataset_name == 'Chemical':
        initial_featureset = ['group_dataset_size',
                               'pairwise_improvement_average','pairwise_improvement_stddev',
                                    'group_variance']
    if dataset_name == 'Landmine':
        initial_featureset =  ['pairwise_improvement_average',
                               'group_variance', 'number_of_tasks']
    if dataset_name == 'Parkinsons':
        initial_featureset = ['pairwise_improvement_average',
                              'pairwise_improvement_stddev',
                                'group_variance','group_distance']


    DataPath = f'../Results/partition_sample/{modelName}/group_sample/train/'
    entire_predictor_data = pd.read_csv(f'{DataPath}Groupwise_Features_{dataset_name}_train.csv')

    print(f'\n\n******* Training Samples = {len(entire_predictor_data)} *******\n\n')
    print(f'shape of entire data = {np.shape(entire_predictor_data)}')
    # print(f'predictor_data.columns = {entire_predictor_data.columns}')

    entire_predictor_data.dropna(inplace=True)

    # !/bin/bash


    entire_featureSet = list(entire_predictor_data.columns)
    entire_featureSet.remove('change')

    target = entire_predictor_data.pop('change')
    Sample_Label = np.array(target)
    print(f'entire_featureSet = {entire_featureSet}')
    # Sample_Inputs = np.array(predictor_data)


    Usefulness_Score = []
    kfold_MSE = []
    NAS_SCORE = []
    Iteration = []
    ARCHITECTURE = []
    Features = []
    Number_of_Features = []
    Prev_Solution = []
    Random_Seed = []
    Prev_iter = []
    Variance = []
    Tweaked_HP = []
    Switch = []
    network_architecture = {'FF_Layers': 2,
                            'FF_Neurons': [50,25],
                            'learning_rate': 0.0013850679916489607,
                            'activation_function': 'sigmoid',
                            'output_activation': 'linear',
                            'Features': initial_featureset}

    PROBABILITIES_RANDOM_NEIGHBOUR = [0.25, 0.4, 0.04, 0.03, 0.03, 0.25]
    # PROBABILITIES_RANDOM_NEIGHBOUR = [0, 0,0,0,0, 1]
    print(f'PROBABILITIES_RANDOM_NEIGHBOUR = {PROBABILITIES_RANDOM_NEIGHBOUR}, {np.sum(PROBABILITIES_RANDOM_NEIGHBOUR)}')
    candidate_activation_hidden = ['relu', 'sigmoid', 'tanh']
    candidate_activation_output = ['linear', 'sigmoid', 'tanh']

    INCREASE_DECREASE_HP_VALUE = ['INCREASE', 'DECREASE']
    switch_architecture = ['SWITCH', 'NO-SWITCH']
    neuron_change_list = [ele for ele in range(-100, 100) if ele != 0]
    NEURON_CHANGE_PERC = random.choice(neuron_change_list)
    print(f'NEURON_CHANGE_PERC = {NEURON_CHANGE_PERC}')
    LEARNING_RATE_CHANGE_PERC = 0.2
    MAX_EVALS = 800
    Scores_all_Folds = []

    random_search()

    # ['DatasetSize_diff', 'Group_Variance_Combined_Sum_Normalized', 'Group_Variance_Combined_Prod_Normalized',
    #  'Group_Variance_Individual_Sum_Normalized', 'Group_StdDev_Combined_Sum_Normalized',
    #  'Group_StdDev_Combined_Prod_Normalized', 'Group_StdDev_Individual_Sum_Normalized', 'Group_StdDev_avg',
    #  'Group_Variance_avg', 'Euclidean_Distance_between_Tasks_Diff', 'Euclidean_Distance_between_Tasks_Prod',
    #  'Euclidean_Distance_Combined_Sum', 'Euclidean_Distance_Combined_Prod', 'Manhattan_Distance_between_Tasks_Diff',
    #  'Manhattan_Distance_between_Tasks_Prod',
    #  'Manhattan_Distance_Combined_Sum', 'Manhattan_Distance_Combined_Prod',
    #  'Euclidean_Distance_Scaled_between_Tasks_Diff', 'Euclidean_Distance_Scaled_between_Tasks_Prod',
    #  'Euclidean_Distance_Scaled_Combined_Sum', 'Euclidean_Distance_Scaled_Combined_Prod',
    #  'Manhattan_Distance_Scaled_between_Tasks_Diff', 'Manhattan_Distance_Scaled_between_Tasks_Prod',
    #  'Manhattan_Distance_Scaled_Combined_Sum', 'Manhattan_Distance_Scaled_Combined_Prod', 'Fitted_param_a_diff',
    #  'Fitted_param_b_diff', 'train_curve_10', 'train_curve_20', 'train_curve_30', 'train_curve_40', 'train_curve_50',
    #  'train_curve_60', 'train_curve_70', 'train_curve_80', 'train_curve_90', 'Weight', 'InterTaskAffinity']

