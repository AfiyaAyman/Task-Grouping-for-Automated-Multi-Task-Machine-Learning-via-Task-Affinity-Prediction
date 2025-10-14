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

    filepath = f'SavedModels/{dataset_name}_Pair_predictor_{fold}_{modelName}_{which_half}.h5'
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
    if dataset_name == 'Schools':
        batch_size = 1024
    else:
        batch_size = 128
    number_of_epoch = 50
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
    shared_hp = list(current_nn_architecture.keys())

    PROBABILITIES_RANDOM_NEIGHBOUR = [0.15, 0.7, 0.08, 0.04, 0.03]
    candidate_activation_hidden = ['relu', 'sigmoid', 'tanh']
    candidate_activation_output = ['linear', 'sigmoid', 'tanh']
    random_hp = np.random.choice(shared_hp, 1, p=PROBABILITIES_RANDOM_NEIGHBOUR)

    current_layers = current_nn_architecture['FF_Layers']

    layers_FF = [item for item in range(0, current_layers)]

    # print(f'Tweaked_HP = {random_hp}')
    Tweaked_HP.append(random_hp)

    if random_hp == 'FF_Neurons':
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
        tmp_activation_hidden = candidate_activation_hidden.copy()
        tmp_activation_hidden.remove(get_current_activation_function)
        new_activation_function = np.random.choice(tmp_activation_hidden)
        current_nn_architecture['activation_function'] = new_activation_function
    elif random_hp == 'output_activation':
        get_current_activation_function = current_nn_architecture['output_activation']
        tmp_activation_output = candidate_activation_output.copy()
        tmp_activation_output.remove(get_current_activation_function)
        new_activation_function = np.random.choice(tmp_activation_output)
        current_nn_architecture['output_activation'] = new_activation_function

    else:
        change = np.random.choice(INCREASE_DECREASE_HP_VALUE)
        if change == 'INCREASE':
            current_nn_architecture['learning_rate'] = current_nn_architecture['learning_rate'] + (
                    current_nn_architecture['learning_rate'] * LEARNING_RATE_CHANGE_PERC)

        else:  # 'DECREASE'
            current_nn_architecture['learning_rate'] = current_nn_architecture['learning_rate'] - (
                    current_nn_architecture['learning_rate'] * LEARNING_RATE_CHANGE_PERC)

    # print(f'Final = {current_nn_architecture}')
    return current_nn_architecture


def random_search(col):
    ALPHA = 0.0001
    current_best_nn_architecture = network_architecture
    print(f'START_ARCHITECTURE = {network_architecture}')

    random_seed = random.randint(0, 100)

    current_solution, variance, trainable_param, mse = kFold_validation(Sample_Inputs, Sample_Label,
                                                                   current_best_nn_architecture, random_seed)
    # print(f'current_solution = {current_solution}')
    print(f'Trainable Parameter = {trainable_param}')

    best_solution = current_solution

    Iteration.append(0)
    Usefulness_Score.append(current_solution)
    kfold_MSE.append(mse)
    current_solution = current_solution - ALPHA * trainable_param
    print(f'current_solution = {current_solution}')
    NAS_SCORE.append(current_solution)
    Variance.append(variance)
    ARCHITECTURE.append(current_best_nn_architecture.copy())
    prev_solution = current_solution
    Prev_Solution.append(prev_solution)
    Random_Seed.append(random_seed)
    Prev_iter.append(0)
    Switch.append('None')
    Tweaked_HP.append('None')

    # Keep searching until reach max evaluations
    for iteration in range(1, MAX_EVALS):
        current_nn_architecture = copy.deepcopy(current_best_nn_architecture)
        current_nn_architecture = random_neighbour(current_nn_architecture)

        random_seed = random.randint(0, 100)
        timeStart = time.time()
        current_solution, variance, trainable_param, mse = kFold_validation(Sample_Inputs, Sample_Label,
                                                                       current_nn_architecture, random_seed)
        print(f'total time = {time.time() - timeStart}')

        Iteration.append(iteration)
        Usefulness_Score.append(current_solution)
        '''USEFULNESS SCORE'''
        current_solution = current_solution - ALPHA * trainable_param
        NAS_SCORE.append(current_solution)
        kfold_MSE.append(mse)
        Variance.append(variance)
        ARCHITECTURE.append(current_nn_architecture.copy())
        Random_Seed.append(random_seed)

        # K = 30
        # Accept_Probability = min(1, math.exp((prev_solution - current_solution) * K))
        # PROBABILITIES_ACCEPT = [Accept_Probability, max(0, 1 - Accept_Probability)]

        if current_solution > prev_solution:
            PROBABILITIES_ACCEPT = [1, 0]
        else:
            diff = abs((prev_solution - current_solution) / prev_solution)
            if diff > 0.01 and diff < 0.02:
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
                  len(Random_Seed), len(kfold_MSE))
            # NAS_SCORE = Usefulness_Score + ALPHA* Total_Trainable_param
            results = pd.DataFrame({'Iteration': Iteration,
                                    'Switch': Switch,
                                    'Tweaked_HP': Tweaked_HP,
                                    'Usefulness_Score': Usefulness_Score,
                                    'NAS_SCORE': NAS_SCORE,
                                    'Prev_Solution': Prev_Solution,
                                    'kfold_MSE': kfold_MSE,
                                    'ARCHITECTURE': ARCHITECTURE,
                                    "Last_switch_at_iter": Prev_iter,
                                    "Scores_all_Folds": Scores_all_Folds,
                                    'Variance': Variance,
                                    "Random_Seed": Random_Seed})
            results.to_csv(f'{ResultPath}/{dataset_name}_Pairwise_NAS_{length}_{col}_{modelName}.csv',
                           index=False)

            if length > tail_pointer:
                length = length - tail_pointer
                os.remove(
                    os.path.join(f'{ResultPath}/{dataset_name}_Pairwise_NAS_{length}_{col}_{modelName}.csv'))

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
                            'kfold_MSE': kfold_MSE,
                            'ARCHITECTURE': ARCHITECTURE,
                            "Last_switch_at_iter": Prev_iter,
                            "Scores_all_Folds": Scores_all_Folds,
                            'Variance': Variance,
                            "Random_Seed": Random_Seed})

    results.to_csv(f'{ResultPath}/{dataset_name}_Pairwise_NAS_{col}_{modelName}.csv',
                   index=False)
    length = len(Usefulness_Score)
    os.remove(
        os.path.join(f'{ResultPath}/{dataset_name}_Pairwise_NAS_{length}_{col}_{modelName}.csv'))


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    modelName = sys.argv[2]
    which_half = sys.argv[3]

    # dataset_name = 'Chemical'
    # modelName = 'SVM'

    DataPath = f'../Results/Pairwise/{modelName}/train/'
    ResultPath = '../Results/NAS_Results/'
    # ALPHA = 1 / 1000

    network_architecture = {'FF_Layers': 2,
                            'FF_Neurons': [5,3],
                            'learning_rate': 0.0025,
                            'activation_function': 'relu',
                            'output_activation': 'sigmoid',
                            }

    predictor_data = pd.read_csv(f'{DataPath}Pairwise_Task_Relation_Features_{dataset_name}_train.csv')
    print(f'\n\n******* Training Samples = {len(predictor_data)} *******\n\n')
    print(f'shape of entire data = {np.shape(predictor_data)}')
    print(f'predictor_data.columns = {predictor_data.columns}')

    # !/bin/bash
    target = predictor_data.pop('Change')

    predictor_data.dropna(inplace=True)
    Sample_Label = np.array(target)
    print(len(predictor_data.columns))
    # Sample_Inputs = np.array(predictor_data)
    ALL_cols = list(predictor_data.columns)
    if which_half == 'first':
        ALL_cols = ALL_cols[:int(len(ALL_cols)/2)]
    else:
        ALL_cols = ALL_cols[int(len(ALL_cols)/2):]

    for col in ALL_cols:
        predictor_data[col] = predictor_data[col].astype(float)
        Sample_Inputs = np.array(predictor_data[col]).reshape(-1, 1)

        # print(f'col = {col}, shape of Sample_Inputs = {np.shape(Sample_Inputs)}, shape of Sample_Label = {np.shape(Sample_Label)}')

        Usefulness_Score = []
        kfold_MSE = []
        NAS_SCORE = []
        Iteration = []
        ARCHITECTURE = []
        Prev_Solution = []
        Random_Seed = []
        Prev_iter = []
        Variance = []
        Tweaked_HP = []
        Switch = []
        INCREASE_DECREASE_HP_VALUE = ['INCREASE', 'DECREASE']
        switch_architecture = ['SWITCH', 'NO-SWITCH']
        neuron_change_list = [ele for ele in range(-50, 50) if ele != 0]
        NEURON_CHANGE_PERC = random.choice(neuron_change_list)
        # print(f'NEURON_CHANGE_PERC = {NEURON_CHANGE_PERC}')
        # exit(0)

        LEARNING_RATE_CHANGE_PERC = 0.1
        MAX_EVALS = 100
        Scores_all_Folds = []

        random_search(col)
