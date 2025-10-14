import pandas as pd
import os
import sys
import pandas as pd
import numpy as np
import os
import time
import ast
import tqdm
import itertools
from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def train_and_evaluate_predictor(architecture, x_train, y_train, x_test, y_test, analysis_type):
    filepath = f'SavedModels/{dataset_name}_Group_predictor.h5'
    number_of_features = np.shape(x_train)[1]

    Input_FF = tf.keras.layers.Input(shape=(number_of_features,))
    hidden_FF = tf.keras.layers.Dense(architecture['FF_Neurons'][0],
                                      activation=architecture['activation_function'])(Input_FF)
    for h in range(1, architecture['FF_Layers']):
        hidden_FF = tf.keras.layers.Dense(architecture['FF_Neurons'][h],
                                          activation=architecture['activation_function'])(hidden_FF)

    output = tf.keras.layers.Dense(1, activation=architecture['output_activation'])(hidden_FF)

    finalModel = Model(inputs=Input_FF, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate=architecture['learning_rate'])
    finalModel.compile(optimizer=opt, loss='mse')
    checkpoint = ModelCheckpoint(filepath, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')

    batch_size = 64
    if analysis_type == 'individual':
        number_of_epoch = 100
    else:
        batch_size = 16
        number_of_epoch = 200
    finalModel.fit(x=x_train,
                   y=y_train,
                   shuffle=True,
                   epochs=number_of_epoch,
                   batch_size=batch_size,
                   validation_data=(x_test,y_test),
                   callbacks=checkpoint,
                   verbose=0)

    finalModel = tf.keras.models.load_model(filepath)
    y_pred = finalModel.predict(x_test, verbose=0)
    # print(f'np.shape(y_pred) = {np.shape(y_pred)}')
    # print(f'np.shape(y_test) = {np.shape(y_test)}')
    r_square = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    if os.path.exists(filepath):
        os.remove(os.path.join(filepath))

    train_r_sq = r2_score(y_train, finalModel.predict(x_train, verbose=0))
    # print(f'train_r_sq = {train_r_sq}')

    return r_square, mse, np.var(y_test)


def individual_usefulness(train_data, test_data, dataset_name):
    analysis_type = 'individual'
    columns = list(train_data.columns)
    columns.remove('change')

    y_train = train_data.pop('change')
    y_test = test_data.pop('change')
    # print(f'np.shape(y_train) = {np.shape(y_train)}, np.shape(y_test) = {np.shape(y_test)}')

    Arch_dict = {}
    Max_Usefulness = {}

    R_Square = []
    Feature = []
    MSE = []
    Variance = []
    ARCH = []
    # cols = ['group_variance', 'group_distance']
    for col in columns:
        # if dataset_name=='Landmine' and col != 'group_distance':
        #     continue
        # if dataset_name=='Parkinsons' and col != 'group_variance':
        #     continue
        if modelname == 'NN':
            nas_data = pd.read_csv(f'{architecture_datapath}{dataset_name}_Groupwise_NAS_{col}.csv')
        else:
            nas_data = pd.read_csv(f'{architecture_datapath}{dataset_name}_Groupwise_NAS_{col}_{modelname}.csv')

        # nas_data = pd.read_csv(f'{architecture_datapath}{dataset_name}_Groupwise_NAS_{col}.csv')
        print(f'\n\n*****\n\n')
        best_idx = nas_data['Usefulness_Score'].idxmax()
        best_architecture = nas_data.ARCHITECTURE[best_idx]
        best_architecture = ast.literal_eval(best_architecture)
        print(f'Highest Usefulness for {col} = {max(nas_data["Usefulness_Score"])} with best_architecture = {best_architecture}')
        Arch_dict[col] = best_architecture

        Max_Usefulness[col] = max(nas_data["Usefulness_Score"])

        x_train = train_data[col].values.reshape(-1, 1)
        x_test = test_data[col].values.reshape(-1, 1)

        r_sq, mse, var = train_and_evaluate_predictor(best_architecture, x_train, y_train, x_test, y_test, analysis_type)
        print(f'Final r_sq = {r_sq}, mse = {mse}, var = {var}')

        Feature.append(col)
        R_Square.append(r_sq)
        MSE.append(mse)
        Variance.append(var)
        ARCH.append(best_architecture)

        if r_sq<0:
            nas_data = nas_data.sort_values(by=['Usefulness_Score'], ascending=False)
            for j in range(1,6):
                best_architecture = nas_data.ARCHITECTURE[j]
                best_architecture =  ast.literal_eval(best_architecture)
                Arch_dict[col] = best_architecture
                Max_Usefulness[col] = max(nas_data["Usefulness_Score"])
                # best_architecture = network_architecture = {'FF_Layers': 2, 'FF_Neurons': [20, 10], 'learning_rate': 0.005,
                #                                             'activation_function': 'sigmoid', 'output_activation': 'tanh'}

                x_train = train_data[col].values.reshape(-1, 1)
                x_test = test_data[col].values.reshape(-1, 1)

                r_sq, mse, var = train_and_evaluate_predictor(best_architecture, x_train, y_train, x_test, y_test,
                                                              analysis_type)
                print(f'Final r_sq = {r_sq}, mse = {mse}, var = {var}')

                Feature.append(col)
                R_Square.append(r_sq)
                MSE.append(mse)
                Variance.append(var)
                ARCH.append(best_architecture)

    result_df = pd.DataFrame({'Feature': Feature, 'R_Square': R_Square, 'MSE': MSE, 'Variance': Variance})
    return result_df


def get_average_usefulness(dataset_name):
    Avg_R_SQUARE = []
    Feature = []
    Avg_MSE = []
    Variance = []
    usefulness_dict = {}
    for runs in range(1, 6):
        res = pd.read_csv(f'{group_study_path}Groupwise_Individual_Usefulness_{dataset_name}_run_{runs}_{modelname}.csv')
        res = res.sort_values('R_Square').drop_duplicates('Feature', keep='last').reset_index(drop=True)
        for i in range(len(res)):
            feature = res['Feature'][i]
            if runs == 1:
                usefulness_dict[feature] = [res['R_Square'][i]]
            else:
                usefulness_dict[feature].append(res['R_Square'][i])

    for k,v in usefulness_dict.items():
        Feature.append(k)
        Avg_R_SQUARE.append(np.mean(v))

    Feature = change_features(Feature)

    avg_res = pd.DataFrame({'Feature': Feature, 'Avg_R_SQUARE': Avg_R_SQUARE})
    avg_res.sort_values(by=['Avg_R_SQUARE'], inplace=True, ascending=False)
    avg_res['Avg_R_SQUARE'] = avg_res['Avg_R_SQUARE'].apply(lambda x: round(x*100, 2))
    avg_res.to_csv(f'{group_study_path}Groupwise_Individual_Usefulness_{dataset_name}_avg_{modelname}.csv', index=False)



    print(f'*******{dataset_name}*******, avg_res = \n{avg_res}')


    # plt.figure(figsize=(8,4))
    # plt.bar(avg_res.Feature, avg_res.Avg_R_SQUARE)
    # plt.xticks(rotation=30)
    # plt.xlabel('Feature')
    # plt.ylabel('Average R_Square')
    # plt.title(f'Average R_Square for {dataset_name}')
    # plt.show()

def final_predictor_usefulness(train_data, test_data, dataset_name):
    analysis_type = 'final'
    y_train = train_data.pop('change')
    y_test = test_data.pop('change')

    nas_all_datapath = f'../Results/NAS_Results/groupwise/{modelname}/'
    if modelname == 'SVM':
        iter_dict = {'School': 405, 'Chemical': 405, 'Landmine': 435, 'Parkinsons': 500}
    if modelname == 'XGBoost':
        iter_dict = {'School': 540, 'Chemical': 720, 'Landmine': 825, 'Parkinsons': 1000}
    if modelname == 'NN':
        iter_dict = {'School': 1500, 'Chemical': 725, 'Landmine': 1245, 'Parkinsons': 1500}

    nas_data = pd.read_csv(f'{nas_all_datapath}{dataset_name}_Groupwise_NAS_{iter_dict[dataset_name]}_ALL_{modelname}.csv')
    print(f'Highest Usefulness for {dataset_name} = {max(nas_data["Usefulness_Score"])}')
    nas_data.sort_values(by=['Usefulness_Score'], inplace=True, ascending=False)
    print(nas_data[['ARCHITECTURE', 'Usefulness_Score']][:3])
    ARCHITECTURE = list(nas_data.ARCHITECTURE.unique())


    mse_scores = []
    r_square_scores = []
    archs = []
    features = []
    variance = []
    already_seen = set()
    for i in range(0,10):
        print(f'{i + 1}th best architecture = {ARCHITECTURE[i]}')
        best_architecture = ast.literal_eval(ARCHITECTURE[i])


        # Feature = ['group_dataset_size', 'pairwise_improvement_average', 'pairwise_improvement_stddev', 'group_variance', 'pairwise_Weight_average']
        Feature = best_architecture['Features']
        # Feature = ['group_variance', 'pairwise_improvement_average', 'pairwise_improvement_stddev','pairwise_Weight_average']
        print(f'Feature = {Feature}')
        new_train_data = train_data[Feature]
        new_test_data = test_data[Feature]
        x_train = np.array(new_train_data)
        x_test = np.array(new_test_data)
        # print(f'np.shape(x_train) = {np.shape(x_train)}, np.shape(x_test) = {np.shape(x_test)}')

        r_sq, mse, var = train_and_evaluate_predictor(best_architecture, x_train, y_train, x_test, y_test, analysis_type)
        print(f'r_sq = {r_sq}, mse = {mse}, var = {var}')
        mse_scores.append(mse)
        archs.append(best_architecture)
        features.append(Feature)
        r_square_scores.append(r_sq)
        variance.append(var)

        if len(r_square_scores) == 5:
            break

    print(f'r_sq = {r_square_scores}')
    return archs, features, r_square_scores, mse_scores, variance

def get_average_usefulness_for_all(dataset_name):
    Avg_R_SQUARE = []
    Feature = []
    Architecture = []
    Avg_MSE = []
    Variance = []
    usefulness_dict = {}
    for runs in range(1, 4):
        res = pd.read_csv(f'{group_study_path}Groupwise_Final_Usefulness_{dataset_name}_run_{runs}_{modelname}.csv')
        for i in range(len(res)):
            arch = res['Architecture'][i]
            if runs == 1:
                usefulness_dict[arch] = [res['R_Square'][i]]
            else:
                usefulness_dict[arch].append(res['R_Square'][i])


    for k, v in usefulness_dict.items():
        Architecture.append(k)
        Avg_R_SQUARE.append(np.mean(v))

    for arch in Architecture:
        arch = ast.literal_eval(arch)
        Feature.append(arch['Features'])


    avg_res = pd.DataFrame({'Avg_R_SQUARE': Avg_R_SQUARE, 'Architecture': Architecture, 'Feature': Feature})
    avg_res.sort_values(by=['Avg_R_SQUARE'], inplace=True, ascending=False)
    avg_res['Avg_R_SQUARE'] = avg_res['Avg_R_SQUARE'].apply(lambda x: round(x*100, 2))

    print(f'dataset = {dataset_name}, final arch = {avg_res["Architecture"][0]}')
    # print(f'final features = {avg_res.Feature[0]}')
    Feat = avg_res["Architecture"][0]
    Feat = ast.literal_eval(Feat)
    Feat = Feat['Features']
    Feat = change_features(Feat)
    print(f'Feat = {Feat}')
    return avg_res.Feature[0], avg_res['Avg_R_SQUARE'][0]

def change_features(Features):
    for idx,feat in enumerate(Features):
        # print(idx, feat)
        feat = feat.title()
        feat = feat.replace('_',' ')
        # print(feat)

        feat = feat.replace('Pairwise Improvement', 'Pairwise MTL gain')
        feat = feat.replace('Average', '$(\mu)$')
        feat = feat.replace('Group Variance', 'Group-target $\sigma^2$ $(\mu)$')
        feat = feat.replace('Group Stddev', 'Group-target $\sigma$ $(\mu)$')

        feat = feat.replace('Variance', '$(\sigma^2)$')
        feat = feat.replace('Stddev', '$(\sigma)$')
        feat = feat.replace('Number Of Tasks', '\#Tasks')
        feat = feat.replace('Group Dataset Size', '\#Samples')
        feat = feat.replace('Group Distance', 'Group distance ($\mu$)')

        feat = feat.replace('Ita', 'ITA')

        feat = feat.replace('ITA', '$\mathcal{Z}_{t_i \leftrightarrow t_j}$')
        feat = feat.replace('Weight', '$\mathcal{W}_{t_i}\cdot\mathcal{W}_{t_j}$')

        # print(idx,feat)
        Features[idx] = feat
    return Features

def pearson_correlation(dataset_name):
    Feature = []
    Target = []
    Pearson_Corr = []

    predictor_Data = pd.read_csv(f'../Results/partition_sample/{modelname}/group_sample/Data_for_GroupPredictor_{dataset_name}_{modelname}.csv',
                                 low_memory=False)
    predictor_Data = predictor_Data.dropna()
    # print(predictor_Data.columns)

    cols_to_consider = list(predictor_Data.columns)
    cols_to_consider.remove('change')
    for col in cols_to_consider:
        # col = list(col)
        corr, _ = pearsonr(predictor_Data[col], predictor_Data['change'])
        Feature.append(col)
        # Target.append(col[1])
        Pearson_Corr.append(round(corr, 5))

    Feature = change_features(Feature)

    Corr = pd.DataFrame({"Feature": Feature,
                         # "GroupwiseMTLGain": Target,
                         f"Pearson_Corr": Pearson_Corr})
    Corr['PearsonCorrelationABS'] = round(abs(Corr['Pearson_Corr']), 4)
    Corr = Corr[['Feature', 'Pearson_Corr', 'PearsonCorrelationABS']]
    Corr = Corr.sort_values(by=['PearsonCorrelationABS'], ascending=False)
    print(f'{dataset_name} = \n{Corr}')
    Corr.to_csv(f'{group_study_path}Groupwise_PearsonCorrelation_{dataset_name}_{modelname}.csv', index=False)

def standley_baseline_comparison():
    simple_avg = []
    predictor_with_average = []
    final_predictor_performance = []
    for dataset_name in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
        train_datapath = f'../Results/partition_sample/{modelname}/group_sample/train/'
        testdatapath = f'../Results/partition_sample/{modelname}/group_sample/test/'
        train_data = pd.read_csv(f'{train_datapath}Groupwise_Features_{dataset_name}_train.csv')
        test_data = pd.read_csv(f'{testdatapath}Groupwise_Features_{dataset_name}_test.csv')
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)

        # print(f'train_data = \n{train_data.columns}')
        # pairwise_improvement_average = list(train_data['pairwise_improvement_average'])
        # change = list(train_data['change'])
        # r_sq = r2_score(change, pairwise_improvement_average)
        # print(f'**********Train Data**********')
        # print(f'dataset = {dataset_name}, r_sq = {r_sq}')
        pairwise_improvement_average = list(test_data['pairwise_improvement_average'])
        change = list(test_data['change'])
        r_sq = r2_score(change, pairwise_improvement_average)
        # print(f'**********Test Data**********')
        print(f'dataset = {dataset_name}, r_sq with simple average of Pairwise_MTL_Gain = {round(r_sq*100,2)}')

        usefulness_data = pd.read_csv(f'{group_study_path}Groupwise_Individual_Usefulness_{dataset_name}_avg_{modelname}.csv', low_memory=False)
        print(f'result from NN predictor with average of Pairwise_MTL_Gain = {round(usefulness_data.Avg_R_SQUARE[0],2)}')

        final_predictor_usefulness_data = pd.read_csv(f'{group_study_path}Groupwise_Final_Usefulness_{modelname}.csv', low_memory=False)
        final_predictor_usefulness_data = final_predictor_usefulness_data[final_predictor_usefulness_data.Dataset == dataset_name].reset_index(drop=True)
        print(f'result from final predictor = {round(final_predictor_usefulness_data.R_Square[0],2)}')

        simple_avg.append(round(r_sq*100,2))
        predictor_with_average.append(round(usefulness_data.Avg_R_SQUARE[0],2))
        final_predictor_performance.append(round(final_predictor_usefulness_data.R_Square[0],2))


    dataset = ['School', 'Chemical', 'Landmine', 'Parkinsons']
    x = np.arange(len(dataset))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, simple_avg, width, label='Simple Avg. of Pairwise_MTL_Gain')
    rects2 = ax.bar(x , predictor_with_average, width, label='NN Predictor with Avg. of Pairwise_MTL_Gain')
    rects3 = ax.bar(x + width / 2, final_predictor_performance, width, label='Final Predictor')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('R_squared')
    ax.set_title('Comparison of R_squared between simple average and NN predictor for each benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset)
    ax.legend()

    plt.show()


if __name__ == '__main__':

    group_study_path = '../Results/Groupwise_Study/'

    modelname = 'NN'

    if modelname == 'NN':
        train_datapath = f'../Results/partition_sample/group_sample/train/'
        testdatapath = f'../Results/partition_sample/group_sample/test/'
        architecture_datapath = f'../Results/NAS_Results/groupwise/'

    else:
        train_datapath = f'../Results/partition_sample/{modelname}/group_sample/train/'
        testdatapath = f'../Results/partition_sample/{modelname}/group_sample/test/'
        architecture_datapath = f'../Results/NAS_Results/groupwise/{modelname}/'

    FINAL_USEFULNESS = []
    FINAL_FEATURES = []
    DATASET_NAME = []

    print('Pearson Correlation for Groupwise-task features - each dataset')
    for dataset_name in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
        pearson_correlation(dataset_name)

    print('Individual Usefulness for Groupwise-task features')
    for dataset_name in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
        for run in range(1, 6):
            train_data = pd.read_csv(f'{train_datapath}Groupwise_Features_{dataset_name}_train.csv')
            test_data = pd.read_csv(f'{testdatapath}Groupwise_Features_{dataset_name}_test.csv')
            train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            train_data.dropna(inplace=True)
            test_data.dropna(inplace=True)
        #
            result_df = individual_usefulness(train_data, test_data, dataset_name)
            result_df.sort_values(by=['R_Square'], ascending=False, inplace=True)
            result_df.to_csv(f'{group_study_path}Groupwise_Individual_Usefulness_{dataset_name}_run_{run}_{modelname}.csv', index=False)

    print('Final Usefulness for Groupwise-task Affinity Predictor')
    for dataset_name in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
        train_data = pd.read_csv(f'{train_datapath}Groupwise_Features_{dataset_name}_train.csv')
        test_data = pd.read_csv(f'{testdatapath}Groupwise_Features_{dataset_name}_test.csv')
        train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)

        for run in range(1, 6):

            MSE = []
            R_Square = []
            Variance = []
            Feature = []
            Architecture = []
            archs, features, r_square_scores, mse_scores, variance = final_predictor_usefulness(train_data, test_data, dataset_name)
            for i in range(0, len(archs)):
                MSE.append(mse_scores[i])
                R_Square.append(r_square_scores[i])
                Feature.append(features[i])
                Architecture.append(archs[i])
                Variance.append(variance[i])

            result_df = pd.DataFrame({'Architecture': Architecture, 'Feature': Feature, 'R_Square': R_Square, 'MSE': MSE, 'Variance': Variance})
            result_df.to_csv(f'{group_study_path}Groupwise_Final_Usefulness_{dataset_name}_run_{run}_{modelname}.csv', index=False)
        #
    print('Average Usefulness for Groupwise-task Affinity Predictor')
    for dataset_name in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
        train_data = pd.read_csv(f'{train_datapath}Groupwise_Features_{dataset_name}_train.csv')
        test_data = pd.read_csv(f'{testdatapath}Groupwise_Features_{dataset_name}_test.csv')
        train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_data.dropna(inplace=True)
        test_data.dropna(inplace=True)
        get_average_usefulness(dataset_name)
        DATASET_NAME.append(dataset_name)
        feat, r_sq = get_average_usefulness_for_all(dataset_name)
        FINAL_USEFULNESS.append(r_sq)
        FINAL_FEATURES.append(feat)
    res_all = pd.DataFrame({'Dataset': DATASET_NAME,
                            'R_Square': FINAL_USEFULNESS})
    print(f'res_all = \n{res_all}')
    res_all.to_csv(f'{group_study_path}Groupwise_Final_Usefulness_{modelname}.csv', index=False)


    standley_baseline_comparison()


