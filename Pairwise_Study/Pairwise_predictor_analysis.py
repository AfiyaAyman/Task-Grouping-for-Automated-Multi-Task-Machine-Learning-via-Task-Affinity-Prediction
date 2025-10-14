import pandas as pd
import numpy as np
import os
import ast

from scipy.stats import pearsonr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from sklearn.metrics import mean_squared_error, r2_score


def train_and_evaluate_predictor(architecture, x_train, y_train, x_test, y_test, analysis_type):
    filepath = f'SavedModels/{dataset_name}_Pair_predictor.h5'

    if analysis_type != 'final':
        architecture['activation_function'] = 'sigmoid'
        architecture['output_activation'] = 'linear'
        number_of_epoch = 100

    else:
        number_of_epoch = 200

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

    if dataset_name == 'School':
        batch_size = 264
    else:
        batch_size = 16

    finalModel.fit(x=x_train,
                   y=y_train,
                   shuffle=True,
                   epochs=number_of_epoch,
                   batch_size=batch_size,
                   validation_data=(x_test, y_test),
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

    return r_square, mse, np.var(y_test)


def individual_usefulness(train_data, test_data, dataset_name):
    analysis_type = 'individual'
    columns = list(train_data.columns)
    columns.remove('Change')

    y_train = train_data.pop('Change')
    y_test = test_data.pop('Change')
    # print(f'np.shape(y_train) = {np.shape(y_train)}, np.shape(y_test) = {np.shape(y_test)}')

    Arch_dict = {}
    Max_Usefulness = {}

    R_Square = []
    Feature = []
    MSE = []
    Variance = []
    Best_Arch = []
    print(f'len(columns) = {len(columns)}')
    for col in columns:
        if modelName == 'NN':
            csv_name = f'{architecture_datapath}{dataset_name}_Pairwise_NAS_{col}.csv'
        else:
            csv_name = f'{architecture_datapath}{dataset_name}_Pairwise_NAS_{col}_{modelName}.csv'
        if not os.path.exists(csv_name):
            continue

        nas_data = pd.read_csv(csv_name)
        max_r_sq = max(nas_data['Usefulness_Score'])

        # another_csv = f'{architecture_datapath}/Extra/{dataset_name}_Pairwise_NAS_{col}.csv'
        # if os.path.exists(another_csv):
        #     nas_data2 = pd.read_csv(another_csv)
        #     max_r_sq2 = max(nas_data2['Usefulness_Score'])
        #
        #     if max_r_sq2 > max(nas_data['Usefulness_Score']):
        #         nas_data = nas_data2
        #         max_r_sq = max_r_sq2
        best_idx = nas_data['Usefulness_Score'].idxmax()
        best_architecture = nas_data.ARCHITECTURE[best_idx]
        best_architecture = ast.literal_eval(best_architecture)
        print(f'col = {col}, Max_Usefulness = {max(nas_data["Usefulness_Score"])}')
        print(f'best_architecture = {best_architecture}')
        # best_architecture = {'FF_Layers': 2,
        #                         'FF_Neurons': [3, 5],
        #                         'learning_rate': 0.0025}
        Arch_dict[col] = best_architecture
        Max_Usefulness[col] = max(nas_data["Usefulness_Score"])

        x_train = train_data[col].values.reshape(-1, 1)
        x_test = test_data[col].values.reshape(-1, 1)

        r_sq, mse, var = train_and_evaluate_predictor(best_architecture, x_train, y_train, x_test, y_test, analysis_type)
        print(f'col = {col}, r_sq = {r_sq}, mse = {mse}, var = {var}')

        Feature.append(col)
        R_Square.append(r_sq)
        MSE.append(mse)
        Variance.append(var)

    result_df = pd.DataFrame({'Feature': Feature, 'R_Square': R_Square, 'MSE': MSE, 'Variance': Variance})
    return result_df
def change_features(Features):
    for idx,feat in enumerate(Features):
        print(feat)
        feat = feat.replace('_',' ')
        feat = feat.replace('Scaled between Tasks', 'between Tasks Scaled')
        feat = feat.replace('Scaled Combined Sum', 'Combined Scaled Sum')
        feat = feat.replace('Scaled Combined Prod', 'Combined Scaled Prod')
        feat = feat.replace('Scaled', 'scaled')
        feat = feat.replace('Prod', 'nP')
        feat = feat.replace('Diff', 'nS')
        # feat = feat.replace('Combined Prod', 'comb-nP')
        # feat = feat.replace('Combined Sum', 'comb-nS')

        feat = feat.replace(' Normalized', '')
        feat = feat.replace('diff', 'nS')
        feat = feat.replace('Sum', 'nS')

        feat = feat.replace('Manhattan Distance between Tasks','$d_{M_{(t_i\leftrightarrow t_j)}}$')
        feat = feat.replace('Euclidean Distance between Tasks', '$d_{E_{(t_i\leftrightarrow t_j)}}$')
        feat = feat.replace('Hamming Distance between Tasks', '$d_{H_{(t_i\leftrightarrow t_j)}}$')
        feat = feat.replace('Manhattan Distance Combined', '$d_{M_{(t_i+t_j)}}$')
        feat = feat.replace('Euclidean Distance Combined', '$d_{E_{(t_i+t_j)}}$')
        feat = feat.replace('Hamming Distance Combined', '$d_{H_{(t_i+t_j)}}$')
        # feat = feat.replace('Manhattan Distance Scaled between Tasks', '$d_{M_{(t_i,t_j)}}$ scaled')
        # feat = feat.replace('Euclidean Distance Scaled between Tasks', '$d_{E_{(t_i,t_j)}}$ scaled')
        # feat = feat.replace('Hamming Distance Scaled between Tasks', '$d_{H_{(t_i,t_j)}}$ scaled')

        feat = feat.replace('Sum Normalized', 'nS')
        feat = feat.replace('Prod Normalized', 'nP')
        feat = feat.replace('Individual Sum Normalized', 'indiv-nS')
        feat = feat.replace('Individual Prod Normalized', 'indiv-nP')
        feat = feat.replace('Manhattan Distance','$d_M$')
        feat = feat.replace('Euclidean Distance', '$d_E$')
        feat = feat.replace('Hamming Distance', '$d_H$')


        feat = feat.replace('avg', '$(\mu)$')

        feat = feat.replace('Group Variance Combined', '${\sigma^{2}}_{(t_i+t_j)}$')
        feat = feat.replace('Group StdDev Combined', '${\sigma}_{(t_i+t_j)}$')
        feat = feat.replace('Group Variance Individual', '$|{\sigma^{2}}_{t_i}-{\sigma^{2}}_{t_j}|$')
        feat = feat.replace('Group StdDev Individual', '$|{\sigma}_{t_i}-{\sigma}_{t_j}|$')
        feat = feat.replace('Group Variance', '$\sigma^2$')
        feat = feat.replace('Group StdDev', '$\sigma$')
        feat = feat.replace('InterTaskAffinity', '$\mathcal{Z}_{t_i \leftrightarrow t_j}$')
        feat = feat.replace('Weight', '$\mathcal{W}_{t_i}\cdot\mathcal{W}_{t_j}$')
        feat = feat.replace('train curve ', 'Curve grad (')
        feat = feat.replace('0', '0\%)')
        feat = feat.replace('DatasetSize', '$|D_{t_i}-D_{t_j}|$')
        feat = feat.replace('Fitted param b nS', 'Param $|b_{t_i} - b_{t_j}|$')
        feat = feat.replace('Fitted param a nS', 'Param $|a_{t_i} - a_{t_j}|$')


        # print(idx,feat)
        # if 'Distance' in feat:
        #     print(feat)

        Features[idx] = feat
    # for f in Features:
    #     print(f)
    return Features
def get_average_usefulness(dataset_name):
    Avg_R_SQUARE = []
    Feature = []
    Avg_MSE = []
    Variance = []
    usefulness_dict = {}
    for runs in range(1, 6):
        if modelName!='NN':
            res = pd.read_csv(f'{pairwise_study_path}Pairwise_Individual_Usefulness_{dataset_name}_run_{runs}_{modelName}.csv')
        else:
            res = pd.read_csv(f'{pairwise_study_path}Pairwise_Individual_Usefulness_{dataset_name}_run_{runs}.csv')
        for i in range(len(res)):
            feature = res['Feature'][i]
            if runs == 1:
                usefulness_dict[feature] = [res['R_Square'][i]]
            else:
                usefulness_dict[feature].append(res['R_Square'][i])

    for k, v in usefulness_dict.items():
        Feature.append(k)
        Avg_R_SQUARE.append(np.mean(v))

    change_features(Feature)

    avg_res = pd.DataFrame({'Feature': Feature, 'Avg_R_SQUARE': Avg_R_SQUARE})

    avg_res['Avg_R_SQUARE'] = avg_res['Avg_R_SQUARE'].apply(lambda x: round(x*100, 2))
    avg_res.to_csv(f'{pairwise_study_path}Pairwise_Individual_Usefulness_{dataset_name}_avg_{modelName}_all.csv', index=False)

    avg_res.sort_values(by=['Avg_R_SQUARE'], inplace=True, ascending=False)
    print(f'avg_res = {avg_res}')


    only_top_10 = avg_res[:8]
    only_top_10.to_csv(f'{pairwise_study_path}Pairwise_Individual_Usefulness_{dataset_name}_avg_{modelName}.csv', index=False)
    #
    # # plt.figure(figsize=(8,4))
    # plt.bar(only_top_10.Feature, only_top_10.Avg_R_SQUARE)
    # plt.xticks(rotation=45)
    # plt.xlabel('Feature')
    # plt.ylabel('Average R_Square')
    # plt.title(f'Average R_Square for {dataset_name}')
    # plt.show()
def get_average_usefulness_for_all(dataset_name):
    Avg_R_SQUARE = []
    Feature = []
    Architecture = []
    Avg_MSE = []
    Variance = []
    usefulness_dict = {}
    for runs in range(1, 4):
        if modelName!='NN':
            res = pd.read_csv(f'{pairwise_study_path}Pairwise_Final_Usefulness_{dataset_name}_run_{runs}_{modelName}.csv')
        else:
            res = pd.read_csv(f'{pairwise_study_path}Pairwise_Final_Usefulness_{dataset_name}_run_{runs}.csv')
        for i in range(len(res)):
            arch = res['Architecture'][i]
            if runs == 1:
                usefulness_dict[arch] = [res['R_Square'][i]]
            else:
                # tmp = (res['R_Square'][i],res['Feature'][i])
                usefulness_dict[arch].append(res['R_Square'][i])


    for k, v in usefulness_dict.items():
        Architecture.append(k)
        Avg_R_SQUARE.append(np.mean(v))

    for arch in Architecture:
        arch = ast.literal_eval(arch)
        Feature.append(arch['Features'])


    avg_res = pd.DataFrame({'Avg_R_SQUARE': Avg_R_SQUARE, 'Architecture': Architecture, 'Feature': Feature})
    avg_res.sort_values(by=['Avg_R_SQUARE'], inplace=True, ascending=False)
    # avg_res['Avg_R_SQUARE'] = avg_res['Avg_R_SQUARE'].apply(lambda x: round(x, 2))

    print(f'avg_res = \n{avg_res}')

    print(f'final arch = {avg_res["Architecture"][0]}')
    print(f'final features = {avg_res.Feature[0]}')
    return avg_res.Feature[0], avg_res['Avg_R_SQUARE'][0]

def final_predictor_usefulness(train_data, test_data, dataset_name):
    analysis_type = 'final'
    y_train = train_data.pop('Change')
    y_test = test_data.pop('Change')

    nas_all_datapath = f'../Results/NAS_Results/pairwise/'
    if modelName == 'NN':
        iter_dict = {'School': 1185, 'Chemical': 2105, 'Landmine': 2745, 'Parkinsons': 3000}

    if modelName == 'XGBoost':
        iter_dict = {'School': 1000, 'Chemical': 1000, 'Landmine': 1000, 'Parkinsons': 1000}
    if modelName == 'SVM':
        iter_dict = {'School': 885, 'Chemical': 975, 'Landmine': 1000, 'Parkinsons': 1000}

    if modelName == 'NN':
        nas_all_datapath = f'../Results/NAS_Results/pairwise/'
        nas_data = pd.read_csv(f'{nas_all_datapath}{dataset_name}_Pairwise_NAS_{iter_dict[dataset_name]}_ALL.csv')
    else:
        nas_all_datapath = f'../Results/NAS_Results/pairwise/{modelName}/'
        nas_data = pd.read_csv(
            f'{nas_all_datapath}{dataset_name}_Pairwise_NAS_{iter_dict[dataset_name]}_ALL_{modelName}.csv')
    print(f'Highest Usefulness for {dataset_name} = {max(nas_data["Usefulness_Score"])}')
    best_idx = nas_data['Usefulness_Score'].idxmax()
    best_architecture = nas_data.ARCHITECTURE[best_idx]
    best_architecture = ast.literal_eval(best_architecture)

    nas_data.sort_values(by=['Usefulness_Score'], inplace=True, ascending=False)
    print(nas_data[['ARCHITECTURE', 'Usefulness_Score']][:3])
    ARCHITECTURE = list(nas_data.ARCHITECTURE.values)
    print(f'ARCHITECTURE = {ARCHITECTURE[:3]}')
    ARCHITECTURE = list(nas_data.ARCHITECTURE.unique())
    print(f'ARCHITECTURE = {ARCHITECTURE[:3]}')

    mse_scores = []
    r_square_scores = []
    archs = []
    features = []
    variance = []
    for i in range(0, 5):
        print(f'{i + 1}th best architecture = {ARCHITECTURE[i]}')
        best_architecture = ast.literal_eval(ARCHITECTURE[i])

        Feature = best_architecture['Features']
        new_train_data = train_data[Feature]
        new_test_data = test_data[Feature]
        x_train = np.array(new_train_data)
        x_test = np.array(new_test_data)
        # print(f'np.shape(x_train) = {np.shape(x_train)}, np.shape(x_test) = {np.shape(x_test)}')

        r_sq, mse, var = train_and_evaluate_predictor(best_architecture, x_train, y_train, x_test, y_test,
                                                      analysis_type)
        print(f'r_sq = {r_sq}, mse = {mse}, var = {var}')
        mse_scores.append(mse)
        archs.append(best_architecture)
        features.append(Feature)
        r_square_scores.append(r_sq)
        variance.append(var)

    return archs, features, r_square_scores, mse_scores, variance

def pearson_correlation(dataset_name):
    Features = []
    Target = []
    Pearson_Corr = []
    predictor_Data = pd.read_csv(f'../Results/Pairwise/{modelName}/Pairwise_Task_Relation_Features_{dataset_name}.csv',
                                 low_memory=False)
    # print(predictor_Data.columns)

    cols_to_consider = list(predictor_Data.columns)
    cols_to_consider.remove('Change')
    for col in cols_to_consider:
        corr, _ = pearsonr(predictor_Data[col], predictor_Data['Change'])
        Features.append(col)
        Pearson_Corr.append(round(corr, 5))

    Features = change_features(Features)

    Corr = pd.DataFrame({"Feature": Features,
                         # "GroupwiseMTLGain": Target,
                         f"Pearson_Corr": Pearson_Corr})
    Corr['PearsonCorrelationABS'] = round(abs(Corr['Pearson_Corr']), 4)
    Corr = Corr[['Feature', 'Pearson_Corr', 'PearsonCorrelationABS']]
    Corr = Corr.sort_values(by=['PearsonCorrelationABS'], ascending=False)
    print(f'{dataset_name} = \n{Corr[["Feature", "Pearson_Corr"]]}')
    Corr = Corr.head(9)
    Corr.to_csv(f'{pairwise_study_path}Pairwise_PearsonCorrelation_{dataset_name}_{modelName}_top.csv', index=False)

def dataset_average_usefulness():
    FEATURES = []
    TEST_LOSS_Dict = {}
    variation_dict = {}
    R_Square_Dict = {}
    Pearson_Dict = {}

    for dataset in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
        avg_res = pd.read_csv(f'{pairwise_study_path}Pairwise_Individual_Usefulness_{dataset}_avg_{modelName}_all.csv', low_memory=False)

        for i in range(len(avg_res)):
            feat = avg_res.Feature[i]
            if feat not in R_Square_Dict.keys():
                R_Square_Dict[feat] = []

    print(len(R_Square_Dict.keys()))
    Sch_rsq = []
    Chem_rsq = []
    Land_rsq = []
    Park_rsq = []
    Sch_feat_dict = {}
    Chem_feat_dict = {}
    Land_feat_dict = {}
    Park_feat_dict = {}

    for dataset in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
        avg_res = pd.read_csv(f'{pairwise_study_path}Pairwise_Individual_Usefulness_{dataset}_avg_{modelName}_all.csv', low_memory=False)
        if dataset == 'School':
            SchFeatures = list(avg_res.Feature.values)
            for f in SchFeatures:
                Sch_feat_dict[f] = avg_res[avg_res.Feature == f].Avg_R_SQUARE.values[0]
        elif dataset == 'Chemical':
            ChemFeatures = list(avg_res.Feature.values)
            for f in ChemFeatures:
                Chem_feat_dict[f] = avg_res[avg_res.Feature == f].Avg_R_SQUARE.values[0]
        elif dataset == 'Landmine':
            LandFeatures = list(avg_res.Feature.values)
            for f in LandFeatures:
                Land_feat_dict[f] = avg_res[avg_res.Feature == f].Avg_R_SQUARE.values[0]
        else:
            ParkFeatures = list(avg_res.Feature.values)
            for f in ParkFeatures:
                Park_feat_dict[f] = avg_res[avg_res.Feature == f].Avg_R_SQUARE.values[0]
        for i in range(len(avg_res)):
            feat = avg_res.Feature[i]
            r_sq = avg_res.Avg_R_SQUARE[i]
            R_Square_Dict[feat].append(r_sq)

    FEATURES = []
    Sch_rsq = []
    Chem_rsq = []
    Land_rsq = []
    Park_rsq = []
    Avg_R_SQUARE = []
    for feat in R_Square_Dict.keys():
        avg_r_sq = np.mean(R_Square_Dict[feat])
        Avg_R_SQUARE.append(avg_r_sq)
        FEATURES.append(feat)

        if feat in SchFeatures:
            Sch_rsq.append(Sch_feat_dict[feat])
        else:
            Sch_rsq.append(0)
        if feat in ChemFeatures:
            Chem_rsq.append(Chem_feat_dict[feat])
        else:
            Chem_rsq.append(0)

        if feat in LandFeatures:
            Land_rsq.append(Land_feat_dict[feat])
        else:
            Land_rsq.append(0)

        if feat in ParkFeatures:
            Park_rsq.append(Park_feat_dict[feat])
        else:
            Park_rsq.append(0)

    print(len(FEATURES), len(Sch_rsq), len(Chem_rsq), len(Land_rsq), len(Park_rsq), len(Avg_R_SQUARE))

    avg_res = pd.DataFrame({"Feature": FEATURES,
                            "School": Sch_rsq,
                            "Chemical": Chem_rsq,
                            "Landmine": Land_rsq,
                            "Parkinsons": Park_rsq,
                            "Avg_R_SQUARE": Avg_R_SQUARE})
    # avg_res = avg_res.sort_values(by=['Avg_R_SQUARE'], ascending=False)
    avg_res.to_csv(f'{pairwise_study_path}Pairwise_Individual_Usefulness_avg_{modelName}_all.csv', index=False)

def pearson_correlation_single_task(dataset_name):
    Features = []
    Target = []
    Pearson_Corr = []
    if modelName == 'NN':
        predictor_Data = pd.read_csv(f'../Results/Pairwise/{modelName}/Pairwise_Task_Specific_Features_{dataset_name}.csv',
                                     low_memory=False)
    else:
        predictor_Data = pd.read_csv(
            f'../Results/Pairwise/{modelName}/Pairwise_Task_Specific_Features_{dataset_name}_{modelName}.csv',
            low_memory=False)
    # print(predictor_Data.columns)


    cols_to_consider = list(predictor_Data.columns)
    cols_to_consider.remove('Change')
    for col in cols_to_consider:
        # col = list(col)
        corr, _ = pearsonr(predictor_Data[col], predictor_Data['Change'])
        Features.append(col)
        # Target.append(col[1])
        Pearson_Corr.append(round(corr, 5))

    # Features = change_features(Features)

    if modelName != 'NN':
        for i in range(len(Features)):
            F = Features[i]

            if '1' in F or '2' in F:
                F = F.replace('1', '$t_i$')
                F = F.replace('2', '$t_i$')
            # F = F.replace('_', ' ')
            F = F.replace('_Task', ' ')
            F = F.replace('Task', '')
            F = F.replace('StdDev', 'Target $\\sigma$')
            F = F.replace('Variance', 'Target $\\sigma^2$')
            Features[i] = F
    else:
        for i in range(len(Features)):
            F = Features[i]
            print(F)
            F = F.replace('_', ' ')
            # F = F.replace('Task', '')
            F = F.replace('StdDev', 'Target $\\sigma$')
            F = F.replace('Variance', 'Target $\\sigma^2$')
            F = F.replace('lc task1', '$\\text{Curve grad}_{t_i}$')
            F = F.replace('lc task2', '$\\text{Curve grad}_{t_i}$')

            for perc in [10,20,30,40,50,60,70,80,90]:
                perc = str(perc)
                if perc in F:
                    F = F.replace(perc, f'({perc}\%)')
                    break


            if 'Task1' in F or 'Task2' in F:
                F = F.replace('Task1', '$t_i$')
                F = F.replace('Task2', '$t_i$')
            # F = F.replace('_', ' ')
            F = F.replace('Task', '')
            Features[i] = F
            print(F)


    Corr = pd.DataFrame({"Feature": Features,
                         # "GroupwiseMTLGain": Target,
                         f"Pearson_Corr": Pearson_Corr})

    remove_feat = ['InterAffinity','Weight','$t_i$']
    # Corr = Corr[Corr.Feature not in remove_feat]
    Corr = Corr[Corr.Feature!='InterAffinity']
    Corr = Corr[Corr.Feature!='Weight']
    Corr = Corr[Corr.Feature!='$t_i$']
    Corr['PearsonCorrelationABS'] = round(abs(Corr['Pearson_Corr']), 4)
    Corr = Corr[['Feature', 'Pearson_Corr', 'PearsonCorrelationABS']]
    Corr = Corr.sort_values(by=['Pearson_Corr'], ascending=True)


    print(f'{dataset_name} = \n{Corr}')
    print(len(Corr))
    # Corr = Corr.head(10)
    Corr.to_csv(f'{pairwise_study_path}PearsonCorrelation_TaskSpecific_{dataset_name}_{modelName}.csv', index=False)


if __name__ == '__main__':

    for modelName in ['NN', 'SVM', 'XGBoost']:
    
        traindatapath = f'../Results/Pairwise/{modelName}/train/'
        testdatapath = f'../Results/Pairwise/{modelName}/test/'
        if modelName == 'NN':
            architecture_datapath = '../Results/NAS_Results/pairwise/'
        else:
            architecture_datapath = f'../Results/NAS_Results/pairwise/{modelName}/'
            
        pairwise_study_path = '../Results/Pairwise_Study/'
        
    
        print('Get pearson correlation for Task-Relation features')
        for dataset_name in ['School','Chemical','Landmine','Parkinsons']:
            pearson_correlation(dataset_name)
            pearson_correlation_single_task(dataset_name)
        
        print('Get individual usefulness for Task-Relation features')
        for dataset_name in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
            for run in range(1, 6):
                MSE = []
                R_Square = []
                Variance = []
                Feature = []
                Architecture = []
                train_data = pd.read_csv(f'{traindatapath}Pairwise_Task_Relation_Features_{dataset_name}_train.csv')
                test_data = pd.read_csv(f'{testdatapath}Pairwise_Task_Relation_Features_{dataset_name}_test.csv')
    
                result_df = individual_usefulness(train_data, test_data, dataset_name)
                result_df.sort_values(by=['R_Square'], ascending=False, inplace=True)
                result_df.to_csv(f'{pairwise_study_path}Pairwise_Individual_Usefulness_{dataset_name}_run_{run}_{modelName}.csv', index=False)
        
        print('Get final usefulness for pairwise-task affinity predictor')
        for dataset_name in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
            for run in range(1, 4):
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
                result_df.sort_values(by=['R_Square'], ascending=False, inplace=True)
                result_df.to_csv(f'{pairwise_study_path}Pairwise_Final_Usefulness_{dataset_name}_run_{run}_{modelName}.csv', index=False)
    
        print('Get average individual usefulness for Task-Relation features')
        FINAL_USEFULNESS = []
        FINAL_FEATURES = []
        DATASET_NAME = []
        for dataset_name in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
            train_data = pd.read_csv(f'{traindatapath}Pairwise_Task_Relation_Features_{dataset_name}_train.csv')
            test_data = pd.read_csv(f'{testdatapath}Pairwise_Task_Relation_Features_{dataset_name}_test.csv')
            get_average_usefulness(dataset_name)
            DATASET_NAME.append(dataset_name)
            feat, r_sq = get_average_usefulness_for_all(dataset_name)
            FINAL_USEFULNESS.append(round(r_sq * 100, 2))
            FINAL_FEATURES.append(feat)
        print(len(FINAL_USEFULNESS), len(FINAL_FEATURES), len(DATASET_NAME))
        res_all = pd.DataFrame({'Dataset': DATASET_NAME,
                                'R_Square': FINAL_USEFULNESS})
        print(f'res_all = \n{res_all}')
        res_all.to_csv(f'{pairwise_study_path}Pairwise_Final_Usefulness_{modelName}.csv', index=False)
        
        dataset_average_usefulness()
