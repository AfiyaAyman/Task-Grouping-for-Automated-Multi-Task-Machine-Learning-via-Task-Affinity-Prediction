import ast

import pandas as pd
import os
import numpy as np




# df = pd.read_csv(f'../Results/partition_sample/{modelName}/Chemical_URS_MTL_{modelName}_500_v_50.csv')
#
# group_sample = df.sample(200, random_state=0)
# group_sample.to_csv(f'../Results/partition_sample/{modelName}/Chemical_group_sample_{modelName}.csv', index=False)

# for cluster_type in ['Exponential', 'NonNegative']:
#     df_0 = pd.read_csv(f'../Results/Clusters/{modelName}/Chemical_Cluster_Results_{modelName}_{cluster_type}_Hierarchical_50_0.csv')
#     df_3 = pd.read_csv(f'../Results/Clusters/{modelName}/Chemical_Cluster_Results_{modelName}_{cluster_type}_Hierarchical_run_0_50.csv')
#     LIST = [df_0, df_3]
#     df = pd.concat(LIST)
#     print(len(df))
#     df.to_csv(f'../Results/Clusters/{modelName}/Chemical_Cluster_Results_{modelName}_{cluster_type}_Hierarchical_run_0.csv', index=False)
modelName = 'xgBoost'
for dataset in ['School','Chemical','Landmine', 'Parkinsons']:
    algo = 'Hierarchical'
    # algo = 'KMeans'
    run = 0
    for cluster_type in ['Exponential', 'NonNegative']:
        if modelName == 'SVM':
            if algo == 'Hierarchical':
                csv = f'../Results/Clusters/{modelName}/{dataset}_Cluster_Results_{modelName}_{cluster_type}_{algo}_run_{run}.csv'
            else:
                csv = f'../Results/Clusters/{modelName}/{dataset}_Cluster_Results_{modelName}_{cluster_type}_{algo}_run_{run}_0.csv'
        else:
            csv = f'../Results/Clusters/{modelName}/{dataset}_Cluster_Results_{modelName}_{cluster_type}_{algo}.csv'

        if os.path.exists(csv):
            df = pd.read_csv(csv)
            print(f'dataset = {dataset}, algo = {algo}, cluster_type = {cluster_type}, Best Loss = {min(df.Total_Loss)}')
            if dataset == 'Chemical':
                best_idx = list(df.Total_Loss).index(min(df.Total_Loss))
                err = df.Individual_Error_Rate[best_idx]
                err = ast.literal_eval(err)
                print(f'Error rate = {np.mean(list(err.values()))}')
                # print(f'Error rate = {min(df.Individual_Error_Rate[0])}')
            if dataset == 'Landmine':
                # print(df.columns)
                best_idx = list(df.Total_Loss).index(min(df.Total_Loss))
                auc = df.Individual_AUC[best_idx]
                auc = ast.literal_eval(auc)
                print(f'AUC = {auc}, {np.mean(list(auc.values()))}')
