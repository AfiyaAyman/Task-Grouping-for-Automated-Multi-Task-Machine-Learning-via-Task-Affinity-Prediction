import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


for modelName in ['NN', 'XGBoost', 'SVM']:
    pairwise_results_path = f'../Results/Pairwise/{modelName}/'
    if not os.path.exists(f'{pairwise_results_path}train'):
        os.makedirs(f'{pairwise_results_path}train')
    if not os.path.exists(f'{pairwise_results_path}test'):
        os.makedirs(f'{pairwise_results_path}test')
    for dataset in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
        entire_dataset = pd.read_csv(f'{pairwise_results_path}Pairwise_Task_Relation_Features_{dataset}.csv', low_memory=False)
        print(f'shape of entire data = {np.shape(entire_dataset)}')

        # continue


        train,test = train_test_split(entire_dataset, test_size=0.25, random_state=42)
        print(f'dataset = {dataset}, len = {len(entire_dataset)}, train = {len(train)}, test = {len(test)}')
        train.to_csv(f'{pairwise_results_path}train/Pairwise_Task_Relation_Features_{dataset}_train.csv', index=False)
        test.to_csv(f'{pairwise_results_path}test/Pairwise_Task_Relation_Features_{dataset}_test.csv', index=False)
