import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

modelName = 'NN'
group_samples_path = f'../Results/partition_sample/group_sample/'
if not os.path.exists(f'{group_samples_path}train'):
    os.makedirs(f'{group_samples_path}train')
if not os.path.exists(f'{group_samples_path}test'):
    os.makedirs(f'{group_samples_path}test')
for dataset in ['School', 'Chemical', 'Landmine', 'Parkinsons']:
    entire_dataset = pd.read_csv(f'{group_samples_path}Data_for_GroupPredictor_{dataset}_{modelName}.csv', low_memory=False)
    # if dataset == 'School':
    #     entire_dataset = entire_dataset.sample(1200, random_state=0)
    print(f'shape of entire data = {np.shape(entire_dataset)}')

    entire_dataset = pd.read_csv(f'../Results/partition_sample/NN/{dataset}_partition_sample_MTL.csv', low_memory=False)
    print(f'shape of entire data = {np.shape(entire_dataset)}')

    train,test = train_test_split(entire_dataset, test_size=0.4, random_state=999)
    print(f'dataset = {dataset}, len = {len(entire_dataset)}, train = {len(train)}, test = {len(test)}')
    print(f'train index = {train.index}\n\ntest index = {test.index}')

    train.to_csv(f'{group_samples_path}train/Groupwise_Features_{dataset}_train.csv', index=False)
    test.to_csv(f'{group_samples_path}test/Groupwise_Features_{dataset}_test.csv', index=False)

for modelName in ['SVM', 'XGBoost']:
    group_samples_path = f'../Results/partition_sample/{modelName}/group_sample/'
    if not os.path.exists(f'{group_samples_path}train'):
        os.makedirs(f'{group_samples_path}train')
    if not os.path.exists(f'{group_samples_path}test'):
        os.makedirs(f'{group_samples_path}test')
    for dataset in ['School', 'Chemical', 'Landmine', 'Parkinsons']:

        entire_dataset = pd.read_csv(f'{group_samples_path}Data_for_GroupPredictor_{dataset}_{modelName}.csv', low_memory=False)
        # if dataset == 'School':
        #     entire_dataset = entire_dataset.sample(1200, random_state=0)
        print(f'shape of entire data = {np.shape(entire_dataset)}')


        train,test = train_test_split(entire_dataset, test_size=0.4, random_state=999)
        print(f'dataset = {dataset}, len = {len(entire_dataset)}, train = {len(train)}, test = {len(test)}')
        # print(f'train index = {train.index}\n\ntest index = {test.index}')
        # exit(0)
        train.to_csv(f'{group_samples_path}train/Groupwise_Features_{dataset}_train.csv', index=False)
        test.to_csv(f'{group_samples_path}test/Groupwise_Features_{dataset}_test.csv', index=False)
