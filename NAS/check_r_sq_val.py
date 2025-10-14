import sys

import pandas as pd
import os
import sys

#'''
dataset = sys.argv[1]
length = sys.argv[2]
print(f'\n\n***********************dataset = {dataset}************************')
df_all = pd.read_csv(f'../Results/NAS_Results/{dataset}_Groupwise_NAS_{length}_ALL_NN.csv')
print(f'MAX = {max(df_all.Usefulness_Score)}')
'''
predictor_data = pd.read_csv(f'../Results/partition_sample/group_sample/train/Groupwise_Features_School_train.csv')
All_Columns = list(predictor_data.columns)
for dataset in ['School','Chemical', 'Landmine',  'Parkinsons']:
    print(f'\n\n***********************dataset = {dataset}************************')
    for col in All_Columns:
        if os.path.exists(f'../Results/NAS_Results/{dataset}_Groupwise_NAS_{col}_NN.csv'):
            df = pd.read_csv(f'../Results/NAS_Results/{dataset}_Groupwise_NAS_{col}_NN.csv')
            print(f'col = {col}, MAX = {max(df.Usefulness_Score)}')

    print(f'**ALL**')

'''