import pandas as pd

dataset = 'Parkinsons'
df = pd.read_csv(f'../Results/NAS_Results/groupwise/xgBoost/Parkinsons_Groupwise_NAS_105_group_dataset_size_xgBoost.csv')
df.to_csv(f'../Results/NAS_Results/groupwise/xgBoost/{dataset}_Groupwise_NAS_group_dataset_size_xgBoost.csv', index=False)