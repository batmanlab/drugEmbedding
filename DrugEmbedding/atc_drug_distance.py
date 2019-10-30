import pandas as pd
import os
import pickle
from collections import defaultdict
from tqdm import tqdm

def lower(x):
    return x.lower()

data_dir = './data/fda_drugs'

# step 1: read ATC hireachy
df_atc = pd.read_csv(os.path.join(data_dir, 'ATC_LOOKUP.csv'))

# step 2: read FDA drugs
# Load FDA drugs
fda_drugs_file = 'all_drugs.smi'
fda_drugs_path = os.path.join(data_dir, fda_drugs_file)
drugs_dict = {}
with open(fda_drugs_path, 'r') as file:
    for seq in file.readlines():
        if seq != '\n':
            line = seq.split(" ")
            drugs_dict[lower((line[1]))] = line[0]
file.close()
fda_df = pd.DataFrame.from_dict(drugs_dict, orient='index').reset_index()
fda_df.columns = ['active_ingredient','smile_x']


# step 3: join these two datasets
# substance in both ATC and FDA
atc_fda_inner = pd.merge(df_atc, fda_df, how='inner', left_on='ATC_LVL5', right_on='active_ingredient')
atc_fda_inner['ATC_LVL0'] = 'ROOT'
atc_fda_inner.to_csv(os.path.join(data_dir, 'atc_fda_all.csv'))

# step 4: calculate the path lengths to measure the semantic similarity of drugs
def drugs_sp(d1, d2):
    if d1[0] == d2[0]:
        return 0
    elif d1[1]['ATC_LVL4'] == d2[1]['ATC_LVL4'] and d1[1]['ATC_LVL3'] == d2[1]['ATC_LVL3'] and d1[1]['ATC_LVL2'] == d2[1]['ATC_LVL2'] and d1[1]['ATC_LVL1'] == d2[1]['ATC_LVL1'] and d1[1]['ATC_LVL0'] == d2[1]['ATC_LVL0']:
        return 2
    elif d1[1]['ATC_LVL3'] == d2[1]['ATC_LVL3'] and d1[1]['ATC_LVL2'] == d2[1]['ATC_LVL2'] and d1[1]['ATC_LVL1'] == d2[1]['ATC_LVL1'] and d1[1]['ATC_LVL0'] == d2[1]['ATC_LVL0']:
        return 4
    elif d1[1]['ATC_LVL2'] == d2[1]['ATC_LVL2'] and d1[1]['ATC_LVL1'] == d2[1]['ATC_LVL1'] and d1[1]['ATC_LVL0'] == d2[1]['ATC_LVL0']:
        return 6
    elif d1[1]['ATC_LVL1'] == d2[1]['ATC_LVL1'] and d1[1]['ATC_LVL0'] == d2[1]['ATC_LVL0']:
        return 8
    else:
        return 10

# pairwise distance
drugs = atc_fda_inner['ATC_LVL5'].unique().tolist()
sp_lst = []
for idx1, row1 in tqdm(atc_fda_inner.iterrows()):
    for idx2, row2 in atc_fda_inner.iterrows():
        sp_lst.append([row1['ATC_LVL5'], row2['ATC_LVL5'], drugs_sp((idx1, row1), (idx2, row2))])

df_sp = pd.DataFrame(sp_lst, columns=['drug_target', 'drug_comparison', 'sp'])
df_sp.to_csv(os.path.join(data_dir, 'drugs_sp_all.csv'))