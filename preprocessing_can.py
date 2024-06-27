import os
from glob import glob
import sys
import pandas as pd

import yaml
with open('columns_rename.yaml') as f:
    rename_dic = yaml.load(f, Loader=yaml.FullLoader)

with open('columns_replace.yaml') as f:
    replace_dics = yaml.load(f, Loader=yaml.FullLoader)
    
with open('columns_drop.yaml') as f:
    drop_cols = yaml.load(f, Loader=yaml.FullLoader)

data_path = '/media/imlab/HDD/gear_anomaly/2. Raw Data/'
to_path = '/media/imlab/HDD/gear_anomaly/5. DS Preprocessed Data/TestSet/'

drop_col = []

def replace_categorical_value(df):
    for col, from_list in replace_dics.items():
        if col == 'AccDep':
            df[col].replace(to_replace=from_list, value=[0, 254, 255], inplace=True)
        elif col == 'VehSpdClu':
            df[col].replace(to_replace=from_list, value=[0, 255], inplace=True)
        else:
            df[col].replace(to_replace=from_list, value=range(len(from_list)), inplace=True)
    return df

def drop_columns(df):
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df

def main():
    sports_csv=[]
    for folder in os.listdir(data_path):
        can_path = glob(os.path.join(data_path, folder, 'CAN', '*.csv'))[0]
        
        df = pd.read_csv(can_path)
        
        # rename column
        df.rename(columns=rename_dic, inplace=True)
        
        df.reset_index(inplace=True, drop=True)
        
        df = replace_categorical_value(df)
        
        if not 2. in df['DriveMode'].unique():
            print(f'{folder} data include Sports mode !!!')
            sports_csv.append(folder)
            continue

        new_df = drop_columns(df)
        
        new_df = new_df[['Timestamp', 'TarGear', 'LatAccel', 'LongAccel', 'YawRate', 'SAS', 'EngStat',
                        'BrkDep', 'AccDep', 'EngRPM', 'WhlSpdFL', 'WhlSpdFR', 'WhlSpdRL',
                        'WhlSpdRR', 'EngColTemp', 'VehSpdClu', 'DriveMode']]
        
        to_csv_path = os.path.join(to_path, f'{folder}.csv')
        new_df.to_csv(to_csv_path, index=False)
    
if __name__ == '__main__':
    if not os.path.isdir(to_path):
        os.makedirs(to_path)
    
    main()