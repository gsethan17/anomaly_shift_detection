import os
import sys

import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from config import DATA_PATH, COI, Input_columns


class DataLoader(object):
    def __init__(self, name="None", down=True):
        self.name = name
        self.raw_df = pd.DataFrame([])
        
        self.dfs = []
        self.anomaly = []
        self.shift = []
        self.cur_gear = []
        self.tar_gear = []
        self.folder = []
        
        self.get_data(down)
        
    def get_shift_index(self, df):
        prev_gear = df['TarGear'].values[:-1]
        cur_gear = df['TarGear'].values[1:]
        
        shift = cur_gear - prev_gear
        
        up_shift = np.where(shift == 1)[0]
        down_shift = np.where(shift == -1)[0]
        
        return up_shift, down_shift
        
    def get_interest_rows(self, df, idx):
        if idx >= 50:
            sub_df = df[(idx-50):(idx+200)].copy()
            
            if len(sub_df.loc[idx:(idx+50), 'DriveMode'].unique()) == 1:
                if sub_df.loc[idx, 'DriveMode'] == 2:
                    return True, True, sub_df
                elif sub_df.loc[idx, 'DriveMode'] == 0:
                    return True, False, sub_df
                else:
                    return False, None, None
                    
            else:
                return False, None, None
        else:
            return False, None, None
            
    def stack_data(self, folder, anomaly, shift, cur, tar, df):
        self.folder.append(folder)
        self.anomaly.append(anomaly)
        self.shift.append(shift)
        self.cur_gear.append(cur)
        self.tar_gear.append(tar)
        self.dfs.append(df[COI])
        
    def get_shift_data(self, path, down):
        df = pd.read_csv(path)
        
        # insert folder name
        folder = os.path.basename(os.path.dirname(path))
        
        up_shift, down_shift = self.get_shift_index(df)
        
        for up_idx in up_shift:
            if down:
                break
            
            is_data, anomaly, sub_df = self.get_interest_rows(df, up_idx)
            if not is_data:
                continue
            cur_gear = sub_df['TarGear'].values[0]
            tar_gear = sub_df['TarGear'].values[1]
            self.stack_data(folder, anomaly, 'U', cur_gear, tar_gear, sub_df.copy())
            
        for down_idx in down_shift:
            is_data, anomaly, sub_df = self.get_interest_rows(df, down_idx)
            if not is_data:
                continue
            cur_gear = sub_df['TarGear'].values[0]
            tar_gear = sub_df['TarGear'].values[1]
            self.stack_data(folder, anomaly, 'D', cur_gear, tar_gear, sub_df.copy())
                
    def get_raw_data_paths(self):
        return glob(os.path.join(DATA_PATH, '*', 'total_log.csv'))
        
    def get_data(self, down):
        data_paths = self.get_raw_data_paths()
        print("[I] Loading Raw Data...")
        for data_path in tqdm(data_paths):
            self.get_shift_data(data_path, down)
            
    def split_train_test(self):
        
        num_anomaly = self.anomaly.count(True)
        
        total_y = np.array(self.anomaly)
        normal_idx = np.where(total_y == False)[0]
        anomaly_idx = np.where(total_y == True)[0]
        
        # random choice of normal data for test
        test_normal = np.random.choice(normal_idx, num_anomaly)
        
        # merge for total test data
        self.test_idx = np.concatenate([test_normal, anomaly_idx], axis=0)
        self.test_Y = np.array([False for i in range(num_anomaly)]+[True for i in range(num_anomaly)])
        
        # train Data
        self.train_idx = np.delete(normal_idx, test_normal)
        
    def get_x(self, i):
        df = self.dfs[i]
        x = df[Input_columns].values
        
        return x
        
    
    def get_y(self, i):
        y = self.anomaly[i]
        
        return y
        
    def get_train_data(self, batch_size):
        mini_i = np.random.choice(self.train_idx, batch_size)
        batch_x = np.array([self.get_x(i) for i in mini_i])
        
        return batch_x
    
    def get_test_data(self):
        batch_x = np.array([self.get_x(i) for i in self.test_idx])
        batch_y = np.array([self.get_y(i) for i in self.test_idx])
        
        return batch_x, batch_y
    ######################################################################################
    def get_scaler(self):
        transformer = MinMaxScaler()
        train_all_df = pd.DataFrame(columns=Input_columns)
        
        for i in self.train_idx:
            tmp_df = pd.DataFrame(self.get_x(i), columns=Input_columns)
            train_all_df = pd.concat([train_all_df, tmp_df], ignore_index=True)
            
        scaler = transformer.fit(train_all_df.values)
        
        return scaler
    
    def get_scaled_train_data(self, scaler):
        train_data = np.array([self.get_x(i) for i in self.train_idx if self.get_x(i).shape[0]==250])
        
        scaled_train_data=[]
        for i in train_data:
            scaled_train_data.append(scaler.transform(i))
        
        scaled_train_data = np.array(scaled_train_data)
        
        return scaled_train_data
    
    def get_scaled_test_data(self, scaler):
        test_data = np.array([self.get_x(i) for i in self.test_idx if self.get_x(i).shape[0]==250])
        
        scaled_test_data=[]
        for i in test_data:
            scaled_test_data.append(scaler.transform(i))
        
        scaled_test_data = np.array(scaled_test_data)
        
        return scaled_test_data