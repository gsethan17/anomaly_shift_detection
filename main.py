from config import DATA_PATH

import os
from glob import glob
import pandas as pd 
def main():
    print(os.listdir(DATA_PATH))
    data_paths = glob(os.path.join(DATA_PATH, '*', 'total_log.csv'))

    for path in data_paths:
        df = pd.read_csv(path)
        print(df.shape)
        print(df.columns)




if __name__ == "__main__":
    main()
