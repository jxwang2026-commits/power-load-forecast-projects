import numpy as np
import pandas as pd
from pathlib import Path

def data_preprocessing(file_path):
    root_path=Path(__file__).resolve().parent.parent
    data_path=root_path/'data'/file_path

    data=pd.read_csv(data_path)

    # data['time']=pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    data['time']=pd.to_datetime(data['time'])

    data.sort_values('time',ascending=True,inplace=True)

    data.drop_duplicates(inplace=True)

    return data

if __name__ == '__main__':
    data_preprocessing()
    