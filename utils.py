import pandas as pd
import numpy as np
from pathlib import Path
import os


def hdf_creater(file_path, out_path, key='data'):
    # for i in range(1,13):
        # file_path = './站点月点位小时数据/南通{}月点位小时数据.xlsx'.format(str(i))
    print(file_path+' --> '+out_path)
    df = pd.read_excel(file_path)
    df = df.infer_objects()
    # if os.path.exists(out_path):
    #     df.to_hdf(out_path, key=key, mode='r+', format='table')
    # else:
    #     df.to_hdf(out_path, key=key, mode='w', format='table')
    df.to_hdf(out_path, key=key, mode='w', format='table')


if __name__ == '__main__':
    file_list = ["./datas/5-8.xlsx","./datas/9-12.xlsx"]
    for file_path in file_list:
        out_path = './'+ str(Path(file_path).with_suffix('.h5'))
        # print(file_path, out_path)
        hdf_creater(file_path, out_path)