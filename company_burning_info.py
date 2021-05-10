import pandas as pd
import numpy as np
import os
import time
from time import strptime
from datetime import datetime
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="inventory prev manager", add_help=False)

    parser.add_argument('--file_list_mode', type=int, default=0,
                        help='0: load file_list from json, 1:auto load from dir')
    parser.add_argument('--file_list_dir', type=str,
                        default='./datas/', help='directory that files in.')
    parser.add_argument('--company_name_mode', type=int, default=0,
                        help='0: load from json, 1:auto load(this will spend a lot of time!)')
    parser.add_argument('--json_name', type=str,
                        default='./datas.json', help='json file from loading')

    args = parser.parse_args()

    return args

def main(args):
    '''
    首先解析 json 文件，读取所需内容
    '''
    json_name = args.json_name
    # print(json_name)
    with open(json_name, 'r') as f:
        json_file = json.load(f)
    company_name_list = json_file['company_name_list']
    file_list = json_file['file_list']
    table_column_dict = json_file['table_column_dict']
    burn_type_dict = json_file['burn_type_dict']
    flow_exception_name_list = json_file['flow_exception_name_list']
    out_dir = json_file['out_dir']
    out_version = json_file['out_version']
    start_time_list = json_file['start_time_list']
    end_time_list = json_file['end_time_list']
    time_format = json_file['time_format']
    time_freq = json_file['time_freq']

    company_dict = {}

    for file_index, filename in enumerate(file_list):
        '''
        解析源文件中，将不同公司名的数据分隔开，放到 df_list 中，之后一一访问处理。
        '''

        df_list = []
        print(f'reading_file :{filename}')
        df = pd.read_excel(filename)
        print(f'Start prev managering.')

        for i in range(len(company_name_list)):
            subdf = df.loc[df[table_column_dict['company_name']] == company_name_list[i]]
            if not subdf.empty:
                df_list.append(subdf)
        '''
        开始处理
        '''
        for subdf in df_list:
            subdf=subdf.reset_index(drop=True)
            company_name = subdf[table_column_dict['company_name']][0]

            smoke_burning_dict = {
                'theory_smoke': float(subdf[table_column_dict['theory_smoke']][0]),
                'year_burning': float(subdf[table_column_dict['year_burning']][0]),
                'fuel_type':subdf[table_column_dict['fuel_type']][0],
            }
            company_dict[company_name] = [smoke_burning_dict['theory_smoke'], smoke_burning_dict['year_burning'], smoke_burning_dict['fuel_type']]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    save_name = out_dir +'company_info'+str(out_version)+'.xlsx'
    print(f'outname:{save_name}')
    df = pd.DataFrame.from_dict(company_dict, orient='index', columns=[table_column_dict['theory_smoke'], table_column_dict
    ['year_burning'], table_column_dict['fuel_type']])
    df = df.reset_index()
    df = df.rename(columns={'index':table_column_dict['company_name']})
    df.to_excel(save_name, index=None)

if __name__ == '__main__':
    args = parse_args()
    main(args)