import pandas as pd
import numpy as np
import time
from time import strptime
from datetime import datetime
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="inventory prev manager", add_help=False)

    parser.add_argument('--json_name', type=str, default='./jsons/datas2017.json', help='json file from loading')

    # parser.add_argument('--crt_from_bg', type=str, default='./datas.json', help='create from beginning')
    parser.add_argument('--input_file_list', type=str, default='./datas/2017电厂/1-3.xlsx')
    parser.add_argument('--save_company_xlsx', type=str, default='./datas/2017电厂/company_name_list.xlsx')
    parser.add_argument('--save_company_json', type=str, default='./datas/2017电厂/company_name_list.json')
    args = parser.parse_args()

    return args

def main(args):
    crt_from_bg = True
    if crt_from_bg:
        json_name = args.json_name
        print(json_name)
        with open(json_name, 'r') as f:
            json_file = json.load(f)
            f.close()
        # file_list = json_file['file_list']
        # file_list = ['./datas/1-4.xlsx']
        table_column_dic = json_file['table_column_dict']
        
        df = pd.read_excel(args.input_file_list)
        df = df.drop_duplicates([table_column_dic['company_name']])
        sr = df[table_column_dic['company_name']]
        # with open('./company_name_list.txt', 'w') as fsr:
        #     # print(sr)
        #     fsr.write(str(sr))
        #     fsr.close()
        # sr.to_excel('./company_name_list.xlsx')
        sr.to_excel(args.save_company_xlsx)

    df = pd.read_excel(args.save_company_xlsx)
    jd = json.dumps(df['企业名称'].astype(str).values.tolist(), ensure_ascii=False)
    print(jd)
    with open(args.save_company_json, 'w+') as f:
        json.dump(jd, f, ensure_ascii=False)
        f.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)