import pandas as pd
import numpy as np
import time
from time import strptime
from datetime import datetime
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="inventory prev manager", add_help=False)

    parser.add_argument('--json_name', type=str, default='./datas.json', help='json file from loading')

    parser.add_argument('--crt_from_bg', type=str, default='./datas.json', help='create from beginning')

    args = parser.parse_args()

    return args

def main(args):
    crt_from_bg = False
    if crt_from_bg:
        json_name = args.json_name
        print(json_name)
        with open(json_name, 'r') as f:
            json_file = json.load(f)
            f.close()
        # file_list = json_file['file_list']
        file_list = ['./datas/1-4.xlsx']
        table_column_dic = json_file['table_column_dict']
        
        df = pd.read_excel(file_list[0])
        df = df.drop_duplicates([table_column_dic['company_name']])
        sr = df[table_column_dic['company_name']]
        # with open('./company_name_list.txt', 'w') as fsr:
        #     # print(sr)
        #     fsr.write(str(sr))
        #     fsr.close()
        sr.to_excel('./company_name_list.xlsx')
    df = pd.read_excel('./company_name_list.xlsx')
    jd = json.dumps(df['企业名称'].astype(str).values.tolist(), ensure_ascii=False)
    print(jd)
    with open('./company_name_list.json', 'w+') as f:
        json.dump(jd, f, ensure_ascii=False)
        f.close()
if __name__ == '__main__':
    args = parse_args()
    main(args)