import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
import os
import time
from time import strptime
from datetime import datetime
import argparse
import json
import scipy.stats as st
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

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

def convert_time2num(t, form='%Y-%m-%d %H:%M:%S.%f'):
    time_struct = strptime(t, form)
    timestamp = time.mktime(time_struct)
    return float(timestamp)
    # return float(t)

def convert_num2time(n, form='%Y-%m-%d %H:%M:%S.%f'):
    t = datetime.fromtimestamp(n).strftime(form)
    return t

def mean_confidence_interval(data, confidence=0.95): 
    a = 1.0 * np.array(data)
    a = a[a!=0]
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    ret = st.t.interval(confidence, n-1, m, se)
    low, high = ret[0], ret[1]
    return low, high

def draw_distribution(data, company_name, column_name, divide_times=10):
    final_name = company_name+'_'+column_name
    data = data[data!=0]

    low, high = int(data.min())-1, int(data.max())+1
    fig=plt.figure()
    plt.hist(data, bins=range(low,high, divide_times))

    plt.title(final_name)
    fig.savefig('./pics/'+final_name+'.png')

def merge_company_list(prev_company_list, cur_company_list, mode=0):
    '''
    if prev_theory_smoke != cur_theory_smoke:
        if mode=0:
            choose the big one
        if mode=1:
            choose average of the two
        if mode=2:
            raise error
    TODO!
    '''
    ret_company_dic = {}
    prev_company_dic = prev_company_list[0]
    cur_company_dic = cur_company_list[0]
    for key in prev_company_dic.keys():
        ret_value = prev_company_dic[key]
        if prev_company_dic[key] != prev_company_dic[key]:
            if mode == 0:
                ret_value = max(ret_value, cur_company_dic[key])
            elif mode == 1:
                ret_value = (ret_value + cur_company_dic[key])/2
            elif mode == 2:
                raise Exception(
                    f'The mode==2, and prev_theory_smoke != cur_theory_smoke! please check!')
            else:
                raise Exception(f'The mode num should be in [0,1,2]!')
        ret_company_dic[key] = ret_value

    ret_df = pd.concat(
        [prev_company_list[1], cur_company_list[1]]).reset_index(drop=True)
    ret_company_list = [ret_company_dic, ret_df]
    return ret_company_list

def screen_filter(company_dict, table_column_dict, mode=None, selected_list=None, largest_per=0.05, smallest_per=0.05):
    '''
    mode = 0:
        autoly by month
    mode = 1:
        by selected month single year
        selected_list :[
            [month1, month2, month3],
            [month4, month5],...
        ]
    mode = 2:
        by selected month multiply year
    '''
    print(f'Start screen and filter!')
    def linear_inter(np_sr):
        '''
        这部分是使用置信区间筛选，由于数据分布问题，未采用。
        new_np = np_sr[np_sr!=0]
        low, high = mean_confidence_interval(new_np)
        new_np[new_np<low]=0
        new_np[new_np>high]=0
        total_size = np.size(new_np)
        in_confidence_size = np.size(new_np[new_np!=0])
        print(f'count {column} : {in_confidence_size} / {total_size} = {(in_confidence_size/total_size*100):.3f} %')
        '''
        '''
        线性插值
        '''
        np_sr = df[column].to_numpy()
        np_sr[np_sr<0.0] = 0.0
        inter_array = np.arange(len(np_sr))
        if np_sr[0]==0:
            # inter_sr = np.interp(inter_array, np.concatenate((np.array([0]),inter_array[np_sr!=0])), np.concatenate((np.array([0]), np_sr[np_sr!=0]))) 
            if column == table_column_dict['flow']:
                np_sr[0] = 1
            else:
                np_sr[0] = 1e-5
        inter_sr = np.interp(inter_array, inter_array[np_sr!=0], np_sr[np_sr!=0])
        return inter_sr


    for key, value in company_dict.items():
        df = value[1]
        column_list = [table_column_dict['flow'],table_column_dict['pm'], table_column_dict['nox'], table_column_dict['so2'], 'pm_by_factor', 'nox_by_factor', 'so2_by_factor']
        for column in column_list:
            np_sr = df[column].to_numpy()
            inter_sr = linear_inter(np_sr)
            df[column] = pd.Series(inter_sr)
        # delete flow
        column_list = [table_column_dict['pm'], table_column_dict['nox'], table_column_dict['so2'], 'pm_by_factor', 'nox_by_factor', 'so2_by_factor']

        if mode is not None:
            # def set_top_zero(sub_df, largest_per, smallest_per, column_list):
            #     largest_numbers = int(len(sub_df.index)*largest_per)
            #     smallest_numbers = int(len(sub_df.index)*smallest_per)
            #     for column in column_list:
            #         sub_df[column].mask(sub_df[column].isin(sub_df[column].nsmallest(smallest_numbers)), other=0, inplace=True)
            #         sub_df[column].mask(sub_df[column].isin(sub_df[column].nlargest(largest_numbers)), other=0, inplace=True)
            df['year'] = df[table_column_dict['monitor_time']].dt.year
            df['month'] = df[table_column_dict['monitor_time']].dt.month
            year_list = pd.unique(df['year'])
            month_list = pd.unique(df['month'])
            if mode == 0:
                # print('refilter autoly by month')
                for year in year_list:
                    for month in month_list:
                        sub_df = df.loc[(df['year']==year) & (df['month']==month)].copy()
                        largest_numbers = int(len(sub_df.index)*largest_per)
                        smallest_numbers = int(len(sub_df.index)*smallest_per)
                        for column in column_list:
                            sub_df[column].mask(sub_df[column].isin(sub_df[column].nsmallest(smallest_numbers)), other=0, inplace=True)
                            sub_df[column].mask(sub_df[column].isin(sub_df[column].nlargest(largest_numbers)), other=0, inplace=True)
                            df.loc[(df['year']==year) & (df['month']==month), column]=sub_df

            if mode == 1:
                print('refilter by month single year')
                for year in year_list:
                    for months in selected_list:
                        sub_df = df.loc[df['year']==year & df['month'].isin(months)]
                        sub_df = df.loc[(df['year']==year) & (df['month']==month)].copy()
                        largest_numbers = int(len(sub_df.index)*largest_per)
                        smallest_numbers = int(len(sub_df.index)*smallest_per)
                        for column in column_list:
                            sub_df[column].mask(sub_df[column].isin(sub_df[column].nsmallest(smallest_numbers)), other=0, inplace=True)
                            sub_df[column].mask(sub_df[column].isin(sub_df[column].nlargest(largest_numbers)), other=0, inplace=True)
                            df.loc[(df['year']==year) & (df['month']==month), column]=sub_df
        for column in column_list:
            np_sr = df[column].to_numpy()
            inter_sr = linear_inter(np_sr)
            df[column] = pd.Series(inter_sr)
        df.drop(columns=['year', 'month'], inplace=True)
        value[1] = df
        
def compute_inventory(company_dict, table_column_dict, out_dir, out_version):
    print(f'Start compute output inventory!')
    '''
    按照计算公式计算
    '''
    for key, value in company_dict.items():
        df = value[1]

        df[table_column_dict['year_burning']]=value[0]['year_burning']

        total_flow = df[table_column_dict['flow']].sum()
        df[table_column_dict['hour_burn']]=df[table_column_dict['flow']]*value[0]['year_burning']/total_flow

        df[table_column_dict['out_pm_concentration']]=df[table_column_dict['pm']]/df[table_column_dict['flow']]

        df[table_column_dict['out_pm']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * (df['pm_by_factor']/df[table_column_dict['flow']])
        
        df[table_column_dict['out_so2_concentration']]=df[table_column_dict['so2']]/df[table_column_dict['flow']]

        df[table_column_dict['out_so2']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * (df['so2_by_factor']/df[table_column_dict['flow']])

        df[table_column_dict['out_nox_concentration']]=df[table_column_dict['nox']]/df[table_column_dict['flow']]

        df[table_column_dict['out_nox']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * (df['nox_by_factor']/df[table_column_dict['flow']])

        df.drop(columns=['pm_by_factor', 'so2_by_factor', 'nox_by_factor'], inplace=True)

        value[1] = df
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        save_name = out_dir +str(key)+str(out_version)+'.xlsx'
        value[1].to_excel(save_name, index=None)
        print(f'output {save_name}')

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
    company_burning_info_path = json_file['company_burning_info_path'] 
    table_column_dict = json_file['table_column_dict']
    burn_type_dict = json_file['burn_type_dict']
    flow_exception_name_list = json_file['flow_exception_name_list']
    out_dir = json_file['out_dir']
    out_version = json_file['out_version']
    start_time_list = json_file['start_time_list']
    end_time_list = json_file['end_time_list']
    time_format = json_file['time_format']
    time_freq = json_file['time_freq']
    # print(f'file_list:{file_list}, out_dir:{out_dir}, out_version:{out_version}')
    
    '''
    company_dict 用于存放公司及其 pandas.DataFrame。其形式为
    company_dic={
        company_name:   company_list=[
                            smoke_burning_dict={
                                'theory_smoke' : **,
                                'year_burning' : **,
                            },
                            sub_df
                        ]
    }
    company_dict 是一个字典，key 为 company_name。value 是一个 company_list，其中有两个元素。company_list[0] 为一个字典，记录理论烟气量与年燃烧量；company_list[1] 为 pandas.DataFrame，用于存放该公司相关数据。

    '''
    company_dict = {}

    print(f'reading burning info from:{company_burning_info_path}')
    company_burning_df = pd.read_excel(company_burning_info_path)

    smoke_burning_dict_total = {}
    for index, row in company_burning_df.iterrows():
        cp_name = row[table_column_dict['company_name']]
        smoke_burning_dict = {
            'theory_smoke' : float(row[table_column_dict['theory_smoke']]),
            'year_burning' : float(row[table_column_dict['year_burning']]),
            'fuel_type' : row[table_column_dict['fuel_type']]
        }
        smoke_burning_dict_total[cp_name] = smoke_burning_dict
        

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
            smoke_burning_dict = smoke_burning_dict_total[company_name]
            # smoke_burning_dict = {
            #     'theory_smoke': float(subdf[table_column_dict['theory_smoke']][0]),
            #     'year_burning': float(subdf[table_column_dict['year_burning']][0]),
            #     'fuel_type':subdf[table_column_dict['fuel_type']][0],
            # }

            '''
            由于后续 groupby 操作，会把燃烧类型抹去，因此先根据燃烧类型，将乘了系数之后的污染物记录。
            'pm_by_factor' = 'pm' * factor
            'so2_by_factor' = 'so2' * factor
            'nox_by_factor' = 'nox' * factor
            '''
            # subdf['burn_type_factor']=subdf[table_column_dict['fuel_type']].apply(lambda x: burn_type_dict[x])
            subdf['burn_type_factor'] = float(burn_type_dict[smoke_burning_dict['fuel_type']])

            subdf[['pm_by_factor', 'so2_by_factor', 'nox_by_factor']]=subdf[[table_column_dict['pm'], table_column_dict['so2'], table_column_dict['nox']]].multiply(subdf['burn_type_factor'], axis='index')

            '''
            处理不正常的值
            '''
            for flow_exception_name in flow_exception_name_list:
                subdf[table_column_dict['flow']] = subdf[table_column_dict['flow']].replace(flow_exception_name,float(0))
            subdf[table_column_dict['flow']] = subdf[table_column_dict['flow']].astype(float)

            '''
            补上缺的时间
            将时间相同的项合并，各项相加。
            由于时间的格式，不能直接 groupby，因此将其转换为数字形式的timestamp, 然后合并。合并完了再抓换回来。
            '''
            # subdf.drop(columns=[table_column_dict['company_name'],table_column_dict['theory_smoke'],table_column_dict['fuel_type'], 'burn_type_factor'], inplace=True)
            subdf.drop(columns=[table_column_dict['company_name'], 'burn_type_factor'], inplace=True)
                       
           
            subdf[table_column_dict['monitor_time']] = subdf[table_column_dict['monitor_time']].apply(
                lambda x: convert_time2num(x, time_format))
            
            subdf = subdf.groupby(table_column_dict['monitor_time']).sum()

            subdf = subdf.sort_values(by=[table_column_dict['monitor_time']]).reset_index()

            subdf[table_column_dict['monitor_time']] = subdf[table_column_dict['monitor_time']].apply(
                lambda x: convert_num2time(x))
            
            subdf[table_column_dict['monitor_time']] = pd.to_datetime(subdf[table_column_dict['monitor_time']], format=time_format)
            
            dateindex = pd.date_range(start_time_list[file_index], end_time_list[file_index], freq=time_freq)

            subdf = subdf.set_index(table_column_dict['monitor_time'])

            subdf = subdf.reindex(dateindex, fill_value=0)
            subdf = subdf.reset_index()
            subdf = subdf.rename(columns={'index':table_column_dict['monitor_time']})

            subdf[table_column_dict['monitor_time']] = pd.to_datetime(subdf[table_column_dict['monitor_time']], format=time_format)

            # subdf[table_column_dict['monitor_time']] = subdf[table_column_dict['monitor_time']].dt.strftime(time_format)

            '''


            处理完了之后将其放到 company_dict 中，用于后续处理
            '''
            company_list = [smoke_burning_dict, subdf]

            if not company_name in company_dict:
                company_dict[company_name] = company_list
            else:
                company_dict[company_name] = merge_company_list(
                    company_dict[company_name], company_list)
    # 清洗与填补
    screen_filter(company_dict, table_column_dict, mode=0, largest_per=0.05, smallest_per=0.05)

    # 计算与生成处理后文件
    compute_inventory(company_dict, table_column_dict, out_dir, out_version)

if __name__ == '__main__':
    args = parse_args()
    main(args)
