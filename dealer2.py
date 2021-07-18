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
from utils import cls_elimate_list

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
                        default='./jsons/datas2017.json', help='json file from loading')
    # parser.add_argument('--l_per', type=int, default=0.975, help='largest_per for filter')
    # parser.add_argument('--s_per', type=int, default=0.025, help='smallest_per for filter')

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

# def linear_inter(np_sr, category, table_column_dict):
#         '''
#         这部分是使用置信区间筛选，由于数据分布问题，未采用。
#         new_np = np_sr[np_sr!=0]
#         low, high = mean_confidence_interval(new_np)
#         new_np[new_np<low]=0
#         new_np[new_np>high]=0
#         total_size = np.size(new_np)
#         in_confidence_size = np.size(new_np[new_np!=0])
#         print(f'count {column} : {in_confidence_size} / {total_size} = {(in_confidence_size/total_size*100):.3f} %')
#         '''
#         '''
#         线性插值
#         '''
#         # np_sr = df[column].to_numpy()
#         np_sr[np_sr<0.0] = 0.0
#         inter_array = np.arange(len(np_sr))
#         if np_sr[0]==0:
#             # inter_sr = np.interp(inter_array, np.concatenate((np.array([0]),inter_array[np_sr!=0])), np.concatenate((np.array([0]), np_sr[np_sr!=0]))) 
#             if category == table_column_dict['flow']:
#                 np_sr[0] = 1
#             else:
#                 np_sr[0] = 1e-5
#         inter_sr = np.interp(inter_array, inter_array[np_sr>0], np_sr[np_sr>0])
#         return inter_sr

def linear_inter_sr(sr, is_flow=False):
    np_sr = sr.to_numpy()
    if np.count_nonzero(np_sr) == 0:
        return pd.Series(np_sr)
    # print(f'np_sr:{np_sr}')
    np_sr[np_sr<0.0] = 0.0
    inter_array = np.arange(len(np_sr))
    # if np_sr[0]==0:
        # if is_flow:
            # np_sr[0] = 1
        # else:
            # np_sr[0] = 1e-5
    
    inter_sr = np.interp(inter_array, inter_array[np_sr!=0], np_sr[np_sr!=0])
    # print(f'inter_sr:{inter_sr}')
    return pd.Series(inter_sr)

def merge_company_list(prev_company_list, cur_company_list, table_column_dict,mode=0):
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
        # if prev_company_dic[key] != prev_company_dic[key]:
        if ret_value != cur_company_dic[key]:
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
    if len(prev_company_list)>2 and len(cur_company_list)>2:
        if table_column_dict['use_limitation']:
            ret_company_list = [ret_company_dic, ret_df, cur_company_list[2], cur_company_list[3]]
        else:
            ret_company_list = [ret_company_dic, ret_df, cur_company_list[2]]
    else:
        ret_company_list = [ret_company_dic, ret_df]

    return ret_company_list

def screen_filter(company_dict, table_column_dict, out_dir, out_version, mode=None, selected_list=None, largest_per=0.975, smallest_per=0.025, interval_confidence=None):
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
    
    def filter_sr(sr, smallest_per, largest_per, table_column_dict,use_limit=False, limitation=None, interval_confidence=None):
        '''
        e1 只用上下限，返回（上下限+异常值）/全部数据
        e2 上下限+95%，返回（上下限+95%+异常值）/全部数据
        异常值被赋成<0，原本数据中=0的不算异常值，一起算入95%
        
        '''
        print(f'smallest_per:{smallest_per}, largest_per:{largest_per}')
        if interval_confidence is None:
            q_low = sr[sr>=0].quantile(smallest_per).item()
            q_high = sr[sr>=0].quantile(largest_per).item()
            
        else:
            a = 1.0*np.array(sr)
            n = len(a)
            m, se = np.mean(a>=0), st.sem(a>=0)
            h = se * st.ppf((1+interval_confidence)/2.0, n-1)
            q_low = m-h
            q_high = m+h
        print(f'interval:{interval_confidence}, q_low:{q_low}, q_high:{q_high}, limitation:{limitation}')

        if use_limit:
            low_sr2 = sr.loc[(sr<q_low) & (sr <limitation[0]) & (sr>=0)]
            high_sr2 = sr.loc[(sr>q_high) & (sr>limitation[1])]

            # low_sr = sr.loc[(sr<q_low) & (sr!=0)]
            # high_sr = sr.loc[(sr>q_high)]
            low_sr = sr.loc[(sr<limitation[0]) & (sr>=0)]
            high_sr = sr.loc[(sr>limitation[1])]

            zero_sr = sr.loc[sr<0]
            if table_column_dict['use_95_filter']:
                sr.mask((sr<q_low) & (sr<limitation[0]), other=0.0, inplace=True)
                sr.mask((sr>q_high) & (sr>limitation[1]), other=0.0, inplace=True)
            else:
                sr.mask((sr<limitation[0]), other=0.0, inplace=True)
                sr.mask((sr>limitation[1]), other=0.0, inplace=True)

            # e1 = len(low_sr.index)+len(high_sr.index)
            # e2 = len(low_sr2.index)+len(high_sr2.index)
            # tl = len(sr.index) - len(zero_sr.index)
            e1 = len(low_sr.index)+len(high_sr.index)+len(zero_sr.index)
            e2 = len(low_sr2.index)+len(high_sr2.index)+len(zero_sr.index)
            tl = len(sr.index)
            print(f'use_limit:low_sr.in{len(low_sr.index)}, high_sr.in:{len(high_sr.index)}, low_sr2.in:{len(low_sr2.index)}, high_sr2.in:{len(high_sr2.index)}, sr.in:{len(sr.index)}, zero:{len(zero_sr.index)}, e1:{e1}, e2:{e2}, tl:{tl}')
        else:
            low_sr = sr.loc[(sr<q_low) & (sr>=0)]
            high_sr = sr.loc[(sr>q_high)]
            zero_sr = sr.loc[sr<0]

            sr.mask((sr<q_low), other=0.0, inplace=True)
            sr.mask((sr>q_high), other=0.0, inplace=True)

            # e1 = len(low_sr.index)+len(high_sr.index)
            # e2 = 0
            # tl = len(sr.index) - len(zero_sr.index)
            e1 = 0
            e2 = len(low_sr.index)+len(high_sr.index)+len(zero_sr.index)
            tl = len(sr.index)
        
        return sr, e1, e2, tl

    def get_limitation_name(input_category, category_list=['pm','nox','so2'], em_limit_module=['emi_{}_low', 'emi_{}_high']):
        input_category = input_category.lower()
        # print(f'in_cat:{input_category}')
        for category in category_list:
            # print(f'cat:{category}')
            if category in input_category:

                return [item.format(category) for item in em_limit_module]

    # month_filter_df = pd.DataFrame(columns=['company_name', 'year', table_column_dict['pm']+'_e1', table_column_dict['pm']+'_e2', table_column_dict['nox']+'_e1', table_column_dict['nox']+'_e2', table_column_dict['so2']+'_e1', table_column_dict['so2']+'_e2'])
    # year_filter_df = pd.DataFrame(columns=['company_name', 'year', 'month', table_column_dict['pm']+'_e1', table_column_dict['pm']+'_e2', table_column_dict['nox']+'_e1', table_column_dict['nox']+'_e2', table_column_dict['so2']+'_e1', table_column_dict['so2']+'_e2'])
    
    month_filter_df = None
    year_filter_df = None

    for key, value in company_dict.items():
        df = value[1]
        # print(df)
        sub_month_filter_df = None
        sub_year_filter_df = None
        # print(f'key：{key}\nvalue:{value}')
        # column_list = [table_column_dict['flow'],table_column_dict['pm'], table_column_dict['nox'], table_column_dict['so2'], 'pm_by_factor', 'nox_by_factor', 'so2_by_factor']
        # # filter by month

        # for column in column_list:
        #     np_sr = df[column].to_numpy()
        #     inter_sr = linear_inter(np_sr)
        #     df[column] = pd.Series(inter_sr)
        # delete flow
        # column_list = [table_column_dict['pm'], table_column_dict['nox'], table_column_dict['so2'], 'pm_by_factor', 'nox_by_factor', 'so2_by_factor']
        # column_list = [table_column_dict['pm'], table_column_dict['nox'], table_column_dict['so2']]
        # filter_column_list = [table_column_dict['pm'], table_column_dict['nox'], table_column_dict['so2']]
        filter_column_list = [table_column_dict['out_pm_concentration'], table_column_dict['out_so2_concentration'], table_column_dict['out_nox_concentration']]

        if mode is not None:
            # def set_top_zero(sub_df, largest_per, smallest_per, column_list):
            #     largest_numbers = int(len(sub_df.index)*largest_per)
            #     smallest_numbers = int(len(sub_df.index)*smallest_per)
            #     for column in column_list:
            #         sub_df[column].mask(sub_df[column].isin(sub_df[column].nsmallest(smallest_numbers)), other=0, inplace=True)
            #         sub_df[column].mask(sub_df[column].isin(sub_df[column].nlargest(largest_numbers)), other=0, inplace=True)
            df['year'] = df[table_column_dict['monitor_time']].dt.year
            df['month'] = df[table_column_dict['monitor_time']].dt.month
            df[table_column_dict['out_pm_concentration']] = df[table_column_dict['pm']]/df[table_column_dict['flow']]

            df[table_column_dict['out_so2_concentration']]=df[table_column_dict['so2']]/df[table_column_dict['flow']]

            df[table_column_dict['out_nox_concentration']]=df[table_column_dict['nox']]/df[table_column_dict['flow']]
            # print(df)
            year_list = pd.unique(df['year'])
            month_list = pd.unique(df['month'])
            if mode == 0:
                # print('refilter autoly by month')
                for year in year_list:
                    for month in month_list:
                        sub_df = df.loc[(df['year']==year) & (df['month']==month)].copy()
                        # flow
                        print(f'comp:{key}, year:{year}, month:{month}')
                        # print(sub_df)
                        print(f'filter flow')
                        sr, e1, e2, tl = filter_sr(sub_df[table_column_dict['flow']].copy(), smallest_per, largest_per, table_column_dict,use_limit=False, interval_confidence=interval_confidence)
                        df.loc[(df['year']==year) & (df['month']==month), table_column_dict['flow']] = sr
                        # print(f'year:{type(year)}, {year}, month:{type(month)}, {month}')
                        time_stamp = pd.to_datetime(str(year.item())+str(month.item()).zfill(2), format='%Y%m', errors='ignore')
                        value[2].add(table_column_dict['flow'],time_stamp, e1, e2, tl)

                        for column in filter_column_list:
                            has_limit_flag = False
                            print(f'category:{column}')
                            if table_column_dict['use_limitation'] and value[3]['has_limit']:
                                limitation_list = get_limitation_name(column)
                                # print(f'limitation_list:{limitation_list}')
                                em_limitation = [value[3][item] for item in limitation_list]
                                has_limit_flag = value[3]['has_limit']
                                # print(f'comp:{key}, em_limitation:{em_limitation}')
                                print(f'filter column')
                                sr, e1, e2, tl = filter_sr(sub_df[column].copy(), smallest_per, largest_per, table_column_dict,  use_limit=True, limitation=em_limitation, interval_confidence=interval_confidence)

                            else:
                                sr, e1, e2, tl = filter_sr(sub_df[column].copy(), smallest_per, largest_per, table_column_dict, use_limit=True, limitation=[0.0, 1e16], interval_confidence=interval_confidence)
                            # print(sr)
                            
                            df.loc[(df['year']==year) & (df['month']==month), column] = sr
                            
                            time_stamp = pd.to_datetime(str(year.item())+str(month.item()).zfill(2), format='%Y%m', errors='ignore')
                            if column in filter_column_list:
                                value[2].add(column, time_stamp, e1, e2, tl, has_limit_flag)
                


                        # largest_numbers = int(len(sub_df.index)*largest_per)
                        # smallest_numbers = int(len(sub_df.index)*smallest_per)
                        # for column in column_list:
                        #     sub_df[column].mask(sub_df[column].isin(sub_df[column].nsmallest(smallest_numbers)), other=0, inplace=True)
                        #     sub_df[column].mask(sub_df[column].isin(sub_df[column].nlargest(largest_numbers)), other=0, inplace=True)
                        # df.loc[(df['year']==year) & (df['month']==month), column]=sub_df

            # if mode == 1:
            #     print('refilter by month single year')
            #     for year in year_list:
            #         for months in selected_list:
            #             sub_df = df.loc[df['year']==year & df['month'].isin(months)]
            #             sub_df = df.loc[(df['year']==year) & (df['month']==month)].copy()
            #             largest_numbers = int(len(sub_df.index)*largest_per)
            #             smallest_numbers = int(len(sub_df.index)*smallest_per)
            #             for column in column_list:
            #                 sub_df[column].mask(sub_df[column].isin(sub_df[column].nsmallest(smallest_numbers)), other=0, inplace=True)
            #                 sub_df[column].mask(sub_df[column].isin(sub_df[column].nlargest(largest_numbers)), other=0, inplace=True)
            #                 df.loc[(df['year']==year) & (df['month']==month), column]=sub_df
            ret_dict = value[2].cal_month_year()
            # print(f'ret_dict:{ret_dict}')
            for column in [table_column_dict['flow']]+filter_column_list:
                # if sub_month_filter_df is None:
                   
                local_month_filter_df = ret_dict[column].copy()
                local_month_filter_df.drop(local_month_filter_df.tail(1).index, inplace=True)
                # print(f"local time:{local_month_filter_df['time']}, {type(local_month_filter_df['time'][0])}")
                local_month_filter_df['time'] = pd.to_datetime(local_month_filter_df['time'], errors='ignore')
                local_month_filter_df['year']=local_month_filter_df['time'].dt.year
                local_month_filter_df['month']=local_month_filter_df['time'].dt.month
                local_month_filter_df[column+'_e1']=local_month_filter_df['elimate1_percent']
                local_month_filter_df[column+'_e2']=local_month_filter_df['elimate2_percent']
                local_month_filter_df = local_month_filter_df[['year', 'month', column+'_e1', column+'_e2']]
                # local_month_filter_df.drop(local_month_filter_df.tail(1).index, inplave=True)

                local_year_filter_df = ret_dict[column].tail(1).copy()
                local_year_filter_df['time'] = pd.to_datetime(local_year_filter_df['time'], errors='ignore')
                # local_year_filter_df['time'] = local_month_filter_df['time'][0]
                local_year_filter_df['year'] = local_year_filter_df['time'].dt.year
                local_year_filter_df[column+'_e1'] = local_year_filter_df['elimate1_percent']
                local_year_filter_df[column+'_e2'] = local_year_filter_df['elimate2_percent']
                local_year_filter_df = local_year_filter_df[['year', column+'_e1', column+'_e2']]

                if sub_month_filter_df is None:
                    sub_month_filter_df = local_month_filter_df
                    sub_year_filter_df = local_year_filter_df
                    # print(f'l_m_f:{local_month_filter_df},\n l_y_f:{local_year_filter_df}')

                else:
                    local_month_filter_df.drop(columns=['year', 'month'], inplace=True)
                    local_year_filter_df.drop(columns=['year'], inplace=True)
                    # print(f'l_m_f:{local_month_filter_df},\n l_y_f:{local_year_filter_df}')

                    sub_month_filter_df = pd.concat([sub_month_filter_df, local_month_filter_df], axis=1)
                    sub_year_filter_df = pd.concat([sub_year_filter_df, local_year_filter_df], axis=1)
                
                # print(f's_m_f:{sub_month_filter_df},\n s_y_f:{sub_year_filter_df}')
            sub_month_filter_df.insert(0,table_column_dict['company_name'], key)
            # sub_month_filter_df.insert(-1, 'use_limit', str(ret_dict['has_limit']))
            sub_month_filter_df.loc[:,'use_limit'] = str(ret_dict['has_limit'])
            sub_year_filter_df.insert(0, table_column_dict['company_name'], key)
            # sub_year_filter_df.insert(-1, 'use_limit', str(ret_dict['has_limit']))
            sub_year_filter_df.loc[:,'use_limit'] = str(ret_dict['has_limit'])
            
            if month_filter_df is None:
                month_filter_df = sub_month_filter_df
                year_filter_df = sub_year_filter_df
            else:
                month_filter_df = pd.concat([month_filter_df, sub_month_filter_df], axis=0)
                year_filter_df = pd.concat([year_filter_df, sub_year_filter_df],  axis=0)
                # print(f'm_m_f:{month_filter_df},\ny_y_f:{year_filter_df}')
            # print(f'month_f_Df:{month_filter_df}')
            # print(f'year_f_df:{year_filter_df}')
        df[table_column_dict['flow']] = linear_inter_sr(df[table_column_dict['flow']], is_flow=True)
        for column in filter_column_list:
            # np_sr = df[column].to_numpy()
            # inter_sr = linear_inter(np_sr, column, table_column_dict)
            # df[column] = pd.Series(inter_sr)
            try:
                df[column] = linear_inter_sr(df[column], is_flow=False)
            except ValueError:
                print(f'key:{key}')
                print(f'column:{column}, \ndf[c]:{df[column]}, \ndf:{df}')
                assert(0)
        df.drop(columns=['year', 'month'], inplace=True)
        df[table_column_dict['pm']] = df[table_column_dict['out_pm_concentration']]*df[table_column_dict['flow']]
        df[table_column_dict['so2']] = df[table_column_dict['out_so2_concentration']]*df[table_column_dict['flow']]
        df[table_column_dict['nox']] = df[table_column_dict['out_nox_concentration']]*df[table_column_dict['flow']]
        value[1] = df
    if table_column_dict['use_limitation']:
        month_save_name = out_dir+'month_filter_use_limit'+str(out_version)+'.xlsx'
        year_save_name = out_dir+'year_filter_use_limit'+ str(out_version)+'.xlsx'
    else:
        month_save_name = out_dir+'month_filter_no_limit'+str(out_version)+'.xlsx'
        year_save_name = out_dir+'year_filter_no_limit'+ str(out_version)+'.xlsx'
    month_filter_df.to_excel(month_save_name, index=None)
    year_filter_df.to_excel(year_save_name, index=None)
    print(f'out month_filter:{month_save_name}\nout year_filter:{year_save_name}')


        
def compute_inventory(company_dict, table_column_dict, out_dir, out_version):
    def is_unique(sr):
        a = sr.to_numpy()
        return (a[0]==a).all()

    print(f'Start compute output inventory!')
    '''
    按照计算公式计算
    
    '''
    total_sum_df = None
    for key, value in company_dict.items():
        df = value[1]

        df[table_column_dict['year_burning']]=value[0]['year_burning']
        # df['burn_type_factor'] = df['burn_type_factor'].mean()
        # assert(is_unique(df['burn_type_factor']))
        total_flow = df[table_column_dict['flow']].sum()
        df[table_column_dict['hour_burn']]=df[table_column_dict['flow']]*value[0]['year_burning']/total_flow

        # df[table_column_dict['out_pm_concentration']]=df[table_column_dict['pm']]/df[table_column_dict['flow']]

        # df[table_column_dict['out_pm']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * (df['pm_by_factor']/df[table_column_dict['flow']])
        # df[table_column_dict['out_pm']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * (df[table_column_dict['pm']]*df['burn_type_factor']/df[table_column_dict['flow']])
        df[table_column_dict['out_pm']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * df[table_column_dict['out_pm_concentration']]*df['burn_type_factor']
        
        # df[table_column_dict['out_so2_concentration']]=df[table_column_dict['so2']]/df[table_column_dict['flow']]

        # df[table_column_dict['out_so2']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * (df['so2_by_factor']/df[table_column_dict['flow']])
        # df[table_column_dict['out_so2']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * (df[table_column_dict['so2']]*df['burn_type_factor']/df[table_column_dict['flow']])
        df[table_column_dict['out_so2']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * df[table_column_dict['out_so2_concentration']]*df['burn_type_factor']

        # df[table_column_dict['out_nox_concentration']]=df[table_column_dict['nox']]/df[table_column_dict['flow']]

        # df[table_column_dict['out_nox']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * (df['nox_by_factor']/df[table_column_dict['flow']])
        # df[table_column_dict['out_nox']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * (df[table_column_dict['nox']]*df['burn_type_factor']/df[table_column_dict['flow']])
        df[table_column_dict['out_nox']]=df[table_column_dict['hour_burn']] * value[0]['theory_smoke'] * df[table_column_dict['out_nox_concentration']]*df['burn_type_factor']

        # df.drop(columns=['pm_by_factor', 'so2_by_factor', 'nox_by_factor'], inplace=True)
        df.drop(columns=['burn_type_factor'], inplace=True)

        df_time_month = df[table_column_dict['monitor_time']].dt.month.astype(int).values

        month_list = [i for i in range(1,13)]
        missing_month_list = []
        missing_month_str =''
        for month in month_list:
            if not month in df_time_month:
                missing_month_list.append(month)
        start_month = None
        for month in missing_month_list:
            if start_month is None:
                start_month = month
                missing_month_str+=str(month)
            else:
                if month==start_month+1:
                    start_month+=1
                else:
                    add_str = '-'+str(start_month)+', '+str(month)
                    missing_month_str+=add_str
                    start_month = month
        if not missing_month_str == '':
            last_month = int(missing_month_str.split(', ')[-1])
            if start_month>last_month:
                missing_month_str = missing_month_str+'-'+str(start_month)
        # print(f'm_m_l:{missing_month_list}')
        df_sum = df.sum(axis=0)
        # print(f'df_sum:{type(df_sum)}\n{df_sum}')
        df_sum[table_column_dict['monitor_time']] = str(df[table_column_dict['monitor_time']][0].year)
        df_sum[table_column_dict['year_burning']] = df_sum[table_column_dict['year_burning']]/len(df.index)
        df_sum[table_column_dict['out_pm_concentration']] = df_sum[table_column_dict['out_pm_concentration']]/len(df.index)
        df_sum[table_column_dict['out_so2_concentration']] = df_sum[table_column_dict['out_so2_concentration']]/len(df.index)
        df_sum[table_column_dict['out_nox_concentration']] = df_sum[table_column_dict['out_nox_concentration']]/len(df.index)
        df = df.append(df_sum, ignore_index=True)
        df_sum = df_sum.append(pd.Series(missing_month_str, index=['missing_month']))
        df_sum = df_sum.to_frame().T
        df_sum.insert(0, table_column_dict['company_name'], key)
        if total_sum_df is None:
            total_sum_df = df_sum
        else:
            total_sum_df = pd.concat([total_sum_df, df_sum], axis=0)
        # print(f'df.index:{df.index},\n{df}')

        value[1] = df
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        save_name = out_dir +str(key)+str(out_version)+'.xlsx'
        value[1].to_excel(save_name, index=None)
        print(f'output {save_name}')

    if table_column_dict['use_limitation']:
        total_sum_df_save_name = out_dir + 'total_sum_use_limit' + str(out_version)+'.xlsx'
    else:
        total_sum_df_save_name = out_dir + 'total_sum_no_limit' + str(out_version)+'.xlsx'

    total_sum_df.to_excel(total_sum_df_save_name, index=None)

    print(f'output total_sum:{total_sum_df_save_name}')
    # print(f'total_sum_df:{total_sum_df}')

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
    company_emission_limitation_path = json_file['company_emission_limitation_path']
    table_column_dict = json_file['table_column_dict']
    burn_type_dict = json_file['burn_type_dict']
    # flow_exception_name_list = json_file['flow_exception_name_list']
    exception_name_list = json_file['exception_name_list']
    use_total_empty_month = json_file['use_total_empty_month']
    out_dir = json_file['out_dir']
    out_version = json_file['out_version']
    start_time_list = json_file['start_time_list']
    end_time_list = json_file['end_time_list']
    time_format = json_file['time_format']
    time_freq = json_file['time_freq']
    emi_factor = json_file['emi_factor']
    smallest_per = json_file['smallest_per']
    largest_per = json_file['largest_per']
    if use_total_empty_month:
        out_version += '_use_empty'
    else:
        out_version += '_no_empty'
    if table_column_dict['use_95_filter']:
        out_version += '_e2'
    else:
        out_version += 'e1'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # print(f'file_list:{file_list}, out_dir:{out_dir}, out_version:{out_version}')
    
    '''
    company_dict 用于存放公司及其 pandas.DataFrame。其形式为
    company_dict={
        company_name:   company_list=[
                            smoke_burning_dict={
                                'theory_smoke' : **,
                                'year_burning' : **,
                            },
                            sub_df, 
                            elimate_list=cls_elimate_list,
                            em_limit_dict:{
                                'emi_so2_high':**,
                                'emi_so2_low':**,
                                'emi_nox_high':**,
                                'emi_nox_low':**,
                                'emi_pm_high':**,
                                'emi_pm_low':**,
                            }
                        ]
    }
    company_dict 是一个字典，key 为 company_name。value 是一个 company_list，其中有两个元素。company_list[0] 为一个字典，记录理论烟气量与年燃烧量；company_list[1] 为 pandas.DataFrame，用于存放该公司相关数据。company_list[2] 为一个class，记录筛除元素数，并提供计算比例方法。company_list[3] 为筛选时上下限

    '''
    '''
        e1 只用上下限，返回（上下限+异常值）/全部数据
        e2 上下限+95%，返回（上下限+95%+异常值）/全部数据
        异常值被赋成<0，原本数据中=0的不算异常值，一起算入95%
        
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
    # print(f'smoke_burning_dict_total:{smoke_burning_dict_total}')
    if table_column_dict['use_limitation']:
        print(f'reading emission limitation info from:{company_emission_limitation_path}')

        em_limit_dict_total = {}
        em_limit_total_df = pd.read_excel(company_emission_limitation_path)
        for index, row in em_limit_total_df.iterrows():
            cp_name = row[table_column_dict['emi_company_name']]
            em_limit_dict ={
                'emi_so2_high':float(row[table_column_dict['emi_so2_high']])*emi_factor,
                'emi_so2_low':float(row[table_column_dict['emi_so2_low']])*emi_factor,
                'emi_nox_high':float(row[table_column_dict['emi_nox_high']])*emi_factor,
                'emi_nox_low':float(row[table_column_dict['emi_nox_low']])*emi_factor,
                'emi_pm_high':float(row[table_column_dict['emi_pm_high']])*emi_factor,
                'emi_pm_low' :float(row[table_column_dict['emi_pm_low']])*emi_factor
            }
            em_limit_dict_total[cp_name] = em_limit_dict
    # print(f'em_li_tl:{em_limit_dict_total}')

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
            if table_column_dict['use_limitation']:
                if company_name in em_limit_dict_total.keys():
                    em_limit_dict = em_limit_dict_total[company_name]
                    em_limit_dict['has_limit'] = True
                else:
                    em_limit_dict = {
                        'emi_so2_high':1e16,
                        'emi_so2_low':0.0,
                        'emi_nox_high':1e16,
                        'emi_nox_low':0.0,
                        'emi_pm_high':1e16,
                        'emi_pm_low' :0.0
                    }
                    em_limit_dict['has_limit'] = False
            # smoke_burning_dict = {
            #     'theory_smoke': float(subdf[table_column_dict['theory_smoke']][0]),
            #     'year_burning': float(subdf[table_column_dict['year_burning']][0]),
            #     'fuel_type':subdf[table_column_dict['fuel_type']][0],
            # }
            
            def deal_abnormal(sr, exception_name_list, other, is_flow=False):
                sr.replace(exception_name_list, other, inplace=True)
                sr = sr.astype(float)
                sr.fillna(other, inplace=True)
                # sr[sr<0.0] = other
                if is_flow:
                    sr[sr<0.0] = 0.0
                return sr
                
            subdf[table_column_dict['flow']] = deal_abnormal(subdf[table_column_dict['flow']], exception_name_list, float(0), is_flow=True)

            abnormal_list=[table_column_dict['pm'], table_column_dict['nox'], table_column_dict['so2']]

            for abnormal_category in abnormal_list:
                subdf[abnormal_category] = deal_abnormal(subdf[abnormal_category], exception_name_list, float(-1))

            '''
            补上缺的时间
            将时间相同的项合并，各项相加。
            由于时间的格式，不能直接 groupby，因此将其转换为数字形式的timestamp, 然后合并。合并完了再抓换回来。
            '''
            # subdf.drop(columns=[table_column_dict['company_name'],table_column_dict['theory_smoke'],table_column_dict['fuel_type'], 'burn_type_factor'], inplace=True)
            # subdf.drop(columns=[table_column_dict['company_name'], 'burn_type_factor'], inplace=True)
            subdf.drop(columns=[table_column_dict['company_name']], inplace=True)

            # subdf[table_column_dict['flow']] = subdf[table_column_dict['flow']].astype(float)
            subdf[table_column_dict['monitor_time']] = subdf[table_column_dict['monitor_time']].apply(
                lambda x: convert_time2num(x, time_format))
            
            subdf = subdf.groupby(table_column_dict['monitor_time']).sum()

            subdf = subdf.sort_values(by=[table_column_dict['monitor_time']]).reset_index()

            subdf[table_column_dict['monitor_time']] = subdf[table_column_dict['monitor_time']].apply(
                lambda x: convert_num2time(x))
            
            subdf[table_column_dict['monitor_time']] = pd.to_datetime(subdf[table_column_dict['monitor_time']], format=time_format)

            if use_total_empty_month:
                dateindex = pd.date_range(start_time_list[file_index], end_time_list[file_index], freq=time_freq)

                subdf = subdf.set_index(table_column_dict['monitor_time'])

                subdf = subdf.reindex(dateindex, fill_value=0)
                subdf = subdf.reset_index()
                subdf = subdf.rename(columns={'index':table_column_dict['monitor_time']})

                subdf[table_column_dict['monitor_time']] = pd.to_datetime(subdf[table_column_dict['monitor_time']], format=time_format)
            
            
            '''
            处理不正常的值
            '''
            # for flow_exception_name in flow_exception_name_list:
                # subdf[table_column_dict['flow']] = subdf[table_column_dict['flow']].replace(flow_exception_name,float(0))
            # subdf[table_column_dict['flow']] = subdf[table_column_dict['flow']].astype(float)
            #nan, negative
            
            subdf[table_column_dict['flow']] = linear_inter_sr(subdf[table_column_dict['flow']], is_flow=True)
            
            
                
            '''
            由于后续 groupby 操作，会把燃烧类型抹去，因此先根据燃烧类型，将乘了系数之后的污染物记录。
            'pm_by_factor' = 'pm' * factor
            'so2_by_factor' = 'so2' * factor
            'nox_by_factor' = 'nox' * factor
            不用by_factor了，后续还原burn_factor
            '''
            # subdf['burn_type_factor']=subdf[table_column_dict['fuel_type']].apply(lambda x: burn_type_dict[x])
            subdf['burn_type_factor'] = float(burn_type_dict[smoke_burning_dict['fuel_type']])

            # subdf[['pm_by_factor', 'so2_by_factor', 'nox_by_factor']]=subdf[[table_column_dict['pm'], table_column_dict['so2'], table_column_dict['nox']]].multiply(subdf['burn_type_factor'], axis='index')

            # '''
            # 去掉by_factor，重新获得burn_type_factor
            # '''
            # subdf['burn_type_factor'] = subdf['pm_by_factor']/subdf[table_column_dict['pm']]
            # subdf['burn_type_factor'] = subdf['burn_type_factor'].mean()
            # subdf.drop(columns=['pm_by_factor', 'so2_by_factor', 'nox_by_factor'], inplace=True)
            # print(subdf)
            # subdf[table_column_dict['monitor_time']] = subdf[table_column_dict['monitor_time']].dt.strftime(time_format)

            '''


            处理完了之后将其放到 company_dict 中，用于后续处理
            '''
            if table_column_dict['use_limitation']:
                company_list = [smoke_burning_dict, subdf, cls_elimate_list(), em_limit_dict]
            else:
                company_list = [smoke_burning_dict, subdf, cls_elimate_list()]


            if not company_name in company_dict:
                company_dict[company_name] = company_list
            else:
                company_dict[company_name] = merge_company_list(
                    company_dict[company_name], company_list, table_column_dict)
    # 清洗与填补
    screen_filter(company_dict, table_column_dict, out_dir, out_version, mode=0, largest_per=largest_per, smallest_per=smallest_per, interval_confidence=None)

    # 计算与生成处理后文件
    compute_inventory(company_dict, table_column_dict, out_dir, out_version)

if __name__ == '__main__':
    args = parse_args()
    main(args)
