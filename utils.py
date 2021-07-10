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

class cls_elimate_list:
    def __init__(self) -> None:
        '''
        self.elimate_list={
            f{category}:f{month_time_stamp}:{
                [elimate1, elimate2, total]
            }
        }
        '''
        self.elimate_list={}
        self.has_limit = False
    def add(self, category, month_time_stamp, elimate1, elimate2, total, has_limit_flag=False):
        '''
            elimate1:上下浓度范围+异常值，筛去的元素数
            elimate2:每个月95%和上下浓度范围+异常值，筛去的元素数
            total:总元素数
        '''
        if has_limit_flag:
            self.has_limit = True
        if not category in self.elimate_list.keys():
        # if not month_time_stamp in self.elimate_list.keys():
            self.elimate_list[category] = {}
        self.elimate_list[category][month_time_stamp]=[elimate1, elimate2, total]
        # else:
            # old_single = self.elimate_list[category][month_time_stamp]
            # old_single[0] = old_single[0] + elimate1
            # old_single[1] = old_single[1] + elimate2
            # old_single[2] = old_single[2] + total
            # self.elimate_list[category][month_time_stamp] = old_single
        return self.elimate_list
    # def df_add(self, )
    def cal_month_year(self):
        category_keys = self.elimate_list.keys()
        ret_dict = {}
        for cat_key in category_keys:
            sorted_elimate_list = [(key, value) for(key, value) in sorted(self.elimate_list[cat_key].items(), key=lambda x:x[0])]
            df = pd.DataFrame(columns=['time','elimate1', 'elimate2', 'total', 'elimate1_percent', 'elimate2_percent'])
            year_e1 = 0
            year_e2 = 0
            year_tl = 0
            # print(f'sorted_elimate_list:{sorted_elimate_list}')
            for elimate_single in sorted_elimate_list:
                time_stamp = elimate_single[0]
                e1 = elimate_single[1][0]
                e2 = elimate_single[1][1]
                tl = elimate_single[1][2]
                if tl==0:
                    tl = 1
                # print(f'e1:{e1}, e2:{e2}, tl:{tl}')
                df = df.append({'time':time_stamp,'elimate1':e1,'elimate2':e2,'total':tl, 'elimate1_percent':e1/tl, 'elimate2_percent':e2/tl},ignore_index=True)
                year_e1 += e1
                year_e2 += e2
                year_tl += tl
            # print(f'time_stamp:{type(time_stamp)}, {time_stamp}')
            df = df.append({'time':time_stamp, 'elimate1':year_e1, 'elimate2':year_e2, 'total':year_tl, 'elimate1_percent':year_e1/year_tl,'elimate2_percent':year_e2/year_tl}, ignore_index=True)
            ret_dict[cat_key] = df
            ret_dict['has_limit'] = self.has_limit
        return ret_dict


if __name__ == '__main__':
    file_list = ["./datas/5-8.xlsx","./datas/9-12.xlsx"]
    for file_path in file_list:
        out_path = './'+ str(Path(file_path).with_suffix('.h5'))
        # print(file_path, out_path)
        hdf_creater(file_path, out_path)