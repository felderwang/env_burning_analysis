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
    parser.add_argument('--l_per', type=int, default=0.975, help='largest_per for filter')
    parser.add_argument('--s_per', type=int, default=0.025, help='smallest_per for filter')

    args = parser.parse_args()

    return args

def main(args):
    