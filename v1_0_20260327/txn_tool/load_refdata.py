# -*- coding: utf-8 -*-
import os
import pandas as pd
from . import config


def __load_csvdata(folder_name):
    """读取文件夹下所有csv数据.

    Parameters
    ----------
    folder_name : str
        文件夹名称
    """
    folder_path = os.path.join(config.ref_dir, folder_name)
    files = list(filter(lambda x: ".csv" in x, os.listdir(folder_path)))
    df = pd.concat([pd.read_csv(os.path.join(folder_path, f)) for f in files])

    return df


def load_third_category(folder_name):
    """读取业务组产出APP分类数据.
    Parameters
    ----------
    folder_name : str
        文件夹名称,
    """
    
    df = __load_csvdata(folder_name=folder_name)
    # df["app_category"] = [str(x).upper() for x in df["app_category"]]

    return df
