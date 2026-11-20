#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fundo 用户画像数据与图表模块。

供 build_profile_decks.py / build_compare_decks.py 生成幻灯片。
"""

from . import profile_report_base as _base
from .profile_report_base import *  # noqa: F401,F403

META_COLS = _base.META_COLS.copy()
META_COLS.insert(3, "sample_month")
TEXT_COLS = _base.TEXT_COLS | {"sample_month"}
ALL_COLUMNS = _base.all_columns(META_COLS)


def load_data(path, log=print, *, extra_columns=()):
    return _base.load_data(path, log=log, extra_columns=extra_columns, columns=ALL_COLUMNS)


def clean_frame(df):
    return _base.clean_frame(df, text_cols=TEXT_COLS)


def build_metrics(df, n_apps):
    return _base.build_metrics(df, n_apps, sample_month_col="sample_month")
