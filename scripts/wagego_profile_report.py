#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WageGo 用户画像数据与图表模块。

供 build_profile_decks.py / build_compare_decks.py 生成幻灯片。
WageGo 的样本月份由 sample_datetime 推导，因此直接复用共享基建的默认实现。
"""

from . import profile_report_base as _base
from .profile_report_base import *  # noqa: F401,F403

META_COLS = list(_base.META_COLS)
TEXT_COLS = set(_base.TEXT_COLS)
ALL_COLUMNS = list(_base.ALL_COLUMNS)

load_data = _base.load_data
clean_frame = _base.clean_frame
build_metrics = _base.build_metrics
