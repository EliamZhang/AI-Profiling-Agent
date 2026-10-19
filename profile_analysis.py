#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用户画像分析：流式读取宽表特征，输出可复用的中间统计结果。

本脚本只消费 txn_tool 已计算好的特征列，不重复实现特征公式。
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.1.0"
PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "input_path": str(PROJECT_ROOT / "sample_with_feature_all_1k.csv"),
    "dict_path": str(PROJECT_ROOT / "dict.csv"),
    "output_dir": str(PROJECT_ROOT / "analysis" / "out"),
    "cache_dir": str(PROJECT_ROOT / "analysis" / "cache"),
    "chunk_size": 5000,
    "reservoir_size": 20000,
    "segment_reservoir_size": 3000,
    "subset_cache": "auto",              # auto | on | off
    "subset_cache_threshold_mb": 300,
    "target_column": "duedate_1m_5",
    "target_label": "首期逾期≥5天",
    "target_positive_value": 1,
    "gate_column": "txn_raw_input_cnt",
    "gate_min": 1,
    "export_user_scores": True,
    "seed": 42,
    "min_feature_valid": 30,
    "min_dimension_coverage": 0.30,
    "min_health_coverage": 0.60,
    "hist_bins": 20,
    "allow_missing_features": False,
    "max_unique_track": 1000000,
    "report_title": "交易行为用户画像分析报告",
    "report_name": "profiling_report.html",
    "score_low": 40,
    "score_high": 70,
    "lender_net_tolerance": 0.05,
    "lender_top_n": 10,
    "lender_recent_share_high": 0.50,
    "lender_institution_high": 3,
    "income_stable_min_months": 5,
    "income_stable_cv": 0.50,
    "income_stable_wage_share": 0.50,
    "income_unstable_wage_share": 0.30,
    "segment_thresholds": {
        "premium_health": 68,
        "premium_max_risk_flags": 0,
        "stable_income_score": 50,
        "stable_cashflow_score": 50,
        "stable_debt_score": 45,
        "tight_cashflow_score": 40,
        "tight_liquidity_score": 40,
        "highrisk_debt_score": 30,
        "highrisk_min_flags": 3,
        "moon_saving_rate": 0.05,
    },
    "flag_thresholds": {
        "high_debt_score": 40,
        "multi_lender_cnt": 3,
        "gambling_share": 0.01,
        "low_buffer_negative_share": 0.20,
        "fees_share": 0.02,
        "income_unstable_cv": 0.50,
        "welfare_income_share": 0.50,
    },
}

PATH_KEYS = ("input_path", "dict_path", "output_dir", "cache_dir")

MODULE_LABELS = {
    "balance": "账户余额",
    "category": "交易分类",
    "income": "收入",
    "expense": "支出",
    "surplus": "结余",
    "worthiness": "资质与消费",
    "dishonour": "拒付",
    "industry": "行业与雇主",
    "lender": "借贷机构",
    "derived": "派生指标",
    "meta": "元数据",
    "other": "其他",
}

DIMENSION_META = {
    "income": {
        "label": "收入稳定性",
        "desc": "工资占比、发薪规律、收入波动与来源集中度",
        "scored": True,
        "weight": 0.24,
        "color": "#2f6fed",
    },
    "cashflow": {
        "label": "收支储蓄",
        "desc": "储蓄率、收支结余、固定支出负担与支出波动",
        "scored": True,
        "weight": 0.26,
        "color": "#0f8b8d",
    },
    "liquidity": {
        "label": "流动性缓冲",
        "desc": "账户余额厚度、负余额天数、发薪后余额与备用月数",
        "scored": True,
        "weight": 0.20,
        "color": "#7c5cff",
    },
    "debt": {
        "label": "负债压力",
        "desc": "借贷频率、负债提款、拒付与费用负担；得分越高压力越低",
        "scored": True,
        "weight": 0.22,
        "color": "#d97706",
    },
    "engagement": {
        "label": "消费活跃度",
        "desc": "活跃天数、交易频率与消费类别覆盖",
        "scored": True,
        "weight": 0.08,
        "color": "#0891b2",
    },
    "consumption": {
        "label": "消费结构",
        "desc": "各类支出占比与品质生活支出，仅作观察不参与健康分",
        "scored": False,
        "weight": 0.0,
        "color": "#64748b",
    },
    "confidence": {
        "label": "数据置信度",
        "desc": "活跃密度、账户数与最近工资时间，仅用于判断评分可信度",
        "scored": False,
        "weight": 0.0,
        "color": "#94a3b8",
    },
    "trend": {
        "label": "近期收支变化", "desc": "近28天与此前154天的日均收支比较，仅作观察",
        "scored": False, "weight": 0.0, "color": "#0d9488",
    },
    "resilience": {
        "label": "支出负担与缓冲", "desc": "刚性支出对收入的负担、期末余额可覆盖的支出天数，仅作观察",
        "scored": False, "weight": 0.0, "color": "#9333ea",
    },
}

SCORED_DIMS = [k for k, v in DIMENSION_META.items() if v["scored"]]
HEALTH_WEIGHTS = {k: DIMENSION_META[k]["weight"] for k in SCORED_DIMS}

SEGMENT_ORDER = [
    "优质均衡",
    "稳健工薪",
    "均衡普通",
    "月光消费",
    "现金流紧张",
    "高风险承压",
    "数据不足",
]

SEGMENT_COLORS = {
    "优质均衡": "#15803d",
    "稳健工薪": "#2f6fed",
    "均衡普通": "#64748b",
    "月光消费": "#d97706",
    "现金流紧张": "#c2410c",
    "高风险承压": "#b91c1c",
    "数据不足": "#94a3b8",
}

COMPETITOR_PATTERN = r"^bank_txn_lender_disburse_competitor_.+_count$"
COMPETITOR_AMOUNT_PATTERN = r"^bank_txn_lender_disburse_competitor_.+_amount_sum$"
COMPETITOR_COUNT_RE = re.compile(r"^bank_txn_lender_disburse_competitor_(.+)_count$")
LENDER_PRODUCT_PATTERN = (
    r"^bank_txn_lender_(?:disburse|repay)_(?:loan|bnpl|advance|bank)"
    r"_(?:amount_sum|count)_l182d$"
)
LENDER_WINDOW_PATTERN = (
    r"^bank_txn_lender_(?:disburse|repay)_(?:amount_sum|amount_count|jgs)"
    r"_l(?:7|28|84|182)d$"
)
LENDER_ANALYSIS_PATTERNS = (
    COMPETITOR_AMOUNT_PATTERN,
    LENDER_PRODUCT_PATTERN,
    LENDER_WINDOW_PATTERN,
)
LENDER_PRODUCTS = [
    ("loan", "个人贷款"),
    ("advance", "现金贷/工资贷"),
    ("bnpl", "BNPL"),
    ("bank", "银行"),
]
LENDER_BEHAVIORS = [
    "无借贷记录",
    "近28天无借贷",
    "仅偿还",
    "仅新增借款",
    "净偿还",
    "借还平衡",
    "净融资扩张",
    "数据不足",
]
UNRESOLVED_COLUMNS = [
    "bank_txn_dishonour_personal_loan_unresolved_cnt_3gp_182d",
    "bank_txn_dishonour_bnpl_unresolved_cnt_3gp_182d",
    "bank_txn_dishonour_cash_wage_advance_unresolved_cnt_3gp_182d",
    "bank_txn_dishonour_bank_unresolved_cnt_3gp_182d",
    "bank_txn_dishonour_debt_collection_unresolved_cnt_3gp_182d",
    "bank_txn_dishonour_other_unresolved_cnt_3gp_182d",
]
DISHONOUR_TYPES = [
    ("personal_loan", "个人贷款"),
    ("bnpl", "BNPL"),
    ("cash_wage_advance", "现金贷/工资贷"),
    ("bank", "银行"),
    ("debt_collection", "债务催收"),
    ("other", "其他"),
]
DISHONOUR_STATUSES = ["无拒付", "已全部解决", "部分解决", "未解决", "数据不足"]
RESOLVED_CNT_COLUMNS = [
    f"bank_txn_dishonour_{code}_resolved_cnt_3gp_182d" for code, _ in DISHONOUR_TYPES
]
RESOLVED_AMT_COLUMNS = [
    f"bank_txn_dishonour_{code}_resolved_amt_3gp_182d" for code, _ in DISHONOUR_TYPES
]
UNRESOLVED_AMT_COLUMNS = [
    f"bank_txn_dishonour_{code}_unresolved_amt_3gp_182d" for code, _ in DISHONOUR_TYPES
]
UNRESOLVED_INST_COLUMNS = [
    f"bank_txn_dishonour_{code}_unresolved_institution_cnt_3gp_182d" for code, _ in DISHONOUR_TYPES
]
CATCHUP_CNT_COLUMNS = [
    f"bank_txn_dishonour_{code}_resolved_catchup_cnt_3gp_182d" for code, _ in DISHONOUR_TYPES
]
INCOME_ARCHETYPES = ["稳定工资", "不稳定工资", "多源/非工资", "福利依赖", "无收入", "数据不足"]
PAY_RHYTHMS = ["月薪", "双周薪", "周薪", "不规律", "无工资/未识别"]
INCOME_ANALYSIS_COLUMNS = (
    "bank_txn_industry_pattern_monthly",
    "bank_txn_industry_pattern_fortnightly",
    "bank_txn_industry_pattern_weekly",
    "bank_txn_industry_pattern_irregular",
)


def infer_module(column: str) -> str:
    m = re.match(r"bank_txn_([a-z_]+?)_", column)
    return m.group(1) if m else "other"


def _f(
    key: str,
    column: str,
    label: str,
    dimension: str,
    direction: int = 0,
    weight: float = 0.0,
    fmt: str = "ratio",
    definition: str = "",
    hist_scale: str = "linear",
) -> dict:
    module = infer_module(column)
    return {
        "key": key,
        "column": column,
        "label": label,
        "dimension": dimension,
        "direction": direction,
        "weight": float(weight),
        "fmt": fmt,
        "definition": definition,
        "hist_scale": hist_scale,
        "module": module,
        "sentinel": "lender" if module == "lender" else "std",
        "sources": [column],
        "derived": False,
    }


def _d(
    key: str,
    label: str,
    dimension: str,
    kind: str,
    direction: int = 0,
    weight: float = 0.0,
    fmt: str = "ratio",
    definition: str = "",
    numerator: str | None = None,
    denominator: str | None = None,
    numerator_sources: list[str] | None = None,
    denominator_sources: list[str] | None = None,
    sources: list[str] | None = None,
    source_pattern: str | None = None,
    hist_scale: str = "linear",
) -> dict:
    src = list(sources or [])
    if numerator:
        src.append(numerator)
    if denominator:
        src.append(denominator)
    src.extend(numerator_sources or [])
    src.extend(denominator_sources or [])
    return {
        "key": key,
        "column": "derived",
        "label": label,
        "dimension": dimension,
        "direction": direction,
        "weight": float(weight),
        "fmt": fmt,
        "definition": definition,
        "hist_scale": hist_scale,
        "module": "derived",
        "sentinel": "derived",
        "sources": src,
        "source_pattern": source_pattern,
        "kind": kind,
        "numerator": numerator,
        "denominator": denominator,
        "numerator_sources": numerator_sources or [],
        "denominator_sources": denominator_sources or [],
        "derived": True,
    }


# ---------------------------------------------------------------------------
# 特征注册表：只引用 txn_tool 已产出的列；direction=0 仅观察，不参与评分
# ---------------------------------------------------------------------------

FEATURES = [
    # 收入稳定性
    _f("inc_wage_share", "bank_txn_worthiness_wage_income_amt_share_in_income_tag_182d", "工资收入占比", "income", 1, 1.2, "pct", "工资收入 / 全部收入（182天，金额口径）"),
    _f("inc_wage_sum", "bank_txn_income_wages_sum_182d", "工资收入总额", "income", 1, 0.8, "amount", "近182天工资入账金额合计", "log"),
    _f("inc_salary_cv", "bank_txn_worthiness_salary_amount_cv_182d", "工资金额变异系数", "income", -1, 1.0, "ratio", "工资入账金额标准差 / 均值（182天）"),
    _f("inc_salary_interval_std", "bank_txn_worthiness_salary_interval_std_days_182d", "发薪间隔标准差", "income", -1, 1.0, "days", "工资入账日期间隔的标准差（天，182天）"),
    _f("inc_salary_active_months", "bank_txn_worthiness_salary_active_months_182d", "有工资月份数", "income", 1, 0.8, "count", "182天内发生工资入账的月份数"),
    _f("inc_global_cv", "bank_txn_income_global_cv_182d", "总收入波动率", "income", -1, 0.8, "ratio", "总收入标准差 / 均值（182天）"),
    _f("inc_source_count", "bank_txn_worthiness_income_source_count_182d", "收入来源数", "income", 1, 0.6, "count", "有收入来源标识的收入来源数（182天）"),
    _f("inc_primary_concentration", "bank_txn_worthiness_primary_income_concentration_182d", "主收入来源集中度", "income", -1, 0.6, "pct", "最大收入来源金额 / 总收入（182天）"),
    _f("inc_primary_employer_ratio", "bank_txn_industry_primary_employer_ratio", "主雇主工资占比", "income", 1, 0.6, "pct", "主雇主工资 / 全部识别工资"),
    _f("inc_pay_interval_std", "bank_txn_industry_pay_interval_std", "发薪间隔标准差（行业）", "income", -1, 0.8, "days", "行业识别模块给出的工资发放间隔标准差（天）"),
    _f("inc_non_salary_share", "bank_txn_worthiness_non_salary_income_share_182d", "非工资收入占比", "income", 0, 0.0, "pct", "非工资收入 / 全部收入（182天）"),
    _f("inc_global_sum", "bank_txn_income_global_sum_182d", "总收入", "income", 0, 0.0, "amount", "近182天全部收入金额", "log"),
    _f("inc_centrelink_sum", "bank_txn_income_centrelink_sum_182d", "福利收入", "income", 0, 0.0, "amount", "近182天 Centrelink 收入金额", "log"),
    _f("inc_employer_cnt", "bank_txn_industry_employer_cnt", "雇主数", "income", 0, 0.0, "count", "识别到的工资雇主数量"),
    _f("inc_wages_latest_vs_3m", "bank_txn_income_wages_latest_vs_3m", "最新工资 / 3月均值", "income", 0, 0.0, "ratio", "最新一笔工资 / 近90天平均工资"),
    _f("inc_consecutive_wage_months", "bank_txn_worthiness_wage_income_consecutive_month_cnt_182d", "连续有工资月数", "income", 0, 0.0, "count", "截至观察窗口末连续发生工资入账的月数（182天）"),
    _f("inc_employer_streak_months", "bank_txn_worthiness_primary_employer_streak_months_182d", "主雇主连续月数", "income", 0, 0.0, "count", "主要雇主连续发薪的月数（182天）"),
    _f("inc_pay_interval_median", "bank_txn_industry_pay_interval_median", "发薪间隔中位数", "income", 0, 0.0, "days", "识别到的相邻工资入账间隔中位数（天）"),

    # 收支储蓄
    _f("cf_saving_rate", "bank_txn_balance_global_savings_rate_182d", "储蓄率（余额变动口径）", "cashflow", 1, 1.2, "pct", "(余额变动口径收入 - 支出) / 收入（182天）"),
    _f("cf_net_after_expense", "bank_txn_worthiness_net_income_after_expense_ratio_182d", "收支结余率（流水口径）", "cashflow", 1, 1.0, "pct", "(收入 - 支出) / 收入（182天，流水分类口径）"),
    _f("cf_income_to_expense", "bank_txn_worthiness_income_to_expense_ratio_182d", "收入支出比", "cashflow", 1, 0.8, "ratio", "收入 / 支出（182天，流水分类口径）"),
    _f("cf_burn_rate", "bank_txn_category_global_burn_rate_182d", "支出收入比", "cashflow", -1, 0.8, "pct", "支出 / 收入（182天，流水分类口径）"),
    _f("cf_surplus_sum", "bank_txn_surplus_surplus_sum_182d", "净结余金额", "cashflow", 1, 0.6, "amount", "收入 - 支出 - 负债还款 - 拒付费用 + 负债提款（182天）"),
    _f("cf_expense_cv", "bank_txn_expense_global_cv_182d", "支出波动率", "cashflow", -1, 0.5, "ratio", "支出金额标准差 / 均值（182天）"),
    _f("cf_rent_to_wage", "bank_txn_worthiness_rent_to_wage_ratio_182d", "房租 / 工资", "cashflow", -1, 0.6, "pct", "房租支出 / 工资收入（182天）"),
    _f("cf_daily_living_to_wage", "bank_txn_worthiness_daily_living_to_wage_ratio_182d", "日常必需支出 / 工资", "cashflow", -1, 0.4, "pct", "日常必需支出 / 工资收入（182天）"),
    _f("cf_expense_sum", "bank_txn_expense_global_sum_182d", "总支出", "cashflow", 0, 0.0, "amount", "近182天全部支出金额", "log"),
    _f("cf_expense_avg_ticket", "bank_txn_category_global_expense_avg_ticket_182d", "平均单笔支出", "cashflow", 0, 0.0, "amount", "近182天平均单笔支出金额", "log"),
    _f("cf_income_avg_ticket", "bank_txn_category_global_income_avg_ticket_182d", "平均单笔收入", "cashflow", 0, 0.0, "amount", "近182天平均单笔收入金额", "log"),

    # 流动性缓冲
    _f("lq_balance_mean", "bank_txn_worthiness_balance_mean_avg_acct_182d", "平均账户余额", "liquidity", 1, 0.8, "amount", "账户日均余额的均值（182天，多账户平均）", "log"),
    _f("lq_balance_median", "bank_txn_worthiness_balance_median_avg_acct_182d", "余额中位数", "liquidity", 1, 0.8, "amount", "账户余额中位数的均值（182天，多账户平均）", "log"),
    _f("lq_negative_share", "bank_txn_worthiness_balance_negative_share_avg_acct_182d", "负余额天数占比", "liquidity", -1, 1.2, "pct", "余额为负的天数占比（182天，多账户平均）"),
    _f("lq_ge1000_share", "bank_txn_worthiness_balance_ge1000_share_avg_acct_182d", "余额≥1000占比", "liquidity", 1, 0.8, "pct", "余额不低于1000的天数占比（182天，多账户平均）"),
    _f("lq_ge2000_share", "bank_txn_worthiness_balance_ge2000_share_avg_acct_182d", "余额≥2000占比", "liquidity", 1, 0.8, "pct", "余额不低于2000的天数占比（182天，多账户平均）"),
    _f("lq_post_salary_min_ratio", "bank_txn_worthiness_post_salary_7d_min_balance_ratio_182d", "发薪后7天最低余额/工资", "liquidity", 1, 0.8, "ratio", "每次发薪后7天最低余额 / 该笔工资的均值（182天）"),
    _f("lq_balance_cv", "bank_txn_worthiness_balance_cv_avg_acct_182d", "余额波动率", "liquidity", -1, 0.6, "ratio", "账户余额变异系数（182天，多账户平均）"),
    _f("lq_reserve_months", "bank_txn_worthiness_reserve_months_182d", "固定支出备用月数", "liquidity", 1, 1.2, "months", "平均余额 / 月均固定承诺支出（182天）"),
    _f("lq_max_drawdown", "bank_txn_worthiness_max_balance_drawdown_182d", "最大余额回撤", "liquidity", -1, 0.6, "pct", "余额从峰值到谷底的最大回撤比例（182天）"),
    _f("lq_above_fixed_share", "bank_txn_worthiness_balance_above_1x_monthly_fixed_spend_days_share_182d", "余额高于1倍月固定支出占比", "liquidity", 1, 0.6, "pct", "余额高于1倍月固定账单的天数占比（182天）"),
    _f("lq_balance_snapshot", "bank_txn_balance_snapshot_last_182d", "期末余额", "liquidity", 0, 0.0, "amount", "182天窗口末余额（多账户合计）", "log"),
    _f("lq_balance_min", "bank_txn_balance_min_182d", "区间最低余额", "liquidity", 0, 0.0, "amount", "182天最低余额", "linear"),
    _f("lq_balance_max", "bank_txn_balance_max_182d", "区间最高余额", "liquidity", 0, 0.0, "amount", "182天最高余额", "log"),

    # 负债压力
    _f("db_debt_share", "bank_txn_category_cluster_debt_share_182d", "债务类支出占比", "debt", -1, 1.2, "pct", "贷款、信用卡还款、债务合并等支出 / 总支出（182天）"),
    _f("db_disburse_sum", "bank_txn_lender_disburse_amount_sum", "历史放款总额", "debt", -1, 0.8, "amount", "全部可用窗口的借贷放款金额合计", "log"),
    _f("db_disburse_count", "bank_txn_lender_disburse_amount_count", "历史放款次数", "debt", -1, 1.2, "count", "全部可用窗口的借贷放款交易笔数"),
    _f("db_liability_drawdown", "bank_txn_worthiness_liability_drawdown_amt_sum_182d", "负债提款金额", "debt", -1, 0.8, "amount", "负债类提款金额（182天）", "log"),
    _f("db_dishonour_ratio", "bank_txn_dishonour_lender_ratio_182d", "拒付比例", "debt", -1, 1.2, "pct", "拒付交易占借贷相关交易的比例（182天）"),
    _f("db_fees_overdraft_share", "bank_txn_category_cluster_fees_overdraft_share_182d", "费用/透支支出占比", "debt", -1, 0.8, "pct", "费用、透支、拒付类支出 / 总支出（182天）"),
    _f("db_gambling_share", "bank_txn_category_cluster_gambling_share_182d", "赌博支出占比", "debt", -1, 0.6, "pct", "赌博类支出 / 总支出（182天）"),
    _f("db_collections_share", "bank_txn_category_cluster_collections_share_182d", "催收支出占比", "debt", -1, 0.4, "pct", "债务催收类支出 / 总支出（182天）"),
    _f("db_cash_advance_sum", "bank_txn_lender_disburse_advance_amount_sum", "现金贷/工资贷放款额", "debt", -1, 0.6, "amount", "现金贷或工资贷类放款金额（全部可用窗口）", "log"),
    _f("db_bnpl_sum", "bank_txn_lender_disburse_bnpl_amount_sum", "BNPL 放款额", "debt", -1, 0.4, "amount", "BNPL 类放款金额（全部可用窗口）", "log"),
    _f("db_repay_sum", "bank_txn_lender_repay_amount_sum", "历史还款总额", "debt", 0, 0.0, "amount", "全部可用窗口的借贷还款金额合计", "log"),
    _f("db_repay_count", "bank_txn_lender_repay_amount_count", "历史还款次数", "debt", 0, 0.0, "count", "全部可用窗口的借贷还款交易笔数"),
    _f("db_liability_repay", "bank_txn_worthiness_liability_repayment_amt_sum_182d", "负债还款金额", "debt", 0, 0.0, "amount", "负债类还款金额（182天）", "log"),
    _f("db_single_lender_max", "bank_txn_lender_disburse_single_count_max", "单机构最大放款次数", "debt", -1, 0.4, "count", "单一借贷机构的最大累计放款笔数（全部可用窗口）"),
    _f("db_disburse_jgs", "bank_txn_lender_disburse_jgs_l182d", "放款机构数（182天）", "debt", -1, 0.0, "count", "有放款记录的借贷机构数（182天）"),
    _f("db_dishonour_bnpl_ratio", "bank_txn_dishonour_bnpl_ratio_182d", "BNPL 拒付比例", "debt", 0, 0.0, "pct", "BNPL 拒付交易占该类交易比例（182天）"),
    _f("db_dishonour_personal_ratio", "bank_txn_dishonour_personal_loan_ratio_182d", "个人贷款拒付比例", "debt", 0, 0.0, "pct", "个人贷款拒付交易占比（182天）"),
    _f("db_dishonour_advance_ratio", "bank_txn_dishonour_cash_wage_advance_ratio_182d", "现金贷拒付比例", "debt", 0, 0.0, "pct", "现金贷/工资贷拒付交易占比（182天）"),
    _f("db_dishonour_bank_ratio", "bank_txn_dishonour_bank_ratio_182d", "银行拒付比例", "debt", 0, 0.0, "pct", "银行类拒付交易占比（182天）"),
    _f("db_dishonour_collection_ratio", "bank_txn_dishonour_debt_collection_ratio_182d", "催收拒付比例", "debt", 0, 0.0, "pct", "债务催收类拒付交易占比（182天）"),

    # 消费活跃度
    _f("en_active_days", "bank_txn_category_active_days_182d", "活跃交易天数", "engagement", 1, 1.0, "count", "182天内有交易的去重天数（应用级，按天去重）"),
    _f("en_txn_freq", "bank_txn_category_txn_freq_per_active_day_182d", "活跃日交易频次", "engagement", 1, 0.8, "ratio", "交易笔数 / 活跃天数（182天）"),
    _f("en_category_distinct", "bank_txn_worthiness_category_distinct_total_182d", "覆盖高价值消费类别数", "engagement", 1, 0.8, "count", "命中的高价值消费类别数（182天）"),
    _f("en_expense_count", "bank_txn_expense_global_count_182d", "支出笔数", "engagement", 1, 0.6, "count", "近182天支出交易笔数"),
    _f("en_weekend_share", "bank_txn_category_weekend_share_182d", "周末交易占比", "engagement", 0, 0.0, "pct", "周末交易笔数 / 全部交易笔数（182天）"),
    _f("en_bank_account_cnt", "bank_txn_worthiness_bank_account_cnt", "银行账户数", "confidence", 0, 0.0, "count", "可观测到的银行账户数"),
    _f("en_days_since_wage", "bank_txn_worthiness_days_since_last_wage_income_182d", "距最近工资天数", "confidence", 0, 0.0, "days", "距最近一次工资入账的天数（182天）"),

    # 消费结构（仅观察）
    _f("cs_essentials_share", "bank_txn_category_cluster_essentials_share_182d", "必需支出占比", "consumption", 0, 0.0, "pct", "食品、公用事业、房租、交通、医疗支出 / 总支出（182天）"),
    _f("cs_discretionary_share", "bank_txn_category_cluster_discretionary_share_182d", "可选消费占比", "consumption", 0, 0.0, "pct", "餐饮、娱乐、旅行、零售、百货、个护、宠物、捐赠 / 总支出（182天）"),
    _f("cs_housing_share", "bank_txn_category_cluster_housing_share_182d", "住房支出占比", "consumption", 0, 0.0, "pct", "住房类支出 / 总支出（182天）"),
    _f("cs_utilities_share", "bank_txn_category_cluster_utilities_share_182d", "公用事业支出占比", "consumption", 0, 0.0, "pct", "水电气网等支出 / 总支出（182天）"),
    _f("cs_transport_share", "bank_txn_category_cluster_transport_share_182d", "交通支出占比", "consumption", 0, 0.0, "pct", "交通类支出 / 总支出（182天）"),
    _f("cs_healthcare_share", "bank_txn_category_cluster_healthcare_share_182d", "医疗支出占比", "consumption", 0, 0.0, "pct", "医疗健康类支出 / 总支出（182天）"),
    _f("cs_education_share", "bank_txn_category_cluster_education_share_182d", "教育支出占比", "consumption", 0, 0.0, "pct", "教育类支出 / 总支出（182天）"),
    _f("cs_dining_share", "bank_txn_category_cluster_dining_share_182d", "餐饮支出占比", "consumption", 0, 0.0, "pct", "餐饮外出支出 / 总支出（182天）"),
    _f("cs_retail_share", "bank_txn_category_cluster_retail_share_182d", "零售支出占比", "consumption", 0, 0.0, "pct", "零售类支出 / 总支出（182天）"),
    _f("cs_travel_share", "bank_txn_category_cluster_travel_share_182d", "旅行支出占比", "consumption", 0, 0.0, "pct", "旅行类支出 / 总支出（182天）"),
    _f("cs_entertainment_share", "bank_txn_category_cluster_entertainment_share_182d", "娱乐支出占比", "consumption", 0, 0.0, "pct", "娱乐类支出 / 总支出（182天）"),
    _f("cs_personal_care_share", "bank_txn_category_cluster_personal_care_share_182d", "个护支出占比", "consumption", 0, 0.0, "pct", "个人护理类支出 / 总支出（182天）"),
    _f("cs_household_daily_share", "bank_txn_category_cluster_household_daily_share_182d", "家庭日用支出占比", "consumption", 0, 0.0, "pct", "家庭日用类支出 / 总支出（182天）"),
    _f("cs_recurring_bills_share", "bank_txn_category_cluster_recurring_bills_share_182d", "周期性账单占比", "consumption", 0, 0.0, "pct", "周期性账单支出 / 总支出（182天）"),
    _f("cs_income_wages_share", "bank_txn_category_cluster_income_wages_share_182d", "工资收入占比（分类口径）", "consumption", 0, 0.0, "pct", "工资收入 / 总收入（182天，交易分类口径）"),
    _f("cs_income_benefits_share", "bank_txn_category_cluster_income_benefits_share_182d", "福利收入占比（分类口径）", "consumption", 0, 0.0, "pct", "福利收入 / 总收入（182天，交易分类口径）"),
    _f("cs_other_credits_share", "bank_txn_category_cluster_other_credits_share_182d", "其他入账占比", "consumption", 0, 0.0, "pct", "其他贷方入账 / 总收入（182天）"),
    _f("cs_transfers_share", "bank_txn_category_cluster_transfers_share_182d", "转账占比", "consumption", 0, 0.0, "pct", "转账类交易 / 总交易（182天）"),
    _f("cs_transfer_external_share", "bank_txn_category_cluster_transfer_external_share_182d", "外部转账占比", "consumption", 0, 0.0, "pct", "外部转账 / 总交易（182天）"),
    _f("cs_risk_obligations_share", "bank_txn_category_cluster_risk_obligations_share_182d", "风险刚性支出占比", "consumption", 0, 0.0, "pct", "风险类刚性支出 / 总支出（182天）"),
    _f("cs_risk_discretionary_share", "bank_txn_category_cluster_risk_discretionary_share_182d", "风险可选消费占比", "consumption", 0, 0.0, "pct", "风险类可选消费 / 总支出（182天）"),
    _f("cs_large_expense_cnt", "bank_txn_worthiness_large_expense_cnt_182d", "大额支出笔数", "consumption", 0, 0.0, "count", "超过大额阈值的支出笔数（182天）"),
    _f("cs_large_expense_sum", "bank_txn_worthiness_large_expense_amt_sum_182d", "大额支出金额", "consumption", 0, 0.0, "amount", "大额支出金额合计（182天）", "log"),
    _f("cs_premium_lifestyle_sum", "bank_txn_worthiness_premium_lifestyle_amt_sum_182d", "品质生活支出", "consumption", 0, 0.0, "amount", "品质生活类支出金额（182天）", "log"),
    _f("cs_high_commitment_sum", "bank_txn_worthiness_high_commitment_amt_sum_182d", "高承诺支出", "consumption", 0, 0.0, "amount", "高承诺/刚性支出金额（182天）", "log"),
    _f("cs_fixed_bill_sum", "bank_txn_worthiness_fixed_bill_amt_sum_182d", "固定账单支出", "consumption", 0, 0.0, "amount", "固定账单类支出金额（182天）", "log"),
]

DERIVED = [
    _d(
        "inc_centrelink_share",
        "福利收入占比",
        "income",
        "ratio",
        direction=0,
        weight=0.0,
        fmt="pct",
        definition="Centrelink 收入 / 全部收入（182天，金额口径）",
        numerator="bank_txn_income_centrelink_sum_182d",
        denominator="bank_txn_income_global_sum_182d",
    ),
    _d(
        "db_unresolved_cnt",
        "未结清拒付次数",
        "debt",
        "sum",
        direction=-1,
        weight=1.0,
        fmt="count",
        definition="六类拒付中未结清次数之和（182天，3GP 口径）",
        sources=UNRESOLVED_COLUMNS,
    ),
    _d(
        "db_active_lender_cnt",
        "活跃借贷机构数",
        "debt",
        "count_positive",
        direction=-1,
        weight=1.0,
        fmt="count",
        definition="有放款记录的借贷机构数（机构名单，全部可用窗口）",
        source_pattern=COMPETITOR_PATTERN,
    ),
    _d(
        "db_debt_service_ratio",
        "负债还款 / 收入",
        "debt",
        "ratio",
        direction=-1,
        weight=0.6,
        fmt="pct",
        definition="负债类还款金额 / 全部收入（182天，流水口径）",
        numerator="bank_txn_worthiness_liability_repayment_amt_sum_182d",
        denominator="bank_txn_worthiness_income_tag_amt_sum_182d",
    ),
    _d(
        "db_recent_disburse_share",
        "近28天放款占182天比例",
        "debt",
        "ratio",
        direction=-1,
        weight=0.0,
        fmt="pct",
        definition="近28天放款金额 / 近182天放款金额；越高说明借款越集中在近期",
        numerator="bank_txn_lender_disburse_amount_sum_l28d",
        denominator="bank_txn_lender_disburse_amount_sum_l182d",
    ),
    _d(
        "db_recent_disburse_count_share",
        "近28天放款笔数占比",
        "debt",
        "ratio",
        direction=-1,
        weight=0.0,
        fmt="pct",
        definition="近28天放款笔数 / 近182天放款笔数",
        numerator="bank_txn_lender_disburse_amount_count_l28d",
        denominator="bank_txn_lender_disburse_amount_count_l182d",
    ),
    _d(
        "db_net_borrow_28d",
        "近28天净融资",
        "debt",
        "diff",
        direction=-1,
        weight=0.0,
        fmt="amount",
        definition="近28天放款金额 − 近28天还款金额；正值表示债务净扩张",
        numerator="bank_txn_lender_disburse_amount_sum_l28d",
        denominator="bank_txn_lender_repay_amount_sum_l28d",
    ),
    _d(
        "db_borrow_repay_ratio_28d",
        "近28天借还比",
        "debt",
        "ratio",
        direction=-1,
        weight=0.0,
        fmt="ratio",
        definition="近28天放款金额 / 近28天还款金额；大于1表示新增借款超过还款",
        numerator="bank_txn_lender_disburse_amount_sum_l28d",
        denominator="bank_txn_lender_repay_amount_sum_l28d",
    ),
    _d(
        "db_repay_per_disburse",
        "还款笔数 / 放款笔数",
        "debt",
        "ratio",
        direction=0,
        weight=0.0,
        fmt="ratio",
        definition="近182天还款笔数 / 放款笔数，用于观察分期还款频率",
        numerator="bank_txn_lender_repay_amount_count_l182d",
        denominator="bank_txn_lender_disburse_amount_count_l182d",
    ),
    _d(
        "db_lender_concentration",
        "单机构放款集中度",
        "debt",
        "ratio",
        direction=0,
        weight=0.0,
        fmt="pct",
        definition="上游单机构/产品最大历史累计放款金额 / 历史放款总额；分子分母均为全部可用窗口",
        numerator="bank_txn_lender_disburse_single_amount_max",
        denominator="bank_txn_lender_disburse_amount_sum",
    ),
    _d(
        "db_loan_disburse_share",
        "个人贷款占放款比例",
        "debt",
        "ratio",
        direction=0,
        weight=0.0,
        fmt="pct",
        definition="个人贷款放款金额 / 近182天放款总额",
        numerator="bank_txn_lender_disburse_loan_amount_sum_l182d",
        denominator="bank_txn_lender_disburse_amount_sum_l182d",
    ),
    _d(
        "db_advance_disburse_share",
        "现金贷占放款比例",
        "debt",
        "ratio",
        direction=-1,
        weight=0.0,
        fmt="pct",
        definition="现金贷/工资贷放款金额 / 近182天放款总额",
        numerator="bank_txn_lender_disburse_advance_amount_sum_l182d",
        denominator="bank_txn_lender_disburse_amount_sum_l182d",
    ),
    _d(
        "db_bnpl_disburse_share",
        "BNPL占放款比例",
        "debt",
        "ratio",
        direction=0,
        weight=0.0,
        fmt="pct",
        definition="BNPL 放款金额 / 近182天放款总额",
        numerator="bank_txn_lender_disburse_bnpl_amount_sum_l182d",
        denominator="bank_txn_lender_disburse_amount_sum_l182d",
    ),
    _d(
        "db_bnpl_repay_share",
        "BNPL占还款比例",
        "debt",
        "ratio",
        direction=0,
        weight=0.0,
        fmt="pct",
        definition="BNPL 还款金额 / 近182天还款总额",
        numerator="bank_txn_lender_repay_bnpl_amount_sum_l182d",
        denominator="bank_txn_lender_repay_amount_sum_l182d",
    ),
    _d(
        "db_resolved_cnt",
        "已结清拒付次数",
        "debt",
        "sum",
        direction=0,
        weight=0.0,
        fmt="count",
        definition="六类拒付中已识别为结清的次数之和（182天，3GP 口径）",
        sources=RESOLVED_CNT_COLUMNS,
    ),
    _d(
        "db_resolved_amt",
        "已结清拒付金额",
        "debt",
        "sum",
        direction=0,
        weight=0.0,
        fmt="amount",
        definition="六类拒付中已结清金额之和（182天，3GP 口径）",
        sources=RESOLVED_AMT_COLUMNS,
        hist_scale="log",
    ),
    _d(
        "db_unresolved_amt",
        "未结清拒付金额",
        "debt",
        "sum",
        direction=-1,
        weight=0.0,
        fmt="amount",
        definition="六类拒付中未结清金额之和（182天，3GP 口径）",
        sources=UNRESOLVED_AMT_COLUMNS,
        hist_scale="log",
    ),
    _d(
        "db_unresolved_inst_max",
        "单类未结清机构数最大值",
        "debt",
        "max",
        direction=-1,
        weight=0.0,
        fmt="count",
        definition="六类拒付中单一类型未结清涉及的机构数最大值（182天）",
        sources=UNRESOLVED_INST_COLUMNS,
    ),
    _d(
        "db_resolved_catchup_cnt",
        "补扣结清次数",
        "debt",
        "sum",
        direction=0,
        weight=0.0,
        fmt="count",
        definition="结清拒付中通过多次补扣完成的次数之和（182天，3GP 口径）",
        sources=CATCHUP_CNT_COLUMNS,
    ),
    _d(
        "db_dishonour_resolved_ratio",
        "拒付结清率",
        "debt",
        "ratio_sum",
        direction=1,
        weight=0.0,
        fmt="pct",
        definition="已结清拒付次数 /（已结清 + 未结清）拒付次数（182天，3GP 口径）",
        numerator_sources=RESOLVED_CNT_COLUMNS,
        denominator_sources=RESOLVED_CNT_COLUMNS + UNRESOLVED_COLUMNS,
    ),
    _d(
        "db_dishonour_unresolved_amt_ratio",
        "拒付未结清金额占比",
        "debt",
        "ratio_sum",
        direction=-1,
        weight=0.0,
        fmt="pct",
        definition="未结清拒付金额 /（已结清 + 未结清）拒付金额（182天，3GP 口径）",
        numerator_sources=UNRESOLVED_AMT_COLUMNS,
        denominator_sources=RESOLVED_AMT_COLUMNS + UNRESOLVED_AMT_COLUMNS,
    ),
]

# 新增观察指标沿用上游特征，近期与历史基线使用互不重叠的窗口。
DERIVED.extend([
    _d("tr_income_daily_ratio", "近期日均收入 / 前期日均收入", "trend", "daily_trend",
       definition="(近28天收入/28) / ((近182天收入-近28天收入)/154)；前期收入须为正",
       numerator="bank_txn_income_global_sum_28d", denominator="bank_txn_income_global_sum_182d"),
    _d("tr_expense_daily_ratio", "近期日均支出 / 前期日均支出", "trend", "daily_trend",
       definition="(近28天支出/28) / ((近182天支出-近28天支出)/154)；前期支出须为正",
       numerator="bank_txn_expense_global_sum_28d", denominator="bank_txn_expense_global_sum_182d"),
    _d("tr_recent_burn", "近28天支出收入比", "trend", "ratio", fmt="pct",
       definition="近28天支出 / 近28天收入；收入为0时留空",
       numerator="bank_txn_expense_global_sum_28d", denominator="bank_txn_income_global_sum_28d"),
    _d("rs_commitment_income", "高承诺支出 / 收入", "resilience", "ratio", fmt="pct",
       definition="近182天高承诺支出 / 同窗口资质模块收入；不与固定账单相加，以免重复计算",
       numerator="bank_txn_worthiness_high_commitment_amt_sum_182d",
       denominator="bank_txn_worthiness_income_tag_amt_sum_182d"),
    _d("rs_fixed_income", "固定账单 / 收入", "resilience", "ratio", fmt="pct",
       definition="近182天固定账单支出 / 同窗口资质模块收入",
       numerator="bank_txn_worthiness_fixed_bill_amt_sum_182d",
       denominator="bank_txn_worthiness_income_tag_amt_sum_182d"),
    _d("rs_expense_buffer_days", "期末余额支出覆盖天数", "resilience", "buffer_days", fmt="days",
       definition="max(182天窗口期末合计余额,0) / (近182天支出/182)；负余额按0天，非未来预测",
       numerator="bank_txn_balance_snapshot_last_182d", denominator="bank_txn_expense_global_sum_182d"),
])

# 区间左闭右开；所有占比以可用流水申请为分母，并单列无法计算的记录。
OBSERVATION_BANDS = {
    "tr_income_daily_ratio": ([0.8, 1.2], ["下降超过20%", "大致持平", "上升至少20%"]),
    "tr_expense_daily_ratio": ([0.8, 1.2], ["下降超过20%", "大致持平", "上升至少20%"]),
    "tr_recent_burn": ([0.8, 1.0], ["支出低于收入80%", "支出接近收入", "支出达到或超过收入"]),
    "rs_commitment_income": ([0.3, 0.6], ["低于30%", "30%至60%", "至少60%"]),
    "rs_fixed_income": ([0.2, 0.4], ["低于20%", "20%至40%", "至少40%"]),
    "rs_expense_buffer_days": ([7.0, 30.0, 90.0], ["不足7天", "7至30天", "30至90天", "至少90天"]),
}


def observation_band(values: np.ndarray, key: str) -> np.ndarray:
    edges, labels = OBSERVATION_BANDS[key]
    out = np.full(values.size, "数据不足", dtype=object)
    valid = np.isfinite(values)
    out[valid] = np.asarray(labels, dtype=object)[np.searchsorted(edges, values[valid], side="right")]
    return out


CATEGORY_MIX_KEYS = [
    "cs_income_wages_share",
    "cs_income_benefits_share",
    "cs_other_credits_share",
    "cs_essentials_share",
    "cs_discretionary_share",
    "cs_housing_share",
    "cs_utilities_share",
    "cs_transport_share",
    "cs_healthcare_share",
    "cs_dining_share",
    "cs_retail_share",
    "cs_travel_share",
    "cs_entertainment_share",
    "cs_household_daily_share",
    "cs_recurring_bills_share",
    "cs_transfers_share",
    "db_debt_share",
    "db_gambling_share",
    "db_fees_overdraft_share",
    "db_collections_share",
]

CORE_METRIC_KEYS = [
    "inc_global_sum",
    "inc_wage_sum",
    "inc_centrelink_sum",
    "inc_centrelink_share",
    "inc_wage_share",
    "inc_salary_cv",
    "inc_consecutive_wage_months",
    "inc_employer_streak_months",
    "inc_pay_interval_median",
    "cf_saving_rate",
    "cf_net_after_expense",
    "cf_income_to_expense",
    "cf_burn_rate",
    "cf_surplus_sum",
    "cf_expense_sum",
    "lq_balance_mean",
    "lq_balance_median",
    "lq_negative_share",
    "lq_reserve_months",
    "db_debt_share",
    "db_disburse_sum",
    "db_disburse_count",
    "db_active_lender_cnt",
    "db_disburse_jgs",
    "db_unresolved_cnt",
    "db_resolved_cnt",
    "db_resolved_amt",
    "db_unresolved_amt",
    "db_unresolved_inst_max",
    "db_resolved_catchup_cnt",
    "db_dishonour_resolved_ratio",
    "db_dishonour_unresolved_amt_ratio",
    "db_dishonour_ratio",
    "db_fees_overdraft_share",
    "db_gambling_share",
    "db_debt_service_ratio",
    "db_recent_disburse_share",
    "db_recent_disburse_count_share",
    "db_net_borrow_28d",
    "db_borrow_repay_ratio_28d",
    "db_repay_per_disburse",
    "db_lender_concentration",
    "db_loan_disburse_share",
    "db_advance_disburse_share",
    "db_bnpl_disburse_share",
    "db_bnpl_repay_share",
    "cs_essentials_share",
    "cs_discretionary_share",
    "en_active_days",
    "en_txn_freq",
] + list(OBSERVATION_BANDS)

HIST_KEYS = [
    "cf_saving_rate",
    "inc_global_sum",
    "lq_balance_median",
    "inc_salary_cv",
    "db_debt_share",
    "en_active_days",
    "cf_net_after_expense",
    "db_disburse_count",
]

FLAG_DEFS = [
    ("flag_unresolved_dishonour", "存在未结清拒付", "六类拒付中未结清次数 > 0", True),
    ("flag_high_debt", "负债压力高", "负债压力维度得分低于阈值", True),
    ("flag_multi_lender", "多头借贷", "有放款记录的借贷机构数 ≥ 阈值", True),
    ("flag_payday", "现金贷/工资贷使用", "历史存在现金贷或工资贷放款", True),
    ("flag_negative_cashflow", "现金净流出", "储蓄率（余额变动口径）< 0", True),
    ("flag_low_buffer", "流动性缓冲不足", "负余额天数占比高或余额中位数为负", True),
    ("flag_gambling", "赌博类支出", "赌博支出占总支出比例 ≥ 阈值", True),
    ("flag_income_unstable", "收入不稳定", "工资金额变异系数 > 阈值", True),
    ("flag_fees_heavy", "费用/透支负担", "费用与透支支出占比 ≥ 阈值", True),
    ("flag_welfare_dependent", "福利收入依赖", "Centrelink 收入占总收入 ≥ 阈值", False),
    ("flag_no_wage", "无工资入账", "近182天无工资收入", False),
    ("flag_no_income", "无收入入账", "近182天无收入记录", False),
    ("flag_no_txn", "无交易数据", "txn_raw_input_cnt 未达到门槛", False),
]


# ---------------------------------------------------------------------------
# 配置与输入
# ---------------------------------------------------------------------------

def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p)


def load_config(config_path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    base = PROJECT_ROOT
    if config_path:
        cp = Path(config_path).expanduser().resolve()
        if not cp.exists():
            raise FileNotFoundError(f"配置文件不存在: {cp}")
        with cp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        base = cp.parent
        cfg["_config_path"] = str(cp)

    for key in PATH_KEYS:
        cfg[key] = str(_resolve_path(base, str(cfg[key])))

    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if key in PATH_KEYS:
                value = str(_resolve_path(Path.cwd(), str(value)))
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value

    cfg["chunk_size"] = max(1, int(cfg["chunk_size"]))
    cfg["reservoir_size"] = max(100, int(cfg["reservoir_size"]))
    cfg["seed"] = int(cfg["seed"])
    cfg["hist_bins"] = max(5, int(cfg["hist_bins"]))
    cfg["segment_reservoir_size"] = max(1, int(cfg["segment_reservoir_size"]))
    if not math.isfinite(float(cfg["target_positive_value"])) or float(cfg["target_positive_value"]) == 0:
        raise ValueError("target_positive_value 必须为非零有限数；负样本固定为0")
    if not 0 <= float(cfg["score_low"]) < float(cfg["score_high"]) <= 100:
        raise ValueError("需要 0 ≤ score_low < score_high ≤ 100")
    for key in ("min_dimension_coverage", "min_health_coverage"):
        if not 0 < float(cfg[key]) <= 1:
            raise ValueError(f"{key} 必须在 (0, 1] 范围内")
    return cfg


def read_header(path: str) -> list[str]:
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def read_dictionary(path: str) -> dict:
    try:
        df = pd.read_csv(path, dtype={"feature": str, "model_vars": str})
    except Exception:
        return {}
    out = {}
    for row in df.itertuples(index=False):
        out[str(row.feature)] = {
            "model_vars": str(getattr(row, "model_vars", "")),
            "default": getattr(row, "default", np.nan),
        }
    return out


def _sentinel_for_column(column: str) -> str:
    return "lender" if infer_module(column) == "lender" else "std"


def resolve_registry(header: list[str], cfg: dict) -> tuple[list[dict], list[dict], list[str]]:
    header_set = set(header)
    features = []
    missing = []
    for spec in FEATURES:
        if spec["column"] in header_set:
            features.append(dict(spec))
        else:
            missing.append(spec["column"])

    derived = []
    for spec in DERIVED:
        spec = dict(spec)
        spec_missing = []
        sources = list(spec.get("sources") or [])
        if spec.get("source_pattern"):
            sources = sorted(c for c in header if re.match(spec["source_pattern"], c))
            if not sources:
                spec_missing.append(f"pattern:{spec['source_pattern']}")
        for col in sources:
            if col not in header_set:
                spec_missing.append(col)
        missing.extend(spec_missing)
        spec["sources"] = list(dict.fromkeys(sources))
        if spec["sources"] and not spec_missing:
            derived.append(spec)

    if missing and not cfg.get("allow_missing_features", False):
        raise ValueError(
            "以下特征列在输入文件中不存在，请检查特征版本或开启 allow_missing_features：\n  - "
            + "\n  - ".join(sorted(set(missing)))
        )

    for spec in features:
        spec["in_score"] = spec["direction"] != 0 and spec["weight"] > 0 and spec["dimension"] in SCORED_DIMS
    for spec in derived:
        spec["in_score"] = spec["direction"] != 0 and spec["weight"] > 0 and spec["dimension"] in SCORED_DIMS
    return features, derived, sorted(set(missing))


# 仅用于口径一致性校验，不参与画像指标
EXTRA_SOURCE_COLUMNS = (
    "bank_txn_income_otherincome_sum_182d",
    "bank_txn_surplus_income_sum_182d",
    "bank_txn_surplus_expense_sum_182d",
    "bank_txn_surplus_liabilityrepayment_sum_182d",
    "bank_txn_surplus_dishonoursfee_sum_182d",
    "bank_txn_surplus_liabilitydrawdown_sum_182d",
    "bank_txn_surplus_surplus_sum_182d",
)


def collect_columns(features: list[dict], derived: list[dict], cfg: dict, header: list[str]) -> list[str]:
    cols = []
    seen = set()

    def add(c):
        if c and c not in seen:
            cols.append(c)
            seen.add(c)

    for c in (
        "user_id",
        "application_id",
        "sample_datetime",
        "sample_month",
        "sample_quarter",
        "user_type",
        cfg["gate_column"],
        cfg["target_column"],
        "feature_error",
    ):
        if c in header:
            add(c)

    for spec in features:
        add(spec["column"])
    for spec in derived:
        for c in spec.get("sources", []):
            add(c)
        for c in (spec.get("numerator"), spec.get("denominator")):
            add(c)
    for c in EXTRA_SOURCE_COLUMNS:
        if c in header:
            add(c)
    for c in INCOME_ANALYSIS_COLUMNS:
        if c in header:
            add(c)
    for c in header:
        if any(re.match(p, c) for p in LENDER_ANALYSIS_PATTERNS):
            add(c)
    return cols


def clean_numeric(series: pd.Series, sentinel: str) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan).copy()
    if sentinel == "lender":
        arr[arr < 0] = np.nan
    else:
        arr[np.isclose(arr, -999999.99999, rtol=0, atol=0.0001)] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def clean_source_numeric(series: pd.Series, column: str) -> np.ndarray:
    arr = clean_numeric(series, _sentinel_for_column(column))
    # 上游 lender 的可加总金额/次数/机构数用 -1 表示无匹配记录。
    # 仅恢复这类字段的结构性0；空值与均值/最值等默认值仍保持缺失。
    if infer_module(column) == "lender" and re.search(r"_(?:amount_sum|amount_count|count|jgs)(?:_l\d+d)?$", column):
        raw = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        arr[raw == -1] = 0.0
    return arr


def compute_derived(spec: dict, source: dict[str, np.ndarray], gate: np.ndarray) -> np.ndarray:
    kind = spec["kind"]
    n = gate.size
    if kind == "ratio":
        num = source[spec["numerator"]]
        den = source[spec["denominator"]]
        out = np.full(n, np.nan, dtype=np.float64)
        np.divide(num, den, out=out, where=np.isfinite(num) & np.isfinite(den) & (den > 0))
    elif kind == "diff":
        num = source[spec["numerator"]]
        den = source[spec["denominator"]]
        out = num - den
        out[~np.isfinite(num) | ~np.isfinite(den)] = np.nan
    elif kind == "sum":
        stack = np.vstack([source[c] for c in spec["sources"]])
        out = np.sum(stack, axis=0)
    elif kind == "max":
        stack = np.vstack([source[c] for c in spec["sources"]])
        complete = np.isfinite(stack).all(axis=0)
        out = np.full(stack.shape[1], np.nan, dtype=np.float64)
        if complete.any():
            out[complete] = np.max(stack[:, complete], axis=0)
    elif kind == "ratio_sum":
        num = np.sum(np.vstack([source[c] for c in spec["numerator_sources"]]), axis=0)
        den = np.sum(np.vstack([source[c] for c in spec["denominator_sources"]]), axis=0)
        out = np.full(n, np.nan, dtype=np.float64)
        np.divide(num, den, out=out, where=np.isfinite(num) & np.isfinite(den) & (den > 0))
    elif kind == "count_positive":
        stack = np.vstack([source[c] for c in spec["sources"]])
        out = np.sum(stack > 0, axis=0).astype(np.float64)
        out[~np.isfinite(stack).all(axis=0)] = np.nan
    elif kind == "daily_trend":
        recent = source[spec["numerator"]]
        previous = source[spec["denominator"]] - recent
        out = np.full(n, np.nan)
        np.divide(recent / 28.0, previous / 154.0, out=out,
                  where=np.isfinite(recent) & (recent >= 0) & np.isfinite(previous) & (previous > 0))
    elif kind == "buffer_days":
        balance = source[spec["numerator"]]
        expense = source[spec["denominator"]]
        out = np.full(n, np.nan)
        np.divide(np.maximum(balance, 0.0), expense / 182.0, out=out,
                  where=np.isfinite(balance) & np.isfinite(expense) & (expense > 0))
    else:
        raise ValueError(f"未知派生类型: {kind}")
    out[~gate | ~np.isfinite(out)] = np.nan
    return out


def build_arrays(chunk: pd.DataFrame, features: list[dict], derived: list[dict], gate: np.ndarray) -> tuple[dict, dict]:
    source_cols = set()
    for spec in features:
        source_cols.add(spec["column"])
    for spec in derived:
        source_cols.update(spec.get("sources", []))
        if spec.get("numerator"):
            source_cols.add(spec["numerator"])
        if spec.get("denominator"):
            source_cols.add(spec["denominator"])
    source_cols.update(EXTRA_SOURCE_COLUMNS)
    source_cols.update(INCOME_ANALYSIS_COLUMNS)
    for col in chunk.columns:
        if any(re.match(p, col) for p in LENDER_ANALYSIS_PATTERNS):
            source_cols.add(col)

    source = {}
    for col in source_cols:
        if col in chunk.columns:
            source[col] = clean_source_numeric(chunk[col], col)

    arrays = {}
    for spec in features:
        if spec["column"] in source:
            arrays[spec["key"]] = np.where(gate, source[spec["column"]], np.nan)
    for spec in derived:
        try:
            arrays[spec["key"]] = compute_derived(spec, source, gate)
        except KeyError:
            continue
    return arrays, source


def read_chunks(path: str, columns: list[str], chunk_size: int, header: list[str], id_columns: list[str]):
    usecols = [c for c in columns if c in set(header)]
    dtype_map = {c: "string" for c in id_columns if c in usecols}
    return pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunk_size,
        encoding="utf-8-sig",
        low_memory=False,
        dtype=dtype_map or None,
    )


# ---------------------------------------------------------------------------
# 累加器
# ---------------------------------------------------------------------------

class Reservoir:
    __slots__ = ("k", "rng", "keys", "vals")

    def __init__(self, k: int, rng: np.random.Generator):
        self.k = int(k)
        self.rng = rng
        self.keys = np.empty(0, dtype=np.float64)
        self.vals = np.empty(0, dtype=np.float64)

    def add(self, values: np.ndarray) -> None:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        keys = self.rng.random(values.size)
        if self.keys.size:
            keys = np.concatenate([self.keys, keys])
            values = np.concatenate([self.vals, values])
        if keys.size > self.k:
            idx = np.argpartition(keys, self.k - 1)[: self.k]
            keys = keys[idx]
            values = values[idx]
        self.keys = keys
        self.vals = values


class FeatureStats:
    __slots__ = ("key", "n", "total", "total_sq", "min", "max", "zeros", "negs", "res")

    def __init__(self, key: str, reservoir_size: int, rng: np.random.Generator):
        self.key = key
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.zeros = 0
        self.negs = 0
        self.res = Reservoir(reservoir_size, rng)

    def add(self, values: np.ndarray) -> None:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.n += int(values.size)
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())
        self.min = min(self.min, float(values.min()))
        self.max = max(self.max, float(values.max()))
        self.zeros += int(np.count_nonzero(values == 0))
        self.negs += int(np.count_nonzero(values < 0))
        self.res.add(values)

    def summary(self) -> dict:
        if self.n == 0:
            return {
                "n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
                "p01": np.nan, "p05": np.nan, "p10": np.nan, "p25": np.nan,
                "p50": np.nan, "p75": np.nan, "p90": np.nan, "p95": np.nan,
                "p99": np.nan, "zero_pct": np.nan, "negative_pct": np.nan,
                "reference_n": 0,
            }
        mean = self.total / self.n
        if self.n > 1:
            var = max(0.0, (self.total_sq - self.n * mean * mean) / (self.n - 1))
            std = math.sqrt(var)
        else:
            std = np.nan
        qs = np.percentile(self.res.vals, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
        return {
            "n": self.n,
            "mean": mean,
            "std": std,
            "min": self.min,
            "max": self.max,
            "p01": float(qs[1]),
            "p05": float(qs[2]),
            "p10": float(qs[3]),
            "p25": float(qs[4]),
            "p50": float(qs[5]),
            "p75": float(qs[6]),
            "p90": float(qs[7]),
            "p95": float(qs[8]),
            "p99": float(qs[9]),
            "zero_pct": 100.0 * self.zeros / self.n,
            "negative_pct": 100.0 * self.negs / self.n,
            "reference_n": int(self.res.vals.size),
        }


class MeanVar:
    __slots__ = ("n", "total", "total_sq")

    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0

    def add(self, values: np.ndarray) -> None:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.n += int(values.size)
        self.total += float(values.sum())
        self.total_sq += float(np.square(values).sum())

    def stats(self) -> dict:
        if self.n == 0:
            return {"n": 0, "mean": np.nan, "var": np.nan}
        mean = self.total / self.n
        var = max(0.0, (self.total_sq - self.n * mean * mean) / (self.n - 1)) if self.n > 1 else np.nan
        return {"n": self.n, "mean": mean, "var": var}


class PairStats:
    __slots__ = ("n", "sx", "sy", "sxx", "syy", "sxy")

    def __init__(self):
        self.n = 0
        self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            return
        xv = x[mask]
        yv = y[mask]
        self.n += int(xv.size)
        self.sx += float(xv.sum())
        self.sy += float(yv.sum())
        self.sxx += float(np.square(xv).sum())
        self.syy += float(np.square(yv).sum())
        self.sxy += float((xv * yv).sum())

    def corr(self) -> float:
        if self.n < 3:
            return np.nan
        cov = (self.sxy - self.sx * self.sy / self.n) / (self.n - 1)
        vx = (self.sxx - self.sx * self.sx / self.n) / (self.n - 1)
        vy = (self.syy - self.sy * self.sy / self.n) / (self.n - 1)
        if vx <= 0 or vy <= 0:
            return np.nan
        return cov / math.sqrt(vx * vy)


class UniqueTracker:
    def __init__(self, cap: int):
        self.cap = int(cap)
        self.seen = set()
        self.capped = False

    def add(self, values) -> None:
        if self.capped:
            return
        for v in values:
            if pd.isna(v) or not str(v).strip():
                continue
            if str(v) in self.seen:
                continue
            if len(self.seen) >= self.cap:
                self.capped = True
                return
            self.seen.add(str(v))

    @property
    def n(self) -> int:
        return len(self.seen)


class TargetStats:
    def __init__(self, positive_value: float = 1.0):
        self.positive_value = float(positive_value)
        self.valid = 0
        self.pos = 0
        self.neg = 0
        self.other = 0

    def add(self, values: np.ndarray) -> None:
        v = values[np.isfinite(values)]
        if v.size == 0:
            return
        pos = v == self.positive_value
        neg = v == 0
        self.valid += int((pos | neg).sum())
        self.pos += int(pos.sum())
        self.neg += int(neg.sum())
        self.other += int((~pos & ~neg).sum())

    def as_dict(self) -> dict:
        rate = self.pos / self.valid if self.valid else np.nan
        lo, hi = wilson_interval(self.pos, self.valid)
        return {
            "valid_n": self.valid,
            "positives": self.pos,
            "negatives": self.neg,
            "other": self.other,
            "rate": rate,
            "ci_low": lo,
            "ci_high": hi,
        }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ---------------------------------------------------------------------------
# 第一遍：全量统计 + 抽样分位 + 可选窄表缓存
# ---------------------------------------------------------------------------

def run_pass1(cfg: dict, header: list[str], features: list[dict], derived: list[dict],
              columns: list[str], cache_path: Path | None, cache_ready: bool) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    total_rows = 0
    txn_rows = 0
    chunk_count = 0
    stats = {spec["key"]: FeatureStats(spec["key"], cfg["reservoir_size"], rng) for spec in features + derived}
    target_stats = TargetStats(float(cfg["target_positive_value"]))
    target_col = cfg["target_column"]
    gate_col = cfg["gate_column"]
    gate_present = gate_col in header
    gate_stats = FeatureStats("__gate__", min(cfg["reservoir_size"], 20000), rng)
    user_tracker = UniqueTracker(cfg["max_unique_track"])
    app_tracker = UniqueTracker(cfg["max_unique_track"])

    period = {"datetime_min": None, "datetime_max": None, "months": set(), "user_types": set()}
    feature_error_n = 0
    gate_false_with_data = 0
    gate_true_without_data = 0
    invariants = Counter()
    invariant_rows = Counter()

    id_cols = [c for c in ("user_id", "application_id") if c in header]
    cache_handle = None
    cache_writer_first = True
    if cache_path is not None and not cache_ready:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_handle = gzip.open(cache_path, "wt", encoding="utf-8", newline="")

    t0 = time.time()
    try:
        scan_path = str(cache_path) if cache_ready and cache_path is not None else cfg["input_path"]
        for chunk in read_chunks(scan_path, columns, cfg["chunk_size"], header, id_cols):
            n = len(chunk)
            if n == 0:
                continue
            chunk_count += 1
            total_rows += n

            if id_cols:
                if "user_id" in chunk.columns:
                    user_tracker.add(chunk["user_id"].tolist())
                if "application_id" in chunk.columns:
                    app_tracker.add(chunk["application_id"].tolist())

            if gate_present:
                gate_vals = pd.to_numeric(chunk[gate_col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                gate = np.isfinite(gate_vals) & (gate_vals >= float(cfg["gate_min"]))
                gate_stats.add(gate_vals)
            else:
                gate_vals = np.ones(n, dtype=np.float64)
                gate = np.ones(n, dtype=bool)
                gate_stats.add(gate_vals)
            txn_rows += int(gate.sum())

            arrays, source = build_arrays(chunk, features, derived, gate)
            for spec in features + derived:
                if spec["key"] in arrays:
                    stats[spec["key"]].add(arrays[spec["key"]])

            if target_col in chunk.columns:
                target_stats.add(pd.to_numeric(chunk[target_col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan))

            if "sample_datetime" in chunk.columns:
                vals = chunk["sample_datetime"].dropna().astype(str).str.strip()
                vals = vals[vals.ne("")]
                if len(vals):
                    vmin, vmax = str(vals.min()), str(vals.max())
                    period["datetime_min"] = vmin if period["datetime_min"] is None else min(period["datetime_min"], vmin)
                    period["datetime_max"] = vmax if period["datetime_max"] is None else max(period["datetime_max"], vmax)
            if "sample_month" in chunk.columns:
                period["months"].update(str(v) for v in chunk["sample_month"].dropna().unique()[:120])
            if "user_type" in chunk.columns:
                period["user_types"].update(str(v) for v in chunk["user_type"].dropna().unique()[:20])
            if "feature_error" in chunk.columns:
                s = chunk["feature_error"].dropna().astype(str).str.strip()
                feature_error_n += int((s.ne("") & s.str.lower().ne("nan") & s.ne("None") & s.ne("<NA>")).sum())

            any_valid = np.zeros(n, dtype=bool)
            for arr in arrays.values():
                any_valid |= np.isfinite(arr)
            raw_valid = np.zeros(n, dtype=bool)
            for spec in features:
                raw_valid |= np.isfinite(clean_numeric(chunk[spec["column"]], spec["sentinel"]))
            gate_false_with_data += int((~gate & raw_valid).sum())
            gate_true_without_data += int((gate & ~any_valid).sum())

            # 口径一致性检查：只检查交易数据可用且源字段完整的行
            g = source.get("bank_txn_income_global_sum_182d", np.full(n, np.nan))
            w = source.get("bank_txn_income_wages_sum_182d", np.full(n, np.nan))
            c = source.get("bank_txn_income_centrelink_sum_182d", np.full(n, np.nan))
            o = source.get("bank_txn_income_otherincome_sum_182d", np.full(n, np.nan))
            m = gate & np.isfinite(g) & np.isfinite(w) & np.isfinite(c) & np.isfinite(o)
            if m.any():
                diff = np.abs((w + c + o) - g)
                bad = m & (diff > np.maximum(1.0, np.abs(g) * 1e-3))
                invariants["income_decomposition_mismatch"] += int(bad.sum())
                invariant_rows["income_decomposition"] += int(m.sum())

            tag = source.get("bank_txn_worthiness_income_tag_amt_sum_182d", np.full(n, np.nan))
            m = gate & np.isfinite(g) & np.isfinite(tag)
            if m.any():
                bad = m & (np.abs(tag - g) > np.maximum(1.0, np.abs(g) * 1e-3))
                invariants["worthiness_income_mismatch"] += int(bad.sum())
                invariant_rows["worthiness_income"] += int(m.sum())

            inc = source.get("bank_txn_surplus_income_sum_182d", np.full(n, np.nan))
            exp = source.get("bank_txn_surplus_expense_sum_182d", np.full(n, np.nan))
            rep = source.get("bank_txn_surplus_liabilityrepayment_sum_182d", np.full(n, np.nan))
            fee = source.get("bank_txn_surplus_dishonoursfee_sum_182d", np.full(n, np.nan))
            draw = source.get("bank_txn_surplus_liabilitydrawdown_sum_182d", np.full(n, np.nan))
            sur = source.get("bank_txn_surplus_surplus_sum_182d", np.full(n, np.nan))
            m = gate & np.isfinite(inc) & np.isfinite(exp) & np.isfinite(rep) & np.isfinite(fee) & np.isfinite(draw) & np.isfinite(sur)
            if m.any():
                expected = inc - exp - rep - fee + draw
                bad = m & (np.abs(expected - sur) > np.maximum(1.0, np.abs(inc) * 1e-3))
                invariants["surplus_mismatch"] += int(bad.sum())
                invariant_rows["surplus"] += int(m.sum())

            if cache_handle is not None:
                out_cols = [c for c in columns if c in chunk.columns]
                chunk[out_cols].to_csv(cache_handle, index=False, header=cache_writer_first)
                cache_writer_first = False

            if chunk_count % 10 == 0:
                print(f"  pass1: {total_rows:,} 行已处理 ...", flush=True)
    finally:
        if cache_handle is not None:
            cache_handle.close()

    elapsed = time.time() - t0
    period["months"] = sorted(period["months"])
    period["user_types"] = sorted(period["user_types"])
    return {
        "stats": stats,
        "total_rows": total_rows,
        "txn_rows": txn_rows,
        "chunk_count": chunk_count,
        "target": target_stats,
        "gate_present": gate_present,
        "gate_stats": gate_stats,
        "user_tracker": user_tracker,
        "app_tracker": app_tracker,
        "period": period,
        "feature_error_n": feature_error_n,
        "gate_false_with_data": gate_false_with_data,
        "gate_true_without_data": gate_true_without_data,
        "invariants": invariants,
        "invariant_rows": invariant_rows,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# 统计结果固化：分位断点与直方图
# ---------------------------------------------------------------------------

def build_breakpoints(stats: FeatureStats, direction: int, min_valid: int) -> dict | None:
    if stats.n < min_valid or stats.n < 2:
        return None
    ref = np.sort(stats.res.vals[np.isfinite(stats.res.vals)])
    if ref.size < 2 or ref[0] == ref[-1]:
        return None
    qs = np.linspace(0, 100, 201)
    xs = np.unique(np.percentile(ref, qs))
    if xs.size < 2:
        return None
    if direction >= 0:
        ys = 100.0 * np.searchsorted(ref, xs, side="right") / ref.size
    else:
        ys = 100.0 * (1.0 - np.searchsorted(ref, xs, side="left") / ref.size)
    ys = np.clip(ys, 0.0, 100.0)
    if np.allclose(ys, ys[0]):
        return None
    return {"xs": [float(v) for v in xs], "ys": [float(v) for v in ys], "valid_n": int(stats.n)}


def make_histogram(values: np.ndarray, spec: dict, bins: int) -> tuple[list[dict], dict]:
    vals = values[np.isfinite(values)]
    meta = {"sample_n": int(vals.size), "scale": "linear", "n_in_range": 0, "n_out_of_range": 0}
    if vals.size == 0:
        return [], meta
    scale = spec.get("hist_scale", "linear")
    if scale == "log" and vals.min() >= 0 and vals.max() > 0:
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
        n_out = int(((vals < lo) | (vals > hi)).sum())
        if hi > lo:
            t = np.log1p(vals[(vals >= lo) & (vals <= hi)])
            edges_t = np.linspace(np.log1p(lo), np.log1p(hi), bins + 1)
            counts, _ = np.histogram(t, bins=edges_t)
            edges = np.expm1(edges_t)
            meta["scale"] = "log"
        else:
            counts, edges = np.histogram(vals, bins=bins)
            n_out = 0
    else:
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
        n_out = int(((vals < lo) | (vals > hi)).sum())
        if hi <= lo:
            counts = np.array([vals.size])
            edges = np.array([lo, hi if hi > lo else lo + 1.0])
        else:
            in_range = vals[(vals >= lo) & (vals <= hi)]
            counts, edges = np.histogram(in_range, bins=bins, range=(lo, hi))
    meta["n_in_range"] = int(vals.size - n_out)
    meta["n_out_of_range"] = n_out
    rows = []
    for i, c in enumerate(counts):
        rows.append({
            "key": spec["key"],
            "bin_index": i,
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i + 1]),
            "count": int(c),
            "sample_n": int(vals.size),
            "scale": meta["scale"],
        })
    return rows, meta


def finalize_stats(cfg: dict, features: list[dict], derived: list[dict], pass1: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    total_rows = pass1["total_rows"]
    txn_rows = pass1["txn_rows"]
    breakpoints = {}
    rows = []
    hist_rows = []
    for spec in features + derived:
        st = pass1["stats"][spec["key"]]
        summ = st.summary()
        bp = None
        if spec["in_score"]:
            bp = build_breakpoints(st, spec["direction"], int(cfg["min_feature_valid"]))
            if bp:
                breakpoints[spec["key"]] = bp
        rows.append({
            "key": spec["key"],
            "label": spec["label"],
            "dimension": spec["dimension"],
            "dimension_label": DIMENSION_META.get(spec["dimension"], {}).get("label", spec["dimension"]),
            "module": spec["module"],
            "module_label": MODULE_LABELS.get(spec["module"], spec["module"]),
            "column": spec["column"] if not spec["derived"] else "派生指标",
            "sources": ";".join(spec.get("sources", [])),
            "definition": spec["definition"],
            "direction": spec["direction"],
            "direction_label": "正向" if spec["direction"] > 0 else ("负向" if spec["direction"] < 0 else "观察"),
            "fmt": spec["fmt"],
            "in_score": bool(spec["in_score"]),
            "weight": spec["weight"],
            "score_available": bool(bp is not None),
            "valid_n": summ["n"],
            "coverage_all_pct": 100.0 * summ["n"] / total_rows if total_rows else np.nan,
            "coverage_txn_pct": 100.0 * summ["n"] / txn_rows if txn_rows else np.nan,
            "missing_pct": 100.0 * (1 - summ["n"] / total_rows) if total_rows else np.nan,
            "mean": summ["mean"],
            "std": summ["std"],
            "min": summ["min"],
            "p01": summ["p01"],
            "p05": summ["p05"],
            "p10": summ["p10"],
            "p25": summ["p25"],
            "p50": summ["p50"],
            "p75": summ["p75"],
            "p90": summ["p90"],
            "p95": summ["p95"],
            "p99": summ["p99"],
            "max": summ["max"],
            "zero_pct": summ["zero_pct"],
            "negative_pct": summ["negative_pct"],
            "reference_n": summ["reference_n"],
        })
        hrows, hmeta = make_histogram(st.res.vals, spec, int(cfg["hist_bins"]))
        hist_rows.extend(hrows)
    return pd.DataFrame(rows), breakpoints, pd.DataFrame(hist_rows, columns=[
        "key", "bin_index", "bin_left", "bin_right", "count", "sample_n", "scale"])


# ---------------------------------------------------------------------------
# 第二遍：评分、分群、目标分析
# ---------------------------------------------------------------------------

def compute_scores(arrays: dict[str, np.ndarray], features: list[dict], breakpoints: dict, cfg: dict) -> tuple[dict, dict, dict, np.ndarray, np.ndarray]:
    n = len(next(iter(arrays.values()))) if arrays else 0
    feature_scores = {}
    for spec in features:
        if not spec["in_score"] or spec["key"] not in arrays:
            continue
        vals = arrays[spec["key"]]
        out = np.full(n, np.nan, dtype=np.float64)
        bp = breakpoints.get(spec["key"])
        if bp is not None:
            valid = np.isfinite(vals)
            if valid.any():
                out[valid] = np.interp(vals[valid], np.asarray(bp["xs"]), np.asarray(bp["ys"]))
                out[valid] = np.clip(out[valid], 0.0, 100.0)
        feature_scores[spec["key"]] = out

    dim_scores = {}
    dim_coverage = {}
    for dim in SCORED_DIMS:
        specs = [s for s in features if s["dimension"] == dim and s["in_score"] and s["key"] in feature_scores]
        if not specs:
            dim_scores[dim] = np.full(n, np.nan)
            dim_coverage[dim] = np.zeros(n)
            continue
        matrix = np.column_stack([feature_scores[s["key"]] for s in specs])
        weights = np.asarray([s["weight"] for s in specs], dtype=np.float64)
        valid = np.isfinite(matrix)
        valid_w = (valid * weights).sum(axis=1)
        total_w = sum(s["weight"] for s in FEATURES + DERIVED
                      if s["dimension"] == dim and s["direction"] != 0 and s["weight"] > 0)
        coverage = valid_w / total_w if total_w > 0 else np.zeros(n)
        raw = np.where(valid, matrix, 0.0) @ weights
        score = np.full(n, np.nan)
        np.divide(raw, valid_w, out=score, where=valid_w > 0)
        score[coverage < float(cfg["min_dimension_coverage"])] = np.nan
        dim_scores[dim] = score
        dim_coverage[dim] = coverage

    weights = np.asarray([HEALTH_WEIGHTS[d] for d in SCORED_DIMS], dtype=np.float64)
    matrix = np.column_stack([dim_scores[d] for d in SCORED_DIMS])
    valid = np.isfinite(matrix)
    valid_w = (valid * weights).sum(axis=1)
    total_w = weights.sum()
    health_cov = valid_w / total_w
    raw = np.where(valid, matrix, 0.0) @ weights
    health = np.full(n, np.nan)
    np.divide(raw, valid_w, out=health, where=valid_w > 0)
    health[health_cov < float(cfg["min_health_coverage"])] = np.nan

    return feature_scores, dim_scores, dim_coverage, health, health_cov


def build_flags(arrays: dict[str, np.ndarray], dim_scores: dict, health: np.ndarray, gate: np.ndarray, cfg: dict) -> tuple[dict, np.ndarray]:
    th = cfg["flag_thresholds"]
    n = gate.size

    def a(key):
        return arrays.get(key, np.full(n, np.nan))

    saving = a("cf_saving_rate")
    neg_share = a("lq_negative_share")
    bal_med = a("lq_balance_median")
    active_lender = a("db_active_lender_cnt")
    payday = a("db_cash_advance_sum")
    unresolved = a("db_unresolved_cnt")
    gambling = a("db_gambling_share")
    fees = a("db_fees_overdraft_share")
    salary_cv = a("inc_salary_cv")
    welfare = a("inc_centrelink_share")
    wage_sum = a("inc_wage_sum")
    income_sum = a("inc_global_sum")

    flags = {
        "flag_unresolved_dishonour": np.nan_to_num(unresolved, nan=0.0) > 0,
        "flag_high_debt": np.isfinite(dim_scores["debt"]) & (dim_scores["debt"] < float(th["high_debt_score"])),
        "flag_multi_lender": np.nan_to_num(active_lender, nan=0.0) >= float(th["multi_lender_cnt"]),
        "flag_payday": np.nan_to_num(payday, nan=0.0) > 0,
        "flag_negative_cashflow": np.isfinite(saving) & (saving < 0),
        "flag_low_buffer": (
            (np.isfinite(neg_share) & (neg_share >= float(th["low_buffer_negative_share"])))
            | (np.isfinite(bal_med) & (bal_med < 0))
        ),
        "flag_gambling": np.isfinite(gambling) & (gambling >= float(th["gambling_share"])),
        "flag_income_unstable": np.isfinite(salary_cv) & (salary_cv > float(th["income_unstable_cv"])),
        "flag_fees_heavy": np.isfinite(fees) & (fees >= float(th["fees_share"])),
        "flag_welfare_dependent": np.isfinite(welfare) & (welfare >= float(th["welfare_income_share"])),
        "flag_no_wage": gate & np.isfinite(wage_sum) & (wage_sum <= 0),
        "flag_no_income": gate & np.isfinite(income_sum) & (income_sum <= 0),
        "flag_no_txn": ~gate,
    }
    for k in flags:
        flags[k] = np.asarray(flags[k], dtype=bool)
        if k != "flag_no_txn":
            flags[k] &= gate
    risk_names = [k for k, _, _, is_risk in FLAG_DEFS if is_risk]
    risk_count = np.sum([flags[k].astype(np.int16) for k in risk_names], axis=0).astype(np.int16) if risk_names else np.zeros(n, dtype=np.int16)
    return flags, risk_count


def assign_segments(dim_scores: dict, health: np.ndarray, arrays: dict, flags: dict, risk_count: np.ndarray, gate: np.ndarray, cfg: dict) -> np.ndarray:
    th = cfg["segment_thresholds"]
    n = gate.size
    saving = arrays.get("cf_saving_rate", np.full(n, np.nan))
    income_score = dim_scores["income"]
    cashflow_score = dim_scores["cashflow"]
    liquidity_score = dim_scores["liquidity"]
    debt_score = dim_scores["debt"]

    seg = np.full(n, "均衡普通", dtype=object)
    moon = gate & np.isfinite(saving) & (saving < float(th["moon_saving_rate"]))
    stable = (
        gate
        & (income_score >= float(th["stable_income_score"]))
        & (cashflow_score >= float(th["stable_cashflow_score"]))
        & (debt_score >= float(th["stable_debt_score"]))
    )
    premium = gate & (health >= float(th["premium_health"])) & (risk_count <= int(th["premium_max_risk_flags"]))
    tight = gate & (
        (np.isfinite(saving) & (saving < 0))
        | (cashflow_score < float(th["tight_cashflow_score"]))
        | (liquidity_score < float(th["tight_liquidity_score"]))
    )
    highrisk = gate & (
        (risk_count >= int(th["highrisk_min_flags"]))
        | (flags["flag_unresolved_dishonour"] & (flags["flag_multi_lender"] | flags["flag_high_debt"]))
        | (np.isfinite(debt_score) & (debt_score < float(th["highrisk_debt_score"])))
    )

    seg[moon] = "月光消费"
    seg[stable] = "稳健工薪"
    seg[premium] = "优质均衡"
    seg[tight] = "现金流紧张"
    seg[highrisk] = "高风险承压"
    seg[~gate | ~np.isfinite(health)] = "数据不足"
    return seg


def run_pass2(cfg: dict, header: list[str], features: list[dict], derived: list[dict], breakpoints: dict,
              columns: list[str], pass1: dict, read_path: str | None = None) -> dict:
    rng = np.random.default_rng(cfg["seed"] + 1)
    path = read_path or cfg["input_path"]
    gate_col = cfg["gate_column"]
    gate_present = gate_col in header
    target_col = cfg["target_column"]
    target_present = target_col in header
    id_cols = [c for c in ("user_id", "application_id") if c in header]

    seg_k = int(cfg["segment_reservoir_size"])
    segment_metric_keys = CORE_METRIC_KEYS + [f"score_{d}" for d in SCORED_DIMS] + ["health_score", "risk_count"]
    seg_store = {
        seg: {
            "n": 0,
            "metrics": {k: FeatureStats(k, seg_k, rng) for k in segment_metric_keys},
            "flags": Counter(),
            "target_matured": 0,
            "target_pos": 0,
        }
        for seg in SEGMENT_ORDER
    }
    dim_stats = {d: FeatureStats(d, seg_k, rng) for d in SCORED_DIMS}
    health_stats = FeatureStats("health", seg_k, rng)
    score_hist = {d: np.zeros(10, dtype=np.int64) for d in SCORED_DIMS}
    score_hist["health"] = np.zeros(10, dtype=np.int64)
    score_hist_nan = {d: 0 for d in SCORED_DIMS}
    score_hist_nan["health"] = 0
    score_levels = {d: np.zeros(3, dtype=np.int64) for d in SCORED_DIMS}
    score_bands = {d: np.zeros((5, 3), dtype=np.int64) for d in SCORED_DIMS}
    score_bands["health"] = np.zeros((5, 3), dtype=np.int64)
    pair_keys = SCORED_DIMS + ["health"]
    pair_stats = {(a, b): PairStats() for i, a in enumerate(pair_keys) for b in pair_keys[i:]}
    smd_keys = [s["key"] for s in features + derived] + [f"score_{d}" for d in SCORED_DIMS] + ["health_score"]
    smd = {k: {"pos": MeanVar(), "neg": MeanVar()} for k in smd_keys}
    flag_totals = Counter()
    target_total = TargetStats(float(cfg["target_positive_value"]))
    total_rows = 0
    txn_rows = 0
    offset = 0
    observation_counts = Counter()
    observation_targets = Counter()
    observation_pos = Counter()

    user_path = Path(cfg["output_dir"]) / "user_scores.csv"
    user_handle = None
    user_first = True
    if cfg.get("export_user_scores", True):
        user_handle = open(user_path, "w", encoding="utf-8-sig", newline="")

    lender_cov_seg = {seg: Counter() for seg in SEGMENT_ORDER}
    lender_behavior_seg = {seg: Counter() for seg in SEGMENT_ORDER}
    lender_behavior_target = {b: {"matured": 0, "pos": 0} for b in LENDER_BEHAVIORS}
    lender_jgs = np.zeros(5, dtype=np.int64)
    lender_product = {
        (code, direction): {"users": 0, "amount": 0.0, "count": 0.0}
        for code, _ in LENDER_PRODUCTS for direction in ("disburse", "repay")
    }
    lender_comp: dict[str, dict] = {}
    lender_totals = Counter()
    lender_quality = Counter()

    income_arch_counts = Counter()
    income_arch_seg = {seg: Counter() for seg in SEGMENT_ORDER}
    income_rhythm_counts = Counter()
    income_rhythm_seg = {seg: Counter() for seg in SEGMENT_ORDER}
    income_rhythm_arch = {r: Counter() for r in PAY_RHYTHMS}
    income_arch_target = {a: {"matured": 0, "pos": 0} for a in INCOME_ARCHETYPES}
    income_arch_stats = {
        a: {
            key: FeatureStats(key, seg_k, rng)
            for key in (
                "inc_wage_share", "inc_salary_cv", "inc_salary_active_months",
                "inc_consecutive_wage_months", "inc_employer_streak_months",
                "inc_pay_interval_median", "inc_global_sum", "inc_centrelink_share",
                "inc_non_salary_share", "inc_source_count", "inc_primary_concentration",
            )
        }
        for a in INCOME_ARCHETYPES
    }
    income_rhythm_stats = {
        r: {key: FeatureStats(key, seg_k, rng) for key in ("inc_wage_share", "inc_salary_cv", "inc_global_sum")}
        for r in PAY_RHYTHMS
    }
    income_quality = Counter()

    dishonour_status_counts = Counter()
    dishonour_status_seg = {seg: Counter() for seg in SEGMENT_ORDER}
    dishonour_status_target = {s: {"matured": 0, "pos": 0} for s in DISHONOUR_STATUSES}
    dishonour_totals = Counter()
    dishonour_unresolved_amt_stats = FeatureStats("db_unresolved_amt", seg_k, rng)
    dishonour_seg_amt_stats = {seg: FeatureStats("db_unresolved_amt", seg_k, rng) for seg in SEGMENT_ORDER}
    dishonour_seg_counts = {seg: Counter() for seg in SEGMENT_ORDER}
    dishonour_type_stats = {
        code: {
            "users": 0, "unresolved_cnt": 0.0, "unresolved_amt": 0.0,
            "unresolved_inst_max": 0.0, "resolved_cnt": 0.0, "resolved_amt": 0.0,
            "catchup_cnt": 0.0,
        }
        for code, _ in DISHONOUR_TYPES
    }

    t0 = time.time()
    try:
        for chunk in read_chunks(path, columns, cfg["chunk_size"], header, id_cols):
            n = len(chunk)
            if n == 0:
                continue
            total_rows += n
            if gate_present:
                gate_vals = pd.to_numeric(chunk[gate_col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                gate = np.isfinite(gate_vals) & (gate_vals >= float(cfg["gate_min"]))
            else:
                gate = np.ones(n, dtype=bool)
            txn_rows += int(gate.sum())

            arrays, source = build_arrays(chunk, features, derived, gate)
            feature_scores, dim_scores, dim_coverage, health, health_cov = compute_scores(arrays, features + derived, breakpoints, cfg)
            flags, risk_count = build_flags(arrays, dim_scores, health, gate, cfg)
            seg = assign_segments(dim_scores, health, arrays, flags, risk_count, gate, cfg)

            if target_present:
                target = pd.to_numeric(chunk[target_col], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                target_total.add(target)
            else:
                target = np.full(n, np.nan)
            target_valid = np.isfinite(target) & ((target == 0) | (target == float(cfg["target_positive_value"])))

            observation_labels = {}
            for key in OBSERVATION_BANDS:
                labels = observation_band(arrays.get(key, np.full(n, np.nan)), key)
                observation_labels[key] = labels
                for segment in ["全部"] + SEGMENT_ORDER:
                    mask = gate if segment == "全部" else gate & (seg == segment)
                    for label in OBSERVATION_BANDS[key][1] + ["数据不足"]:
                        band_mask = mask & (labels == label)
                        group = (key, segment, label)
                        observation_counts[group] += int(band_mask.sum())
                        observation_targets[group] += int((band_mask & target_valid).sum())
                        observation_pos[group] += int((band_mask & target_valid & (target == float(cfg["target_positive_value"]))).sum())

            # ---- lender 结构、机构分布与近期融资动态 ----
            d182 = source.get("bank_txn_lender_disburse_amount_sum_l182d", np.full(n, np.nan))
            r182 = source.get("bank_txn_lender_repay_amount_sum_l182d", np.full(n, np.nan))
            d28 = source.get("bank_txn_lender_disburse_amount_sum_l28d", np.full(n, np.nan))
            r28 = source.get("bank_txn_lender_repay_amount_sum_l28d", np.full(n, np.nan))
            jgs = source.get("bank_txn_lender_disburse_jgs_l182d", np.full(n, np.nan))
            has_d = gate & np.isfinite(d182) & (d182 > 0)
            has_r = gate & np.isfinite(r182) & (r182 > 0)

            lender_totals["has_disburse"] += int(has_d.sum())
            lender_totals["has_repay"] += int(has_r.sum())
            known_lender = gate & np.isfinite(d182) & np.isfinite(r182)
            lender_totals["repay_only"] += int((known_lender & ~has_d & has_r).sum())
            lender_totals["no_lender"] += int((known_lender & ~has_d & ~has_r).sum())
            lender_totals["identified_borrowers"] += int((has_d & np.isfinite(jgs) & (jgs > 0)).sum())
            lender_totals["disburse_amount"] += float(np.nansum(d182[gate]))
            lender_totals["repay_amount"] += float(np.nansum(r182[gate]))
            lender_totals["disburse_amount_28d"] += float(np.nansum(d28[gate]))
            lender_totals["repay_amount_28d"] += float(np.nansum(r28[gate]))
            recent_share = np.divide(
                d28, d182, out=np.full(n, np.nan, dtype=np.float64),
                where=np.isfinite(d182) & (d182 > 0),
            )
            lender_totals["recent_share_high"] += int(
                (gate & np.isfinite(recent_share)
                 & (recent_share >= float(cfg["lender_recent_share_high"]))).sum()
            )

            for segment in SEGMENT_ORDER:
                mask = gate & (seg == segment)
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                bucket = lender_cov_seg[segment]
                bucket["users"] += cnt
                bucket["has_disburse"] += int((mask & has_d).sum())
                bucket["has_repay"] += int((mask & has_r).sum())
                bucket["repay_only"] += int((mask & known_lender & ~has_d & has_r).sum())
                bucket["no_lender"] += int((mask & known_lender & ~has_d & ~has_r).sum())
                bucket["multi_lender"] += int(
                    (mask & has_d & np.isfinite(jgs)
                     & (jgs >= float(cfg["lender_institution_high"]))).sum()
                )

            jv = jgs[has_d & np.isfinite(jgs)]
            if jv.size:
                jbucket = np.full(jv.shape, 4, dtype=np.int64)
                jbucket[jv <= 0] = 0
                jbucket[jv == 1] = 1
                jbucket[jv == 2] = 2
                jbucket[(jv >= 3) & (jv <= 4)] = 3
                lender_jgs += np.bincount(jbucket, minlength=5)

            d28z = np.where(np.isfinite(d28), d28, 0.0)
            r28z = np.where(np.isfinite(r28), r28, 0.0)
            tol = float(cfg["lender_net_tolerance"])
            behavior = np.full(n, "无借贷记录", dtype=object)
            active_hist = has_d | has_r
            behavior[gate & active_hist & (d28z <= 0) & (r28z <= 0)] = "近28天无借贷"
            behavior[gate & (d28z <= 0) & (r28z > 0)] = "仅偿还"
            behavior[gate & (d28z > 0) & (r28z <= 0)] = "仅新增借款"
            both = gate & (d28z > 0) & (r28z > 0)
            expand = both & (d28z > r28z * (1.0 + tol))
            shrink = both & (r28z > d28z * (1.0 + tol))
            behavior[expand] = "净融资扩张"
            behavior[shrink] = "净偿还"
            behavior[both & ~expand & ~shrink] = "借还平衡"
            behavior[~known_lender | ~np.isfinite(d28) | ~np.isfinite(r28)] = "数据不足"

            for segment in SEGMENT_ORDER:
                mask = gate & (seg == segment)
                if not mask.any():
                    continue
                for bname in LENDER_BEHAVIORS:
                    lender_behavior_seg[segment][bname] += int((mask & (behavior == bname)).sum())

            if target_present:
                pos_value = float(cfg["target_positive_value"])
                for bname in LENDER_BEHAVIORS:
                    bm = gate & (behavior == bname)
                    matured = int((bm & target_valid).sum())
                    lender_behavior_target[bname]["matured"] += matured
                    lender_behavior_target[bname]["pos"] += int(
                        (bm & target_valid & (target == pos_value)).sum()
                    )

            for code, _ in LENDER_PRODUCTS:
                for direction in ("disburse", "repay"):
                    vals = source.get(
                        f"bank_txn_lender_{direction}_{code}_amount_sum_l182d",
                        np.full(n, np.nan),
                    )
                    cnts = source.get(
                        f"bank_txn_lender_{direction}_{code}_count_l182d",
                        np.full(n, np.nan),
                    )
                    valid = gate & np.isfinite(vals) & (vals > 0)
                    entry = lender_product[(code, direction)]
                    entry["users"] += int(valid.sum())
                    entry["amount"] += float(np.nansum(vals[valid]))
                    entry["count"] += float(np.nansum(cnts[valid]))

            for col in chunk.columns:
                m = COMPETITOR_COUNT_RE.match(col)
                if not m:
                    continue
                name = m.group(1)
                cnt = clean_source_numeric(chunk[col], col)
                used = gate & np.isfinite(cnt) & (cnt > 0)
                if not used.any():
                    continue
                amount_col = col[: -len("_count")] + "_amount_sum"
                amt = (
                    clean_source_numeric(chunk[amount_col], amount_col)
                    if amount_col in chunk.columns
                    else np.full(n, np.nan)
                )
                entry = lender_comp.setdefault(name, {"users": 0, "amount": 0.0})
                entry["users"] += int(used.sum())
                entry["amount"] += float(np.nansum(amt[used]))
                lender_totals["competitor_amount"] += float(np.nansum(amt[used]))

            for direction in ("disburse", "repay"):
                for metric in ("amount_sum", "amount_count"):
                    windows = [
                        source.get(
                            f"bank_txn_lender_{direction}_{metric}_l{day}d",
                            np.full(n, np.nan),
                        )
                        for day in (7, 28, 84, 182)
                    ]
                    checked = gate & np.isfinite(np.vstack(windows)).all(axis=0)
                    bad = checked & (
                        (windows[0] > windows[1] + 1e-9)
                        | (windows[1] > windows[2] + 1e-9)
                        | (windows[2] > windows[3] + 1e-9)
                    )
                    lender_quality["window_violations"] += int(bad.sum())
                    lender_quality["window_checked"] += int(checked.sum())
                total = source.get(
                    f"bank_txn_lender_{direction}_amount_sum_l182d",
                    np.full(n, np.nan),
                )
                stack = np.vstack([
                    source.get(
                        f"bank_txn_lender_{direction}_{code}_amount_sum_l182d",
                        np.full(n, np.nan),
                    )
                    for code, _ in LENDER_PRODUCTS
                ])
                parts = np.nansum(stack, axis=0)
                all_nan = np.all(~np.isfinite(stack), axis=0)
                checked = gate & np.isfinite(total) & ~all_nan
                bad = checked & (parts > total + np.maximum(1.0, np.abs(total) * 1e-6))
                lender_quality[f"{direction}_product_overflow"] += int(bad.sum())
                lender_quality[f"{direction}_product_checked"] += int(checked.sum())

            # ---- 收入形态与发薪节律 ----
            income_sum = arrays.get("inc_global_sum", np.full(n, np.nan))
            centrelink_share = arrays.get("inc_centrelink_share", np.full(n, np.nan))
            wage_share = arrays.get("inc_wage_share", np.full(n, np.nan))
            salary_cv = arrays.get("inc_salary_cv", np.full(n, np.nan))
            salary_months = arrays.get("inc_salary_active_months", np.full(n, np.nan))
            no_income = gate & np.isfinite(income_sum) & (income_sum <= 0)
            welfare = (
                gate & ~no_income & np.isfinite(centrelink_share)
                & (centrelink_share >= float(cfg["flag_thresholds"]["welfare_income_share"]))
            )
            stable = (
                gate & ~no_income & ~welfare
                & np.isfinite(wage_share) & (wage_share >= float(cfg["income_stable_wage_share"]))
                & np.isfinite(salary_months) & (salary_months >= float(cfg["income_stable_min_months"]))
                & np.isfinite(salary_cv) & (salary_cv <= float(cfg["income_stable_cv"]))
            )
            unstable = (
                gate & ~no_income & ~welfare & ~stable
                & np.isfinite(wage_share) & (wage_share >= float(cfg["income_unstable_wage_share"]))
            )
            archetype = np.full(n, "多源/非工资", dtype=object)
            archetype[unstable] = "不稳定工资"
            archetype[stable] = "稳定工资"
            archetype[welfare] = "福利依赖"
            archetype[no_income] = "无收入"
            income_known = np.isfinite(income_sum) & (
                no_income | (np.isfinite(wage_share) & np.isfinite(centrelink_share)))
            archetype[~income_known] = "数据不足"

            rhythm = np.full(n, "无工资/未识别", dtype=object)
            pattern_stack = []
            for rhythm_label, rhythm_col in (
                ("月薪", "bank_txn_industry_pattern_monthly"),
                ("双周薪", "bank_txn_industry_pattern_fortnightly"),
                ("周薪", "bank_txn_industry_pattern_weekly"),
                ("不规律", "bank_txn_industry_pattern_irregular"),
            ):
                rhythm_flag = source.get(rhythm_col, np.full(n, np.nan))
                pattern_stack.append(np.where(np.isfinite(rhythm_flag), rhythm_flag, 0.0))
                rhythm[gate & np.isfinite(rhythm_flag) & (rhythm_flag > 0)] = rhythm_label
            pattern_sum = np.sum(np.vstack(pattern_stack), axis=0)
            rhythm[pattern_sum > 1] = "无工资/未识别"
            income_quality["pattern_checked"] += int(gate.sum())
            income_quality["pattern_conflicts"] += int((gate & (pattern_sum > 1)).sum())

            income_arch_counts.update(archetype[gate])
            income_rhythm_counts.update(rhythm[gate])
            for segment in SEGMENT_ORDER:
                mask = gate & (seg == segment)
                if not mask.any():
                    continue
                income_arch_seg[segment].update(archetype[mask])
                income_rhythm_seg[segment].update(rhythm[mask])
            for a in INCOME_ARCHETYPES:
                for r in PAY_RHYTHMS:
                    income_rhythm_arch[r][a] += int((gate & (archetype == a) & (rhythm == r)).sum())
            pos_value = float(cfg["target_positive_value"])
            for a in INCOME_ARCHETYPES:
                am = gate & (archetype == a)
                if not am.any():
                    continue
                for key, st in income_arch_stats[a].items():
                    vals = arrays.get(key)
                    if vals is not None:
                        st.add(vals[am])
                if target_present:
                    tv = am & target_valid
                    income_arch_target[a]["matured"] += int(tv.sum())
                    income_arch_target[a]["pos"] += int((tv & (target == pos_value)).sum())
            for r in PAY_RHYTHMS:
                rm = gate & (rhythm == r)
                if not rm.any():
                    continue
                for key, st in income_rhythm_stats[r].items():
                    vals = arrays.get(key)
                    if vals is not None:
                        st.add(vals[rm])

            # ---- 拒付恢复状态 ----
            ucnt_arr = arrays.get("db_unresolved_cnt", np.full(n, np.nan))
            rcnt_arr = arrays.get("db_resolved_cnt", np.full(n, np.nan))
            ucz = np.where(np.isfinite(ucnt_arr), ucnt_arr, 0.0)
            rcz = np.where(np.isfinite(rcnt_arr), rcnt_arr, 0.0)
            dish_status = np.full(n, "无拒付", dtype=object)
            any_dish = gate & ((ucz > 0) | (rcz > 0))
            dish_status[any_dish & (ucz <= 0)] = "已全部解决"
            dish_status[gate & (ucz > 0) & (rcz > 0)] = "部分解决"
            dish_status[gate & (ucz > 0) & (rcz <= 0)] = "未解决"
            dish_known = gate & np.isfinite(ucnt_arr) & np.isfinite(rcnt_arr)
            dish_status[~dish_known] = "数据不足"
            any_dish &= dish_known

            dishonour_status_counts.update(dish_status[gate])
            for segment in SEGMENT_ORDER:
                mask = gate & (seg == segment)
                if not mask.any():
                    continue
                dishonour_status_seg[segment].update(dish_status[mask])
            if target_present:
                for s in DISHONOUR_STATUSES:
                    sm = gate & (dish_status == s) & target_valid
                    dishonour_status_target[s]["matured"] += int(sm.sum())
                    dishonour_status_target[s]["pos"] += int((sm & (target == pos_value)).sum())

            uamt_arr = arrays.get("db_unresolved_amt", np.full(n, np.nan))
            ramt_arr = arrays.get("db_resolved_amt", np.full(n, np.nan))
            catch_arr = arrays.get("db_resolved_catchup_cnt", np.full(n, np.nan))
            inst_arr = arrays.get("db_unresolved_inst_max", np.full(n, np.nan))
            unres = gate & (ucz > 0)
            dishonour_totals["any_dishonour"] += int(any_dish.sum())
            dishonour_totals["resolved_only"] += int((dish_status[gate] == "已全部解决").sum())
            dishonour_totals["partial"] += int((dish_status[gate] == "部分解决").sum())
            dishonour_totals["unresolved_only"] += int((dish_status[gate] == "未解决").sum())
            dishonour_totals["unresolved_users"] += int(unres.sum())
            dishonour_totals["unresolved_cnt"] += float(np.nansum(ucz[gate]))
            dishonour_totals["resolved_cnt"] += float(np.nansum(rcz[gate]))
            dishonour_totals["unresolved_amt"] += float(np.nansum(uamt_arr[gate]))
            dishonour_totals["resolved_amt"] += float(np.nansum(ramt_arr[gate]))
            dishonour_totals["catchup_cnt"] += float(np.nansum(catch_arr[gate]))
            if np.isfinite(inst_arr[gate]).any():
                dishonour_totals["unresolved_inst_max"] = max(
                    dishonour_totals["unresolved_inst_max"], float(np.nanmax(inst_arr[gate]))
                )
            if unres.any():
                dishonour_unresolved_amt_stats.add(uamt_arr[unres])
            for segment in SEGMENT_ORDER:
                sm = unres & (seg == segment)
                if sm.any():
                    dishonour_seg_amt_stats[segment].add(uamt_arr[sm])
                seg_mask = gate & (seg == segment)
                if seg_mask.any():
                    bucket = dishonour_seg_counts[segment]
                    bucket["unresolved_cnt"] += float(np.nansum(ucz[seg_mask]))
                    bucket["resolved_cnt"] += float(np.nansum(rcz[seg_mask]))
                    bucket["unresolved_amt"] += float(np.nansum(uamt_arr[seg_mask]))
                    bucket["resolved_amt"] += float(np.nansum(ramt_arr[seg_mask]))

            for code, _ in DISHONOUR_TYPES:
                entry = dishonour_type_stats[code]
                ucnt_t = source.get(
                    f"bank_txn_dishonour_{code}_unresolved_cnt_3gp_182d", np.full(n, np.nan))
                rcnt_t = source.get(
                    f"bank_txn_dishonour_{code}_resolved_cnt_3gp_182d", np.full(n, np.nan))
                uamt_t = source.get(
                    f"bank_txn_dishonour_{code}_unresolved_amt_3gp_182d", np.full(n, np.nan))
                ramt_t = source.get(
                    f"bank_txn_dishonour_{code}_resolved_amt_3gp_182d", np.full(n, np.nan))
                inst_t = source.get(
                    f"bank_txn_dishonour_{code}_unresolved_institution_cnt_3gp_182d", np.full(n, np.nan))
                catch_t = source.get(
                    f"bank_txn_dishonour_{code}_resolved_catchup_cnt_3gp_182d", np.full(n, np.nan))
                used = gate & (
                    (np.isfinite(ucnt_t) & (ucnt_t > 0)) | (np.isfinite(rcnt_t) & (rcnt_t > 0))
                )
                entry["users"] += int(used.sum())
                entry["unresolved_cnt"] += float(np.nansum(ucnt_t[gate]))
                entry["unresolved_amt"] += float(np.nansum(uamt_t[gate]))
                entry["resolved_cnt"] += float(np.nansum(rcnt_t[gate]))
                entry["resolved_amt"] += float(np.nansum(ramt_t[gate]))
                entry["catchup_cnt"] += float(np.nansum(catch_t[gate]))
                valid_inst = inst_t[gate & np.isfinite(inst_t)]
                if valid_inst.size:
                    entry["unresolved_inst_max"] = max(
                        entry["unresolved_inst_max"], float(valid_inst.max())
                    )

            # 指标矩阵（分群画像 + SMD + 相关性）
            metric_arrays = {k: arrays.get(k, np.full(n, np.nan)) for k in CORE_METRIC_KEYS}
            for d in SCORED_DIMS:
                metric_arrays[f"score_{d}"] = dim_scores[d]
            metric_arrays["health_score"] = health
            metric_arrays["risk_count"] = risk_count.astype(np.float64)

            score_arrays = {d: dim_scores[d] for d in SCORED_DIMS}
            score_arrays["health"] = health

            # 分群聚合
            for segment in SEGMENT_ORDER:
                mask = seg == segment
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                bucket = seg_store[segment]
                bucket["n"] += cnt
                for key in CORE_METRIC_KEYS:
                    bucket["metrics"][key].add(metric_arrays[key][mask])
                for d in SCORED_DIMS:
                    bucket["metrics"][f"score_{d}"].add(dim_scores[d][mask])
                bucket["metrics"]["health_score"].add(health[mask])
                bucket["metrics"]["risk_count"].add(risk_count[mask].astype(np.float64))
                for flag_name, _, _, _ in FLAG_DEFS:
                    bucket["flags"][flag_name] += int(flags[flag_name][mask].sum())
                if target_present:
                    tv = target_valid & mask
                    bucket["target_matured"] += int(tv.sum())
                    bucket["target_pos"] += int((tv & (target == float(cfg["target_positive_value"]))).sum())

            # 全局维度分布与目标分层
            for d in SCORED_DIMS:
                arr = dim_scores[d]
                valid = np.isfinite(arr)
                if valid.any():
                    dim_stats[d].add(arr[valid])
                    score_levels[d] += np.array([
                        np.count_nonzero(arr[valid] < float(cfg["score_low"])),
                        np.count_nonzero((arr[valid] >= float(cfg["score_low"])) & (arr[valid] < float(cfg["score_high"]))),
                        np.count_nonzero(arr[valid] >= float(cfg["score_high"])),
                    ])
                    bands = np.clip((arr[valid] // 10).astype(int), 0, 9)
                    score_hist[d] += np.bincount(bands, minlength=10)
                    b20 = np.clip((arr[valid] // 20).astype(int), 0, 4)
                    score_bands[d][:, 0] += np.bincount(b20, minlength=5)
                score_hist_nan[d] += int((~valid).sum())
                if target_present and valid.any():
                    tmask = target_valid & valid
                    b20 = np.zeros(n, dtype=np.int64)
                    b20[valid] = np.clip((arr[valid] // 20).astype(int), 0, 4)
                    for b in range(5):
                        bm = tmask & (b20 == b)
                        score_bands[d][b, 1] += int(bm.sum())
                        score_bands[d][b, 2] += int((bm & (target == float(cfg["target_positive_value"]))).sum())

            valid = np.isfinite(health)
            if valid.any():
                health_stats.add(health[valid])
                bands = np.clip((health[valid] // 10).astype(int), 0, 9)
                score_hist["health"] += np.bincount(bands, minlength=10)
                b20 = np.clip((health[valid] // 20).astype(int), 0, 4)
                score_bands["health"][:, 0] += np.bincount(b20, minlength=5)
            score_hist_nan["health"] += int((~valid).sum())
            if target_present and valid.any():
                tmask = target_valid & valid
                b20 = np.zeros(n, dtype=np.int64)
                b20[valid] = np.clip((health[valid] // 20).astype(int), 0, 4)
                for b in range(5):
                    bm = tmask & (b20 == b)
                    score_bands["health"][b, 1] += int(bm.sum())
                    score_bands["health"][b, 2] += int((bm & (target == float(cfg["target_positive_value"]))).sum())

            # 维度相关性
            for i, a in enumerate(pair_keys):
                for b in pair_keys[i:]:
                    pair_stats[(a, b)].add(score_arrays[a], score_arrays[b])

            # 目标 SMD
            if target_present:
                pos_mask = target_valid & (target == float(cfg["target_positive_value"]))
                neg_mask = target_valid & (target == 0)
                for key in smd_keys:
                    if key in metric_arrays:
                        vals = metric_arrays[key]
                    else:
                        vals = arrays.get(key, np.full(n, np.nan))
                    smd[key]["pos"].add(vals[pos_mask])
                    smd[key]["neg"].add(vals[neg_mask])

            for flag_name, _, _, _ in FLAG_DEFS:
                flag_totals[flag_name] += int(flags[flag_name].sum())

            if user_handle is not None:
                user_id = chunk["user_id"].astype(str).to_numpy() if "user_id" in chunk.columns else np.arange(offset, offset + n).astype(str)
                app_id = chunk["application_id"].astype(str).to_numpy() if "application_id" in chunk.columns else np.arange(offset, offset + n).astype(str)
                out = pd.DataFrame({
                    "row_no": np.arange(offset, offset + n),
                    "user_id": user_id,
                    "application_id": app_id,
                    "segment": seg,
                    "health_score": np.round(health, 2),
                    "health_coverage": np.round(health_cov, 4),
                    "risk_flag_count": risk_count,
                })
                for d in SCORED_DIMS:
                    out[f"score_{d}"] = np.round(dim_scores[d], 2)
                    out[f"coverage_{d}"] = np.round(dim_coverage[d], 4)
                for flag_name, _, _, _ in FLAG_DEFS:
                    out[flag_name] = flags[flag_name].astype(np.int8)
                for key in ["cf_saving_rate", "lq_balance_median", "db_disburse_count",
                            "db_active_lender_cnt", "db_unresolved_cnt", "inc_wage_share",
                            "cs_discretionary_share", "en_active_days"]:
                    out[key] = arrays.get(key, np.full(n, np.nan))
                out["income_archetype"] = np.where(gate, archetype, "数据不足")
                out["lender_behavior"] = np.where(gate, behavior, "数据不足")
                out["dishonour_status"] = np.where(gate, dish_status, "数据不足")
                for key, labels in observation_labels.items():
                    out[key] = arrays.get(key, np.full(n, np.nan))
                    out[f"{key}_band"] = labels
                out.to_csv(user_handle, index=False, header=user_first, float_format="%.4f")
                user_first = False

            offset += n
            if offset % max(cfg["chunk_size"] * 10, 50000) == 0:
                print(f"  pass2: {offset:,} 行已处理 ...", flush=True)
    finally:
        if user_handle is not None:
            user_handle.close()
            if user_first:
                # 没有数据时也保留表头，避免下游误判文件缺失
                pd.DataFrame(columns=["row_no", "user_id", "application_id", "segment", "health_score"]).to_csv(user_path, index=False)

    elapsed = time.time() - t0

    # 维度汇总
    dim_rows = []
    for d in SCORED_DIMS:
        st = dim_stats[d].summary()
        n_valid = st["n"]
        hist = score_hist[d]
        low, mid, high = [int(v) for v in score_levels[d]]
        specs = [s for s in features + derived if s["dimension"] == d]
        scored_specs = [s for s in specs if s["in_score"] and s["key"] in breakpoints]
        dim_rows.append({
            "dimension": d,
            "label": DIMENSION_META[d]["label"],
            "desc": DIMENSION_META[d]["desc"],
            "weight": HEALTH_WEIGHTS[d],
            "scored_users": n_valid,
            "scored_pct": 100.0 * n_valid / txn_rows if txn_rows else np.nan,
            "mean": st["mean"],
            "std": st["std"],
            "p10": st["p10"],
            "p25": st["p25"],
            "p50": st["p50"],
            "p75": st["p75"],
            "p90": st["p90"],
            "low_pct": 100.0 * low / n_valid if n_valid else np.nan,
            "mid_pct": 100.0 * mid / n_valid if n_valid else np.nan,
            "high_pct": 100.0 * high / n_valid if n_valid else np.nan,
            "features_total": len(specs),
            "features_scored": len(scored_specs),
        })
    dim_df = pd.DataFrame(dim_rows)

    hist_rows = []
    for d, hist in score_hist.items():
        for i, c in enumerate(hist):
            hist_rows.append({"dimension": d, "bin_low": i * 10, "bin_high": (i + 1) * 10, "users": int(c)})
    hist_df = pd.DataFrame(hist_rows)

    # 分群汇总与画像
    seg_rows = []
    profile_rows = []
    for segment in SEGMENT_ORDER:
        bucket = seg_store[segment]
        cnt = bucket["n"]
        hsum = bucket["metrics"]["health_score"].summary()
        target_rate = bucket["target_pos"] / bucket["target_matured"] if bucket["target_matured"] else np.nan
        lo, hi = wilson_interval(bucket["target_pos"], bucket["target_matured"])
        row = {
            "segment": segment,
            "users": cnt,
            "share_pct": 100.0 * cnt / total_rows if total_rows else np.nan,
            "health_median": hsum["p50"],
            "health_mean": hsum["mean"],
            "risk_flags_mean": bucket["metrics"]["risk_count"].summary()["mean"],
            "target_matured": bucket["target_matured"],
            "target_pos": bucket["target_pos"],
            "target_rate": target_rate,
            "target_ci_low": lo,
            "target_ci_high": hi,
        }
        for d in SCORED_DIMS:
            row[f"score_{d}_median"] = bucket["metrics"][f"score_{d}"].summary()["p50"]
        for flag_name, _, _, is_risk in FLAG_DEFS:
            row[f"{flag_name}_pct"] = 100.0 * bucket["flags"][flag_name] / cnt if cnt else np.nan
        seg_rows.append(row)
        for key in CORE_METRIC_KEYS:
            s = bucket["metrics"][key].summary()
            profile_rows.append({
                "segment": segment,
                "metric": key,
                "n": s["n"],
                "mean": s["mean"],
                "p25": s["p25"],
                "p50": s["p50"],
                "p75": s["p75"],
            })
    seg_df = pd.DataFrame(seg_rows)
    profile_df = pd.DataFrame(profile_rows)

    # 目标分层
    band_rows = []
    for key, arr in score_bands.items():
        for b in range(5):
            users, matured, pos = [int(v) for v in arr[b]]
            rate = pos / matured if matured else np.nan
            lo, hi = wilson_interval(pos, matured)
            band_rows.append({
                "score_key": key,
                "score_label": DIMENSION_META.get(key, {}).get("label", "综合健康分"),
                "band": b,
                "band_label": f"{b * 20}-{b * 20 + 20}",
                "users": users,
                "matured": matured,
                "positives": pos,
                "rate": rate,
                "ci_low": lo,
                "ci_high": hi,
            })
    band_df = pd.DataFrame(band_rows)

    # SMD
    smd_rows = []
    for key in smd_keys:
        ps = smd[key]["pos"].stats()
        ns = smd[key]["neg"].stats()
        if ps["n"] >= 2 and ns["n"] >= 2 and np.isfinite(ps["var"]) and np.isfinite(ns["var"]):
            pooled = math.sqrt(max(0.0, (ps["var"] + ns["var"]) / 2.0))
            value = (ps["mean"] - ns["mean"]) / pooled if pooled > 0 else np.nan
        else:
            value = np.nan
        spec = next((s for s in features + derived if s["key"] == key), None)
        if spec:
            label = spec["label"]
        elif key == "health_score":
            label = "综合健康分"
        elif key.startswith("score_"):
            label = DIMENSION_META.get(key[6:], {}).get("label", key) + "得分"
        else:
            label = key
        smd_rows.append({
            "key": key,
            "label": label,
            "dimension": spec["dimension"] if spec else ("score" if key.startswith("score_") or key == "health_score" else ""),
            "n_pos": ps["n"],
            "n_neg": ns["n"],
            "mean_pos": ps["mean"],
            "mean_neg": ns["mean"],
            "smd": value,
            "abs_smd": abs(value) if np.isfinite(value) else np.nan,
        })
    smd_df = pd.DataFrame(smd_rows).sort_values("abs_smd", ascending=False, na_position="last")

    # 相关性
    corr_rows = []
    for (a, b), ps in pair_stats.items():
        c = ps.corr()
        corr_rows.append({"row": a, "col": b, "corr": c, "n": ps.n})
        if a != b:
            corr_rows.append({"row": b, "col": a, "corr": c, "n": ps.n})
    corr_df = pd.DataFrame(corr_rows)

    # 风险标签
    flag_rows = []
    for name, label, desc, is_risk in FLAG_DEFS:
        users = int(flag_totals.get(name, 0))
        denom = total_rows if name == "flag_no_txn" else txn_rows
        flag_rows.append({
            "flag": name,
            "label": label,
            "description": desc,
            "is_risk": bool(is_risk),
            "users": users,
            "denominator": denom,
            "prevalence_pct": 100.0 * users / denom if denom else np.nan,
        })
    flag_df = pd.DataFrame(flag_rows)

    # 消费结构
    cat_rows = []
    for key in CATEGORY_MIX_KEYS:
        spec = next((s for s in features if s["key"] == key), None)
        if not spec:
            continue
        summ = pass1["stats"][key].summary()
        cat_rows.append({
            "key": key,
            "label": spec["label"],
            "dimension": spec["dimension"],
            "mean_share": summ["mean"],
            "median_share": summ["p50"],
            "p75_share": summ["p75"],
            "users_nonzero": summ["n"] - int(summ["zero_pct"] * summ["n"] / 100.0) if summ["n"] else 0,
            "nonzero_pct": 100.0 - summ["zero_pct"] if summ["n"] else np.nan,
            "coverage_txn_pct": 100.0 * summ["n"] / txn_rows if txn_rows else np.nan,
        })
    cat_df = pd.DataFrame(cat_rows, columns=["key", "label", "dimension", "mean_share", "median_share", "p75_share",
                                             "users_nonzero", "nonzero_pct", "coverage_txn_pct"]).sort_values("mean_share", ascending=False)

    # lender 负债结构汇总
    def stat_p50(key):
        st = pass1["stats"].get(key)
        return st.summary()["p50"] if st else np.nan

    gate_n = txn_rows
    has_d_n = int(lender_totals["has_disburse"])
    has_r_n = int(lender_totals["has_repay"])
    no_lender_n = int(lender_totals["no_lender"])
    repay_only_n = int(lender_totals["repay_only"])
    behavior_totals = Counter()
    for segment in SEGMENT_ORDER:
        behavior_totals.update(lender_behavior_seg[segment])

    def pct(n, d):
        return 100.0 * n / d if d else np.nan

    summary_rows = [
        ("gate_users", "可用流水用户", gate_n, "count", "进入评分与分群"),
        ("has_disburse", "有放款记录用户", has_d_n, "count", f"占可用流水 {pct(has_d_n, gate_n):.1f}%"),
        ("repay_only", "仅有还款记录用户", repay_only_n, "count", f"占可用流水 {pct(repay_only_n, gate_n):.1f}%"),
        ("no_lender", "无借贷记录用户", no_lender_n, "count", f"占可用流水 {pct(no_lender_n, gate_n):.1f}%"),
        ("data_unknown", "借贷参与数据不足", gate_n - has_d_n - repay_only_n - no_lender_n,
         "count", "真实空值或缺列，不能解释为无借贷记录"),
        ("identified_borrowers", "可识别放款机构用户", int(lender_totals["identified_borrowers"]), "count",
         f"占有放款用户 {pct(int(lender_totals['identified_borrowers']), has_d_n):.1f}%"),
        ("disburse_amount", "放款总额（182天）", float(lender_totals["disburse_amount"]), "amount", "全部可用流水用户合计"),
        ("repay_amount", "还款总额（182天）", float(lender_totals["repay_amount"]), "amount", "全部可用流水用户合计"),
        ("disburse_jgs_median", "放款机构数中位数", stat_p50("db_disburse_jgs"), "count", "有放款记录用户"),
        ("multi_lender_share", "放款机构数≥3 占比",
         pct(int(sum(b["multi_lender"] for b in lender_cov_seg.values())), has_d_n), "pct", "占有放款记录用户"),
        ("recent_disburse_share_median", "近28天放款占182天比例中位数",
         stat_p50("db_recent_disburse_share"), "pct", "有放款记录用户"),
        ("recent_share_high", "近28天放款占比≥50%用户", int(lender_totals["recent_share_high"]), "count",
         f"占有放款记录用户 {pct(int(lender_totals['recent_share_high']), has_d_n):.1f}%"),
        ("recent_net_expand", "近28天净融资扩张用户", int(behavior_totals["净融资扩张"]), "count",
         f"占可用流水 {pct(int(behavior_totals['净融资扩张']), gate_n):.1f}%"),
        ("recent_net_shrink", "近28天净偿还用户", int(behavior_totals["净偿还"]), "count",
         f"占可用流水 {pct(int(behavior_totals['净偿还']), gate_n):.1f}%"),
        ("net_borrow_median", "近28天净融资中位数", stat_p50("db_net_borrow_28d"), "amount", "正值为净扩张，负值为净偿还"),
        ("lender_concentration_median", "单机构放款集中度中位数",
         stat_p50("db_lender_concentration"), "pct", "单机构最大累计放款 / 放款总额"),
    ]
    lender_summary_df = pd.DataFrame(
        summary_rows, columns=["metric", "label", "value", "unit", "note"]
    )

    inst_labels = ["0（未识别）", "1", "2", "3–4", "5+"]
    inst_total = int(lender_jgs.sum())
    lender_inst_df = pd.DataFrame([
        {
            "bucket": label,
            "users": int(lender_jgs[i]),
            "share_pct": pct(int(lender_jgs[i]), inst_total),
        }
        for i, label in enumerate(inst_labels)
    ])

    behavior_rows = []
    for bname in LENDER_BEHAVIORS:
        behavior_rows.append({
            "segment": "全部",
            "behavior": bname,
            "users": int(behavior_totals[bname]),
            "share_pct": pct(int(behavior_totals[bname]), gate_n),
        })
    for segment in SEGMENT_ORDER:
        users = int(lender_cov_seg[segment]["users"])
        for bname in LENDER_BEHAVIORS:
            n = int(lender_behavior_seg[segment][bname])
            behavior_rows.append({
                "segment": segment,
                "behavior": bname,
                "users": n,
                "share_pct": pct(n, users),
            })
    lender_behavior_df = pd.DataFrame(behavior_rows)

    target_rows = []
    for bname in LENDER_BEHAVIORS:
        users = int(behavior_totals[bname])
        matured = int(lender_behavior_target[bname]["matured"])
        pos = int(lender_behavior_target[bname]["pos"])
        lo, hi = wilson_interval(pos, matured)
        target_rows.append({
            "behavior": bname,
            "users": users,
            "matured": matured,
            "positives": pos,
            "rate": pos / matured if matured else np.nan,
            "ci_low": lo,
            "ci_high": hi,
        })
    lender_target_df = pd.DataFrame(target_rows)

    product_rows = []
    for direction, user_denom in (("disburse", has_d_n), ("repay", has_r_n)):
        amount_denom = sum(lender_product[(code, direction)]["amount"] for code, _ in LENDER_PRODUCTS)
        for code, label in LENDER_PRODUCTS:
            entry = lender_product[(code, direction)]
            product_rows.append({
                "product": code,
                "label": label,
                "direction": direction,
                "users": int(entry["users"]),
                "user_share_pct": pct(int(entry["users"]), user_denom),
                "amount": entry["amount"],
                "amount_share_pct": pct(entry["amount"], amount_denom),
                "count": entry["count"],
                "avg_ticket": entry["amount"] / entry["count"] if entry["count"] else np.nan,
            })
    lender_product_df = pd.DataFrame(product_rows)

    comp_rows = []
    comp_denom_amount = float(lender_totals["competitor_amount"])
    for name, entry in sorted(lender_comp.items(), key=lambda kv: (-kv[1]["users"], -kv[1]["amount"])):
        comp_rows.append({
            "competitor": name.replace("_", " "),
            "users": int(entry["users"]),
            "user_share_pct": pct(int(entry["users"]), int(lender_totals["identified_borrowers"])),
            "amount": entry["amount"],
            "amount_share_pct": pct(entry["amount"], comp_denom_amount),
        })
    lender_top_df = pd.DataFrame(
        comp_rows[: max(1, int(cfg.get("lender_top_n", 10)))],
        columns=["competitor", "users", "user_share_pct", "amount", "amount_share_pct"],
    )

    seg_lender_rows = []
    for segment in SEGMENT_ORDER:
        bucket = lender_cov_seg[segment]
        users = int(bucket["users"])
        metrics = seg_store[segment]["metrics"]

        def mp50(key):
            return metrics[key].summary()["p50"] if key in metrics else np.nan

        seg_lender_rows.append({
            "segment": segment,
            "users": users,
            "has_disburse_pct": pct(int(bucket["has_disburse"]), users),
            "repay_only_pct": pct(int(bucket["repay_only"]), users),
            "no_lender_pct": pct(int(bucket["no_lender"]), users),
            "multi_lender_pct": pct(int(bucket["multi_lender"]), int(bucket["has_disburse"])),
            "net_expand_pct": pct(int(lender_behavior_seg[segment]["净融资扩张"]), users),
            "net_shrink_pct": pct(int(lender_behavior_seg[segment]["净偿还"]), users),
            "disburse_jgs_median": mp50("db_disburse_jgs"),
            "recent_disburse_share_median": mp50("db_recent_disburse_share"),
            "net_borrow_median": mp50("db_net_borrow_28d"),
            "borrow_repay_ratio_median": mp50("db_borrow_repay_ratio_28d"),
            "lender_concentration_median": mp50("db_lender_concentration"),
            "advance_disburse_share_median": mp50("db_advance_disburse_share"),
            "bnpl_repay_share_median": mp50("db_bnpl_repay_share"),
        })
    lender_seg_df = pd.DataFrame(seg_lender_rows)

    # 收入形态与发薪节律
    def arch_stat(archetype, key):
        return income_arch_stats[archetype][key].summary()

    income_summary_rows = [
        ("gate_users", "可用流水用户", gate_n, "count", "进入评分与分群"),
        ("stable_wage", "稳定工资型", int(income_arch_counts["稳定工资"]), "count",
         f"占可用流水 {pct(int(income_arch_counts['稳定工资']), gate_n):.1f}%"),
        ("unstable_wage", "不稳定工资型", int(income_arch_counts["不稳定工资"]), "count",
         f"占可用流水 {pct(int(income_arch_counts['不稳定工资']), gate_n):.1f}%"),
        ("multi_source", "多源/非工资型", int(income_arch_counts["多源/非工资"]), "count",
         f"占可用流水 {pct(int(income_arch_counts['多源/非工资']), gate_n):.1f}%"),
        ("welfare", "福利依赖型", int(income_arch_counts["福利依赖"]), "count",
         f"占可用流水 {pct(int(income_arch_counts['福利依赖']), gate_n):.1f}%"),
        ("no_income", "无收入型", int(income_arch_counts["无收入"]), "count",
         f"占可用流水 {pct(int(income_arch_counts['无收入']), gate_n):.1f}%"),
        ("monthly_pay", "月薪", int(income_rhythm_counts["月薪"]), "count", "发薪节律识别结果"),
        ("fortnightly_pay", "双周薪", int(income_rhythm_counts["双周薪"]), "count", "发薪节律识别结果"),
        ("weekly_pay", "周薪", int(income_rhythm_counts["周薪"]), "count", "发薪节律识别结果"),
        ("irregular_pay", "不规律发薪", int(income_rhythm_counts["不规律"]), "count", "发薪节律识别结果"),
        ("unrecognized_pay", "无工资/未识别节律", int(income_rhythm_counts["无工资/未识别"]), "count",
         "含无工资入账与工资流水不足以识别节律的用户"),
        ("median_consecutive_months", "连续有工资月数中位数", stat_p50("inc_consecutive_wage_months"),
         "months", "可用流水用户"),
        ("median_employer_streak", "主雇主连续月数中位数", stat_p50("inc_employer_streak_months"),
         "months", "有可识别雇主的用户"),
        ("median_pay_interval", "发薪间隔中位数", stat_p50("inc_pay_interval_median"),
         "days", "有可识别工资节律的用户"),
    ]
    income_summary_df = pd.DataFrame(
        income_summary_rows, columns=["metric", "label", "value", "unit", "note"]
    )

    income_arch_rows = []
    for a in INCOME_ARCHETYPES:
        users = int(income_arch_counts[a])
        income_arch_rows.append({
            "archetype": a,
            "users": users,
            "share_pct": pct(users, gate_n),
            "wage_share_median": arch_stat(a, "inc_wage_share")["p50"],
            "salary_cv_median": arch_stat(a, "inc_salary_cv")["p50"],
            "salary_active_months_median": arch_stat(a, "inc_salary_active_months")["p50"],
            "consecutive_months_median": arch_stat(a, "inc_consecutive_wage_months")["p50"],
            "employer_streak_median": arch_stat(a, "inc_employer_streak_months")["p50"],
            "pay_interval_median": arch_stat(a, "inc_pay_interval_median")["p50"],
            "income_median": arch_stat(a, "inc_global_sum")["p50"],
            "centrelink_share_median": arch_stat(a, "inc_centrelink_share")["p50"],
            "non_salary_share_median": arch_stat(a, "inc_non_salary_share")["p50"],
            "source_count_median": arch_stat(a, "inc_source_count")["p50"],
            "primary_concentration_median": arch_stat(a, "inc_primary_concentration")["p50"],
        })
    income_arch_df = pd.DataFrame(income_arch_rows)

    rhythm_rows = []
    for r in PAY_RHYTHMS:
        users = int(income_rhythm_counts[r])
        mix = income_rhythm_arch[r]
        rhythm_rows.append({
            "rhythm": r,
            "users": users,
            "share_pct": pct(users, gate_n),
            "wage_share_median": income_rhythm_stats[r]["inc_wage_share"].summary()["p50"],
            "salary_cv_median": income_rhythm_stats[r]["inc_salary_cv"].summary()["p50"],
            "income_median": income_rhythm_stats[r]["inc_global_sum"].summary()["p50"],
            **{f"{a}_pct": pct(mix[a], users) for a in INCOME_ARCHETYPES},
        })
    income_rhythm_df = pd.DataFrame(rhythm_rows)

    income_seg_rows = []
    for segment in SEGMENT_ORDER:
        arch_mix = income_arch_seg[segment]
        rhythm_mix = income_rhythm_seg[segment]
        users = int(sum(arch_mix.values()))
        metrics = seg_store[segment]["metrics"]

        def seg_p50(key):
            return metrics[key].summary()["p50"] if key in metrics else np.nan

        income_seg_rows.append({
            "segment": segment,
            "users": users,
            **{f"{a}_pct": pct(arch_mix[a], users) for a in INCOME_ARCHETYPES},
            "irregular_pay_pct": pct(rhythm_mix["不规律"], users),
            "unrecognized_pay_pct": pct(rhythm_mix["无工资/未识别"], users),
            "wage_share_median": seg_p50("inc_wage_share"),
            "salary_cv_median": seg_p50("inc_salary_cv"),
            "consecutive_months_median": seg_p50("inc_consecutive_wage_months"),
            "employer_streak_median": seg_p50("inc_employer_streak_months"),
            "pay_interval_median": seg_p50("inc_pay_interval_median"),
        })
    income_seg_df = pd.DataFrame(income_seg_rows)

    income_target_rows = []
    for a in INCOME_ARCHETYPES:
        users = int(income_arch_counts[a])
        matured = int(income_arch_target[a]["matured"])
        pos = int(income_arch_target[a]["pos"])
        lo, hi = wilson_interval(pos, matured)
        income_target_rows.append({
            "archetype": a,
            "users": users,
            "matured": matured,
            "positives": pos,
            "rate": pos / matured if matured else np.nan,
            "ci_low": lo,
            "ci_high": hi,
        })
    income_target_df = pd.DataFrame(income_target_rows)

    # 拒付恢复
    resolved_total = float(dishonour_totals["resolved_cnt"])
    unresolved_total = float(dishonour_totals["unresolved_cnt"])
    resolved_amt_total = float(dishonour_totals["resolved_amt"])
    unresolved_amt_total = float(dishonour_totals["unresolved_amt"])
    dishonour_summary_rows = [
        ("gate_users", "可用流水用户", gate_n, "count", "进入评分与分群"),
        ("any_dishonour", "有拒付记录用户", int(dishonour_totals["any_dishonour"]), "count",
         f"占可用流水 {pct(int(dishonour_totals['any_dishonour']), gate_n):.1f}%"),
        ("resolved_only", "拒付已全部解决", int(dishonour_totals["resolved_only"]), "count",
         f"占有拒付记录 {pct(int(dishonour_totals['resolved_only']), int(dishonour_totals['any_dishonour'])):.1f}%"),
        ("partial", "部分未解决", int(dishonour_totals["partial"]), "count",
         f"占有拒付记录 {pct(int(dishonour_totals['partial']), int(dishonour_totals['any_dishonour'])):.1f}%"),
        ("unresolved_only", "全部未解决", int(dishonour_totals["unresolved_only"]), "count",
         f"占有拒付记录 {pct(int(dishonour_totals['unresolved_only']), int(dishonour_totals['any_dishonour'])):.1f}%"),
        ("unresolved_users", "存在未结清拒付用户", int(dishonour_totals["unresolved_users"]), "count",
         f"占可用流水 {pct(int(dishonour_totals['unresolved_users']), gate_n):.1f}%"),
        ("resolved_ratio", "拒付结清率", resolved_total / (resolved_total + unresolved_total)
         if (resolved_total + unresolved_total) > 0 else np.nan, "pct", "结清次数 / 全部拒付次数"),
        ("unresolved_amt_ratio", "未结清金额占比", unresolved_amt_total / (resolved_amt_total + unresolved_amt_total)
         if (resolved_amt_total + unresolved_amt_total) > 0 else np.nan, "pct", "未结清金额 / 全部拒付金额"),
        ("resolved_cnt", "已结清拒付次数", resolved_total, "count", "全部可用流水用户合计"),
        ("unresolved_cnt", "未结清拒付次数", unresolved_total, "count", "全部可用流水用户合计"),
        ("resolved_amt", "已结清拒付金额", resolved_amt_total, "amount", "全部可用流水用户合计"),
        ("unresolved_amt", "未结清拒付金额", unresolved_amt_total, "amount", "全部可用流水用户合计"),
        ("catchup_cnt", "补扣结清次数", float(dishonour_totals["catchup_cnt"]), "count",
         "通过多次补扣完成的结清次数"),
        ("unresolved_inst_max", "单类未结清机构数最大值", float(dishonour_totals["unresolved_inst_max"]),
         "count", "单一拒付类型下单用户未结清机构数的最大值"),
        ("unresolved_amt_median", "未结清金额中位数", dishonour_unresolved_amt_stats.summary()["p50"],
         "amount", "存在未结清拒付的用户"),
    ]
    dishonour_summary_df = pd.DataFrame(
        dishonour_summary_rows, columns=["metric", "label", "value", "unit", "note"]
    )

    dishonour_recovery_rows = []
    for s in DISHONOUR_STATUSES:
        n = int(dishonour_status_counts[s])
        dishonour_recovery_rows.append({
            "segment": "全部", "status": s, "users": n, "share_pct": pct(n, gate_n),
        })
    for segment in SEGMENT_ORDER:
        users = int(sum(dishonour_status_seg[segment].values()))
        for s in DISHONOUR_STATUSES:
            n = int(dishonour_status_seg[segment][s])
            dishonour_recovery_rows.append({
                "segment": segment, "status": s, "users": n, "share_pct": pct(n, users),
            })
    dishonour_recovery_df = pd.DataFrame(dishonour_recovery_rows)

    dishonour_seg_rows = []
    for segment in SEGMENT_ORDER:
        mix = dishonour_status_seg[segment]
        users = int(sum(mix.values()))
        counts = dishonour_seg_counts[segment]
        any_n = sum(int(mix[s]) for s in ("已全部解决", "部分解决", "未解决"))
        dishonour_seg_rows.append({
            "segment": segment,
            "users": users,
            "any_dishonour_pct": pct(any_n, users),
            "resolved_only_pct": pct(int(mix["已全部解决"]), users),
            "partial_pct": pct(int(mix["部分解决"]), users),
            "unresolved_only_pct": pct(int(mix["未解决"]), users),
            "unresolved_user_pct": pct(int(mix["部分解决"]) + int(mix["未解决"]), users),
            "resolved_ratio": (counts["resolved_cnt"] / (counts["resolved_cnt"] + counts["unresolved_cnt"])
                               if (counts["resolved_cnt"] + counts["unresolved_cnt"]) > 0 else np.nan),
            "unresolved_amt_median": dishonour_seg_amt_stats[segment].summary()["p50"],
        })
    dishonour_seg_df = pd.DataFrame(dishonour_seg_rows)

    dishonour_type_rows = []
    for code, label in DISHONOUR_TYPES:
        entry = dishonour_type_stats[code]
        total_cnt = entry["resolved_cnt"] + entry["unresolved_cnt"]
        dishonour_type_rows.append({
            "type": code,
            "label": label,
            "users": int(entry["users"]),
            "unresolved_cnt": entry["unresolved_cnt"],
            "unresolved_amt": entry["unresolved_amt"],
            "unresolved_inst_max": entry["unresolved_inst_max"],
            "resolved_cnt": entry["resolved_cnt"],
            "resolved_amt": entry["resolved_amt"],
            "catchup_cnt": entry["catchup_cnt"],
            "recovery_rate": entry["resolved_cnt"] / total_cnt if total_cnt > 0 else np.nan,
        })
    dishonour_type_df = pd.DataFrame(dishonour_type_rows)

    dishonour_target_rows = []
    for s in DISHONOUR_STATUSES:
        users = int(dishonour_status_counts[s])
        matured = int(dishonour_status_target[s]["matured"])
        pos = int(dishonour_status_target[s]["pos"])
        lo, hi = wilson_interval(pos, matured)
        dishonour_target_rows.append({
            "status": s,
            "users": users,
            "matured": matured,
            "positives": pos,
            "rate": pos / matured if matured else np.nan,
            "ci_low": lo,
            "ci_high": hi,
        })
    dishonour_target_df = pd.DataFrame(dishonour_target_rows)

    income_quality_summary = {
        "pattern_checked": int(income_quality.get("pattern_checked", 0)),
        "pattern_conflicts": int(income_quality.get("pattern_conflicts", 0)),
    }

    lender_quality = {
        "window_checked": int(lender_quality.get("window_checked", 0)),
        "window_violations": int(lender_quality.get("window_violations", 0)),
        "disburse_product_checked": int(lender_quality.get("disburse_product_checked", 0)),
        "disburse_product_overflow": int(lender_quality.get("disburse_product_overflow", 0)),
        "repay_product_checked": int(lender_quality.get("repay_product_checked", 0)),
        "repay_product_overflow": int(lender_quality.get("repay_product_overflow", 0)),
    }

    observation_rows = []
    for (key, segment, band), count in observation_counts.items():
        group = (key, segment, band)
        denominator = sum(v for (k, s, _), v in observation_counts.items() if k == key and s == segment)
        matured = observation_targets[group]
        pos = observation_pos[group]
        lo, hi = wilson_interval(pos, matured)
        observation_rows.append({
            "key": key, "segment": segment, "band": band, "users": count,
            "denominator": denominator, "share_pct": 100.0 * count / denominator if denominator else np.nan,
            "matured": matured, "positives": pos, "rate": pos / matured if matured else np.nan,
            "ci_low": lo, "ci_high": hi,
        })

    return {
        "total_rows": total_rows,
        "txn_rows": txn_rows,
        "target": target_total,
        "health_summary": health_stats.summary(),
        "dim_df": dim_df,
        "dim_hist_df": hist_df,
        "seg_df": seg_df,
        "profile_df": profile_df,
        "band_df": band_df,
        "smd_df": smd_df,
        "corr_df": corr_df,
        "flag_df": flag_df,
        "cat_df": cat_df,
        "lender_summary_df": lender_summary_df,
        "lender_inst_df": lender_inst_df,
        "lender_seg_df": lender_seg_df,
        "lender_behavior_df": lender_behavior_df,
        "lender_target_df": lender_target_df,
        "lender_product_df": lender_product_df,
        "lender_top_df": lender_top_df,
        "lender_quality": lender_quality,
        "income_summary_df": income_summary_df,
        "income_arch_df": income_arch_df,
        "income_rhythm_df": income_rhythm_df,
        "income_seg_df": income_seg_df,
        "income_target_df": income_target_df,
        "income_quality": income_quality_summary,
        "dishonour_summary_df": dishonour_summary_df,
        "dishonour_recovery_df": dishonour_recovery_df,
        "dishonour_seg_df": dishonour_seg_df,
        "dishonour_type_df": dishonour_type_df,
        "dishonour_target_df": dishonour_target_df,
        "observation_df": pd.DataFrame(observation_rows),
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# 洞察生成
# ---------------------------------------------------------------------------

def _fnum(v, fmt="ratio", digits=1):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if fmt == "pct":
        return f"{v * 100:.{digits}f}%"
    if fmt == "amount":
        return f"A${v:,.0f}"
    if fmt == "count":
        return f"{v:,.0f}"
    if fmt == "days":
        return f"{v:.{digits}f} 天"
    if fmt == "months":
        return f"{v:.{digits}f} 个月"
    if fmt == "score":
        return f"{v:.{digits}f}"
    return f"{v:,.2f}"


def generate_insights(cfg: dict, overview: dict, feature_df: pd.DataFrame, seg_df: pd.DataFrame,
                      flag_df: pd.DataFrame, target: dict, smd_df: pd.DataFrame, corr_df: pd.DataFrame,
                      lender_summary: pd.DataFrame | None = None,
                      lender_behavior: pd.DataFrame | None = None,
                      lender_products: pd.DataFrame | None = None,
                      income_archetypes: pd.DataFrame | None = None,
                      income_pay_rhythm: pd.DataFrame | None = None,
                      dishonour_summary: pd.DataFrame | None = None,
                      dishonour_types: pd.DataFrame | None = None) -> dict:
    fmap = {r["key"]: r for _, r in feature_df.iterrows()}
    smap = {r["segment"]: r for _, r in seg_df.iterrows()}
    lsum = {}
    if lender_summary is not None and len(lender_summary):
        lsum = {r["metric"]: r for _, r in lender_summary.iterrows()}

    def lv(key):
        row = lsum.get(key)
        return float(row["value"]) if row is not None and pd.notna(row["value"]) else np.nan

    def f(key, field):
        row = fmap.get(key)
        return row[field] if row is not None else np.nan

    def flag_pct(name):
        row = flag_df[flag_df["flag"] == name]
        return float(row["prevalence_pct"].iloc[0]) if len(row) else np.nan

    scored_seg = seg_df[seg_df["segment"] != "数据不足"] if len(seg_df) else seg_df
    top_seg = scored_seg.sort_values("users", ascending=False).iloc[0] if len(scored_seg) else None
    findings = []
    caveats = []

    findings.append({
        "title": "数据可用性",
        "text": (
            f"共读取 {overview['rows']:,} 条申请，其中 {overview['txn_rows']:,} 条（{overview['txn_share_pct']:.1f}%）"
            f"具备可用交易流水；其余 {overview['rows'] - overview['txn_rows']:,} 条归为数据不足；有流水但评分覆盖不足的记录也归入数据不足。"
        ),
    })
    if top_seg is not None:
        findings.append({
            "title": "最大客群",
            "text": (
                f"「{top_seg['segment']}」为最大客群，共 {int(top_seg['users']):,} 条（{top_seg['share_pct']:.1f}%），"
                f"健康分中位数 {_fnum(top_seg['health_median'], 'score')}，"
                f"风险标签均值 {_fnum(top_seg['risk_flags_mean'], 'ratio', 2)} 个。"
            ),
        })
    findings.append({
        "title": "收支与储蓄",
        "text": (
            f"储蓄率（余额变动口径）中位数 {_fnum(f('cf_saving_rate', 'p50'), 'pct')}，"
            f"有效用户中 {f('cf_saving_rate', 'negative_pct'):.1f}% 为净流出；"
            f"收支结余率（流水口径）中位数 {_fnum(f('cf_net_after_expense', 'p50'), 'pct')}。"
        ),
    })
    findings.append({
        "title": "负债与拒付",
        "text": (
            f"机构名单内多头借贷（≥3家）触发率 {flag_pct('flag_multi_lender'):.1f}%，"
            f"存在未结清拒付 {flag_pct('flag_unresolved_dishonour'):.1f}%，"
            f"现金贷/工资贷使用率 {flag_pct('flag_payday'):.1f}%；"
            f"历史放款次数中位数 {_fnum(f('db_disburse_count', 'p50'), 'count')} 次。"
        ),
    })
    if lsum:
        gate_n = lv("gate_users")
        has_d = lv("has_disburse")
        findings.append({
            "title": "负债参与结构",
            "text": (
                f"可用流水用户中 {has_d:,.0f} 条（{(100.0 * has_d / gate_n if gate_n else 0.0):.1f}%）有放款记录，"
                f"{lv('repay_only'):,.0f} 条仅有还款记录，{lv('no_lender'):,.0f} 条无借贷记录；"
                f"有放款用户的放款机构数中位数 {_fnum(lv('disburse_jgs_median'), 'count')} 家，"
                f"机构数≥3 的占 {lv('multi_lender_share'):.1f}%。"
                "风险标签“多头借贷”仅统计机构名单内竞品，与此处全量去重机构数口径不同。"
            ),
        })
        findings.append({
            "title": "近期融资动态",
            "text": (
                f"近 28 天净融资扩张 {lv('recent_net_expand'):,.0f} 条、净偿还 {lv('recent_net_shrink'):,.0f} 条；"
                f"近 28 天放款占 182 天放款比例中位数 {_fnum(lv('recent_disburse_share_median'), 'pct')}，"
                f"单机构放款集中度中位数 {_fnum(lv('lender_concentration_median'), 'pct')}。"
            ),
        })
    if lender_products is not None and len(lender_products):
        dis = lender_products[lender_products["direction"] == "disburse"].copy()
        if len(dis):
            dis = dis.sort_values("amount", ascending=False)
            parts = "、".join(
                f"{r['label']} {r['amount_share_pct']:.1f}%"
                for _, r in dis.head(3).iterrows()
                if pd.notna(r["amount_share_pct"])
            )
            findings.append({
                "title": "放款产品结构",
                "text": (
                    f"按可识别产品的放款金额，{parts}；"
                    "产品分项未覆盖全部放款，占比仅在已分类金额内计算。"
                ),
            })
    findings.append({
        "title": "收入结构",
        "text": (
            f"工资收入占比中位数 {_fnum(f('inc_wage_share', 'p50'), 'pct')}，"
            f"无工资入账触发率 {flag_pct('flag_no_wage'):.1f}%，"
            f"福利收入依赖（≥50%）触发率 {flag_pct('flag_welfare_dependent'):.1f}%。"
        ),
    })
    if income_archetypes is not None and len(income_archetypes):
        arch = income_archetypes.sort_values("users", ascending=False)
        top_arch = arch.iloc[0]
        stable_row = arch[arch["archetype"] == "稳定工资"]
        stable_share = float(stable_row["share_pct"].iloc[0]) if len(stable_row) else np.nan
        rhythm_txt = ""
        if income_pay_rhythm is not None and len(income_pay_rhythm):
            rhythm_top = income_pay_rhythm[income_pay_rhythm["users"] > 0].sort_values(
                "users", ascending=False
            )
            if len(rhythm_top):
                rr = rhythm_top.iloc[0]
                rhythm_txt = f"发薪节律以「{rr['rhythm']}」为主（{rr['share_pct']:.1f}%）；"
        findings.append({
            "title": "收入形态与发薪节律",
            "text": (
                f"最大收入形态为「{top_arch['archetype']}」，占 {top_arch['share_pct']:.1f}%；"
                f"稳定工资型占 {_fnum(stable_share / 100.0, 'pct')}。{rhythm_txt}"
                "收入形态只基于流水可识别的工资、福利与收入来源，现金收入可能导致低估。"
            ),
        })
    if dishonour_summary is not None and len(dishonour_summary):
        dsum = {r["metric"]: r for _, r in dishonour_summary.iterrows()}

        def dv(key):
            row = dsum.get(key)
            return float(row["value"]) if row is not None and pd.notna(row["value"]) else np.nan

        any_n = dv("any_dishonour")
        gate_n = dv("gate_users")
        unres_n = dv("unresolved_users")
        unres_share = 100.0 * unres_n / any_n if any_n > 0 else np.nan
        type_txt = ""
        if dishonour_types is not None and len(dishonour_types):
            ranked = dishonour_types[dishonour_types["unresolved_amt"] > 0].sort_values(
                "unresolved_amt", ascending=False
            )
            if len(ranked):
                tr = ranked.iloc[0]
                type_txt = f"未结清金额最高的是{tr['label']}（A${tr['unresolved_amt']:,.0f}）。"
        findings.append({
            "title": "拒付与恢复",
            "text": (
                f"有拒付记录 {any_n:,.0f} 条（占可用流水 {(100.0 * any_n / gate_n if gate_n else 0.0):.1f}%），"
                f"其中 {unres_n:,.0f} 条存在未结清拒付（{_fnum(unres_share / 100.0, 'pct')}）；"
                f"拒付结清率 {_fnum(dv('resolved_ratio'), 'pct')}，"
                f"未结清金额合计 A${dv('unresolved_amt'):,.0f}。{type_txt}"
            ),
        })
    findings.append({
        "title": "流动性缓冲",
        "text": (
            f"余额中位数 {_fnum(f('lq_balance_median', 'p50'), 'amount')}，"
            f"负余额天数占比中位数 {_fnum(f('lq_negative_share', 'p50'), 'pct')}，"
            f"固定支出备用月数中位数 {_fnum(f('lq_reserve_months', 'p50'), 'months')}。"
        ),
    })
    findings.append({
        "title": "消费结构",
        "text": (
            f"可选消费占比中位数 {_fnum(f('cs_discretionary_share', 'p50'), 'pct')}，"
            f"必需支出占比中位数 {_fnum(f('cs_essentials_share', 'p50'), 'pct')}，"
            f"赌博类支出触发率 {flag_pct('flag_gambling'):.1f}%。"
        ),
    })
    if any(key in fmap for key in OBSERVATION_BANDS):
        findings.append({
            "title": "近期收支与支出缓冲",
            "text": (
                f"近28天相对此前154天的日均收入倍数中位数 {_fnum(f('tr_income_daily_ratio', 'p50'), 'ratio', 2)}，"
                f"日均支出倍数中位数 {_fnum(f('tr_expense_daily_ratio', 'p50'), 'ratio', 2)}；"
                f"高承诺支出/收入中位数 {_fnum(f('rs_commitment_income', 'p50'), 'pct')}，"
                f"期末余额支出覆盖天数中位数 {_fnum(f('rs_expense_buffer_days', 'p50'), 'days')}。"
                "这些指标仅描述已观测流水，不计入健康分；需结合窗口完整性解读。"
            ),
        })

    if target.get("valid_n", 0):
        rate = target["rate"]
        findings.append({
            "title": "目标标签（探索性）",
            "text": (
                f"已到期样本 {target['valid_n']:,} 条，正样本 {target['positives']:,} 条"
                f"（{rate * 100:.1f}%，95%CI {target['ci_low'] * 100:.1f}%-{target['ci_high'] * 100:.1f}%）。"
                "正样本量很小，目标关联仅用于提出假设，不构成统计显著结论。"
            ),
        })
        caveats.append("目标标签正样本极少，任何分组逾期率差异都可能由随机波动导致，不能据此做上线决策。")
    else:
        caveats.append("目标列不存在或没有已到期样本，未做目标关联分析。")

    if len(corr_df):
        pairs = corr_df[(corr_df["row"] != corr_df["col"]) & corr_df["corr"].notna()].copy()
        pairs["abs"] = pairs["corr"].abs()
        top = pairs.sort_values("abs", ascending=False).head(1)
        if len(top):
            r = top.iloc[0]
            label_map = {k: v["label"] for k, v in DIMENSION_META.items()}
            label_map["health"] = "综合健康分"
            findings.append({
                "title": "维度相关性",
                "text": (
                    f"{label_map.get(r['row'], r['row'])}与{label_map.get(r['col'], r['col'])}"
                    f"的相关系数为 {r['corr']:.2f}（n={int(r['n']):,}），"
                    "维度之间存在一定信息重叠，分群阈值需结合业务复核。"
                ),
            })

    if flag_pct("flag_no_txn") > 30:
        caveats.append("超过三成申请没有可用交易流水，评分覆盖率不足，报告中的客群结构仅代表有流水客群。")
    if overview.get("single_month"):
        caveats.append("申请样本集中在单个月份，无法验证跨申请月份的客群迁徙；本报告的近期变化仅比较同一申请内的流水窗口。")
    if overview.get("rows", 0) < 5000:
        caveats.append("当前样本量较小，分位数与分群边界在生产全量数据上应重新校准。")
    caveats.append("借贷机构识别依赖交易分类与第三方机构映射；无放款记录不等于无负债，未映射交易不计入产品结构。")
    caveats.append("拒付“已解决/未解决”依赖特征代码的补扣识别窗口；未结清只表示窗口内未识别到补扣，不等同于当前逾期。")
    caveats.append("收入形态依赖工资、福利与收入来源流水识别，现金收入、家庭内部转账和一次性入账可能造成误分类。")
    caveats.append("金额单位沿用源数据口径（推断为澳元），比率类指标需注意余额变动口径与流水分类口径不可混用。")
    caveats.append("画像结果用于客群理解与策略假设，不替代征信、反欺诈、收入验证与合规审查。")

    narratives = {}
    for segment in SEGMENT_ORDER:
        row = smap.get(segment)
        if row is None:
            continue
        if segment == "数据不足":
            narratives[segment] = (
                f"{int(row['users']):,} 条（{row['share_pct']:.1f}%）未达到流水门槛或健康分覆盖率要求，"
                "未计算健康分与维度评分。"
            )
            continue
        narratives[segment] = (
            f"{int(row['users']):,} 条（{row['share_pct']:.1f}%），"
            f"健康分中位数 {_fnum(row['health_median'], 'score')}，"
            f"收入稳定性 {_fnum(row['score_income_median'], 'score')}、"
            f"收支储蓄 {_fnum(row['score_cashflow_median'], 'score')}、"
            f"流动性 {_fnum(row['score_liquidity_median'], 'score')}、"
            f"负债压力 {_fnum(row['score_debt_median'], 'score')}。"
        )
    narratives.pop("_", None)

    return {
        "key_findings": findings,
        "segment_narratives": narratives,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# 输出与主流程
# ---------------------------------------------------------------------------

def _cache_fingerprint(input_path: str, columns: list[str]) -> tuple[str, dict]:
    st = os.stat(input_path)
    meta = {
        "input_path": str(Path(input_path).resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "columns": sorted(columns),
    }
    raw = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16], meta


def _cache_is_ready(cache_path: Path, meta_path: Path, fingerprint: str) -> bool:
    if not cache_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("fingerprint") == fingerprint
    except Exception:
        return False


def _write_quality(cfg: dict, overview: dict, pass1: dict, feature_df: pd.DataFrame,
                   dim_df: pd.DataFrame, seg_df: pd.DataFrame, target: dict, cache_used: bool,
                   lender_quality: dict | None = None,
                   income_quality: dict | None = None) -> pd.DataFrame:
    rows = []

    def add(check, value, status, detail):
        rows.append({"check": check, "value": value, "status": status, "detail": detail})

    rows_total = overview["rows"]
    txn_rows = overview["txn_rows"]
    share = overview["txn_share_pct"]
    add("申请记录数", f"{rows_total:,}", "info", "分析粒度为申请记录；同一用户可能有多条申请")
    add("可用交易流水记录数", f"{txn_rows:,}（{share:.1f}%）", "ok" if share >= 50 else "warn",
        f"以 {cfg['gate_column']} ≥ {cfg['gate_min']} 判定；无流水记录不参与评分")
    add("特征列数", f"{len(feature_df):,}", "info", f"其中参与评分 {int(feature_df['in_score'].sum()):,} 个")
    add("评分可用特征数", f"{int(feature_df['score_available'].sum()):,}", "ok" if feature_df['score_available'].sum() >= 0.8 * feature_df['in_score'].sum() else "warn",
        "有效样本不足或无区分度的特征不会进入评分")
    add("全量用户去重数", f"{pass1['user_tracker'].n:,}" + ("+" if pass1['user_tracker'].capped else ""),
        "info", "用户 ID 去重计数；达到跟踪上限时以 + 标记")
    add("申请 ID 去重数", f"{pass1['app_tracker'].n:,}" + ("+" if pass1['app_tracker'].capped else ""),
        "info", "申请 ID 去重计数")
    add("交易计数门槛列", cfg["gate_column"] if pass1["gate_present"] else "缺失",
        "ok" if pass1["gate_present"] else "warn",
        "缺失时退化为全部记录参与评分" if not pass1["gate_present"] else "用于区分是否存在原始交易数据")
    add("门槛外但有特征值", f"{pass1['gate_false_with_data']:,}", "ok" if pass1['gate_false_with_data'] == 0 else "warn",
        "门槛判定与特征缺失应保持一致；大于 0 说明存在口径冲突")
    add("门槛内但全特征缺失", f"{pass1['gate_true_without_data']:,}", "ok" if pass1['gate_true_without_data'] == 0 else "warn",
        "门槛内记录理论上应至少有一个可用特征")
    add("目标列", cfg["target_column"], "ok" if target.get("valid_n", 0) else "warn",
        f"已到期 {target.get('valid_n', 0):,} 条，正样本 {target.get('positives', 0):,} 条")
    add("无效目标标签", str(target.get("other", 0)), "warn" if target.get("other", 0) else "ok",
        f"仅0和{cfg['target_positive_value']}进入目标分析，其余非空数值均排除")
    insufficient = int(seg_df.loc[seg_df["segment"] == "数据不足", "users"].sum())
    low_coverage = insufficient - (rows_total - txn_rows)
    add("有流水但评分覆盖不足", str(low_coverage), "warn" if low_coverage else "ok",
        "健康分不可计算时归入数据不足；不能默认归为均衡普通")
    dsum = overview.get("dictionary", {})
    add("数据字典匹配", f"{dsum.get('matched_features', 0):,} / {dsum.get('dictionary_features', 0):,}",
        "ok" if dsum.get("sentinel_conflicts", 0) == 0 else "warn",
        "dict.csv 用于核对缺失值哨兵；冲突数大于 0 时需检查特征版本")
    if target.get("valid_n", 0):
        add("目标正样本比例", f"{target['rate'] * 100:.2f}%", "warn" if target["positives"] < 30 else "ok",
            "正样本少于 30 条时，分组逾期率仅作探索")
    add("收入分解一致性", f"{pass1['invariants'].get('income_decomposition_mismatch', 0):,} / {pass1['invariant_rows'].get('income_decomposition', 0):,}",
        "ok" if pass1['invariants'].get('income_decomposition_mismatch', 0) == 0 else "warn",
        "工资 + 福利 + 其他收入 与总收入在 0.1% 容差内的不一致行数")
    add("收入口径一致性", f"{pass1['invariants'].get('worthiness_income_mismatch', 0):,} / {pass1['invariant_rows'].get('worthiness_income', 0):,}",
        "ok" if pass1['invariants'].get('worthiness_income_mismatch', 0) == 0 else "warn",
        "收入模块总收入与资质模块 income_tag 在 0.1% 容差内的不一致行数")
    add("结余恒等式一致性", f"{pass1['invariants'].get('surplus_mismatch', 0):,} / {pass1['invariant_rows'].get('surplus', 0):,}",
        "ok" if pass1['invariants'].get('surplus_mismatch', 0) == 0 else "warn",
        "收入 - 支出 - 负债还款 - 拒付费用 + 负债提款 与结余在 0.1% 容差内的不一致行数")
    add("特征错误标记", f"{pass1['feature_error_n']:,}", "ok" if pass1['feature_error_n'] == 0 else "warn",
        "源数据 feature_error 非空记录数")
    add("数据期间", overview.get("period_text", "—"), "info", "样本时间范围")
    add("子集缓存", "启用" if cache_used else "未启用", "info",
        "缓存为窄表 gzip，仅用于加速第二遍扫描；不改变统计口径")
    add("分群覆盖", f"{int(seg_df['users'].sum()):,} / {rows_total:,}", "ok" if int(seg_df['users'].sum()) == rows_total else "warn",
        "所有申请应且仅应落入一个分群")
    add("维度评分覆盖率", "；".join(f"{r['label']} {r['scored_pct']:.1f}%" for _, r in dim_df.iterrows()),
        "info", "维度得分非空的记录占可用交易流水记录比例")
    if lender_quality:
        add("借贷窗口单调性",
            f"{lender_quality.get('window_violations', 0):,} / {lender_quality.get('window_checked', 0):,}",
            "ok" if lender_quality.get("window_violations", 0) == 0 else "warn",
            "近7/28/84/182天放款与还款金额、笔数应逐窗口不减少")
        add("借贷产品分项不超总额",
            f"放款 {lender_quality.get('disburse_product_overflow', 0):,} / {lender_quality.get('disburse_product_checked', 0):,}；"
            f"还款 {lender_quality.get('repay_product_overflow', 0):,} / {lender_quality.get('repay_product_checked', 0):,}",
            "ok" if (lender_quality.get("disburse_product_overflow", 0) == 0
                     and lender_quality.get("repay_product_overflow", 0) == 0) else "warn",
            "个人贷款、现金贷、BNPL、银行分项之和不应超过对应总额")
    if income_quality:
        add("发薪节律互斥性",
            f"{income_quality.get('pattern_conflicts', 0):,} / {income_quality.get('pattern_checked', 0):,}",
            "ok" if income_quality.get("pattern_conflicts", 0) == 0 else "warn",
            "monthly/fortnightly/weekly/irregular 四个发薪节律标记应互斥")
    return pd.DataFrame(rows)


def _period_text(period: dict) -> str:
    lo, hi = period.get("datetime_min"), period.get("datetime_max")
    months = period.get("months") or []
    if lo and hi:
        return f"{lo[:10]} 至 {hi[:10]}" if lo != hi else lo[:10]
    if months:
        return "、".join(months[:6]) + (" 等" if len(months) > 6 else "")
    return "—"


def _write_json(path: Path, obj: dict) -> None:
    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    path.write_text(json.dumps(normalize(obj), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def run_analysis(config: dict | None = None) -> dict:
    cfg = config or load_config()
    t_start = time.time()
    input_path = cfg["input_path"]
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 读取表头并校验特征注册表: {Path(input_path).name}", flush=True)
    header = read_header(input_path)
    if cfg["gate_column"] not in header:
        raise ValueError(f"缺少流水门槛列 {cfg['gate_column']}，无法可靠区分可用数据")
    features, derived, missing = resolve_registry(header, cfg)
    if not features and not derived:
        raise ValueError("输入文件没有可识别的画像特征")
    if missing:
        print(f"  警告：{len(missing)} 个特征列缺失，已跳过: {', '.join(missing[:5])}", flush=True)
    dictionary = read_dictionary(cfg["dict_path"])
    dict_matched = 0
    sentinel_conflicts = 0
    for spec in features:
        info = dictionary.get(spec["column"])
        if not info:
            continue
        dict_matched += 1
        try:
            default = float(info["default"])
        except (TypeError, ValueError):
            continue
        expected = -1.0 if spec["sentinel"] == "lender" else -999999.99999
        if abs(default - expected) > 0.01:
            sentinel_conflicts += 1
    dictionary_summary = {
        "dictionary_features": len(dictionary),
        "matched_features": dict_matched,
        "sentinel_conflicts": sentinel_conflicts,
    }
    print(f"  数据字典: {len(dictionary):,} 个特征，匹配 {dict_matched:,} 个，"
          f"缺失值哨兵冲突 {sentinel_conflicts:,} 个", flush=True)
    columns = collect_columns(features, derived, cfg, header)
    file_mb = os.path.getsize(input_path) / 1e6
    print(f"  文件 {file_mb:,.1f} MB，{len(header):,} 列；本次读取 {len(columns):,} 列，"
          f"注册特征 {len(features) + len(derived):,} 个", flush=True)

    # 可选窄表缓存
    cache_path = None
    cache_meta_path = None
    cache_ready = False
    fingerprint, cache_meta = _cache_fingerprint(input_path, columns)
    mode = str(cfg.get("subset_cache", "auto")).lower()
    use_cache = mode == "on" or (mode == "auto" and file_mb >= float(cfg["subset_cache_threshold_mb"]))
    if use_cache:
        cache_path = cache_dir / f"subset_{fingerprint}.csv.gz"
        cache_meta_path = cache_dir / f"subset_{fingerprint}.meta.json"
        cache_ready = _cache_is_ready(cache_path, cache_meta_path, fingerprint)
        print(f"  子集缓存: {'复用' if cache_ready else '本次生成'} {cache_path.name}", flush=True)

    print("[2/5] 第一遍扫描：覆盖度、分位数、口径一致性", flush=True)
    pass1 = run_pass1(cfg, header, features, derived, columns,
                      cache_path if use_cache else None, cache_ready)
    if use_cache and not cache_ready:
        _write_json(cache_meta_path, {**cache_meta, "fingerprint": fingerprint, "columns": columns})
    print(f"  完成 {pass1['total_rows']:,} 行，用时 {pass1['elapsed']:.1f}s；"
          f"可用流水 {pass1['txn_rows']:,} 行", flush=True)
    if pass1["total_rows"] == 0:
        raise ValueError("输入文件没有数据行，无法进行画像分析")

    print("[3/5] 固化特征统计、评分断点与直方图", flush=True)
    feature_df, breakpoints, hist_df = finalize_stats(cfg, features, derived, pass1)
    feature_df.to_csv(output_dir / "feature_stats.csv", index=False, encoding="utf-8-sig")
    hist_df.to_csv(output_dir / "feature_histograms.csv", index=False, encoding="utf-8-sig")
    _write_json(output_dir / "feature_breakpoints.json", {
        "reference": "valid values, bottom-k reservoir",
        "reservoir_size": cfg["reservoir_size"],
        "features": breakpoints,
    })

    print("[4/5] 第二遍扫描：评分、分群、目标分析与分组聚合", flush=True)
    read_path = str(cache_path) if (use_cache and cache_path is not None and cache_path.exists()) else None
    pass2 = run_pass2(cfg, header, features, derived, breakpoints, columns, pass1, read_path=read_path)
    print(f"  完成 {pass2['total_rows']:,} 行，用时 {pass2['elapsed']:.1f}s", flush=True)

    target = pass1["target"].as_dict()
    target.update({
        "column": cfg["target_column"],
        "label": cfg["target_label"],
        "positive_value": cfg["target_positive_value"],
        "available": bool(target.get("valid_n", 0)),
    })

    period_text = _period_text(pass1["period"])
    rows_ = pass1["total_rows"]
    txn_rows = pass1["txn_rows"]
    overview = {
        "version": VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": Path(input_path).name,
        "input_size_mb": round(file_mb, 2),
        "rows": rows_,
        "txn_rows": txn_rows,
        "txn_share_pct": 100.0 * txn_rows / rows_ if rows_ else np.nan,
        "columns_in_file": len(header),
        "columns_read": len(columns),
        "features_registered": len(features) + len(derived),
        "features_scored": len(breakpoints),
        "dictionary": dictionary_summary,
        "period_text": period_text,
        "period": pass1["period"],
        "single_month": len(pass1["period"].get("months") or []) <= 1,
        "user_types": pass1["period"].get("user_types", []),
        "target": target,
        "segment_order": SEGMENT_ORDER,
        "dimensions": DIMENSION_META,
        "health_weights": HEALTH_WEIGHTS,
        "health": pass2["health_summary"],
        "thresholds": {
            "segment": cfg["segment_thresholds"],
            "flag": cfg["flag_thresholds"],
            "score_low": cfg["score_low"],
            "score_high": cfg["score_high"],
        },
        "timing_sec": {
            "pass1": round(pass1["elapsed"], 2),
            "pass2": round(pass2["elapsed"], 2),
            "total_before_report": round(time.time() - t_start, 2),
        },
        "subset_cache_used": bool(read_path),
        "outputs": [
            "feature_stats.csv", "feature_histograms.csv", "feature_breakpoints.json",
            "dimension_summary.csv", "dimension_score_hist.csv", "segment_summary.csv",
            "segment_profiles.csv", "risk_flags.csv", "category_mix.csv", "correlations.csv",
            "target_overall.json", "target_by_segment.csv", "target_by_score_band.csv",
            "target_feature_smd.csv", "quality_checks.csv", "observation_bands.csv",
            "lender_summary.csv", "lender_institutions.csv", "lender_segments.csv",
            "lender_behavior.csv", "lender_behavior_target.csv", "lender_products.csv",
            "lender_top_competitors.csv",
            "income_summary.csv", "income_archetypes.csv", "income_pay_rhythm.csv",
            "income_segments.csv", "income_archetype_target.csv",
            "dishonour_summary.csv", "dishonour_recovery.csv", "dishonour_segments.csv",
            "dishonour_types.csv", "dishonour_recovery_target.csv",
            "overview.json", "insights.json", "run_meta.json",
        ],
    }

    pass2["dim_df"].to_csv(output_dir / "dimension_summary.csv", index=False, encoding="utf-8-sig")
    if cfg.get("export_user_scores", True):
        overview["outputs"].append("user_scores.csv")
    pass2["observation_df"].to_csv(output_dir / "observation_bands.csv", index=False, encoding="utf-8-sig")
    pass2["dim_hist_df"].to_csv(output_dir / "dimension_score_hist.csv", index=False, encoding="utf-8-sig")
    pass2["seg_df"].to_csv(output_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")
    pass2["profile_df"].to_csv(output_dir / "segment_profiles.csv", index=False, encoding="utf-8-sig")
    pass2["flag_df"].to_csv(output_dir / "risk_flags.csv", index=False, encoding="utf-8-sig")
    pass2["cat_df"].to_csv(output_dir / "category_mix.csv", index=False, encoding="utf-8-sig")
    pass2["corr_df"].to_csv(output_dir / "correlations.csv", index=False, encoding="utf-8-sig")
    _write_json(output_dir / "target_overall.json", target)
    pass2["seg_df"][["segment", "users", "target_matured", "target_pos", "target_rate", "target_ci_low", "target_ci_high"]].to_csv(
        output_dir / "target_by_segment.csv", index=False, encoding="utf-8-sig")
    pass2["band_df"].to_csv(output_dir / "target_by_score_band.csv", index=False, encoding="utf-8-sig")
    pass2["smd_df"].to_csv(output_dir / "target_feature_smd.csv", index=False, encoding="utf-8-sig")
    pass2["lender_summary_df"].to_csv(output_dir / "lender_summary.csv", index=False, encoding="utf-8-sig")
    pass2["lender_inst_df"].to_csv(output_dir / "lender_institutions.csv", index=False, encoding="utf-8-sig")
    pass2["lender_seg_df"].to_csv(output_dir / "lender_segments.csv", index=False, encoding="utf-8-sig")
    pass2["lender_behavior_df"].to_csv(output_dir / "lender_behavior.csv", index=False, encoding="utf-8-sig")
    pass2["lender_target_df"].to_csv(output_dir / "lender_behavior_target.csv", index=False, encoding="utf-8-sig")
    pass2["lender_product_df"].to_csv(output_dir / "lender_products.csv", index=False, encoding="utf-8-sig")
    pass2["lender_top_df"].to_csv(output_dir / "lender_top_competitors.csv", index=False, encoding="utf-8-sig")
    pass2["income_summary_df"].to_csv(output_dir / "income_summary.csv", index=False, encoding="utf-8-sig")
    pass2["income_arch_df"].to_csv(output_dir / "income_archetypes.csv", index=False, encoding="utf-8-sig")
    pass2["income_rhythm_df"].to_csv(output_dir / "income_pay_rhythm.csv", index=False, encoding="utf-8-sig")
    pass2["income_seg_df"].to_csv(output_dir / "income_segments.csv", index=False, encoding="utf-8-sig")
    pass2["income_target_df"].to_csv(output_dir / "income_archetype_target.csv", index=False, encoding="utf-8-sig")
    pass2["dishonour_summary_df"].to_csv(output_dir / "dishonour_summary.csv", index=False, encoding="utf-8-sig")
    pass2["dishonour_recovery_df"].to_csv(output_dir / "dishonour_recovery.csv", index=False, encoding="utf-8-sig")
    pass2["dishonour_seg_df"].to_csv(output_dir / "dishonour_segments.csv", index=False, encoding="utf-8-sig")
    pass2["dishonour_type_df"].to_csv(output_dir / "dishonour_types.csv", index=False, encoding="utf-8-sig")
    pass2["dishonour_target_df"].to_csv(output_dir / "dishonour_recovery_target.csv", index=False, encoding="utf-8-sig")

    print("[5/5] 生成洞察与质量检查", flush=True)
    insights = generate_insights(cfg, overview, feature_df, pass2["seg_df"], pass2["flag_df"],
                                 target, pass2["smd_df"], pass2["corr_df"],
                                 lender_summary=pass2["lender_summary_df"],
                                 lender_behavior=pass2["lender_behavior_df"],
                                 lender_products=pass2["lender_product_df"],
                                 income_archetypes=pass2["income_arch_df"],
                                 income_pay_rhythm=pass2["income_rhythm_df"],
                                 dishonour_summary=pass2["dishonour_summary_df"],
                                 dishonour_types=pass2["dishonour_type_df"])
    _write_json(output_dir / "insights.json", insights)

    quality_df = _write_quality(cfg, overview, pass1, feature_df, pass2["dim_df"], pass2["seg_df"],
                                target, read_path is not None, pass2.get("lender_quality"),
                                pass2.get("income_quality"))
    quality_df.to_csv(output_dir / "quality_checks.csv", index=False, encoding="utf-8-sig")

    _write_json(output_dir / "overview.json", overview)
    _write_json(output_dir / "config_used.json", cfg)
    _write_json(output_dir / "run_meta.json", {
        "version": VERSION,
        "generated_at": overview["generated_at"],
        "input_path": str(Path(input_path).resolve()),
        "output_dir": str(output_dir.resolve()),
        "rows": rows_,
        "txn_rows": txn_rows,
        "chunks": pass1["chunk_count"],
        "timing_sec": overview["timing_sec"],
        "config": cfg,
    })

    print(f"分析完成，中间结果已写入: {output_dir}", flush=True)
    return {
        "output_dir": str(output_dir),
        "overview": overview,
        "insights": insights,
        "elapsed": time.time() - t_start,
    }


def load_intermediates(output_dir: str) -> dict:
    out = Path(output_dir)

    def read_json(name):
        p = out / name
        if not p.exists():
            raise FileNotFoundError(f"缺少中间结果 {p}，请先运行 profile_analysis.py")
        return json.loads(p.read_text(encoding="utf-8"))

    def read_csv(name, **kwargs):
        p = out / name
        if not p.exists():
            raise FileNotFoundError(f"缺少中间结果 {p}，请先运行 profile_analysis.py")
        try:
            return pd.read_csv(p, encoding="utf-8-sig", **kwargs)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    return {
        "output_dir": str(out),
        "overview": read_json("overview.json"),
        "insights": read_json("insights.json"),
        "target": read_json("target_overall.json"),
        "run_meta": read_json("run_meta.json"),
        "feature_stats": read_csv("feature_stats.csv"),
        "feature_histograms": read_csv("feature_histograms.csv"),
        "dimension_summary": read_csv("dimension_summary.csv"),
        "dimension_score_hist": read_csv("dimension_score_hist.csv"),
        "segment_summary": read_csv("segment_summary.csv"),
        "segment_profiles": read_csv("segment_profiles.csv"),
        "risk_flags": read_csv("risk_flags.csv"),
        "category_mix": read_csv("category_mix.csv"),
        "correlations": read_csv("correlations.csv"),
        "target_by_segment": read_csv("target_by_segment.csv"),
        "target_by_score_band": read_csv("target_by_score_band.csv"),
        "target_feature_smd": read_csv("target_feature_smd.csv"),
        "quality_checks": read_csv("quality_checks.csv"),
        "lender_summary": read_csv("lender_summary.csv"),
        "lender_institutions": read_csv("lender_institutions.csv"),
        "lender_segments": read_csv("lender_segments.csv"),
        "lender_behavior": read_csv("lender_behavior.csv"),
        "lender_behavior_target": read_csv("lender_behavior_target.csv"),
        "lender_products": read_csv("lender_products.csv"),
        "lender_top_competitors": read_csv("lender_top_competitors.csv"),
        "income_summary": read_csv("income_summary.csv"),
        "income_archetypes": read_csv("income_archetypes.csv"),
        "income_pay_rhythm": read_csv("income_pay_rhythm.csv"),
        "income_segments": read_csv("income_segments.csv"),
        "income_archetype_target": read_csv("income_archetype_target.csv"),
        "dishonour_summary": read_csv("dishonour_summary.csv"),
        "dishonour_recovery": read_csv("dishonour_recovery.csv"),
        "dishonour_segments": read_csv("dishonour_segments.csv"),
        "dishonour_types": read_csv("dishonour_types.csv"),
        "dishonour_recovery_target": read_csv("dishonour_recovery_target.csv"),
        "observation_bands": read_csv("observation_bands.csv") if (out / "observation_bands.csv").exists() else pd.DataFrame(),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="用户画像分析（流式读取宽表特征）")
    p.add_argument("--config", help="JSON 配置文件路径")
    p.add_argument("--input", help="输入特征 CSV")
    p.add_argument("--dict", dest="dict_path", help="dict.csv 数据字典路径")
    p.add_argument("--output-dir", help="中间结果输出目录")
    p.add_argument("--cache-dir", help="子集缓存目录")
    p.add_argument("--chunk-size", type=int, help="分块行数")
    p.add_argument("--reservoir-size", type=int, help="每特征分位抽样上限")
    p.add_argument("--target", dest="target_column", help="目标标签列")
    p.add_argument("--seed", type=int, help="随机种子")
    p.add_argument("--subset-cache", choices=["auto", "on", "off"], help="窄表缓存策略")
    p.add_argument("--no-user-export", action="store_true", help="不导出用户级评分明细")
    p.add_argument("--allow-missing-features", action="store_true", help="缺失特征时跳过而不是报错")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {
        "input_path": args.input,
        "dict_path": args.dict_path,
        "output_dir": args.output_dir,
        "cache_dir": args.cache_dir,
        "chunk_size": args.chunk_size,
        "reservoir_size": args.reservoir_size,
        "target_column": args.target_column,
        "seed": args.seed,
        "subset_cache": args.subset_cache,
        "export_user_scores": False if args.no_user_export else None,
        "allow_missing_features": True if args.allow_missing_features else None,
    }
    try:
        cfg = load_config(args.config, overrides)
        run_analysis(cfg)
    except Exception as exc:
        print(f"分析失败: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
