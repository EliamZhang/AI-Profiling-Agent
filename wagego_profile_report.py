#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WageGo 用户画像报告生成器（精简版）

用法:
    python wagego_profile_report.py
    python wagego_profile_report.py --data data/wagego_sample_with_feature_all.csv --out output

输出:
    output/wagego_profile_report.html   可视化报告（图表内嵌）
    output/charts/*.png                 图表源文件
    output/profile_metrics.csv          关键指标明细
    output/segment_summary.csv          用户分层明细
"""

from __future__ import annotations

import argparse
import html
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 plotly，请先执行: pip install plotly") from exc


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "wagego_sample_with_feature_all.csv"
DEFAULT_OUT = ROOT / "output"

SENTINEL = -999000.0          # 金额类缺失哨兵：-999999.99999
LENDER_SENTINEL = -1          # lender 类缺失哨兵
WINDOW_DAYS = 182
MONTHS = 6.0                  # 182 天按 6 个月折算

# 图表色板（dataviz 参考色板，已通过校验）
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# ---------------------------------------------------------------------------
# 字段清单：只读取报告需要的列
# ---------------------------------------------------------------------------
CONSUMPTION_CATS = {
    "groceries": "超市",
    "rent": "房租",
    "dining_out": "餐饮",
    "transport": "交通",
    "utilities": "公用事业",
    "insurance": "保险",
    "telecommunications": "通信",
    "retail": "零售",
    "department_stores": "百货",
    "travel": "旅游",
    "entertainment": "娱乐",
    "subscription_tv": "订阅电视",
    "gyms_and_other_memberships": "健身/会员",
    "health": "医疗",
    "education": "教育",
    "personal_care": "个人护理",
    "pet_care": "宠物",
    "home_improvement": "家装",
    "automotive": "汽车",
    "gambling": "赌博",
    "fees": "手续费",
    "donations": "捐赠",
}
ESSENTIAL_CATS = ["groceries", "utilities", "rent", "transport", "health"]
DISCRETIONARY_CATS = [
    "dining_out",
    "entertainment",
    "travel",
    "retail",
    "department_stores",
    "personal_care",
    "pet_care",
    "donations",
]

INDUSTRY_NAME = {
    1: "住宿餐饮",
    2: "银行代发",
    3: "清洁/安保/物业",
    4: "建筑",
    5: "教育培训",
    6: "金融保险",
    7: "政府/公共管理",
    8: "医疗与社会援助",
    9: "劳务派遣",
    10: "制造",
    11: "采矿",
    12: "零售",
    13: "社会服务/非营利",
    14: "体育/娱乐/博彩",
    15: "技术服务/信息媒体",
    16: "运输/邮政/仓储",
    17: "未知",
}

# 每个指标只取一个主窗口（182d/l182d）；趋势用 1m_vs_3m、3m_vs_6m 比率，
# 不再读取同一指标的 7/14/28/56/84/168d 版本，避免重复。
META_COLS = [
    "user_id",
    "application_id",
    "sample_datetime",
    "user_type",
    "txn_raw_input_cnt",
    "txn_balance_input_cnt",
    "feature_error",
]
INCOME_COLS = [
    "bank_txn_income_global_cv_182d",
    "bank_txn_income_wages_sum_182d",
    "bank_txn_income_wages_count_182d",
    "bank_txn_income_wages_cv_182d",
    "bank_txn_income_wages_1m_vs_3m",
    "bank_txn_income_wages_3m_vs_6m",
    "bank_txn_income_wages_latest_amount",
    "bank_txn_income_centrelink_sum_182d",
    "bank_txn_income_centrelink_cv_182d",
    "bank_txn_income_otherincome_sum_182d",
    "bank_txn_income_otherincome_category_externaltransfers_sum_182d",
    "bank_txn_income_otherincome_category_internaltransfer_sum_182d",
    "bank_txn_income_otherincome_category_allothercredits_sum_182d",
    "bank_txn_income_otherincome_category_gambling_sum_182d",
    "bank_txn_worthiness_income_source_cnt_182d",
    "bank_txn_worthiness_days_since_last_wage_income_182d",
    "bank_txn_worthiness_wage_income_consecutive_month_cnt_182d",
]
EXPENSE_COLS = [
    "bank_txn_expense_global_count_182d",
    "bank_txn_expense_global_median_182d",
    "bank_txn_expense_global_cv_182d",
    "bank_txn_category_external_transfers_debit_amt_182d",
    "bank_txn_category_internal_transfer_debit_amt_182d",
    "bank_txn_category_global_debit_amt_182d",
    "bank_txn_category_gambling_debit_amt_182d",
    "bank_txn_category_fees_debit_amt_182d",
] + [f"bank_txn_category_{k}_debit_amt_182d" for k in CONSUMPTION_CATS]
SURPLUS_COLS = [
    "bank_txn_surplus_income_sum_182d",
    "bank_txn_surplus_expense_sum_182d",
    "bank_txn_surplus_surplus_sum_182d",
    "bank_txn_surplus_liabilityrepayment_sum_182d",
    "bank_txn_surplus_liabilitydrawdown_sum_182d",
    "bank_txn_surplus_fortnightly_surplus_182d",
    "bank_txn_surplus_surplus_1m_vs_3m",
    "bank_txn_surplus_surplus_3m_vs_6m",
    "bank_txn_surplus_expense_1m_vs_3m",
    "bank_txn_surplus_expense_3m_vs_6m",
    "bank_txn_surplus_liabilitydrawdown_1m_vs_3m",
    "bank_txn_surplus_liabilityrepayment_1m_vs_3m",
]
BALANCE_COLS = [
    "bank_txn_balance_snapshot_last_182d",
    "bank_txn_balance_min_182d",
    "bank_txn_balance_max_182d",
    "bank_txn_balance_sign_change_count_182d",
    "bank_txn_balance_anomaly_days_182d",
    "bank_txn_balance_below_avg_ratio_182d",
    "bank_txn_balance_global_net_burn_per_day_182d",
    "bank_txn_balance_global_savings_rate_182d",
]
LENDER_COLS = [
    "bank_txn_lender_disburse_amount_sum_l182d",
    "bank_txn_lender_repay_amount_sum_l182d",
    "bank_txn_lender_disburse_amount_count_l182d",
    "bank_txn_lender_repay_amount_count_l182d",
    "bank_txn_lender_disburse_amount_avg_l182d",
    "bank_txn_lender_disburse_amount_max_l182d",
    "bank_txn_lender_disburse_advance_count_l182d",
    "bank_txn_lender_disburse_loan_count_l182d",
    "bank_txn_lender_disburse_bnpl_count_l182d",
    "bank_txn_lender_disburse_bank_count_l182d",
    "bank_txn_lender_disburse_advance_amount_sum_l182d",
    "bank_txn_lender_disburse_loan_amount_sum_l182d",
    "bank_txn_lender_disburse_bnpl_amount_sum_l182d",
    "bank_txn_lender_disburse_bank_amount_sum_l182d",
    "bank_txn_worthiness_bank_liability_drawdown_lender_cnt_182d",
]
DISHONOUR_COLS = [
    "bank_txn_dishonour_lender_ratio_182d",
    "bank_txn_dishonour_lender_resolved_cnt_3gp_182d",
    "bank_txn_dishonour_lender_unresolved_cnt_3gp_182d",
    "bank_txn_dishonour_lender_resolved_amt_3gp_182d",
    "bank_txn_dishonour_lender_unresolved_amt_3gp_182d",
    "bank_txn_dishonour_lender_resolved_catchup_cnt_3gp_182d",
    "bank_txn_dishonour_lender_unresolved_institution_cnt_3gp_182d",
    "bank_txn_dishonour_cash_wage_advance_ratio_182d",
    "bank_txn_dishonour_personal_loan_ratio_182d",
    "bank_txn_dishonour_bnpl_ratio_182d",
    "bank_txn_dishonour_bank_ratio_182d",
    "bank_txn_dishonour_debt_collection_ratio_182d",
]
INDUSTRY_COLS = [
    "bank_txn_industry_employer_cnt",
    "bank_txn_industry_pattern_fortnightly",
    "bank_txn_industry_pattern_weekly",
    "bank_txn_industry_pattern_monthly",
    "bank_txn_industry_pattern_irregular",
    "bank_txn_industry_pay_interval_median",
    "bank_txn_industry_pay_interval_std",
    "bank_txn_industry_primary_employer_ratio",
    "bank_txn_industry_primary_industry_code",
]
CATEGORY_COLS = [
    "bank_txn_category_active_days_182d",
    "bank_txn_category_weekend_share_182d",
    "bank_txn_category_inter_txn_gap_p90_182d",
    "bank_txn_category_cluster_fees_overdraft_avg_ticket_182d",
    "bank_txn_category_cluster_gambling_avg_ticket_182d",
    "bank_txn_category_cluster_debt_share_182d",
    "bank_txn_category_cluster_debt_consolidation_share_182d",
    "bank_txn_category_cluster_collections_share_182d",
]
WORTHINESS_COLS = [
    "bank_txn_worthiness_balance_negative_share_avg_acct_182d",
    "bank_txn_worthiness_balance_ge1000_share_avg_acct_182d",
    "bank_txn_worthiness_bank_account_cnt",
    "bank_txn_worthiness_primary_employer_streak_months_182d",
    "bank_txn_worthiness_daily_living_amt_max_182d",
    "bank_txn_worthiness_high_commitment_amt_sum_182d",
    "bank_txn_worthiness_large_expense_amt_max_182d",
    "bank_txn_worthiness_large_expense_cnt_182d",
    "bank_txn_worthiness_days_since_last_liability_drawdown_182d",
    "bank_txn_worthiness_days_since_last_liability_repayment_182d",
]
ALL_COLUMNS = list(
    dict.fromkeys(
        META_COLS
        + INCOME_COLS
        + EXPENSE_COLS
        + SURPLUS_COLS
        + BALANCE_COLS
        + LENDER_COLS
        + DISHONOUR_COLS
        + INDUSTRY_COLS
        + CATEGORY_COLS
        + WORTHINESS_COLS
    )
)
TEXT_COLS = {"user_id", "application_id", "sample_datetime", "user_type", "feature_error"}
COUNT_COLS = {"txn_raw_input_cnt", "txn_balance_input_cnt"}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def fmt_money(x: float, compact: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if compact:
        if x >= 1_000_000:
            return f"{sign}${x / 1_000_000:.1f}M"
        if x >= 1_000:
            return f"{sign}${x / 1_000:.0f}K"
    if 0 < x < 1:
        return f"{sign}${x:.2f}"
    return f"{sign}${x:,.0f}"


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{digits}f}%"


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.div(b.replace(0, np.nan))


def finite(s: pd.Series) -> pd.Series:
    return s.mask(~np.isfinite(s))


def q(s: pd.Series, p: float) -> float:
    return float(s.quantile(p)) if s.notna().any() else np.nan


def neg_share(s: pd.Series) -> float:
    s = s.dropna()
    return float((s < 0).mean()) if len(s) else np.nan


def pos_share(s: pd.Series) -> float:
    s = s.dropna()
    return float((s > 0).mean()) if len(s) else np.nan


def cond_share(cond: pd.Series, s: pd.Series | None = None) -> float:
    if s is None:
        return float(cond.mean())
    ok = s.notna()
    return float((cond & ok).sum() / ok.sum()) if ok.any() else np.nan


def html_table(headers: list[str], rows: list[list[str]], cls: str = "", raw_cols: tuple[int, ...] = ()) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{cell if i in raw_cols else html.escape(str(cell))}</td>"
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------------------
# 数据读取与清洗
# ---------------------------------------------------------------------------
def load_data(path: Path, log=print) -> tuple[pd.DataFrame, list[str], int]:
    if not path.exists():
        raise SystemExit(f"数据文件不存在: {path}")
    t0 = time.time()
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in ALL_COLUMNS if c in header]
    missing = [c for c in ALL_COLUMNS if c not in header]
    log(f"读取数据: {path.name} | 总字段 {len(header):,} | 本次加载 {len(usecols)} 个字段 ...")
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    log(f"读取完成: {len(df):,} 行，用时 {time.time() - t0:.1f}s")
    return df, missing, len(header)


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n_apps = len(df)
    for c in df.columns:
        if c in TEXT_COLS:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if c in COUNT_COLS:
            df[c] = s.fillna(0)
            continue
        s = s.mask(s <= SENTINEL)
        if c.startswith("bank_txn_lender_"):
            s = s.mask(s == LENDER_SENTINEL)
        df[c] = s
    return df, n_apps


def dedupe_users(df: pd.DataFrame) -> pd.DataFrame:
    # 同一 user_id 保留交易笔数最多的申请，避免重复申请稀释画像
    return (
        df.sort_values(["user_id", "txn_raw_input_cnt"], ascending=[True, False])
        .drop_duplicates("user_id", keep="first")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------
def build_metrics(df: pd.DataFrame, n_apps: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    d = df[df["txn_raw_input_cnt"] > 0].copy()
    core = [
        "bank_txn_surplus_income_sum_182d",
        "bank_txn_surplus_expense_sum_182d",
        "bank_txn_surplus_surplus_sum_182d",
    ]
    d = d[d[core].notna().all(axis=1) & (d["bank_txn_surplus_income_sum_182d"] > 0)].copy()
    if d.empty:
        raise SystemExit("没有可用的有效交易样本，请检查数据。")

    d["inc"] = d["bank_txn_surplus_income_sum_182d"]
    d["exp"] = d["bank_txn_surplus_expense_sum_182d"]
    d["sur"] = d["bank_txn_surplus_surplus_sum_182d"]
    d["rep"] = d["bank_txn_surplus_liabilityrepayment_sum_182d"]
    d["draw"] = d["bank_txn_surplus_liabilitydrawdown_sum_182d"]
    d["inc_m"] = d["inc"] / MONTHS
    d["exp_m"] = d["exp"] / MONTHS
    d["sur_m"] = d["sur"] / MONTHS
    d["rep_m"] = d["rep"] / MONTHS
    d["draw_m"] = d["draw"] / MONTHS
    d["pre_debt_cf"] = d["inc"] - d["exp"]
    d["sur_rate"] = finite(safe_div(d["sur"], d["inc"]))
    d["sur_rate_cap"] = d["sur_rate"].clip(-1, 1)
    d["rep_rate"] = finite(safe_div(d["rep"], d["inc"]))
    d["draw_rate"] = finite(safe_div(d["draw"], d["inc"]))
    d["net_borrow"] = d["draw"] - d["rep"]
    d["bal"] = d["bank_txn_balance_snapshot_last_182d"]
    d["buffer_days"] = finite(safe_div(d["bal"], d["exp_m"] / 30))
    d["txn_pm"] = d["bank_txn_expense_global_count_182d"] / MONTHS

    cons_cols = [f"bank_txn_category_{k}_debit_amt_182d" for k in CONSUMPTION_CATS]
    cons_cols = [c for c in cons_cols if c in d.columns]
    d["cons"] = d[cons_cols].sum(axis=1, min_count=1)
    d["cons_m"] = d["cons"] / MONTHS
    d["cons_share_exp"] = finite(safe_div(d["cons"], d["exp"]))
    d["ext_transfer_share"] = finite(
        safe_div(d["bank_txn_category_external_transfers_debit_amt_182d"], d["exp"])
    )
    d["high_commit_share"] = finite(
        safe_div(d["bank_txn_worthiness_high_commitment_amt_sum_182d"], d["inc"])
    )
    cons_share_user = d[cons_cols].div(
        d[cons_cols].sum(axis=1).replace(0, np.nan), axis=0
    )
    d["cat_hhi"] = (cons_share_user ** 2).sum(axis=1, min_count=1)
    d["cat_top3"] = cons_share_user.apply(
        lambda row: row.dropna().nlargest(3).sum() if row.notna().any() else np.nan,
        axis=1,
    )

    def segment_of(x: float) -> str | float:
        if pd.isna(x):
            return np.nan
        if x >= 0.05:
            return "盈余型"
        if x >= -0.05:
            return "平衡型"
        if x >= -0.20:
            return "赤字型"
        return "重压型"

    d["segment"] = d["sur_rate_cap"].map(segment_of)

    wages = d["bank_txn_income_wages_sum_182d"]
    cent = d["bank_txn_income_centrelink_sum_182d"]
    other = d["bank_txn_income_otherincome_sum_182d"]
    total_income = wages.sum() + cent.sum() + other.sum()
    w = wages.fillna(0) > 0
    c = cent.fillna(0) > 0

    cons_total = float(d["cons"].sum()) if d["cons"].notna().any() else np.nan
    cat_totals = {
        k: float(d[f"bank_txn_category_{k}_debit_amt_182d"].sum())
        for k in CONSUMPTION_CATS
        if f"bank_txn_category_{k}_debit_amt_182d" in d.columns
    }
    cat_shares = {k: (v / cons_total if cons_total and np.isfinite(cons_total) else np.nan) for k, v in cat_totals.items()}

    ess_cols = [f"bank_txn_category_{k}_debit_amt_182d" for k in ESSENTIAL_CATS]
    dis_cols = [f"bank_txn_category_{k}_debit_amt_182d" for k in DISCRETIONARY_CATS]
    gamb = d["bank_txn_category_gambling_debit_amt_182d"]
    fees = d["bank_txn_category_fees_debit_amt_182d"]

    emp = d["bank_txn_industry_employer_cnt"].notna()
    ind = d[emp]

    m: dict[str, float | str] = {}
    m["n_apps"] = n_apps
    m["n_users"] = len(df)
    m["n_valid"] = len(d)
    m["coverage"] = len(d) / len(df)
    m["n_no_data"] = len(df) - len(d)
    if "sample_datetime" in df.columns:
        months = sorted(
            pd.to_datetime(df["sample_datetime"], errors="coerce", format="mixed")
            .dt.strftime("%Y-%m")
            .dropna()
            .unique()
            .tolist()
        )
    else:
        months = []
    m["sample_month"] = "、".join(months) if months else "—"
    user_types = sorted(df["user_type"].dropna().astype(str).unique().tolist())
    m["user_types"] = "、".join(user_types) if user_types else "—"
    m["user_type_label"] = "新客" if user_types == ["New"] else m["user_types"]
    m["median_txn_cnt"] = float(d["txn_raw_input_cnt"].median())
    m["median_active_days"] = q(d["bank_txn_category_active_days_182d"], 0.5)
    m["median_txn_pm"] = q(d["txn_pm"], 0.5)

    # 收入
    m["wage_incidence"] = pos_share(wages)
    m["cent_incidence"] = pos_share(cent)
    m["other_incidence"] = pos_share(other)
    m["wage_median_m"] = q(wages[wages > 0], 0.5) / MONTHS
    m["cent_median_m"] = q(cent[cent > 0], 0.5) / MONTHS
    m["other_median_m"] = q(other[other > 0], 0.5) / MONTHS
    m["wage_share_agg"] = wages.sum() / total_income if total_income else np.nan
    m["cent_share_agg"] = cent.sum() / total_income if total_income else np.nan
    m["other_share_agg"] = other.sum() / total_income if total_income else np.nan
    other_parts = {
        "ext": d["bank_txn_income_otherincome_category_externaltransfers_sum_182d"],
        "internal": d["bank_txn_income_otherincome_category_internaltransfer_sum_182d"],
        "credits": d["bank_txn_income_otherincome_category_allothercredits_sum_182d"],
        "gambling": d["bank_txn_income_otherincome_category_gambling_sum_182d"],
    }
    other_total = float(other.sum())
    for key, col in other_parts.items():
        m[f"other_{key}_share"] = float(col.sum() / other_total) if other_total else np.nan
        m[f"other_{key}_inc"] = pos_share(col)
        m[f"other_{key}_med"] = q(col[col > 0], 0.5)
    m["other_known_share"] = (
        float(sum(col.sum() for col in other_parts.values()) / other_total)
        if other_total
        else np.nan
    )
    m["other_residual_share"] = (
        max(0.0, 1.0 - m["other_known_share"])
        if np.isfinite(m["other_known_share"])
        else np.nan
    )
    m["transfer_inc_share"] = (
        float((other_parts["ext"].sum() + other_parts["internal"].sum()) / total_income)
        if total_income
        else np.nan
    )
    m["arch_both"] = float((w & c).mean())
    m["arch_wage_only"] = float((w & ~c).mean())
    m["arch_cent_only"] = float((~w & c).mean())
    m["arch_neither"] = float((~w & ~c).mean())
    m["wage_cv_p50"] = q(d["bank_txn_income_wages_cv_182d"], 0.5)
    m["wage_1m3m_p50"] = q(d["bank_txn_income_wages_1m_vs_3m"], 0.5)
    m["income_cv_p50"] = q(d["bank_txn_income_global_cv_182d"], 0.5)
    m["wage_count_p50"] = q(d["bank_txn_income_wages_count_182d"], 0.5)
    m["wage_count_pm_p50"] = (
        m["wage_count_p50"] / MONTHS if np.isfinite(m["wage_count_p50"]) else np.nan
    )
    m["wage_latest_p50"] = q(d["bank_txn_income_wages_latest_amount"], 0.5)
    m["wage_3m6m_p50"] = q(d["bank_txn_income_wages_3m_vs_6m"], 0.5)
    m["wage_3m6m_decline"] = cond_share(
        d["bank_txn_income_wages_3m_vs_6m"] < 0.9, d["bank_txn_income_wages_3m_vs_6m"]
    )
    m["cent_cv_p50"] = q(
        d.loc[c, "bank_txn_income_centrelink_cv_182d"], 0.5
    )
    m["income_source_p50"] = q(d["bank_txn_worthiness_income_source_cnt_182d"], 0.5)
    m["days_since_wage_p50"] = q(
        d.loc[w, "bank_txn_worthiness_days_since_last_wage_income_182d"], 0.5
    )
    m["wage_consecutive_p50"] = q(
        d.loc[w, "bank_txn_worthiness_wage_income_consecutive_month_cnt_182d"], 0.5
    )
    m["employer_streak_p50"] = q(
        ind["bank_txn_worthiness_primary_employer_streak_months_182d"], 0.5
    )
    m["pay_interval_p50"] = q(ind["bank_txn_industry_pay_interval_median"], 0.5)
    m["pay_interval_std_p50"] = q(ind["bank_txn_industry_pay_interval_std"], 0.5)
    ind_code = d["bank_txn_industry_primary_industry_code"].dropna().astype(int)
    known_code = ind_code[ind_code != 17]
    m["industry_n"] = int(len(known_code))
    m["industry_unknown_share"] = (
        float((ind_code == 17).mean()) if len(ind_code) else np.nan
    )
    ind_top = known_code.value_counts(normalize=True).head(3)
    m["industry_top_text"] = "、".join(
        f"{INDUSTRY_NAME.get(int(k), str(k))} {v:.0%}" for k, v in ind_top.items()
    )
    m["employer_n"] = int(emp.sum())
    m["employer_cnt_p50"] = q(ind["bank_txn_industry_employer_cnt"], 0.5)
    m["primary_ratio_p50"] = q(ind["bank_txn_industry_primary_employer_ratio"], 0.5)
    m["pay_weekly"] = float(ind["bank_txn_industry_pattern_weekly"].mean())
    m["pay_fortnightly"] = float(ind["bank_txn_industry_pattern_fortnightly"].mean())
    m["pay_monthly"] = float(ind["bank_txn_industry_pattern_monthly"].mean())
    m["pay_irregular"] = float(ind["bank_txn_industry_pattern_irregular"].mean())
    m["employer_n"] = int(emp.sum())
    m["inc_p25"] = q(d["inc_m"], 0.25)
    m["inc_p50"] = q(d["inc_m"], 0.5)
    m["inc_p75"] = q(d["inc_m"], 0.75)
    m["exp_p25"] = q(d["exp_m"], 0.25)
    m["exp_p50"] = q(d["exp_m"], 0.5)
    m["exp_p75"] = q(d["exp_m"], 0.75)
    m["cons_p50"] = q(d["cons_m"], 0.5)
    m["cons_share_exp_p50"] = q(d["cons_share_exp"], 0.5)
    m["cons_share_exp_agg"] = float(d["cons"].sum() / d["exp"].sum()) if d["exp"].sum() else np.nan
    m["ext_share_p50"] = q(d["ext_transfer_share"], 0.5)
    m["ext_share_agg"] = (
        float(d["bank_txn_category_external_transfers_debit_amt_182d"].sum() / d["exp"].sum())
        if d["exp"].sum()
        else np.nan
    )
    internal_transfer = d["bank_txn_category_internal_transfer_debit_amt_182d"]
    global_debit = d["bank_txn_category_global_debit_amt_182d"]
    m["ext_transfer_share_global"] = (
        float(d["bank_txn_category_external_transfers_debit_amt_182d"].sum() / global_debit.sum())
        if global_debit.sum()
        else np.nan
    )
    m["internal_transfer_share_global"] = (
        float(internal_transfer.sum() / global_debit.sum()) if global_debit.sum() else np.nan
    )
    m["internal_transfer_inc"] = pos_share(internal_transfer)
    m["ess_share_cons"] = float(d[ess_cols].sum().sum() / cons_total) if cons_total else np.nan
    m["dis_share_cons"] = float(d[dis_cols].sum().sum() / cons_total) if cons_total else np.nan
    m["gamb_incidence"] = pos_share(gamb)
    m["gamb_median_pos"] = q(gamb[gamb > 0], 0.5)
    m["gamb_p90"] = q(gamb, 0.9)
    m["gamb_p99"] = q(gamb, 0.99)
    m["gamb_share_cons"] = float(gamb.sum() / cons_total) if cons_total else np.nan
    m["fees_incidence"] = pos_share(fees)
    m["fees_median_pos"] = q(fees[fees > 0], 0.5)
    m["exp_ticket_p50"] = q(d["bank_txn_expense_global_median_182d"], 0.5)
    m["exp_cv_p50"] = q(d["bank_txn_expense_global_cv_182d"], 0.5)
    m["exp_trend_1m3m_p50"] = q(d["bank_txn_surplus_expense_1m_vs_3m"], 0.5)
    m["exp_trend_3m6m_p50"] = q(d["bank_txn_surplus_expense_3m_vs_6m"], 0.5)
    m["exp_trend_up"] = cond_share(
        d["bank_txn_surplus_expense_1m_vs_3m"] > 1.1,
        d["bank_txn_surplus_expense_1m_vs_3m"],
    )
    m["cat_hhi_p50"] = q(d["cat_hhi"], 0.5)
    m["cat_top3_p50"] = q(d["cat_top3"], 0.5)
    m["weekend_share_p50"] = q(d["bank_txn_category_weekend_share_182d"], 0.5)
    m["txn_gap_p90_p50"] = q(d["bank_txn_category_inter_txn_gap_p90_182d"], 0.5)
    fee_ticket = d["bank_txn_category_cluster_fees_overdraft_avg_ticket_182d"]
    m["fees_ticket_incidence"] = pos_share(fee_ticket)
    m["fees_ticket_p50"] = q(fee_ticket[fee_ticket > 0], 0.5)
    gamb_ticket = d["bank_txn_category_cluster_gambling_avg_ticket_182d"]
    m["gamb_ticket_p50"] = q(gamb_ticket[gamb_ticket > 0], 0.5)
    m["debt_share_p50"] = q(d["bank_txn_category_cluster_debt_share_182d"], 0.5)
    m["debt_consolidation_inc"] = pos_share(
        d["bank_txn_category_cluster_debt_consolidation_share_182d"]
    )
    m["collections_inc"] = pos_share(d["bank_txn_category_cluster_collections_share_182d"])
    m["high_commit_share_p50"] = q(d["high_commit_share"], 0.5)
    m["large_expense_p50"] = q(d["bank_txn_worthiness_large_expense_amt_max_182d"], 0.5)
    m["large_expense_p90"] = q(d["bank_txn_worthiness_large_expense_amt_max_182d"], 0.9)
    large_cnt = d["bank_txn_worthiness_large_expense_cnt_182d"]
    m["large_expense_cnt_p50"] = q(large_cnt, 0.5)
    m["large_expense_cnt_p90"] = q(large_cnt, 0.9)
    m["large_expense_cnt_pm_p50"] = (
        m["large_expense_cnt_p50"] / MONTHS if np.isfinite(m["large_expense_cnt_p50"]) else np.nan
    )
    m["large_expense_cnt_pm_p90"] = (
        m["large_expense_cnt_p90"] / MONTHS if np.isfinite(m["large_expense_cnt_p90"]) else np.nan
    )
    m["daily_living_p50"] = q(d["bank_txn_worthiness_daily_living_amt_max_182d"], 0.5)

    # 资金与负债
    m["pre_debt_cf_p50"] = q(d["pre_debt_cf"] / MONTHS, 0.5)
    m["pre_debt_neg"] = neg_share(d["pre_debt_cf"])
    m["sur_p25"] = q(d["sur_m"], 0.25)
    m["sur_p50"] = q(d["sur_m"], 0.5)
    m["sur_p75"] = q(d["sur_m"], 0.75)
    m["sur_neg"] = neg_share(d["sur"])
    m["sur_rate_p25"] = q(d["sur_rate_cap"], 0.25)
    m["sur_rate_p50"] = q(d["sur_rate_cap"], 0.5)
    m["sur_rate_p75"] = q(d["sur_rate_cap"], 0.75)
    m["fortnight_p50"] = q(d["bank_txn_surplus_fortnightly_surplus_182d"], 0.5)
    m["fortnight_neg"] = neg_share(d["bank_txn_surplus_fortnightly_surplus_182d"])
    m["savings_p50"] = q(d["bank_txn_balance_global_savings_rate_182d"], 0.5)
    m["savings_neg"] = neg_share(d["bank_txn_balance_global_savings_rate_182d"])
    bal = d["bal"]
    m["bal_p10"] = q(bal, 0.10)
    m["bal_p25"] = q(bal, 0.25)
    m["bal_p50"] = q(bal, 0.50)
    m["bal_p75"] = q(bal, 0.75)
    m["bal_p90"] = q(bal, 0.90)
    m["bal_neg"] = neg_share(bal)
    m["bal_lt100"] = cond_share(bal < 100, bal)
    m["bal_lt500"] = cond_share(bal < 500, bal)
    m["bal_n"] = int(bal.notna().sum())
    m["buffer_p50"] = q(d["buffer_days"], 0.5)
    m["buffer_lt7"] = cond_share(d["buffer_days"] < 7, d["buffer_days"])
    m["buffer_lt30"] = cond_share(d["buffer_days"] < 30, d["buffer_days"])
    m["rep_m_p50"] = q(d["rep_m"], 0.5)
    m["rep_rate_p50"] = q(d["rep_rate"], 0.5)
    m["rep_rate_p90"] = q(d["rep_rate"], 0.9)
    m["draw_m_p50"] = q(d["draw_m"], 0.5)
    m["draw_rate_p50"] = q(d["draw_rate"], 0.5)
    days_draw = d["bank_txn_worthiness_days_since_last_liability_drawdown_182d"]
    days_rep = d["bank_txn_worthiness_days_since_last_liability_repayment_182d"]
    m["days_since_drawdown_p50"] = q(days_draw, 0.5)
    m["days_since_drawdown_cov"] = float(days_draw.notna().mean())
    m["days_since_repayment_p50"] = q(days_rep, 0.5)
    m["days_since_repayment_cov"] = float(days_rep.notna().mean())
    m["net_borrow_share"] = cond_share(d["net_borrow"] > 0, d["net_borrow"])
    m["sur_trend_1m3m_p50"] = q(d["bank_txn_surplus_surplus_1m_vs_3m"], 0.5)
    m["sur_trend_3m6m_p50"] = q(d["bank_txn_surplus_surplus_3m_vs_6m"], 0.5)
    draw_trend = d["bank_txn_surplus_liabilitydrawdown_1m_vs_3m"]
    rep_trend = d["bank_txn_surplus_liabilityrepayment_1m_vs_3m"]
    m["draw_trend_p50"] = q(draw_trend[draw_trend > 0], 0.5)
    m["rep_trend_p50"] = q(rep_trend[rep_trend > 0], 0.5)
    m["bal_min_p50"] = q(d["bank_txn_balance_min_182d"], 0.5)
    m["bal_max_p50"] = q(d["bank_txn_balance_max_182d"], 0.5)
    sign_change = d["bank_txn_balance_sign_change_count_182d"]
    m["sign_change_p50"] = q(sign_change, 0.5)
    m["sign_change_p90"] = q(sign_change, 0.9)
    m["sign_change_pm_p50"] = (
        m["sign_change_p50"] / MONTHS if np.isfinite(m["sign_change_p50"]) else np.nan
    )
    m["bal_anomaly_p50"] = q(d["bank_txn_balance_anomaly_days_182d"], 0.5)
    m["bal_below_avg_p50"] = q(d["bank_txn_balance_below_avg_ratio_182d"], 0.5)
    m["net_burn_p50"] = q(d["bank_txn_balance_global_net_burn_per_day_182d"], 0.5)
    m["bal_ge1000_p50"] = q(
        d["bank_txn_worthiness_balance_ge1000_share_avg_acct_182d"], 0.5
    )
    m["neg_balance_days_p50"] = q(
        d["bank_txn_worthiness_balance_negative_share_avg_acct_182d"], 0.5
    )
    acct = d["bank_txn_worthiness_bank_account_cnt"].where(
        d["bank_txn_worthiness_bank_account_cnt"] >= 0
    )
    m["bank_account_p50"] = q(acct, 0.5)

    # lender 字段用 -1 表示“无该类借贷”，清洗后缺失即代表未使用
    m["any_disburse"] = float(d["bank_txn_lender_disburse_amount_sum_l182d"].notna().mean())
    m["any_repay"] = float(d["bank_txn_lender_repay_amount_sum_l182d"].notna().mean())
    m["adv_use"] = float(d["bank_txn_lender_disburse_advance_count_l182d"].notna().mean())
    m["loan_use"] = float(d["bank_txn_lender_disburse_loan_count_l182d"].notna().mean())
    m["bnpl_use"] = float(d["bank_txn_lender_disburse_bnpl_count_l182d"].notna().mean())
    m["bank_use"] = float(d["bank_txn_lender_disburse_bank_count_l182d"].notna().mean())
    disb = d["bank_txn_lender_disburse_amount_sum_l182d"]
    repay = d["bank_txn_lender_repay_amount_sum_l182d"]
    m["disb_count_p50"] = q(d.loc[disb > 0, "bank_txn_lender_disburse_amount_count_l182d"], 0.5)
    m["repay_count_p50"] = q(d.loc[repay > 0, "bank_txn_lender_repay_amount_count_l182d"], 0.5)
    m["disb_ticket_p50"] = q(d.loc[disb > 0, "bank_txn_lender_disburse_amount_avg_l182d"], 0.5)
    m["disb_max_p50"] = q(d.loc[disb > 0, "bank_txn_lender_disburse_amount_max_l182d"], 0.5)
    lcnt = d["bank_txn_worthiness_bank_liability_drawdown_lender_cnt_182d"]
    m["lender_cnt_p50"] = q(lcnt[lcnt > 0], 0.5)
    amount_cols = {
        "adv": "bank_txn_lender_disburse_advance_amount_sum_l182d",
        "loan": "bank_txn_lender_disburse_loan_amount_sum_l182d",
        "bnpl": "bank_txn_lender_disburse_bnpl_amount_sum_l182d",
        "bank": "bank_txn_lender_disburse_bank_amount_sum_l182d",
    }
    disb_total = float(d["bank_txn_lender_disburse_amount_sum_l182d"].sum())
    for key, col in amount_cols.items():
        m[f"{key}_amt_share"] = float(d[col].sum() / disb_total) if disb_total else np.nan
    mix = [("现金贷", m["adv_amt_share"]), ("分期贷", m["loan_amt_share"]),
           ("BNPL", m["bnpl_amt_share"]), ("银行", m["bank_amt_share"])]
    mix = [x for x in mix if np.isfinite(x[1]) and x[1] > 0]
    mix.sort(key=lambda x: x[1], reverse=True)
    m["product_mix_text"] = "、".join(f"{name} {v:.0%}" for name, v in mix)

    m["dish_any"] = cond_share(
        d["bank_txn_dishonour_lender_ratio_182d"] > 0,
        d["bank_txn_dishonour_lender_ratio_182d"],
    )
    m["dish_unres"] = cond_share(
        d["bank_txn_dishonour_lender_unresolved_cnt_3gp_182d"] > 0,
        d["bank_txn_dishonour_lender_unresolved_cnt_3gp_182d"],
    )
    m["dish_unres_amt_total"] = float(
        d["bank_txn_dishonour_lender_unresolved_amt_3gp_182d"].sum()
    )
    unres_income = finite(
        safe_div(d["bank_txn_dishonour_lender_unresolved_amt_3gp_182d"], d["inc"])
    )
    m["dish_unres_income_p50"] = q(unres_income, 0.5)
    m["dish_unres_income_p90"] = q(unres_income, 0.9)
    m["dish_personal_loan"] = cond_share(
        d["bank_txn_dishonour_personal_loan_ratio_182d"] > 0,
        d["bank_txn_dishonour_personal_loan_ratio_182d"],
    )
    m["dish_advance"] = cond_share(
        d["bank_txn_dishonour_cash_wage_advance_ratio_182d"] > 0,
        d["bank_txn_dishonour_cash_wage_advance_ratio_182d"],
    )
    m["dish_bnpl"] = cond_share(
        d["bank_txn_dishonour_bnpl_ratio_182d"] > 0,
        d["bank_txn_dishonour_bnpl_ratio_182d"],
    )
    m["dish_bank"] = cond_share(
        d["bank_txn_dishonour_bank_ratio_182d"] > 0,
        d["bank_txn_dishonour_bank_ratio_182d"],
    )
    m["dish_collection"] = cond_share(
        d["bank_txn_dishonour_debt_collection_ratio_182d"] > 0,
        d["bank_txn_dishonour_debt_collection_ratio_182d"],
    )
    res_cnt = d["bank_txn_dishonour_lender_resolved_cnt_3gp_182d"].fillna(0)
    unres_cnt = d["bank_txn_dishonour_lender_unresolved_cnt_3gp_182d"].fillna(0)
    m["dish_resolved_cnt_total"] = float(res_cnt.sum())
    m["dish_unresolved_cnt_total"] = float(unres_cnt.sum())
    m["dish_resolution_rate"] = (
        float(res_cnt.sum() / (res_cnt.sum() + unres_cnt.sum()))
        if (res_cnt.sum() + unres_cnt.sum())
        else np.nan
    )
    m["dish_resolved_amt_total"] = float(
        d["bank_txn_dishonour_lender_resolved_amt_3gp_182d"].sum()
    )
    m["dish_catchup_inc"] = pos_share(
        d["bank_txn_dishonour_lender_resolved_catchup_cnt_3gp_182d"]
    )
    inst = d["bank_txn_dishonour_lender_unresolved_institution_cnt_3gp_182d"]
    m["dish_institution_p50"] = q(inst[inst > 0], 0.5)

    # 分层
    seg = (
        d.dropna(subset=["segment"])
        .groupby("segment", observed=True)
        .agg(
            n=("user_id", "size"),
            inc_m=("inc_m", "median"),
            exp_m=("exp_m", "median"),
            sur_m=("sur_m", "median"),
            bal=("bal", "median"),
            rep_rate=("rep_rate", "median"),
            adv_use=(
                "bank_txn_lender_disburse_advance_count_l182d",
                lambda s: float(s.notna().mean()),
            ),
            dishon=(
                "bank_txn_dishonour_lender_ratio_182d",
                lambda s: float((s.dropna() > 0).mean()) if s.notna().any() else np.nan,
            ),
        )
        .reindex(["盈余型", "平衡型", "赤字型", "重压型"])
    )
    seg["share"] = seg["n"] / len(d)

    m["cat_shares"] = cat_shares
    m["cat_totals"] = cat_totals
    m["cons_total"] = cons_total
    return d, m, seg


# ---------------------------------------------------------------------------
# 图表（Plotly 交互式）
# ---------------------------------------------------------------------------
CHART_FONT = "Microsoft YaHei, PingFang SC, Noto Sans SC, Arial, sans-serif"
NEUTRAL = "#76746d"
PLOTLY_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def base_layout(
    fig,
    height: int = 430,
    margin: dict | None = None,
    showlegend: bool = False,
) -> None:
    fig.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=CHART_FONT, color=INK2, size=12),
        margin=margin or dict(l=96, r=24, t=56, b=48),
        showlegend=showlegend,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=BASELINE,
            font=dict(family=CHART_FONT, color=INK, size=12),
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=BASELINE,
        linecolor=BASELINE,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=INK2, size=11),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=BASELINE,
        linecolor=BASELINE,
        tickfont=dict(color=INK2, size=12),
        title_font=dict(color=INK2, size=11),
        automargin=True,
    )


def add_title(fig, text: str, xref: str = "x domain", y: float = 1.16) -> None:
    fig.add_annotation(
        x=0,
        y=y,
        xref=xref,
        yref="y domain",
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(family=CHART_FONT, color=INK, size=14),
    )


def fig_html(fig, height: int = 430) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config=PLOTLY_CONFIG,
        default_height=f"{height}px",
    )


def chart_income(m: dict) -> str:
    labels = ["工资", "福利", "其他入账"]
    inc_vals = [
        float(m["wage_incidence"]) * 100,
        float(m["cent_incidence"]) * 100,
        float(m["other_incidence"]) * 100,
    ]
    amt_vals = [
        float(m["wage_share_agg"]) * 100,
        float(m["cent_share_agg"]) * 100,
        float(m["other_share_agg"]) * 100,
    ]

    left = go.Figure()
    left.add_trace(
        go.Bar(
            y=labels,
            x=inc_vals,
            orientation="h",
            name="有该类收入的用户占比",
            marker=dict(color=SERIES[0]),
            text=[f"{v:.0f}%" for v in inc_vals],
            textposition="outside",
            textfont=dict(color=INK2, size=11),
            cliponaxis=False,
            hovertemplate="有该类收入的用户占比 %{x:.1f}%<br>%{y}<extra></extra>",
        )
    )
    left.add_trace(
        go.Bar(
            y=labels,
            x=amt_vals,
            orientation="h",
            name="收入金额占比",
            marker=dict(color=SERIES[1]),
            text=[f"{v:.0f}%" for v in amt_vals],
            textposition="outside",
            textfont=dict(color=INK2, size=11),
            cliponaxis=False,
            hovertemplate="收入金额占比 %{x:.1f}%<br>%{y}<extra></extra>",
        )
    )
    base_layout(left, height=430, margin=dict(l=82, r=18, t=64, b=76), showlegend=True)
    left.update_layout(
        barmode="group",
        bargap=0.32,
        bargroupgap=0.08,
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=-0.18,
            yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    left.update_xaxes(
        range=[0, 112],
        tickvals=[0, 25, 50, 75, 100],
        ticktext=["0", "25%", "50%", "75%", "100%"],
    )
    left.update_yaxes(categoryorder="array", categoryarray=labels, showgrid=False)
    add_title(left, f"收入来源：{fmt_pct(m['other_share_agg'], 0)} 的金额来自其他入账")

    pct = lambda x: float(x) * 100 if np.isfinite(x) else 0.0
    labor = pct(m["wage_share_agg"] + m["cent_share_agg"])
    ext_inc = pct(m["other_ext_share"] * m["other_share_agg"])
    int_inc = pct(m["other_internal_share"] * m["other_share_agg"])
    rest_inc = max(0.0, pct(m["other_share_agg"]) - ext_inc - int_inc)
    ext_dbt = pct(m["ext_transfer_share_global"])
    int_dbt = pct(m["internal_transfer_share_global"])
    rest_dbt = max(0.0, 100.0 - ext_dbt - int_dbt)

    right = go.Figure()
    seg_defs = [
        ("工资+福利", ["入账构成"], [labor], SERIES[0], "#ffffff"),
        ("外部转账", ["入账构成", "借记构成"], [ext_inc, ext_dbt], SERIES[1], "#ffffff"),
        ("内部转账", ["入账构成", "借记构成"], [int_inc, int_dbt], SERIES[3], INK),
        ("其他", ["入账构成", "借记构成"], [rest_inc, rest_dbt], NEUTRAL, "#ffffff"),
    ]
    for name, ys, xs, color, text_color in seg_defs:
        shown = [
            "其他借记" if name == "其他" and y == "借记构成" else name for y in ys
        ]
        right.add_trace(
            go.Bar(
                y=ys,
                x=xs,
                orientation="h",
                name=name,
                marker=dict(color=color, line=dict(color=SURFACE, width=1.5)),
                text=[f"{lab}<br>{v:.0f}%" for lab, v in zip(shown, xs)],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color=text_color, size=11),
                customdata=[[lab, v] for lab, v in zip(shown, xs)],
                hovertemplate="%{y}<br>%{customdata[0]}：%{customdata[1]:.1f}%<extra></extra>",
            )
        )
    base_layout(right, height=430, margin=dict(l=84, r=18, t=64, b=76))
    right.update_layout(barmode="stack", bargap=0.45)
    right.update_xaxes(
        range=[0, 100],
        tickvals=[0, 25, 50, 75, 100],
        ticktext=["0", "25%", "50%", "75%", "100%"],
    )
    right.update_yaxes(
        categoryorder="array",
        categoryarray=["借记构成", "入账构成"],
        showgrid=False,
    )
    add_title(
        right,
        f"转账主导：占入账 {fmt_pct(m['transfer_inc_share'], 0)}、占借记 {fmt_pct(m['ext_transfer_share_global'] + m['internal_transfer_share_global'], 0)}",
    )

    return (
        f'<div class="chart-grid"><div>{fig_html(left)}</div>'
        f"<div>{fig_html(right)}</div></div>"
    )


def chart_consumption(m: dict, top_n: int = 10) -> str:
    items = sorted(m["cat_shares"].items(), key=lambda kv: kv[1])[-top_n:]
    labels = [
        CONSUMPTION_CATS[k] + ("（风险类）" if k == "gambling" else "") for k, _ in items
    ]
    vals = [float(v) * 100 for _, v in items]
    colors = [STATUS["critical"] if k == "gambling" else SERIES[0] for k, _ in items]
    totals = [fmt_money(m["cat_totals"][k], compact=True) for k, _ in items]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=vals,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(color=INK2, size=11),
            cliponaxis=False,
            customdata=[[t] for t in totals],
            hovertemplate="%{y}<br>占可识别消费 %{x:.1f}%<br>6 个月总额 %{customdata[0]}<extra></extra>",
        )
    )
    base_layout(fig, height=430, margin=dict(l=124, r=34, t=64, b=48))
    fig.update_xaxes(
        range=[0, max(vals) * 1.22 if vals else 1],
        ticksuffix="%",
        title="占可识别消费比例",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=labels, showgrid=False)
    top3 = "、".join(CONSUMPTION_CATS[k] for k, _ in items[-3:][::-1])
    add_title(
        fig,
        f"可识别消费结构：{top3}居前（仅占总支出 {fmt_pct(m['cons_share_exp_agg'])}）",
    )
    return fig_html(fig)


def chart_health(d: pd.DataFrame, m: dict) -> str:
    sur = d["sur_m"].dropna()
    lo = q(sur, 0.01)
    hi = q(sur, 0.99)
    lo = float(np.floor(lo / 500) * 500) if np.isfinite(lo) else -4000.0
    hi = float(np.ceil(hi / 500) * 500) if np.isfinite(hi) else 2000.0
    if not hi > lo:
        lo, hi = -4000.0, 2000.0
    vals = sur[(sur >= lo) & (sur <= hi)]
    neg = vals[vals < 0]
    pos = vals[vals >= 0]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[1.35, 1.0],
        horizontal_spacing=0.12,
    )
    bin_size = (hi - lo) / 60
    fig.add_trace(
        go.Histogram(
            x=neg,
            xbins=dict(start=lo, end=hi, size=bin_size),
            name="负盈余",
            marker=dict(color=STATUS["critical"], line=dict(width=0.5, color=SURFACE)),
            hovertemplate="月度盈余 %{x:$,.0f}<br>用户数 %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=pos,
            xbins=dict(start=lo, end=hi, size=bin_size),
            name="正盈余",
            marker=dict(color=SERIES[0], line=dict(width=0.5, color=SURFACE)),
            hovertemplate="月度盈余 %{x:$,.0f}<br>用户数 %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    med = m["sur_p50"]
    fig.add_vline(x=0, line_color=BASELINE, line_width=1, row=1, col=1)
    fig.add_vline(x=med, line_color=INK2, line_width=1.4, row=1, col=1)
    right_side = med >= (lo + hi) / 2
    fig.add_annotation(
        x=med,
        y=1,
        xref="x",
        yref="y domain",
        text=f"中位 {fmt_money(med)}",
        showarrow=False,
        xanchor="right" if right_side else "left",
        yanchor="bottom",
        font=dict(color=INK2, size=11),
    )
    fig.add_annotation(
        x=lo + (hi - lo) * 0.02,
        y=0.55,
        xref="x",
        yref="y domain",
        text=f"{fmt_pct(m['sur_neg'])}<br>为负",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(color=STATUS["critical"], size=12),
    )

    bal = d["bal"]
    bucket_labels = ["透支 (<$0)", "$0–100", "$100–500", "$500–1k", "≥$1k"]
    bucket_masks = [
        bal < 0,
        (bal >= 0) & (bal < 100),
        (bal >= 100) & (bal < 500),
        (bal >= 500) & (bal < 1000),
        bal >= 1000,
    ]
    bucket_vals = [cond_share(mask, bal) * 100 for mask in bucket_masks]
    fig.add_trace(
        go.Bar(
            y=bucket_labels,
            x=bucket_vals,
            orientation="h",
            marker=dict(color=SEQ_BLUE[::-1]),
            text=[f"{v:.1f}%" for v in bucket_vals],
            textposition="outside",
            textfont=dict(color=INK2, size=11),
            cliponaxis=False,
            hovertemplate="%{y}<br>用户占比 %{x:.1f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )

    base_layout(fig, height=430, margin=dict(l=110, r=28, t=78, b=52), showlegend=True)
    fig.update_layout(
        barmode="overlay",
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1.02,
            yanchor="bottom",
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        range=[lo, hi],
        tickformat="$,.0f",
        title="月度盈余（显示 P1–P99，剔除两端各 1%）",
        row=1,
        col=1,
    )
    fig.update_yaxes(title="用户数", row=1, col=1)
    fig.update_xaxes(
        range=[0, max(bucket_vals) * 1.25 if bucket_vals else 1],
        ticksuffix="%",
        title="用户占比",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=bucket_labels,
        autorange="reversed",
        showgrid=False,
        row=1,
        col=2,
    )
    add_title(fig, f"月度盈余分布：{fmt_pct(m['sur_neg'])} 为负", xref="x domain")
    add_title(fig, f"账户余额分布：中位 {fmt_money(m['bal_p50'])}", xref="x2 domain")
    return fig_html(fig)


def chart_segments(seg: pd.DataFrame) -> str:
    colors = {
        "盈余型": STATUS["good"],
        "平衡型": STATUS["warning"],
        "赤字型": STATUS["serious"],
        "重压型": STATUS["critical"],
    }
    fig = go.Figure()
    for name, row in seg.iterrows():
        share = row["share"]
        if not np.isfinite(share) or share <= 0:
            continue
        v = float(share) * 100
        col = colors.get(name, SERIES[0])
        fig.add_trace(
            go.Bar(
                y=["用户分层"],
                x=[v],
                orientation="h",
                name=name,
                marker=dict(color=col, line=dict(color=SURFACE, width=1.5)),
                text=[f"{name}<br>{v:.0f}%"] if v >= 7 else [f"{v:.0f}%"],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(
                    color=INK if col == STATUS["warning"] else "#ffffff",
                    size=11,
                ),
                customdata=[
                    [
                        fmt_pct(share),
                        fmt_money(row["inc_m"]),
                        fmt_money(row["exp_m"]),
                        fmt_money(row["sur_m"]),
                        fmt_money(row["bal"]),
                        fmt_pct(row["rep_rate"]),
                    ]
                ],
                hovertemplate=(
                    "<b>%{name}</b><br>占比 %{customdata[0]}<br>收入中位 %{customdata[1]}"
                    "<br>支出中位 %{customdata[2]}<br>盈余中位 %{customdata[3]}"
                    "<br>余额中位 %{customdata[4]}<br>还款率中位 %{customdata[5]}<extra></extra>"
                ),
            )
        )
    base_layout(fig, height=240, margin=dict(l=12, r=12, t=12, b=58), showlegend=True)
    fig.update_layout(
        barmode="stack",
        bargap=0.35,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.10,
            yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(range=[0, 100], visible=False)
    fig.update_yaxes(visible=False)
    return fig_html(fig, 240)


# ---------------------------------------------------------------------------
# HTML 报告
# ---------------------------------------------------------------------------
CSS = """
:root {
  --surface: #fcfcfb;
  --page: #f9f9f7;
  --ink: #0b0b0b;
  --ink2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11,11,11,.10);
  --blue: #2a78d6;
  --orange: #eb6834;
  --good: #0ca30c;
  --warning: #fab219;
  --serious: #ec835a;
  --critical: #d03b3b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
  font-size: 14px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1120px; margin: 0 auto; padding: 34px 24px 64px; }
.eyebrow { color: var(--muted); font-size: 11px; letter-spacing: .14em; font-weight: 650; text-transform: uppercase; }
h1 { font-size: 30px; line-height: 1.25; margin: 8px 0 6px; letter-spacing: -.02em; }
.subtitle { color: var(--ink2); font-size: 14px; margin: 0 0 16px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 26px; }
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  border: 1px solid var(--border); background: var(--surface);
  border-radius: 999px; padding: 5px 11px; font-size: 12px; color: var(--ink2);
}
.chip b { color: var(--ink); font-weight: 650; }
.kpi-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;
}
.kpi {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 15px 16px 13px;
}
.kpi .label { color: var(--ink2); font-size: 12px; margin-bottom: 3px; }
.kpi .value { font-size: 26px; font-weight: 680; letter-spacing: -.02em; line-height: 1.15; }
.kpi .sub { color: var(--muted); font-size: 11.5px; margin-top: 3px; }
.section { margin-top: 34px; }
h2 {
  font-size: 18px; margin: 0 0 6px; letter-spacing: -.01em;
  display: flex; align-items: center; gap: 10px;
}
h2 .num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 24px; height: 24px; border-radius: 50%;
  background: var(--ink); color: #fff; font-size: 12px; font-weight: 650;
}
.lede { color: var(--ink2); margin: 0 0 14px; font-size: 13.5px; }
.goal { color: var(--muted); font-size: 12.5px; margin: -1px 0 10px; }
.goal b { color: var(--ink2); font-weight: 600; }
h3 {
  font-size: 13px; font-weight: 650; color: var(--ink);
  margin: 18px 0 4px; padding-left: 9px; border-left: 3px solid var(--blue);
  line-height: 1.4;
}
.chart {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 10px 12px 8px; margin: 4px 0 12px;
}
.chart-grid { display: grid; grid-template-columns: 1.3fr 1fr; gap: 10px; align-items: start; }
.notes { margin: 0; padding-left: 18px; }
.notes li { margin: 4px 0; color: var(--ink2); }
.notes b { color: var(--ink); }
.two-col { display: grid; grid-template-columns: 1.05fr .95fr; gap: 24px; align-items: start; }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11.5px; padding: 7px 8px; border-bottom: 1px solid var(--baseline); white-space: nowrap; }
td { padding: 7px 8px; border-bottom: 1px solid var(--grid); color: var(--ink2); }
td:first-child { color: var(--ink); font-weight: 550; }
tr:last-child td { border-bottom: none; }
.table-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 6px 10px; overflow-x: auto; }
.chip-status { display: inline-flex; align-items: center; gap: 6px; color: var(--ink); font-weight: 550; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }
.findings { display: grid; gap: 10px; }
.finding {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 13px 16px; display: grid; grid-template-columns: 28px 1fr; gap: 10px;
}
.finding .n { color: var(--muted); font-weight: 700; font-size: 13px; }
.finding p { margin: 0; color: var(--ink2); }
.finding b { color: var(--ink); }
details { margin-top: 8px; }
summary { cursor: pointer; color: var(--muted); font-size: 12px; user-select: none; }
summary:hover { color: var(--ink2); }
details table { margin-top: 8px; }
.footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--baseline); color: var(--muted); font-size: 11.5px; }
.footer p { margin: 4px 0; }
.note { color: var(--muted); font-size: 11.5px; margin: 6px 0 0; }
@media (max-width: 760px) { .two-col { grid-template-columns: 1fr; } .chart-grid { grid-template-columns: 1fr; } h1 { font-size: 25px; } }
@media print { .wrap { max-width: none; } .chart, .kpi, .finding { break-inside: avoid; } }
"""


def status_chip(level: str, label: str) -> str:
    colors = {"good": "var(--good)", "warning": "var(--warning)", "serious": "var(--serious)", "critical": "var(--critical)"}
    return (
        f'<span class="chip-status"><span class="dot" style="background:{colors.get(level, "var(--muted)")}"></span>{html.escape(label)}</span>'
    )


def build_html(
    m: dict,
    seg: pd.DataFrame,
    figs: dict[str, str],
    missing: list[str],
    n_fields: int,
    generated_at: str,
) -> str:
    plotly_js = get_plotlyjs()
    # 图表数据表
    income_rows = [
        ["工资", fmt_pct(m["wage_incidence"]), fmt_pct(m["wage_share_agg"]), fmt_money(m["wage_median_m"])],
        ["福利", fmt_pct(m["cent_incidence"]), fmt_pct(m["cent_share_agg"]), fmt_money(m["cent_median_m"])],
        ["其他入账", fmt_pct(m["other_incidence"]), fmt_pct(m["other_share_agg"]), fmt_money(m["other_median_m"])],
    ]
    other_rows = [
        ["外部转账", fmt_pct(m["other_ext_share"]), fmt_pct(m["other_ext_inc"]), fmt_money(m["other_ext_med"])],
        ["内部转账", fmt_pct(m["other_internal_share"]), fmt_pct(m["other_internal_inc"]), fmt_money(m["other_internal_med"])],
        ["其他信用入账", fmt_pct(m["other_credits_share"]), fmt_pct(m["other_credits_inc"]), fmt_money(m["other_credits_med"])],
        ["赌博入账", fmt_pct(m["other_gambling_share"]), fmt_pct(m["other_gambling_inc"]), fmt_money(m["other_gambling_med"])],
        ["其余未列示", fmt_pct(m["other_residual_share"]), "—", "—"],
    ]
    cat_items = sorted(m["cat_shares"].items(), key=lambda kv: kv[1], reverse=True)
    top5_text = "、".join(
        f"{CONSUMPTION_CATS[k]} {fmt_pct(v)}" for k, v in cat_items[:5]
    )
    cons_rows = [
        [
            CONSUMPTION_CATS[k],
            fmt_pct(v),
            fmt_money(m["cat_totals"][k] / m["n_valid"]),
        ]
        for k, v in cat_items
    ]
    balance_rows = [[lab, fmt_pct(v)] for lab, v in m.get("balance_buckets", [])]

    def num(x, digits: int = 2) -> str:
        return f"{x:.{digits}f}" if isinstance(x, (int, float, np.floating)) and np.isfinite(x) else "—"

    liability_rows = [
        ["现金贷 / 工资预支", fmt_pct(m["adv_use"]), fmt_pct(m["adv_amt_share"]), fmt_pct(m["dish_advance"])],
        ["个人贷款 / 分期贷", fmt_pct(m["loan_use"]), fmt_pct(m["loan_amt_share"]), fmt_pct(m["dish_personal_loan"])],
        ["BNPL", fmt_pct(m["bnpl_use"]), fmt_pct(m["bnpl_amt_share"]), fmt_pct(m["dish_bnpl"])],
        ["银行", fmt_pct(m["bank_use"]), fmt_pct(m["bank_amt_share"]), "—"],
    ]

    seg_rows = []
    seg_meta = {
        "盈余型": ("good", "盈余型"),
        "平衡型": ("warning", "平衡型"),
        "赤字型": ("serious", "赤字型"),
        "重压型": ("critical", "重压型"),
    }
    for name, row in seg.iterrows():
        level, label = seg_meta.get(name, ("warning", name))
        seg_rows.append(
            [
                status_chip(level, label),
                fmt_pct(row["share"]),
                fmt_money(row["inc_m"]),
                fmt_money(row["exp_m"]),
                fmt_money(row["sur_m"]),
                fmt_money(row["bal"]),
                fmt_pct(row["rep_rate"]),
                fmt_pct(row["adv_use"]),
                fmt_pct(row["dishon"]),
            ]
        )

    def seg_val(name: str, col: str) -> float:
        return float(seg.loc[name, col]) if name in seg.index else np.nan

    findings = [
        f"<b>收支缺口是常态，支出端为主、偿债进一步扩大缺口。</b> {fmt_pct(m['pre_debt_neg'])} 的用户在偿债前已收不抵支，计入还本付息后升至 {fmt_pct(m['sur_neg'])}，中位月盈余 {fmt_money(m['sur_p50'])}。重压型收入中位 {fmt_money(seg_val('重压型', 'inc_m'))}，高于平衡型的 {fmt_money(seg_val('平衡型', 'inc_m'))}，缺口并非单纯收入不足。",
        f"<b>工资覆盖广，但入账金额高度依赖其他来源。</b> {fmt_pct(m['wage_incidence'])} 有工资收入，工资中位 {fmt_money(m['wage_median_m'])}/月，但工资仅占入账金额 {fmt_pct(m['wage_share_agg'])}，“其他入账”占 {fmt_pct(m['other_share_agg'])}，其中外部+内部转账合计占分项收入 {fmt_pct(m['transfer_inc_share'])}。总入账 CV 中位 {num(m['income_cv_p50'])}，收入来源数中位 {num(m['income_source_p50'], 0)} 个；{fmt_pct(m['wage_3m6m_decline'])} 的用户近 3 月工资较近 6 月下降超 10%。",
        f"<b>流动性缓冲接近于零。</b> 中位账户余额仅 {fmt_money(m['bal_p50'])}，只能覆盖 {m['buffer_p50']:.1f} 天支出；{fmt_pct(m['buffer_lt7'])} 的用户缓冲不足 7 天，小额意外支出即可直接转为透支；6 个月内余额正负切换中位 {num(m['sign_change_p50'], 0)} 次，反复进出透支。",
        f"<b>支出刚性强，可压缩空间有限。</b> 按全部借记流水口径，外部转账占 {fmt_pct(m['ext_transfer_share_global'])}、内部转账占 {fmt_pct(m['internal_transfer_share_global'])}；债务相关支出占 {fmt_pct(m['debt_share_p50'])}，高承诺支出占收入中位 {fmt_pct(m['high_commit_share_p50'])}；可识别消费中位仅占总支出 {fmt_pct(m['cons_share_exp_p50'])}，支出端可压缩余地不大。",
        f"<b>借贷以短期现金贷为主，借款扩张快于还款。</b> 放款金额结构为 {m['product_mix_text'] or '—'}；近 1 月/近 3 月新增借款比中位 {num(m['draw_trend_p50'])}，高于还款比的 {num(m['rep_trend_p50'])}，{fmt_pct(m['net_borrow_share'])} 为净借款人，债务滚动特征明显。",
        f"<b>拒付处置能力弱，风险可能继续累积。</b> {fmt_pct(m['dish_unres'])} 的用户存在未解决拒付，未解决金额合计 {fmt_money(m['dish_unres_amt_total'], compact=True)}，未解决金额/收入 P90 达 {fmt_pct(m['dish_unres_income_p90'])}，解决率仅 {fmt_pct(m['dish_resolution_rate'])}；未解决机构数中位 {num(m['dish_institution_p50'], 0)} 家，处置渠道集中但进展有限。",
        f"<b>赌博与手续费渗透广，行为风险不可忽视。</b> {fmt_pct(m['gamb_incidence'])} 有赌博交易，占可识别消费 {fmt_pct(m['gamb_share_cons'])}，P99 单笔达 {fmt_money(m['gamb_p99'])}；{fmt_pct(m['fees_incidence'])} 有手续费支出，手续费/透支类交易发生率达 {fmt_pct(m['fees_ticket_incidence'])}。",
        f"<b>人群分化明显，尾部压力集中。</b> 重压型占 {fmt_pct(seg_val('重压型', 'share'))}，月支出中位 {fmt_money(seg_val('重压型', 'exp_m'))} 高于收入 {fmt_money(seg_val('重压型', 'inc_m'))}，现金贷使用率 {fmt_pct(seg_val('重压型', 'adv_use'))}、拒付率 {fmt_pct(seg_val('重压型', 'dishon'))}；盈余型仅占 {fmt_pct(seg_val('盈余型', 'share'))}，整体均值掩盖了尾部风险。",
    ]

    missing_note = ""
    if missing:
        missing_note = (
            f"缺失字段（已跳过）：{html.escape('、'.join(missing[:8]))}"
            f"{'…' if len(missing) > 8 else ''}。"
        )

    kpis = [
        ("中位月收入", fmt_money(m["inc_p50"]), f"P25 {fmt_money(m['inc_p25'])} · P75 {fmt_money(m['inc_p75'])}"),
        ("中位月支出", fmt_money(m["exp_p50"]), f"P25 {fmt_money(m['exp_p25'])} · P75 {fmt_money(m['exp_p75'])}"),
        ("中位月盈余", fmt_money(m["sur_p50"]), f"近1月/近3月 {num(m['sur_trend_1m3m_p50'])}"),
        ("负盈余用户", fmt_pct(m["sur_neg"]), f"偿债前 {fmt_pct(m['pre_debt_neg'])}"),
        ("中位账户余额", fmt_money(m["bal_p50"]), f"低于均值天数 {fmt_pct(m['bal_below_avg_p50'])}"),
        ("余额不足 7 天支出", fmt_pct(m["buffer_lt7"]), f"中位 {m['buffer_p50']:.1f} 天"),
        ("有借贷放款", fmt_pct(m["any_disburse"]), f"净借款人 {fmt_pct(m['net_borrow_share'])}"),
        ("有拒付记录", fmt_pct(m["dish_any"]), f"未解决金额 {fmt_money(m['dish_unres_amt_total'], compact=True)}"),
        ("有赌博交易", fmt_pct(m["gamb_incidence"]), f"占可识别消费 {fmt_pct(m['gamb_share_cons'])} · 手续费/透支 {fmt_pct(m['fees_ticket_incidence'])}"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'
        for label, value, sub in kpis
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WageGo 用户画像报告（精简版）</title>
<script>{plotly_js}</script>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">WAGEGO · USER PROFILE</div>
  <h1>WageGo 用户画像报告（精简版）</h1>
  <p class="subtitle">基于银行流水特征 · {html.escape(str(m['sample_month']))} {html.escape(str(m['user_type_label']))}样本 · 观察窗口 {WINDOW_DAYS} 天</p>
  <div class="chips">
    <span class="chip">申请 <b>{m['n_apps']:,}</b></span>
    <span class="chip">去重用户 <b>{m['n_users']:,}</b></span>
    <span class="chip">有效流水 <b>{m['n_valid']:,}</b>（{fmt_pct(m['coverage'])}）</span>
    <span class="chip">无有效流水 <b>{m['n_no_data']:,}</b>（{fmt_pct(m['n_no_data']/m['n_users'])}）</span>
    <span class="chip">生成时间 <b>{html.escape(generated_at)}</b></span>
  </div>
  <p class="note">图表可交互：悬停查看明细，右上角工具栏支持缩放、平移与下载。</p>

  <div class="kpi-grid">{kpi_html}</div>

  <section class="section">
    <h2><span class="num">1</span>收入与就业</h2>
    <p class="goal"><b>分析目标</b>：判断还款来源是否充足、稳定、可持续。</p>
    <p class="lede">工资覆盖面广，但入账金额主要由“其他入账”贡献，且其中转账类占比较高；整体入账波动大、来源分散，工资趋势总体平稳，就业信息可识别度有限。</p>
    <div class="chart">{figs['income']}<p class="note">n = {m['n_valid']:,}；入账构成以分项收入（工资+福利+其他入账）为分母，借记构成以全部借记流水为分母；工资与福利合并展示。</p></div>
    <h3>收入水平与来源</h3>
    <ul class="notes">
      <li><b>{fmt_pct(m['wage_incidence'])}</b> 有工资收入，工资中位 <b>{fmt_money(m['wage_median_m'])}/月</b>，但工资仅占入账金额 <b>{fmt_pct(m['wage_share_agg'])}</b>；<b>{fmt_pct(m['cent_incidence'])}</b> 有福利收入，中位 <b>{fmt_money(m['cent_median_m'])}/月</b>；“其他入账”占全部入账 <b>{fmt_pct(m['other_share_agg'])}</b>，是入账金额的主要来源。</li>
      <li>“其他入账”中，外部转账占 <b>{fmt_pct(m['other_ext_share'])}</b>、内部转账 <b>{fmt_pct(m['other_internal_share'])}</b>、其他信用入账 <b>{fmt_pct(m['other_credits_share'])}</b>、赌博入账 <b>{fmt_pct(m['other_gambling_share'])}</b>；外部+内部转账合计占分项收入 <b>{fmt_pct(m['transfer_inc_share'])}</b>，相当部分入账是转账流水而非稳定劳动收入。</li>
      <li>收入组合：仅工资 <b>{fmt_pct(m['arch_wage_only'])}</b>、工资+福利 <b>{fmt_pct(m['arch_both'])}</b>、仅福利 <b>{fmt_pct(m['arch_cent_only'])}</b>、无工资/福利 <b>{fmt_pct(m['arch_neither'])}</b>；收入来源数中位 <b>{num(m['income_source_p50'], 0)} 个</b>。</li>
    </ul>
    <h3>收入稳定性与趋势</h3>
    <ul class="notes">
      <li>总入账 CV 中位 <b>{num(m['income_cv_p50'])}</b>；有工资用户工资 CV <b>{num(m['wage_cv_p50'])}</b>，有福利用户福利 CV <b>{num(m['cent_cv_p50'])}</b>，入账波动整体偏高。</li>
      <li>工资近 1 月/近 3 月比中位 <b>{num(m['wage_1m3m_p50'])}</b>、近 3 月/近 6 月 <b>{num(m['wage_3m6m_p50'])}</b>，<b>{fmt_pct(m['wage_3m6m_decline'])}</b> 下降超 10%；最近一笔工资中位 <b>{fmt_money(m['wage_latest_p50'])}</b>。</li>
      <li>工资到账 <b>{num(m['wage_count_pm_p50'], 1)} 笔/月</b>，距最近工资中位 <b>{num(m['days_since_wage_p50'], 0)} 天</b>，工资连续月数中位 <b>{num(m['wage_consecutive_p50'], 0)} 个月</b>，收入时效性尚可。</li>
    </ul>
    <h3>就业形态与行业</h3>
    <ul class="notes">
      <li>有雇主识别的 {m['employer_n']:,} 人中，周薪/双周薪分别占 <b>{fmt_pct(m['pay_weekly'])}</b>/<b>{fmt_pct(m['pay_fortnightly'])}</b>；发薪间隔中位 <b>{num(m['pay_interval_p50'], 0)} 天</b>（标准差 {num(m['pay_interval_std_p50'])}）。</li>
      <li>中位雇主数 <b>{num(m['employer_cnt_p50'], 0)} 家</b>，主雇主收入占比中位 <b>{fmt_pct(m['primary_ratio_p50'])}</b>，主雇主连续月数中位 <b>{num(m['employer_streak_p50'], 0)} 个月</b>，就业关系总体集中。</li>
      <li>已识别行业 <b>{m['industry_n']:,} 人</b>，主行业为 <b>{m['industry_top_text'] or '—'}</b>；<b>{fmt_pct(m['industry_unknown_share'])}</b> 无法识别行业。</li>
    </ul>
    <details>
      <summary>查看图表数据</summary>
      {html_table(["收入来源", "有该类收入用户占比", "收入金额占比", "中位月收入（有该类收入）"], income_rows)}
      <p class="note">金额占比为样本加总口径；“其他入账”含转账、退款等非工资性入账。</p>
      <p class="note" style="margin-top:12px"><b>其他入账拆分</b>（四类合计覆盖 {fmt_pct(m['other_known_share'])}，其余为未列示类别）</p>
      {html_table(["其他入账构成", "占其他入账", "用户发生率", "有该类用户 6 个月中位金额"], other_rows)}
      <p class="note">占比以“其他入账”为分母；外部+内部转账合计占分项收入 {fmt_pct(m['transfer_inc_share'])}。内部转账为自有账户间划转，不应视为真实收入。</p>
    </details>
  </section>

  <section class="section">
    <h2><span class="num">2</span>支出与消费</h2>
    <p class="goal"><b>分析目标</b>：判断资金流向、消费刚性与可压缩空间。</p>
    <p class="lede">支出中真正占用资金的是外部/内部转账与刚性负担，可识别消费占比有限、类目分散，支出端可压缩空间有限。</p>
    <div class="chart">{figs['consumption']}<p class="note">n = {m['n_valid']:,}；占比以可识别消费为分母，可识别消费仅覆盖总支出 {fmt_pct(m['cons_share_exp_agg'])}。</p></div>
    <h3>支出规模与形态</h3>
    <ul class="notes">
      <li>月支出中位 <b>{fmt_money(m['exp_p50'])}</b>，单笔中位 <b>{fmt_money(m['exp_ticket_p50'])}</b>，单笔金额 CV 中位 <b>{num(m['exp_cv_p50'])}</b>，支出呈小额、高波动特征。</li>
      <li>可识别消费中位 <b>{fmt_money(m['cons_p50'])}/月</b>，用户层面中位占总支出 <b>{fmt_pct(m['cons_share_exp_p50'])}</b>；外部转账占全部支出 <b>{fmt_pct(m['ext_share_agg'])}</b>（用户层面中位 {fmt_pct(m['ext_share_p50'])}）；按全部借记流水口径，外部转账占 <b>{fmt_pct(m['ext_transfer_share_global'])}</b>、内部转账占 <b>{fmt_pct(m['internal_transfer_share_global'])}</b>（发生率 {fmt_pct(m['internal_transfer_inc'])}），转账是资金流出的主要通道。</li>
      <li>支出趋势：近 1 月/近 3 月中位 <b>{num(m['exp_trend_1m3m_p50'])}</b>、近 3 月/近 6 月 <b>{num(m['exp_trend_3m6m_p50'])}</b>，<b>{fmt_pct(m['exp_trend_up'])}</b> 近 1 月支出较近 3 月上升超 10%。</li>
    </ul>
    <h3>消费结构与集中度</h3>
    <ul class="notes">
      <li>主要消费类目：{top5_text}。</li>
      <li>必需品（超市/公用/房租/交通/医疗）占可识别消费 <b>{fmt_pct(m['ess_share_cons'])}</b>，可选消费占 <b>{fmt_pct(m['dis_share_cons'])}</b>；前 3 类占 <b>{fmt_pct(m['cat_top3_p50'])}</b>（HHI {num(m['cat_hhi_p50'])}），集中度中等。</li>
    </ul>
    <h3>刚性支出与固定负担</h3>
    <ul class="notes">
      <li>债务相关支出占 <b>{fmt_pct(m['debt_share_p50'])}</b>，高承诺支出占收入中位 <b>{fmt_pct(m['high_commit_share_p50'])}</b>；固定负担绝对水平不高，但叠加转账后压缩了可识别消费空间。</li>
    </ul>
    <details>
      <summary>查看图表数据（全部消费类别）</summary>
      {html_table(["消费类别", "占可识别消费", "人均 6 个月支出"], cons_rows)}
      <p class="note">人均 = 类别支出加总 / 全部有效用户数（含无该类别消费的用户）。</p>
    </details>
  </section>

  <section class="section">
    <h2><span class="num">3</span>负债与信用履约</h2>
    <p class="goal"><b>分析目标</b>：判断借贷依赖、偿债压力、债务滚动与违约处置。</p>
    <p class="lede">借贷使用面广，短期现金贷是主要资金来源；新增借款增速高于还款，债务滚动特征明显。拒付规模大、解决率低，是负债端最突出的风险点。</p>
    <h3>借贷使用与产品结构</h3>
    <ul class="notes">
      <li>有放款 <b>{fmt_pct(m['any_disburse'])}</b>、有还款 <b>{fmt_pct(m['any_repay'])}</b>；放款笔数中位 <b>{num(m['disb_count_p50'], 0)} 笔</b>、还款笔数中位 <b>{num(m['repay_count_p50'], 0)} 笔</b>，平均单笔 <b>{fmt_money(m['disb_ticket_p50'])}</b>、最大单笔中位 <b>{fmt_money(m['disb_max_p50'])}</b>，涉及 <b>{num(m['lender_cnt_p50'], 0)} 家</b> 放款机构。</li>
    </ul>
    <div class="table-card">
      {html_table(["借贷产品", "使用率", "放款金额占比", "有拒付用户占比"], liability_rows)}
    </div>
    <p class="note">使用率与拒付占比均以有效用户为分母；金额占比以全部放款金额为分母，其余为未单列产品。</p>
    <h3>还款压力与借款滚动</h3>
    <ul class="notes">
      <li>月还款中位 <b>{fmt_money(m['rep_m_p50'])}</b>，偿债/收入中位 <b>{fmt_pct(m['rep_rate_p50'])}</b>、P90 <b>{fmt_pct(m['rep_rate_p90'])}</b>。</li>
      <li>月新增借款中位 <b>{fmt_money(m['draw_m_p50'])}</b>，近 1 月/近 3 月借款比中位 <b>{num(m['draw_trend_p50'])}</b>，高于还款比的 <b>{num(m['rep_trend_p50'])}</b>；<b>{fmt_pct(m['net_borrow_share'])}</b> 为净借款人。</li>
      <li>有对应交易用户中，最近一笔借款距中位 <b>{num(m['days_since_drawdown_p50'], 0)} 天</b>（覆盖 {fmt_pct(m['days_since_drawdown_cov'])}），最近一笔还款距中位 <b>{num(m['days_since_repayment_p50'], 0)} 天</b>（覆盖 {fmt_pct(m['days_since_repayment_cov'])}），借款/还款发生频繁，债务处于持续滚动状态。</li>
    </ul>
    <h3>信用履约与债务处置</h3>
    <ul class="notes">
      <li><b>{fmt_pct(m['dish_any'])}</b> 有拒付记录，<b>{fmt_pct(m['dish_unres'])}</b> 未解决；已解决 <b>{m['dish_resolved_cnt_total']:,.0f}</b> 笔、未解决 <b>{m['dish_unresolved_cnt_total']:,.0f}</b> 笔，解决率 <b>{fmt_pct(m['dish_resolution_rate'])}</b>，追补发生率 <b>{fmt_pct(m['dish_catchup_inc'])}</b>。</li>
      <li>已解决金额 <b>{fmt_money(m['dish_resolved_amt_total'], compact=True)}</b>、未解决金额 <b>{fmt_money(m['dish_unres_amt_total'], compact=True)}</b>；未解决金额/收入 P90 达 <b>{fmt_pct(m['dish_unres_income_p90'])}</b>；未解决拒付涉及机构数中位 <b>{num(m['dish_institution_p50'], 0)} 家</b>，处置渠道集中但进展有限。</li>
      <li><b>{fmt_pct(m['collections_inc'])}</b> 有催收/坏账类交易，<b>{fmt_pct(m['debt_consolidation_inc'])}</b> 有债务整合，债务处置已从拒付延伸到催收与整合环节。</li>
    </ul>
  </section>

  <section class="section">
    <h2><span class="num">4</span>现金流与流动性</h2>
    <p class="goal"><b>分析目标</b>：判断收支是否平衡、账户能否缓冲冲击。</p>
    <p class="lede">偿债前已有大量用户收不抵支，计入还本付息后缺口进一步扩大；盈余边际略有改善，但账户缓冲极薄，收入中断或意外支出都可能直接转为透支。</p>
    <div class="chart">{figs['health']}<p class="note">n = {m['n_valid']:,}；盈余直方图显示 P1–P99 范围（剔除两端各 1%），余额取最近快照。</p></div>
    <h3>收支结余</h3>
    <ul class="notes">
      <li>偿债前现金流中位 <b>{fmt_money(m['pre_debt_cf_p50'])}/月</b>，<b>{fmt_pct(m['pre_debt_neg'])}</b> 为负；偿债后盈余中位 <b>{fmt_money(m['sur_p50'])}/月</b>，<b>{fmt_pct(m['sur_neg'])}</b> 为负。</li>
      <li>盈余近 1 月/近 3 月比中位 <b>{num(m['sur_trend_1m3m_p50'])}</b>、近 3 月/近 6 月 <b>{num(m['sur_trend_3m6m_p50'])}</b>，边际略有改善但未扭转缺口。</li>
      <li>双周盈余中位 <b>{fmt_money(m['fortnight_p50'])}</b>，<b>{fmt_pct(m['fortnight_neg'])}</b> 为负；账户口径储蓄率中位 <b>{fmt_pct(m['savings_p50'])}</b>，<b>{fmt_pct(m['savings_neg'])}</b> 为负。</li>
    </ul>
    <h3>账户余额与流动性</h3>
    <ul class="notes">
      <li>账户余额中位 <b>{fmt_money(m['bal_p50'])}</b>，P25 <b>{fmt_money(m['bal_p25'])}</b>；<b>{fmt_pct(m['bal_lt100'])}</b> 低于 $100，<b>{fmt_pct(m['bal_neg'])}</b> 透支，余额为负的账户天数占比中位 <b>{fmt_pct(m['neg_balance_days_p50'])}</b>；6 个月内余额正负切换中位 <b>{num(m['sign_change_p50'], 0)} 次</b>（约 {num(m['sign_change_pm_p50'], 1)} 次/月），反复进出透支。</li>
      <li>区间最低值中位 <b>{fmt_money(m['bal_min_p50'])}</b>、最高值中位 <b>{fmt_money(m['bal_max_p50'])}</b>；余额低于自身均值的天数占比中位 <b>{fmt_pct(m['bal_below_avg_p50'])}</b>，异常天数中位 <b>{num(m['bal_anomaly_p50'], 0)} 天</b>，净消耗中位 <b>{fmt_money(m['net_burn_p50'])}/天</b>。</li>
      <li>余额 ≥$1,000 的账户天数占比中位仅 <b>{fmt_pct(m['bal_ge1000_p50'])}</b>；中位余额仅覆盖 <b>{m['buffer_p50']:.1f} 天</b>支出，<b>{fmt_pct(m['buffer_lt7'])}</b> 不足 7 天，流动性缓冲极薄。中位银行账户数 <b>{num(m['bank_account_p50'], 0)} 个</b>。</li>
    </ul>
    <details>
      <summary>查看图表数据</summary>
      {html_table(["余额区间", "用户占比"], balance_rows)}
      <p class="note">余额有效用户 {m['bal_n']:,} 人；中位余额可覆盖支出天数 = 账户余额 /（月支出 / 30）。</p>
    </details>
  </section>

  <section class="section">
    <h2><span class="num">5</span>交易行为与风险信号</h2>
    <p class="goal"><b>分析目标</b>：识别交易节奏、异常交易与高风险行为。</p>
    <p class="lede">交易频率高、节奏集中，大额支出与赌博、手续费/透支等行为构成传统收支指标之外的风险信号。</p>
    <h3>交易节奏</h3>
    <ul class="notes">
      <li>月交易笔数中位 <b>{m['median_txn_pm']:.0f}</b> 笔；周末消费占比中位 <b>{fmt_pct(m['weekend_share_p50'])}</b>，交易间隔 P90 中位 <b>{num(m['txn_gap_p90_p50'], 0)} 天</b>，交易频率较高。</li>
    </ul>
    <h3>异常与大额交易</h3>
    <ul class="notes">
      <li>最大单笔支出中位 <b>{fmt_money(m['large_expense_p50'])}</b>、P90 <b>{fmt_money(m['large_expense_p90'])}</b>；大额支出中位 <b>{num(m['large_expense_cnt_pm_p50'], 1)} 笔/月</b>，P90 <b>{num(m['large_expense_cnt_pm_p90'], 1)} 笔/月</b>；日常消费最大单笔中位 <b>{fmt_money(m['daily_living_p50'])}</b>，大额支出对月度现金流的扰动不容忽视。</li>
    </ul>
    <h3>高风险行为</h3>
    <ul class="notes">
      <li>赌博：<b>{fmt_pct(m['gamb_incidence'])}</b> 有交易，占可识别消费 <b>{fmt_pct(m['gamb_share_cons'])}</b>；有记录用户 6 个月中位支出 <b>{fmt_money(m['gamb_median_pos'])}</b>，单笔中位 <b>{fmt_money(m['gamb_ticket_p50'])}</b>，P99 <b>{fmt_money(m['gamb_p99'])}</b>。</li>
      <li>手续费/透支：<b>{fmt_pct(m['fees_incidence'])}</b> 有手续费支出，有支出用户中位 <b>{fmt_money(m['fees_median_pos'])}</b>；手续费/透支类交易发生率 <b>{fmt_pct(m['fees_ticket_incidence'])}</b>，单笔中位 <b>{fmt_money(m['fees_ticket_p50'])}</b>。</li>
    </ul>
  </section>

  <section class="section">
    <h2><span class="num">6</span>用户分层</h2>
    <p class="lede">按偿债后盈余率分为四层：盈余型 ≥5%、平衡型 -5%~5%、赤字型 -20%~-5%、重压型 &lt;-20%。</p>
    <div class="chart">{figs['segments']}<p class="note">n = {m['n_valid']:,}；分层按偿债后盈余率划分，阈值见本章说明；悬停可查看各层收入、支出、余额与还款率。</p></div>
    <div class="table-card">
      {html_table(
        ["分层", "占比", "月收入", "月支出", "月盈余", "中位余额", "偿债/收入", "现金贷使用率", "拒付率"],
        seg_rows,
        raw_cols=(0,),
      )}
    </div>
    <p class="note">盈余率已截尾至 [-100%, +100%]；完整分层指标见 segment_summary.csv。</p>
  </section>

  <section class="section">
    <h2><span class="num">7</span>主要发现</h2>
    <div class="findings">
      {''.join(f'<div class="finding"><div class="n">{i+1:02d}</div><p>{text}</p></div>' for i, text in enumerate(findings))}
    </div>
  </section>

  <div class="footer">
    <p><b>口径说明</b>：样本为 {html.escape(str(m['sample_month']))} 的{html.escape(str(m['user_type_label']))}样本；同一 user_id 保留交易笔数最多的申请。有效样本要求 txn_raw_input_cnt &gt; 0，且收入/支出/盈余非缺失、收入 &gt; 0，共 {m['n_valid']:,} 人（{fmt_pct(m['coverage'])}）。观察窗口 {WINDOW_DAYS} 天，月度指标按 6 个月折算。</p>
    <p><b>缺失处理</b>：金额类字段的 −999,999.99999 与 lender 类字段的 −1 视为缺失；未做插补。无有效流水用户（{m['n_no_data']:,} 人）未纳入画像指标。</p>
    <p><b>指标定义</b>：盈余 = 收入 − 支出 − 还本付息 − 拒付费 + 新增借款；可识别消费 = {len(CONSUMPTION_CATS)} 个消费类目的 debit 金额之和；拒付率 = 拒付笔数 /（拒付笔数 + 还款笔数），解决率 = 已解决笔数 /（已解决 + 未解决）；CV = 交易级金额变异系数；HHI = 各消费类目份额平方和；高承诺支出 = 房租/保险/教育/医疗/健身类支出加总；赌博占比 = 赌博类支出 / 可识别消费；手续费/透支类交易发生率 = 有该类交易的用户占比；催收/债务整合 = 有对应类目交易的用户占比。分位数均为各自分布独立计算，不可直接相减。</p>
    <p><b>新增指标口径</b>：其他入账拆分以“其他入账”为分母；转账类入账占比以分项收入（工资+福利+其他入账）为分母；内部转账为自有账户间划转，不应视为真实收入。支出端外部/内部转账占比均以全部借记流水（global debit）为分母，故不在“总支出”口径下直接相加。最近借款/还款距今天数仅对有对应交易的用户计算；未解决拒付/收入 = 未解决拒付金额 / 收入。余额正负切换次数为 182 天内余额跨越零值的次数；大额支出笔数按 6 个月折算为月均。</p>
    <p><b>数据源</b>：wagego_sample_with_feature_all.csv（共 {n_fields:,} 个字段，本次加载 {len(ALL_COLUMNS) - len(missing)} 个；月份由 sample_datetime 派生）；每类指标只保留 182d / l182d 主窗口，趋势使用 1m_vs_3m 与 3m_vs_6m 比率，未加载 7/14/28/56/84/168d 等重复窗口。变量口径沿用同一套交易特征定义。{missing_note}</p>
    <p>本报告为描述性画像，不构成信用评估结论。生成时间：{html.escape(generated_at)}</p>
  </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def export_metrics(m: dict, seg: pd.DataFrame, out: Path) -> None:
    rows = []
    for k, v in m.items():
        if isinstance(v, (dict, list)):
            continue
        rows.append({"metric": k, "value": v})
    pd.DataFrame(rows).to_csv(out / "profile_metrics.csv", index=False, encoding="utf-8-sig")
    seg.reset_index().to_csv(out / "segment_summary.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 WageGo 用户画像报告（精简版）")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="样本 CSV 路径")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    parser.add_argument("--top-n", type=int, default=10, help="消费结构图展示的类别数")
    args = parser.parse_args()

    data_path = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df_raw, missing, n_fields = load_data(data_path)
    df_raw, n_apps = clean_frame(df_raw)
    df = dedupe_users(df_raw)
    d, m, seg = build_metrics(df, n_apps)

    # 供 HTML 余额分桶表使用
    bal = d["bal"]
    bucket_labels = ["透支 (<$0)", "$0–100", "$100–500", "$500–1k", "≥$1k"]
    bucket_masks = [
        bal < 0,
        (bal >= 0) & (bal < 100),
        (bal >= 100) & (bal < 500),
        (bal >= 500) & (bal < 1000),
        bal >= 1000,
    ]
    m["balance_buckets"] = [(lab, cond_share(mask, bal)) for lab, mask in zip(bucket_labels, bucket_masks)]

    print("生成图表 ...")
    figs = {
        "income": chart_income(m),
        "consumption": chart_consumption(m, top_n=args.top_n),
        "health": chart_health(d, m),
        "segments": chart_segments(seg),
    }

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_html = build_html(m, seg, figs, missing, n_fields, generated_at)
    report_path = out / "wagego_profile_report.html"
    report_path.write_text(report_html, encoding="utf-8")

    export_metrics(m, seg, out)

    print("\n" + "=" * 64)
    print("WageGo 用户画像报告已生成")
    print("=" * 64)
    print(f"样本: {m['n_apps']:,} 份申请 -> {m['n_users']:,} 名用户; 有效流水 {m['n_valid']:,} 人 ({fmt_pct(m['coverage'])})")
    print(f"收入: 月收入中位 {fmt_money(m['inc_p50'])} | 月支出中位 {fmt_money(m['exp_p50'])} | 月盈余中位 {fmt_money(m['sur_p50'])}")
    print(f"风险: 负盈余 {fmt_pct(m['sur_neg'])} | 中位余额 {fmt_money(m['bal_p50'])} | 有借贷 {fmt_pct(m['any_disburse'])} | 有拒付 {fmt_pct(m['dish_any'])}")
    print(f"\n报告: {report_path}")
    print("图表: 已内嵌 Plotly（支持悬停、缩放、下载）")
    print(f"指标: {out / 'profile_metrics.csv'}")
    print(f"分层: {out / 'segment_summary.csv'}")


if __name__ == "__main__":
    main()
