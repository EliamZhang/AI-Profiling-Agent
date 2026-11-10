#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WageGo 用户画像数据与图表模块。

供 build_profile_decks.py / build_compare_decks.py 生成幻灯片。
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 plotly，请先执行: pip install plotly") from exc


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
SENTINEL = -999000.0          # 金额类缺失哨兵：-999999.99999
LENDER_SENTINEL = -1          # lender 类缺失哨兵
WINDOW_DAYS = 182
MONTHS = 6.0                  # 182 天按 6 个月折算

# 图表色板（dataviz 参考色板，已通过校验）
SURFACE = "#fcfcfb"
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
    "bank_txn_surplus_dishonoursfee_sum_182d",
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

    m["inc_mean"] = float(d["inc_m"].mean())
    m["exp_mean"] = float(d["exp_m"].mean())
    m["sur_mean"] = float(d["sur_m"].mean())
    m["bal_mean"] = float(d["bal"].mean())
    m["rep_mean"] = float(d["rep_m"].mean())
    m["draw_mean"] = float(d["draw_m"].mean())
    m["dishfee_mean"] = float(
        d["bank_txn_surplus_dishonoursfee_sum_182d"].div(MONTHS).mean()
    )
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


def chart_liability(m: dict) -> str:
    raw = [
        ("现金贷 / 工资预支", m["adv_use"], m["adv_amt_share"], m["dish_advance"]),
        ("个人贷款 / 分期贷", m["loan_use"], m["loan_amt_share"], m["dish_personal_loan"]),
        ("BNPL", m["bnpl_use"], m["bnpl_amt_share"], m["dish_bnpl"]),
        ("银行", m["bank_use"], m["bank_amt_share"], np.nan),
    ]
    rows = [
        (name, float(use) * 100, float(amt) * 100, float(dish))
        for name, use, amt, dish in raw
        if np.isfinite(use) and np.isfinite(amt)
    ]
    rows.sort(key=lambda row: row[2])
    labels = [row[0] for row in rows]
    usage = [row[1] for row in rows]
    amounts = [row[2] for row in rows]
    dish_text = [fmt_pct(row[3]) if np.isfinite(row[3]) else "—" for row in rows]

    left = go.Figure()
    for name, use, amt, _ in rows:
        left.add_trace(
            go.Scatter(
                x=[use, amt],
                y=[name, name],
                mode="lines",
                line=dict(color=BASELINE, width=2.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    left.add_trace(
        go.Scatter(
            x=usage,
            y=labels,
            mode="markers+text",
            name="使用率",
            marker=dict(color=SERIES[0], size=10, line=dict(color=SURFACE, width=2)),
            text=[f"{v:.0f}%" for v in usage],
            textposition="bottom center",
            textfont=dict(color=INK2, size=10.5),
            cliponaxis=False,
            customdata=[[t] for t in dish_text],
            hovertemplate=(
                "%{y}<br>使用率 %{x:.1f}%<br>有拒付用户占比 %{customdata[0]}<extra></extra>"
            ),
        )
    )
    left.add_trace(
        go.Scatter(
            x=amounts,
            y=labels,
            mode="markers+text",
            name="放款金额占比",
            marker=dict(color=SERIES[1], size=10, line=dict(color=SURFACE, width=2)),
            text=[f"{v:.0f}%" for v in amounts],
            textposition="top center",
            textfont=dict(color=INK2, size=10.5),
            cliponaxis=False,
            hovertemplate="%{y}<br>放款金额占比 %{x:.1f}%<extra></extra>",
        )
    )
    base_layout(left, height=430, margin=dict(l=156, r=40, t=64, b=64), showlegend=True)
    left.update_layout(
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=-0.16,
            yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    xmax = max(usage + amounts) if rows else 1
    left.update_xaxes(range=[0, xmax * 1.22 if xmax else 1], ticksuffix="%")
    left.update_yaxes(categoryorder="array", categoryarray=labels, showgrid=False)
    top = rows[-1] if rows else None
    add_title(
        left,
        f"借贷产品：{top[0]} 占放款金额 {top[2]:.0f}%" if top else "借贷产品结构",
        xref="paper",
    )

    res_cnt = float(m["dish_resolved_cnt_total"])
    unres_cnt = float(m["dish_unresolved_cnt_total"])
    res_amt = float(m["dish_resolved_amt_total"])
    unres_amt = float(m["dish_unres_amt_total"])
    cnt_total = res_cnt + unres_cnt
    amt_total = res_amt + unres_amt
    res_pct = [
        100 * res_cnt / cnt_total if cnt_total else 0.0,
        100 * res_amt / amt_total if amt_total else 0.0,
    ]
    unres_pct = [100 - value for value in res_pct]

    right = go.Figure()
    right.add_trace(
        go.Bar(
            y=["笔数", "金额"],
            x=res_pct,
            orientation="h",
            name="已解决",
            marker=dict(color=STATUS["good"], line=dict(color=SURFACE, width=1.5)),
            text=[f"已解决<br>{v:.0f}%" if v >= 12 else f"{v:.0f}%" for v in res_pct],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#ffffff", size=11),
            customdata=[[f"{res_cnt:,.0f} 笔"], [fmt_money(res_amt, compact=True)]],
            hovertemplate="%{y}<br>已解决 %{customdata[0]}（%{x:.1f}%）<extra></extra>",
        )
    )
    right.add_trace(
        go.Bar(
            y=["笔数", "金额"],
            x=unres_pct,
            orientation="h",
            name="未解决",
            marker=dict(color=STATUS["critical"], line=dict(color=SURFACE, width=1.5)),
            text=[f"未解决<br>{v:.0f}%" if v >= 12 else f"{v:.0f}%" for v in unres_pct],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#ffffff", size=11),
            customdata=[[f"{unres_cnt:,.0f} 笔"], [fmt_money(unres_amt, compact=True)]],
            hovertemplate="%{y}<br>未解决 %{customdata[0]}（%{x:.1f}%）<extra></extra>",
        )
    )
    base_layout(right, height=430, margin=dict(l=72, r=28, t=64, b=64), showlegend=True)
    right.update_layout(
        barmode="stack",
        bargap=0.45,
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=-0.16,
            yanchor="top",
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    right.update_xaxes(
        range=[0, 100],
        tickvals=[0, 25, 50, 75, 100],
        ticktext=["0", "25%", "50%", "75%", "100%"],
    )
    right.update_yaxes(categoryorder="array", categoryarray=["金额", "笔数"], showgrid=False)
    add_title(right, f"拒付处置：解决率 {fmt_pct(m['dish_resolution_rate'])}")

    return (
        f'<div class="chart-grid even"><div>{fig_html(left)}</div>'
        f"<div>{fig_html(right)}</div></div>"
    )


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
    bin_size = (hi - lo) / 36
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


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


