#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swap In vs Swap Out 原始交易明细画像指标模块。

供 build_raw_deck.py 生成幻灯片。
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from existing_swap_in_profile_report import fmt_money, fmt_pct

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


IN_COLOR = "#1c5cab"
OUT_COLOR = "#d14b45"

SEG_ORDER = [
    "借贷周转型",
    "工薪高杠杆型",
    "工薪低杠杆型",
    "转账过路型",
    "现金流脆弱型",
    "混合/低活跃型",
]


def med(df: pd.DataFrame, col: str) -> float:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan


def rate(df: pd.DataFrame, col: str) -> float:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def summary(df: pd.DataFrame) -> dict:
    return {
        "n": len(df),
        "txn": med(df, "n_txn"),
        "accounts": med(df, "n_accounts"),
        "income": med(df, "income"),
        "wage_ratio": med(df, "wage_ratio"),
        "true_income_ratio": med(df, "true_income_ratio"),
        "transfer_in_ratio": med(df, "transfer_in_ratio"),
        "loan_inflow_ratio": med(df, "loan_inflow_ratio"),
        "repay_rate": med(df, "repay_rate"),
        "consumption_share": med(df, "consumption_share"),
        "ext_transfer_share": med(df, "ext_transfer_share"),
        "internal_transfer_share": med(df, "internal_transfer_share"),
        "debt_share": med(df, "debt_share"),
        "fees_share": med(df, "fees_share"),
        "gambling_share": med(df, "gambling_share"),
        "balance": med(df, "median_balance"),
        "buffer": med(df, "buffer_days"),
        "neg_days": med(df, "neg_days_share"),
        "low_days": med(df, "low_balance_days_share"),
        "dishonour_cnt": med(df, "dishonour_cnt"),
        "dishonour_amt": med(df, "dishonour_amt"),
        "any_wage": rate(df, "any_wage"),
        "any_loan_draw": rate(df, "any_loan_draw"),
        "any_repay": rate(df, "any_repay"),
        "any_gambling": rate(df, "any_gambling"),
        "any_dishonour": rate(df, "any_dishonour"),
        "cash_advance": rate(df, "cash_advance"),
        "bnpl": rate(df, "bnpl"),
        "personal_loan": rate(df, "personal_loan"),
        "bank_loan": rate(df, "bank_loan"),
        "active_lender": med(df, "active_lender_cnt"),
        "pay_cov": rate(df.assign(x=(df["pay_events"] > 0).astype(float)), "x"),
        "pay_events": med(df, "pay_events"),
        "pay_outflow": med(df, "pay_outflow_7d"),
        "pay_repay": med(df, "pay_repay_7d"),
        "pay_ext": med(df, "pay_ext_7d"),
        "pay_recycle": med(df, "pay_recycle_7d"),
        "loan_cov": rate(df.assign(x=(df["loan_events"] > 0).astype(float)), "x"),
        "loan_events": med(df, "loan_events"),
        "loan_recycle": med(df, "loan_recycle_7d"),
        "loan_repay": med(df, "loan_repay_7d"),
        "gambling_high": rate(df.assign(x=((df["gambling_share"] >= 0.05) | (df["gambling"] >= 1000)).astype(float)), "x"),
        "buffer_lt7": rate(df.assign(x=(df["buffer_days"] < 7).astype(float)), "x"),
        "buffer_lt1": rate(df.assign(x=(df["buffer_days"] < 1).astype(float)), "x"),
        "essential_share": med(df, "essential_share"),
        "income_cv": med(df, "income_cv"),
    }


def income_mix(df: pd.DataFrame) -> dict:
    inc = float(df["income"].sum())
    ext_credit = float(df["total_credit"].sum() - df["internal_transfer_in"].sum())
    return {
        "wage": float(df["wage"].sum()),
        "welfare": float(df["welfare"].sum()),
        "other": float(df["other_income"].sum()),
        "loan_in": float(df["loan_in"].sum()),
        "ext_in": float(df["ext_transfer_in"].sum()),
        "income": inc,
        "ext_credit": ext_credit,
        "wage_share": float(df["wage"].sum()) / inc if inc else np.nan,
        "welfare_share": float(df["welfare"].sum()) / inc if inc else np.nan,
        "other_share": float(df["other_income"].sum()) / inc if inc else np.nan,
        "true_ratio": inc / ext_credit if ext_credit else np.nan,
        "transfer_in": float(df["ext_transfer_in"].sum()) / ext_credit if ext_credit else np.nan,
        "loan_in_ratio": float(df["loan_in"].sum()) / ext_credit if ext_credit else np.nan,
    }


def outflow_mix(df: pd.DataFrame) -> dict:
    td = float(df["total_debit"].sum())
    keys = ["consumption", "ext_transfer_out", "internal_transfer_out", "debt_repay", "gambling", "fees", "cash", "other_debit"]
    return {k: (float(df[k].sum()) / td if td else np.nan) for k in keys} | {"total": td}


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def seg(row):
        if row["loan_inflow_ratio"] >= 0.10 and (
            row["repay_rate"] >= 0.20 or row["loan_recycle_7d"] >= 3
        ):
            return "借贷周转型"
        if row["wage_ratio"] >= 0.50 and row["repay_rate"] >= 0.15:
            return "工薪高杠杆型"
        if (
            row["wage_ratio"] < 0.50 or row["true_income_ratio"] < 0.45
        ) and row["ext_transfer_share"] >= 0.25:
            return "转账过路型"
        if row["buffer_days"] < 0.20 or row["neg_days_share"] >= 0.20:
            return "现金流脆弱型"
        if row["wage_ratio"] >= 0.80 and row["repay_rate"] < 0.15:
            return "工薪低杠杆型"
        return "混合/低活跃型"

    df["seg"] = df.apply(seg, axis=1)
    return df


def p(x, digits=1):
    return fmt_pct(x, digits)


def m(x):
    return fmt_money(x)


def n(x, digits=0):
    return "—" if not np.isfinite(x) else f"{x:,.{digits}f}"


def ratio(x):
    return "—" if not np.isfinite(x) else f"{x:.2f}x"


def delta(a, b, kind="pct", digits=1):
    if not (np.isfinite(a) and np.isfinite(b)):
        return '<span class="delta muted">—</span>'
    if kind == "pct":
        diff = (b - a) * 100
        text = f"{abs(diff):.{digits}f}pp"
        flat = abs(diff) < 0.05
    elif kind == "pct_rel":
        diff = (b - a) * 100
        text = f"{abs(diff):.{digits}f}pp"
        flat = abs(diff) < 0.05
    elif kind == "money":
        diff = b - a
        text = fmt_money(abs(diff))
        if a > 0 and abs(diff) / a <= 3:
            text += f"（{abs(diff) / a:.0%}）"
        flat = abs(diff) < 1e-9
    elif kind == "count":
        diff = b - a
        text = f"{abs(diff):,.0f}"
        flat = abs(diff) < 0.5
    elif kind == "ratio":
        diff = b - a
        text = f"{abs(diff):.2f}x"
        flat = abs(diff) < 1e-9
    elif kind == "days":
        diff = b - a
        text = f"{abs(diff):.2f} 天"
        flat = abs(diff) < 0.05
    else:
        diff = b - a
        text = f"{abs(diff):.{digits}f}"
        flat = abs(diff) < 1e-9
    if flat:
        return '<span class="delta muted">基本持平</span>'
    who = "Out" if diff > 0 else "In"
    color = OUT_COLOR if diff > 0 else IN_COLOR
    return (
        f'<span class="delta"><span class="dot" style="background:{color}"></span>'
        f"{who} 高 {text}</span>"
    )


