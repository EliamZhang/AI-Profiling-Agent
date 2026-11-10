#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swap In vs Swap Out 原始交易明细分析（指标构建）

用法:
    python swap_in_out_analysis.py

输出:
    output/swap_in_out_user_metrics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DATA = {
    "Swap In": ROOT / "data" / "swap_in.csv",
    "Swap Out": ROOT / "data" / "swap_out.csv",
}
OUT = ROOT / "output" / "swap_in_out_user_metrics.csv"
WINDOW_DAYS = 182

# 结构型类目：不进入消费口径
STRUCTURAL_CATS = {
    "external transfers",
    "internal transfer",
    "non sacc loans",
    "sacc loans",
    "wages",
    "centrelink",
    "all other credits",
    "dishonours",
    "fees",
    "overdrawn",
    "debt collection",
    "credit card repayments",
    "information",
}
ESSENTIAL_CATS = {
    "groceries",
    "rent",
    "utilities",
    "transport",
    "health",
    "insurance",
    "education",
    "telecommunications",
    "automotive",
}
LOAN_TP = (
    "afterpay|zip pay|cba steppay|beforepay|mypaynow|credit corp loan|"
    "nab now pay later|wagepay|wagetap|press pay|fundo loans|cash converters|"
    "swoosh finance|moneyspot|generic loan|paypal pay in 4|credit corp collection|"
    "american express credit card repayment|cba loans"
)
GAMBLING_TP = (
    "sportsbet|dabble|tab|picklebet|betr|betfair|casino|bet365|"
    "ladbrokes|pointsbet|neds|unibet"
)


def lower(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower()


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat = lower(df["category"])
    trx = lower(df["trx_type"])
    tp = lower(df["third_party"])
    text = lower(df["text"])
    debit = df["dr_cr"].str.lower().eq("debit")
    credit = df["dr_cr"].str.lower().eq("credit")

    internal = (
        cat.eq("internal transfer")
        | trx.eq("internal transfer")
        | tp.str.contains("internal transfer|internal funds transfer", regex=True)
        | text.str.contains(r"transfer (?:to|from) xx\d+", regex=True)
    )
    external = cat.eq("external transfers") & ~internal
    loan_cat = cat.isin(["non sacc loans", "sacc loans"])
    loan_tp = tp.str.contains(LOAN_TP, regex=True)
    loan_deposit = trx.str.contains("loan deposit", regex=True)
    wage = (
        cat.eq("wages")
        | trx.str.startswith("wage payments")
        | text.str.contains(r"\bpayroll\b|\bsalary\b|\bwage\b", regex=True)
    )
    welfare = cat.eq("centrelink") | trx.eq("welfare payments")
    gambling = (
        cat.eq("gambling")
        | tp.str.contains(GAMBLING_TP, regex=True)
        | text.str.contains(GAMBLING_TP, regex=True)
    )
    dishonour = cat.eq("dishonours") | trx.str.contains(
        "dishonour|direct debit reversal", regex=True
    )
    fee = cat.isin(["fees", "overdrawn"]) | trx.str.contains(
        "fee|interest charge", regex=True
    )
    debt_other = (
        cat.eq("debt collection")
        | cat.eq("credit card repayments")
        | tp.str.contains("credit corp collection|credit card repayment", regex=True)
    )
    atm = trx.str.contains("atm", regex=True)
    refund = trx.str.contains("refund", regex=True)
    other_credit = cat.eq("all other credits") | trx.isin(
        ["credit", "deposit", "earned interest"]
    )
    consumption = ~cat.isin(STRUCTURAL_CATS | {"gambling"}) & cat.ne("")

    cls = pd.Series("other", index=df.index, dtype="object")
    # credits
    cls[credit & internal] = "internal_credit"
    cls[credit & (loan_cat | loan_tp | loan_deposit)] = "loan_in"
    cls[credit & wage] = "wage"
    cls[credit & welfare] = "welfare"
    cls[credit & refund] = "refund"
    cls[credit & external & ~wage & ~welfare & ~(loan_cat | loan_tp | loan_deposit)] = "ext_transfer_in"
    cls[credit & other_credit & ~wage & ~welfare & ~(loan_cat | loan_tp | loan_deposit) & ~external & ~internal] = "other_income"
    cls[credit & (cls == "other")] = "other_credit"

    # debits
    cls[debit & internal] = "internal_debit"
    cls[debit & gambling] = "gambling"
    cls[debit & (loan_cat | loan_tp | debt_other)] = "debt_repay"
    cls[debit & dishonour] = "dishonour"
    cls[debit & fee] = "fees"
    cls[debit & external] = "ext_transfer_out"
    cls[debit & atm] = "cash"
    cls[debit & consumption] = "consumption"
    cls[debit & (cls == "other")] = "other_debit"
    df["cls"] = cls
    df["is_essential"] = debit & consumption & cat.isin(ESSENTIAL_CATS)
    df["is_gambling"] = debit & gambling
    df["is_dishonour"] = dishonour
    df["is_loan_in"] = credit & (loan_cat | loan_tp | loan_deposit)
    df["is_debt_repay"] = debit & (loan_cat | loan_tp | debt_other)
    df["is_wage"] = credit & wage
    df["is_ext_transfer_out"] = debit & external
    df["is_consumption"] = debit & consumption
    return df


def load_group(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["sample_datetime"] = pd.to_datetime(df["sample_datetime"], errors="coerce")
    df = df.drop_duplicates("transaction_id")
    df = df[df["transaction_date"].notna()].copy()
    start = df["sample_datetime"] - pd.Timedelta(days=WINDOW_DAYS)
    df = df[(df["transaction_date"] >= start) & (df["transaction_date"] <= df["sample_datetime"])].copy()
    df = classify(df)
    df["group"] = name
    return df


def safe_div(a, b):
    return a / b if b not in (0, None) and np.isfinite(b) else np.nan


def user_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, g in df.groupby("user_id", sort=False):
        g = g.sort_values(["transaction_date", "transaction_id"]).reset_index(drop=True)
        amount = g["amount"].astype(float)
        credit = g["dr_cr"].str.lower().eq("credit")
        debit = ~credit
        total_credit = amount[credit].sum()
        total_debit = amount[debit].sum()

        def s(mask):
            return float(amount[mask & debit].sum())

        def c(mask):
            return float(amount[mask & credit].sum())

        wage = c(g["is_wage"])
        welfare = c(g["cls"].eq("welfare") & credit)
        other_income = c(g["cls"].eq("other_income") & credit)
        loan_in = c(g["is_loan_in"])
        ext_in = c(g["cls"].eq("ext_transfer_in") & credit)
        internal_in = c(g["cls"].eq("internal_credit") & credit)
        refund = c(g["cls"].eq("refund") & credit)
        income = wage + welfare + other_income
        ext_credit = total_credit - internal_in

        consumption = s(g["is_consumption"])
        essential = s(g["is_essential"])
        gambling = s(g["is_gambling"])
        debt_repay = s(g["is_debt_repay"])
        ext_out = s(g["is_ext_transfer_out"])
        internal_out = s(g["cls"].eq("internal_debit"))
        fees = s(g["cls"].eq("fees"))
        cash = s(g["cls"].eq("cash"))
        other_debit = s(g["cls"].eq("other_debit"))
        dishonour_cnt = int(g["is_dishonour"].sum())
        dishonour_debit_cnt = int((g["is_dishonour"] & debit).sum())
        dishonour_credit_cnt = int((g["is_dishonour"] & credit).sum())
        dishonour_amt = float(amount[g["is_dishonour"]].sum())

        # monthly income volatility
        inc_rows = g[g["cls"].isin(["wage", "welfare", "other_income"]) & credit].copy()
        inc_rows["month"] = inc_rows["transaction_date"].dt.to_period("M")
        monthly_inc = inc_rows.groupby("month")["amount"].sum()
        income_cv = (
            float(monthly_inc.std() / monthly_inc.mean())
            if len(monthly_inc) >= 2 and monthly_inc.mean() > 0
            else np.nan
        )

        # loan product / lender structure
        loan_tp = lower(g["third_party"])
        loan_cat = lower(g["category"])
        loan_rows = g["is_loan_in"] | g["is_debt_repay"]
        active_lender_cnt = int(g.loc[loan_rows, "third_party"].nunique())
        cash_advance = int(bool((
            loan_rows
            & (
                loan_cat.isin(["sacc loans", "non sacc loans"])
                | loan_tp.str.contains(
                    "beforepay|mypaynow|wagepay|wagetap|press pay|cash converters|swoosh|moneyspot|generic loan",
                    regex=True,
                )
            )
        ).any()))
        bnpl = int(bool((
            loan_rows
            & loan_tp.str.contains(
                "afterpay|zip pay|cba steppay|paypal pay in 4|nab now pay later",
                regex=True,
            )
        ).any()))
        personal_loan = int(bool((
            loan_rows & loan_tp.str.contains("credit corp loan|fundo loans", regex=True)
        ).any()))
        bank_loan = int(bool((
            loan_rows & loan_tp.str.contains("cba loans|bank loan", regex=True)
        ).any()))

        # daily balance: last transaction per account/day, then sum across accounts
        gg = g.dropna(subset=["balance"]).copy()
        gg["date"] = gg["transaction_date"].dt.normalize()
        last = (
            gg.sort_values(["date", "transaction_id"])
            .drop_duplicates(["bank_account_id", "date"], keep="last")
        )
        daily = last.groupby("date")["balance"].sum().sort_index()
        median_bal = float(daily.median()) if len(daily) else np.nan
        min_bal = float(daily.min()) if len(daily) else np.nan
        neg_days = float((daily < 0).mean()) if len(daily) else np.nan
        low_days = float((daily < 100).mean()) if len(daily) else np.nan
        n_days = int(g["transaction_date"].dt.normalize().nunique())
        monthly_debit = total_debit / (n_days / 30) if n_days else np.nan
        buffer_days = median_bal / (monthly_debit / 30) if monthly_debit and np.isfinite(monthly_debit) else np.nan

        # payroll D+7 windows
        wage_events = g[g["is_wage"] & (g["amount"] > 0)]
        pay_ratios = []
        for _, ev in wage_events.iterrows():
            t = ev["transaction_date"]
            if (g["transaction_date"].max() - t).days < 7:
                continue
            win = g[(g["transaction_date"] > t) & (g["transaction_date"] <= t + pd.Timedelta(days=7))]
            wamt = float(ev["amount"])
            if wamt <= 0 or win.empty:
                continue
            win_debit = win[~win["dr_cr"].str.lower().eq("credit")]
            outflow = win_debit.loc[~win_debit["cls"].eq("internal_debit"), "amount"].sum()
            pay_ratios.append({
                "outflow": float(outflow) / wamt,
                "repay": float(win.loc[win["is_debt_repay"], "amount"].sum()) / wamt,
                "ext": float(win.loc[win["is_ext_transfer_out"], "amount"].sum()) / wamt,
                "gamb": float(win.loc[win["is_gambling"], "amount"].sum()) / wamt,
                "cons": float(win.loc[win["is_consumption"], "amount"].sum()) / wamt,
            })
        pay = pd.DataFrame(pay_ratios)

        # loan recycle D+7 windows
        loan_events = g[g["is_loan_in"] & (g["amount"] > 0)]
        loan_ratios = []
        for _, ev in loan_events.iterrows():
            t = ev["transaction_date"]
            if (g["transaction_date"].max() - t).days < 7:
                continue
            win = g[(g["transaction_date"] > t) & (g["transaction_date"] <= t + pd.Timedelta(days=7))]
            lamt = float(ev["amount"])
            if lamt <= 0 or win.empty:
                continue
            win_debit = win[~win["dr_cr"].str.lower().eq("credit")]
            outflow = win_debit.loc[~win_debit["cls"].eq("internal_debit"), "amount"].sum()
            loan_ratios.append({
                "outflow": float(outflow) / lamt,
                "repay": float(win.loc[win["is_debt_repay"], "amount"].sum()) / lamt,
                "ext": float(win.loc[win["is_ext_transfer_out"], "amount"].sum()) / lamt,
                "gamb": float(win.loc[win["is_gambling"], "amount"].sum()) / lamt,
                "cons": float(win.loc[win["is_consumption"], "amount"].sum()) / lamt,
            })
        loan = pd.DataFrame(loan_ratios)

        rows.append({
            "user_id": uid,
            "group": g["group"].iloc[0],
            "application_id": g["application_id"].iloc[0],
            "n_txn": len(g),
            "n_accounts": g["bank_account_id"].nunique(),
            "date_min": g["transaction_date"].min(),
            "date_max": g["transaction_date"].max(),
            "n_days": n_days,
            "total_credit": total_credit,
            "total_debit": total_debit,
            "wage": wage,
            "welfare": welfare,
            "other_income": other_income,
            "income": income,
            "loan_in": loan_in,
            "ext_transfer_in": ext_in,
            "internal_transfer_in": internal_in,
            "refund": refund,
            "consumption": consumption,
            "essential": essential,
            "gambling": gambling,
            "debt_repay": debt_repay,
            "ext_transfer_out": ext_out,
            "internal_transfer_out": internal_out,
            "fees": fees,
            "cash": cash,
            "other_debit": other_debit,
            "dishonour_cnt": dishonour_cnt,
            "dishonour_debit_cnt": dishonour_debit_cnt,
            "dishonour_credit_cnt": dishonour_credit_cnt,
            "dishonour_amt": dishonour_amt,
            "income_cv": income_cv,
            "active_lender_cnt": active_lender_cnt,
            "cash_advance": cash_advance,
            "bnpl": bnpl,
            "personal_loan": personal_loan,
            "bank_loan": bank_loan,
            "median_balance": median_bal,
            "min_balance": min_bal,
            "neg_days_share": neg_days,
            "low_balance_days_share": low_days,
            "buffer_days": buffer_days,
            "monthly_debit": monthly_debit,
            "wage_ratio": safe_div(wage, income),
            "welfare_ratio": safe_div(welfare, income),
            "other_income_ratio": safe_div(other_income, income),
            "true_income_ratio": safe_div(income, ext_credit),
            "loan_inflow_ratio": safe_div(loan_in, ext_credit),
            "transfer_in_ratio": safe_div(ext_in, ext_credit),
            "repay_rate": safe_div(debt_repay, income),
            "loan_in_rate": safe_div(loan_in, income),
            "gambling_share": safe_div(gambling, consumption),
            "consumption_share": safe_div(consumption, total_debit),
            "ext_transfer_share": safe_div(ext_out, total_debit),
            "internal_transfer_share": safe_div(internal_out, total_debit),
            "debt_share": safe_div(debt_repay, total_debit),
            "fees_share": safe_div(fees, total_debit),
            "essential_share": safe_div(essential, consumption),
            "any_wage": int(wage > 0),
            "any_loan": int(loan_in > 0 or debt_repay > 0),
            "any_loan_draw": int(loan_in > 0),
            "any_repay": int(debt_repay > 0),
            "any_gambling": int(gambling > 0),
            "any_dishonour": int(dishonour_cnt > 0),
            "pay_events": len(pay),
            "pay_outflow_7d": pay["outflow"].median() if len(pay) else np.nan,
            "pay_repay_7d": pay["repay"].median() if len(pay) else np.nan,
            "pay_ext_7d": pay["ext"].median() if len(pay) else np.nan,
            "pay_gamb_7d": pay["gamb"].median() if len(pay) else np.nan,
            "pay_cons_7d": pay["cons"].median() if len(pay) else np.nan,
            "pay_recycle_7d": (pay["repay"] + pay["ext"]).median() if len(pay) else np.nan,
            "loan_events": len(loan),
            "loan_outflow_7d": loan["outflow"].median() if len(loan) else np.nan,
            "loan_repay_7d": loan["repay"].median() if len(loan) else np.nan,
            "loan_ext_7d": loan["ext"].median() if len(loan) else np.nan,
            "loan_gamb_7d": loan["gamb"].median() if len(loan) else np.nan,
            "loan_cons_7d": loan["cons"].median() if len(loan) else np.nan,
            "loan_recycle_7d": (loan["repay"] + loan["ext"]).median() if len(loan) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    frames = [load_group(name, path) for name, path in DATA.items()]
    tx = pd.concat(frames, ignore_index=True)
    users = user_metrics(tx)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    users.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"用户指标: {OUT} | {len(users)} users")
    print("\n=== 交易分类规模（笔数 / 金额） ===")
    print(tx.groupby(["group", "cls"])["amount"].agg(["count", "sum"]).round(0).to_string())
    print("\n=== 用户指标中位数 / 发生率 ===")
    show = [
        "n_txn", "n_accounts", "income", "wage", "welfare", "loan_in", "debt_repay",
        "consumption", "gambling", "ext_transfer_out", "internal_transfer_out",
        "median_balance", "buffer_days", "neg_days_share", "low_balance_days_share",
        "wage_ratio", "true_income_ratio", "loan_inflow_ratio", "repay_rate",
        "gambling_share", "ext_transfer_share", "consumption_share",
        "pay_events", "pay_outflow_7d", "pay_repay_7d", "pay_ext_7d",
        "pay_recycle_7d",
        "loan_events", "loan_outflow_7d", "loan_repay_7d", "loan_recycle_7d",
        "active_lender_cnt", "cash_advance", "bnpl", "personal_loan", "bank_loan",
        "any_wage", "any_loan_draw", "any_repay", "any_gambling", "any_dishonour",
    ]
    for g, sub in users.groupby("group"):
        print(f"\n--- {g} | n={len(sub)} ---")
        out = []
        for col in show:
            s = pd.to_numeric(sub[col], errors="coerce")
            if s.notna().sum() == 0:
                continue
            if col.startswith("any_") or col in {"neg_days_share", "low_balance_days_share"}:
                out.append(f"{col}={s.mean():.1%}")
            else:
                out.append(f"{col}={s.median():.3g}")
        print(" | ".join(out))


if __name__ == "__main__":
    main()
