#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swap In vs Swap Out 原始交易明细画像对比报告

用法:
    python swap_in_out_profile_report.py

输出:
    output/swap_in_out_profile_report.html
"""

from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import swap_in_out_analysis as ana
from existing_swap_in_profile_report import fmt_money, fmt_pct

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "swap_in_out_profile_report.html"

IN_COLOR = "#1c5cab"
OUT_COLOR = "#d14b45"
GOLD = "#a16207"
GOLD_LINE = "#d9a441"
GOLD_BG = "#fdf9ec"

SEG_ORDER = [
    "借贷周转型",
    "工薪高杠杆型",
    "工薪低杠杆型",
    "转账过路型",
    "现金流脆弱型",
    "混合/低活跃型",
]
SEG_DESC = {
    "借贷周转型": "贷款流入和偿债负担都高，活跃机构多，贷款到账后资金快速流出，靠滚动周转维持现金流。与工薪高杠杆型的区别：贷款流入、活跃机构和周转链路更突出。",
    "工薪高杠杆型": "工资覆盖高，但偿债/收入高、外部转账流出大，工资到账后被债务和转账迅速吸收。与借贷周转型的区别：工资占比更高、贷款流入较低，但偿债和外部转账仍高。",
    "工薪低杠杆型": "工资稳定、偿债负担低、活跃机构少，是相对最健康的一类；但账户缓冲仍然很薄。与工薪高杠杆型的区别：偿债/收入和活跃机构显著更低。",
    "转账过路型": "工资占比很低，真实收入可解释性弱，外部转账占全部借记超过一半，资金以过路为主。与工薪型人群的区别：工资占比极低，外部转账占借记超过一半。",
    "现金流脆弱型": "余额缓冲极低、负余额天数高，收入并非最弱，但账户几乎没有安全垫。与转账过路型的区别：仍有工资收入，但余额缓冲更薄、负余额天数更高。",
    "混合/低活跃型": "收入、工资和转账活跃度都偏低，画像不突出，更像低活跃或信息不足人群。与转账过路型的区别：外部转账和收入活跃度都低，更像信息不足人群。",
}


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


def metric_row(label: str, a, b, kind="pct", digits=1, note="") -> str:
    if kind == "pct":
        wa = max(0.0, min(100.0, a * 100 if np.isfinite(a) else 0))
        wb = max(0.0, min(100.0, b * 100 if np.isfinite(b) else 0))
        va, vb = p(a, digits), p(b, digits)
    elif kind == "pct_rel":
        mx = max([x for x in (a, b) if np.isfinite(x)] + [1e-9])
        wa = max(0.0, min(100.0, a / mx * 100 if np.isfinite(a) else 0))
        wb = max(0.0, min(100.0, b / mx * 100 if np.isfinite(b) else 0))
        va, vb = p(a, digits), p(b, digits)
    elif kind == "money":
        mx = max([x for x in (a, b) if np.isfinite(x)] + [1e-9])
        wa = max(0.0, min(100.0, a / mx * 100 if np.isfinite(a) else 0))
        wb = max(0.0, min(100.0, b / mx * 100 if np.isfinite(b) else 0))
        va, vb = m(a), m(b)
    elif kind == "count":
        mx = max([x for x in (a, b) if np.isfinite(x)] + [1e-9])
        wa = max(0.0, min(100.0, a / mx * 100 if np.isfinite(a) else 0))
        wb = max(0.0, min(100.0, b / mx * 100 if np.isfinite(b) else 0))
        va, vb = n(a), n(b)
    elif kind == "ratio":
        mx = max([x for x in (a, b) if np.isfinite(x)] + [1e-9])
        wa = max(0.0, min(100.0, a / mx * 100 if np.isfinite(a) else 0))
        wb = max(0.0, min(100.0, b / mx * 100 if np.isfinite(b) else 0))
        va, vb = ratio(a), ratio(b)
    elif kind == "days":
        mx = max([x for x in (a, b) if np.isfinite(x)] + [1e-9])
        wa = max(0.0, min(100.0, a / mx * 100 if np.isfinite(a) else 0))
        wb = max(0.0, min(100.0, b / mx * 100 if np.isfinite(b) else 0))
        va, vb = f"{a:.2f} 天", f"{b:.2f} 天"
    else:
        mx = max([x for x in (a, b) if np.isfinite(x)] + [1e-9])
        wa = max(0.0, min(100.0, a / mx * 100 if np.isfinite(a) else 0))
        wb = max(0.0, min(100.0, b / mx * 100 if np.isfinite(b) else 0))
        va, vb = n(a, digits), n(b, digits)
    note_html = f'<div class="metric-note">{html.escape(note)}</div>' if note else ""
    return f"""
      <div class="metric">
        <div class="metric-label">{html.escape(label)}{note_html}</div>
        <div class="metric-bars">
          <div class="bar-row"><span class="who"><span class="dot" style="background:{IN_COLOR}"></span>In</span><div class="track"><div class="fill" style="width:{wa:.1f}%;background:{IN_COLOR}"></div></div><span class="val">{va}</span></div>
          <div class="bar-row"><span class="who"><span class="dot" style="background:{OUT_COLOR}"></span>Out</span><div class="track"><div class="fill" style="width:{wb:.1f}%;background:{OUT_COLOR}"></div></div><span class="val">{vb}</span></div>
        </div>
        <div class="metric-delta">{delta(a, b, kind, digits)}</div>
      </div>"""


def kpi_card(label: str, a, b, kind="pct", digits=1) -> str:
    if kind == "pct":
        va, vb = p(a, digits), p(b, digits)
    elif kind == "money":
        va, vb = m(a), m(b)
    elif kind == "count":
        va, vb = n(a), n(b)
    elif kind == "ratio":
        va, vb = ratio(a), ratio(b)
    elif kind == "days":
        va, vb = f"{a:.2f} 天", f"{b:.2f} 天"
    else:
        va, vb = n(a, digits), n(b, digits)
    return f"""
      <div class="kpi">
        <div class="kpi-label">{html.escape(label)}</div>
        <div class="kpi-row"><span class="who"><span class="dot" style="background:{IN_COLOR}"></span>Swap In</span><span class="v">{va}</span></div>
        <div class="kpi-row"><span class="who"><span class="dot" style="background:{OUT_COLOR}"></span>Swap Out</span><span class="v">{vb}</span></div>
        <div class="kpi-delta">{delta(a, b, kind, digits)}</div>
      </div>"""


def conclusion(text: str) -> str:
    return f'<div class="key-conclusion"><span class="star">★</span><div>{text}</div></div>'


CSS = """
:root {
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7;
  --in: #1c5cab; --out: #d14b45; --gold: #a16207; --gold-line: #d9a441;
  --gold-bg: #fdf9ec; --hl: #fbeec4;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
  font-size: 14px; line-height: 1.65; -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1120px; margin: 0 auto; padding: 32px 24px 56px; }
.eyebrow { color: var(--muted); font-size: 10.5px; letter-spacing: .14em; font-weight: 650; text-transform: uppercase; }
h1 { font-size: 29px; line-height: 1.25; margin: 7px 0 5px; letter-spacing: -.02em; }
.subtitle { color: var(--ink2); font-size: 13.5px; margin: 0 0 12px; }
.chips { display: flex; flex-wrap: wrap; gap: 7px; margin: 0 0 12px; }
.chip {
  display: inline-flex; align-items: center; gap: 5px;
  border: 1px solid rgba(11,11,11,.10); background: var(--surface);
  border-radius: 999px; padding: 4px 10px; font-size: 11.5px; color: var(--ink2);
}
.chip b { color: var(--ink); font-weight: 650; }
.note-legend { color: var(--muted); font-size: 11px; margin: 0 0 14px; }
.star { color: var(--gold); font-size: 12px; line-height: 1; margin-right: 4px; }
.key-conclusion {
  display: flex; gap: 8px; align-items: flex-start;
  margin: 4px 0 14px; padding: 9px 12px;
  border-left: 3px solid var(--gold-line); background: var(--gold-bg);
  border-radius: 0 8px 8px 0; color: var(--ink); font-size: 13.5px; line-height: 1.65;
}
.key-conclusion .star { margin-top: 2px; flex: none; font-size: 14px; }
mark.hl { background: var(--hl); color: inherit; padding: 0 2px; border-radius: 3px; font-weight: 650; }
.kpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin: 14px 0 4px; }
.kpi { background: var(--surface); border: 1px solid rgba(11,11,11,.10); border-radius: 12px; padding: 12px 13px; }
.kpi-label { color: var(--ink2); font-size: 11.5px; min-height: 34px; }
.kpi-row { display: flex; align-items: baseline; justify-content: space-between; gap: 6px; margin-top: 5px; }
.kpi-row .who { color: var(--ink2); font-size: 10.5px; display: inline-flex; align-items: center; gap: 4px; }
.kpi-row .v { font-size: 17px; font-weight: 650; letter-spacing: -.01em; }
.kpi-delta { color: var(--muted); font-size: 10px; margin-top: 5px; }
.section { margin-top: 30px; }
h2 { font-size: 18px; margin: 0 0 8px; letter-spacing: -.01em; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
h2 .num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 23px; height: 23px; border-radius: 50%;
  background: var(--ink); color: #fff; font-size: 11px; font-weight: 650;
}
h3 { font-size: 13.5px; margin: 18px 0 6px; color: var(--ink); }
.metric {
  display: grid; grid-template-columns: 200px 1fr 126px; gap: 12px;
  align-items: center; padding: 7px 0; border-bottom: 1px solid var(--grid);
}
.metric:last-child { border-bottom: none; }
.metric-label { font-size: 12.5px; color: var(--ink); font-weight: 550; }
.metric-note { color: var(--muted); font-size: 10.5px; font-weight: 400; margin-top: 2px; }
.metric-bars { display: grid; gap: 4px; }
.bar-row { display: grid; grid-template-columns: 34px 1fr 66px; gap: 7px; align-items: center; }
.bar-row .who { font-size: 10.5px; color: var(--muted); display: inline-flex; align-items: center; gap: 4px; }
.track { height: 9px; background: #eeede7; border-radius: 999px; overflow: hidden; }
.fill { height: 100%; border-radius: 999px; }
.bar-row .val { font-size: 11.5px; font-weight: 650; text-align: right; font-variant-numeric: tabular-nums; color: var(--ink); }
.metric-delta { text-align: right; font-size: 10.5px; color: var(--muted); }
.delta { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
.delta.muted { color: var(--muted); }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex: none; }
.table-card { background: var(--surface); border: 1px solid rgba(11,11,11,.10); border-radius: 12px; padding: 4px 10px; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 10.5px; padding: 7px 8px; border-bottom: 1px solid var(--baseline); white-space: nowrap; }
td { padding: 7px 8px; border-bottom: 1px solid var(--grid); color: var(--ink2); }
td:first-child { color: var(--ink); font-weight: 550; }
tr:last-child td { border-bottom: none; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.card { background: var(--surface); border: 1px solid rgba(11,11,11,.10); border-radius: 12px; padding: 12px 14px; }
.card h3 { margin: 0 0 6px; font-size: 13px; }
.card ul { margin: 0; padding-left: 18px; }
.card li { margin: 4px 0; color: var(--ink2); font-size: 12.5px; }
.note-box {
  margin-top: 18px; padding: 10px 12px; border: 1px solid var(--grid);
  border-radius: 10px; background: var(--surface); color: var(--ink2); font-size: 11.5px;
}
.note-box b { color: var(--ink); }
.footer { margin-top: 24px; padding-top: 12px; border-top: 1px solid var(--baseline); color: var(--muted); font-size: 11px; }
.footer p { margin: 3px 0; }
.tag { display: inline-block; padding: 1px 7px; border-radius: 999px; background: #f1f0eb; color: var(--ink2); font-size: 10.5px; }
@media (max-width: 900px) {
  .kpi-grid { grid-template-columns: repeat(3, 1fr); }
  .metric { grid-template-columns: 160px 1fr 100px; }
}
@media (max-width: 680px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .metric { grid-template-columns: 1fr; gap: 4px; }
  .metric-delta { text-align: left; }
  .two-col { grid-template-columns: 1fr; }
  h1 { font-size: 23px; }
}
@media print { .wrap { max-width: none; } .kpi, .metric, .card { break-inside: avoid; } }
"""


def build_html(users: pd.DataFrame, tx: pd.DataFrame, generated_at: str) -> str:
    si = summary(users[users["group"] == "Swap In"])
    so = summary(users[users["group"] == "Swap Out"])
    mi = income_mix(users[users["group"] == "Swap In"])
    mo = income_mix(users[users["group"] == "Swap Out"])
    oi = outflow_mix(users[users["group"] == "Swap In"])
    oo = outflow_mix(users[users["group"] == "Swap Out"])

    seg_ct = pd.crosstab(users["seg"], users["group"]).reindex(SEG_ORDER, fill_value=0)
    for col in ("Swap In", "Swap Out"):
        if col not in seg_ct:
            seg_ct[col] = 0
    seg_ct["total"] = seg_ct["Swap In"] + seg_ct["Swap Out"]
    seg_ct["In%"] = seg_ct["Swap In"] / si["n"]
    seg_ct["Out%"] = seg_ct["Swap Out"] / so["n"]

    seg_cols = ["income", "wage_ratio", "true_income_ratio", "loan_inflow_ratio",
                "repay_rate", "ext_transfer_share", "buffer_days", "dishonour_cnt",
                "active_lender_cnt", "pay_recycle_7d", "loan_recycle_7d"]
    seg_prof = users.groupby("seg")[seg_cols].median().reindex(SEG_ORDER)
    users = users.copy()
    users["gambling_high_flag"] = ((users["gambling_share"] >= 0.05) | (users["gambling"] >= 1000)).astype(float)
    risk_seg = users.groupby("seg").agg(
        dishon_any=("any_dishonour", "mean"),
        dishon_cnt=("dishonour_cnt", "median"),
        gamb_high=("gambling_high_flag", "mean"),
        buffer=("buffer_days", "median"),
    ).reindex(SEG_ORDER)
    risk_seg_rows = "".join(
        f"<tr><td>{html.escape(seg)}</td>"
        f"<td>{p(risk_seg.loc[seg, 'dishon_any'])}</td>"
        f"<td>{n(risk_seg.loc[seg, 'dishon_cnt'])}</td>"
        f"<td>{p(risk_seg.loc[seg, 'gamb_high'])}</td>"
        f"<td>{n(risk_seg.loc[seg, 'buffer'], 2)} 天</td></tr>"
        for seg in SEG_ORDER
    )

    loan = tx[tx["is_loan_in"] | tx["is_debt_repay"]].copy()
    loan["third_party"] = loan["third_party"].fillna("(未标注)")
    top = loan.groupby("third_party")["user_id"].nunique().sort_values(ascending=False).head(10)

    kpis = "".join([
        kpi_card("中位真实收入", si["income"], so["income"], "money"),
        kpi_card("工资覆盖率", si["any_wage"], so["any_wage"], "pct"),
        kpi_card("偿债 / 收入", si["repay_rate"], so["repay_rate"], "pct"),
        kpi_card("外部转账 / 借记", si["ext_transfer_share"], so["ext_transfer_share"], "pct"),
        kpi_card("拒付次数中位", si["dishonour_cnt"], so["dishonour_cnt"], "count"),
        kpi_card("缓冲天数中位", si["buffer"], so["buffer"], "days"),
    ])

    income_rows = "".join([
        metric_row("工资覆盖率", si["any_wage"], so["any_wage"], "pct"),
        metric_row("工资占真实收入（用户中位）", si["wage_ratio"], so["wage_ratio"], "pct"),
        metric_row("真实收入 / 总入账（剔除内部转账）", si["true_income_ratio"], so["true_income_ratio"], "pct"),
        metric_row("外部转账流入 / 总入账", si["transfer_in_ratio"], so["transfer_in_ratio"], "pct"),
        metric_row("贷款流入 / 总入账", si["loan_inflow_ratio"], so["loan_inflow_ratio"], "pct"),
    ])
    income_mix_rows = "".join([
        f"<tr><td>工资</td><td>{p(mi['wage_share'])}</td><td>{p(mo['wage_share'])}</td><td>{delta(mi['wage_share'], mo['wage_share'], 'pct')}</td></tr>",
        f"<tr><td>福利</td><td>{p(mi['welfare_share'])}</td><td>{p(mo['welfare_share'])}</td><td>{delta(mi['welfare_share'], mo['welfare_share'], 'pct')}</td></tr>",
        f"<tr><td>其他入账</td><td>{p(mi['other_share'])}</td><td>{p(mo['other_share'])}</td><td>{delta(mi['other_share'], mo['other_share'], 'pct')}</td></tr>",
    ])

    outflow_rows = "".join([
        metric_row("可识别消费 / 全部借记", si["consumption_share"], so["consumption_share"], "pct"),
        metric_row("外部转账 / 全部借记", si["ext_transfer_share"], so["ext_transfer_share"], "pct"),
        metric_row("内部转账 / 全部借记", si["internal_transfer_share"], so["internal_transfer_share"], "pct"),
        metric_row("债务还款 / 全部借记", si["debt_share"], so["debt_share"], "pct"),
        metric_row("赌博 / 全部借记", oi["gambling"], oo["gambling"], "pct",
                   note="按金额加总；Swap In 的赌博金额集中在少数用户"),
    ])
    debt_rows = "".join([
        metric_row("有借贷放款", si["any_loan_draw"], so["any_loan_draw"], "pct"),
        metric_row("有还款", si["any_repay"], so["any_repay"], "pct"),
        metric_row("活跃借款机构数（中位）", si["active_lender"], so["active_lender"], "count"),
    ])
    product_rows = "".join([
        f"<tr><td>现金贷 / 工资预支</td><td>{p(si['cash_advance'])}</td><td>{p(so['cash_advance'])}</td><td>{delta(si['cash_advance'], so['cash_advance'], 'pct')}</td></tr>",
        f"<tr><td>BNPL</td><td>{p(si['bnpl'])}</td><td>{p(so['bnpl'])}</td><td>{delta(si['bnpl'], so['bnpl'], 'pct')}</td></tr>",
        f"<tr><td>个人贷</td><td>{p(si['personal_loan'])}</td><td>{p(so['personal_loan'])}</td><td>{delta(si['personal_loan'], so['personal_loan'], 'pct')}</td></tr>",
        f"<tr><td>银行贷</td><td>{p(si['bank_loan'])}</td><td>{p(so['bank_loan'])}</td><td>{delta(si['bank_loan'], so['bank_loan'], 'pct')}</td></tr>",
    ])
    lender_rows = "".join(
        f"<tr><td>{html.escape(str(provider))}</td>"
        f"<td>{loan[(loan['third_party'] == provider) & (loan['group'] == 'Swap In')]['user_id'].nunique()}</td>"
        f"<td>{loan[(loan['third_party'] == provider) & (loan['group'] == 'Swap Out')]['user_id'].nunique()}</td>"
        f"<td>{loan[loan['third_party'] == provider]['user_id'].nunique()}</td></tr>"
        for provider in top.index
    )

    pay_rows = "".join([
        metric_row("发薪后 7 天还贷 / 工资", si["pay_repay"], so["pay_repay"], "pct"),
        metric_row("发薪后 7 天外部转账 / 工资", si["pay_ext"], so["pay_ext"], "pct"),
        metric_row("发薪后 7 天还贷+转账 / 工资", si["pay_recycle"], so["pay_recycle"], "pct_rel"),
        metric_row("发薪后 7 天总流出 / 工资", si["pay_outflow"], so["pay_outflow"], "pct_rel"),
        metric_row("贷款到账后 7 天还贷 / 贷款", si["loan_repay"], so["loan_repay"], "ratio"),
        metric_row("贷款到账后 7 天还贷+外部转账 / 贷款", si["loan_recycle"], so["loan_recycle"], "ratio"),
    ])

    risk_rows = "".join([
        metric_row("中位账户余额", si["balance"], so["balance"], "money"),
        metric_row("缓冲天数（中位）", si["buffer"], so["buffer"], "days"),
        metric_row("负余额天数占比", si["neg_days"], so["neg_days"], "pct"),
        metric_row("余额 < $100 天数占比", si["low_days"], so["low_days"], "pct"),
        metric_row("有拒付记录", si["any_dishonour"], so["any_dishonour"], "pct"),
        metric_row("拒付次数（中位）", si["dishonour_cnt"], so["dishonour_cnt"], "count"),
        metric_row("高赌博暴露占比", si["gambling_high"], so["gambling_high"], "pct"),
    ])

    seg_rows = "".join(
        f"<tr><td>{html.escape(seg)}</td>"
        f"<td>{int(seg_ct.loc[seg, 'Swap In'])}（{p(seg_ct.loc[seg, 'In%'])}）</td>"
        f"<td>{int(seg_ct.loc[seg, 'Swap Out'])}（{p(seg_ct.loc[seg, 'Out%'])}）</td>"
        f"<td>{int(seg_ct.loc[seg, 'total'])}</td></tr>"
        for seg in SEG_ORDER
    )
    seg_prof_rows = "".join(
        f"<tr><td>{html.escape(seg)}</td>"
        f"<td>{m(seg_prof.loc[seg, 'income'])}</td>"
        f"<td>{p(seg_prof.loc[seg, 'wage_ratio'])}</td>"
        f"<td>{p(seg_prof.loc[seg, 'repay_rate'])}</td>"
        f"<td>{p(seg_prof.loc[seg, 'ext_transfer_share'])}</td>"
        f"<td>{n(seg_prof.loc[seg, 'buffer_days'], 2)}</td>"
        f"<td>{n(seg_prof.loc[seg, 'dishonour_cnt'])}</td>"
        f"<td>{n(seg_prof.loc[seg, 'active_lender_cnt'], 1)}</td>"
        f"<td>{ratio(seg_prof.loc[seg, 'loan_recycle_7d'])}</td></tr>"
        for seg in SEG_ORDER
    )
    seg_desc = "".join(
        f"<li><b>{html.escape(seg)}</b>：{SEG_DESC[seg]}</li>" for seg in SEG_ORDER
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swap In vs Swap Out 人群画像对比分析</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">SWAP IN × SWAP OUT · RAW TRANSACTION PROFILE</div>
  <h1>Swap In vs Swap Out 人群画像对比分析</h1>
  <p class="subtitle">基于原始银行流水交易明细 · {ana.WINDOW_DAYS} 天观察窗口 · 存量用户 · 描述性对比</p>
  <div class="chips">
    <span class="chip"><span class="dot" style="background:{IN_COLOR}"></span>Swap In <b>{si['n']}</b> 用户 · 窗口内 <b>{int(users[users['group']=='Swap In']['n_txn'].sum()):,}</b> 笔</span>
    <span class="chip"><span class="dot" style="background:{OUT_COLOR}"></span>Swap Out <b>{so['n']}</b> 用户 · 窗口内 <b>{int(users[users['group']=='Swap Out']['n_txn'].sum()):,}</b> 笔</span>
    <span class="chip">交易日期 <b>2026-03 至 2026-10</b></span>
    <span class="chip">生成时间 <b>{html.escape(generated_at)}</b></span>
  </div>
  <p class="note-legend"><span class="star">★</span>金色框为结论，金色高亮为支撑数据；蓝色为 Swap In，红色为 Swap Out。内部转账、贷款放款已从收入中剔除；拒付为事件代理指标。除特别说明外，比率类指标为组内用户中位数，金额为组内用户中位数；构成表按金额加总并单独标注。</p>

  {conclusion(f"<b>一句话结论：</b>Swap Out 收入更高、工资更稳，但借贷、拒付和发薪后外部流出更重；Swap In 收入更弱、内部转账更多，但杠杆和拒付压力相对低。两组共同点是账户缓冲极薄。")}

  <div class="kpi-grid">{kpis}</div>

  <section class="section">
    <h2><span class="num">1</span>执行摘要</h2>
    {conclusion(f"Swap Out 不是“更优质”，而是“更高收入 + 更高杠杆”；Swap In 不是“更差”，而是“更低杠杆 + 更弱收入”。")}
    <ul class="notes">
      <li><b>收入：</b>Swap Out 中位真实收入 <mark class="hl">{m(so['income'])}</mark>，Swap In <mark class="hl">{m(si['income'])}</mark>；工资覆盖率 <mark class="hl">{p(so['any_wage'])}</mark> vs <mark class="hl">{p(si['any_wage'])}</mark>，工资占真实收入 <mark class="hl">{p(so['wage_ratio'])}</mark> vs <mark class="hl">{p(si['wage_ratio'])}</mark>。</li>
      <li><b>资金流向：</b>Swap Out 外部转账占全部借记 <mark class="hl">{p(so['ext_transfer_share'])}</mark>，Swap In <mark class="hl">{p(si['ext_transfer_share'])}</mark>；Swap In 内部转账占比更高（<mark class="hl">{p(si['internal_transfer_share'])}</mark> vs <mark class="hl">{p(so['internal_transfer_share'])}</mark>）。</li>
      <li><b>杠杆：</b>Swap Out 有借贷放款 <mark class="hl">{p(so['any_loan_draw'])}</mark> vs <mark class="hl">{p(si['any_loan_draw'])}</mark>，活跃机构中位 <mark class="hl">{n(so['active_lender'], 0)}</mark> vs <mark class="hl">{n(si['active_lender'], 0)}</mark>，偿债/收入 <mark class="hl">{p(so['repay_rate'])}</mark> vs <mark class="hl">{p(si['repay_rate'])}</mark>。</li>
      <li><b>履约：</b>Swap Out 有拒付记录 <mark class="hl">{p(so['any_dishonour'])}</mark> vs <mark class="hl">{p(si['any_dishonour'])}</mark>，拒付次数中位 <mark class="hl">{n(so['dishonour_cnt'])}</mark> vs <mark class="hl">{n(si['dishonour_cnt'])}</mark>。</li>
      <li><b>发薪后：</b>Swap Out 发薪后 7 天还贷+外部转账/工资 <mark class="hl">{p(so['pay_recycle'])}</mark>，Swap In <mark class="hl">{p(si['pay_recycle'])}</mark>；Swap Out 的工资更快被债务和转账吸收。</li>
      <li><b>流动性：</b>两组中位余额仅 <mark class="hl">{m(si['balance'])}</mark> / <mark class="hl">{m(so['balance'])}</mark>，缓冲天数中位 <mark class="hl">{n(si['buffer'], 2)}</mark> / <mark class="hl">{n(so['buffer'], 2)}</mark> 天，缓冲低于 7 天占比 <mark class="hl">{p(si['buffer_lt7'])}</mark> / <mark class="hl">{p(so['buffer_lt7'])}</mark>。</li>
      <li><b>分层：</b>Swap Out 有 <mark class="hl">{p(seg_ct.loc['借贷周转型', 'Out%'] + seg_ct.loc['工薪高杠杆型', 'Out%'])}</mark> 落在“借贷周转/工薪高杠杆”，Swap In 为 <mark class="hl">{p(seg_ct.loc['借贷周转型', 'In%'] + seg_ct.loc['工薪高杠杆型', 'In%'])}</mark>；Swap In 更多落在转账过路、现金流脆弱和混合人群。</li>
    </ul>
  </section>

  <section class="section">
    <h2><span class="num">2</span>样本整体底盘判断</h2>
    {conclusion(f"两组都是“高流水、低缓冲、重转账”的人群；Swap Out 更偏工薪和高杠杆，Swap In 更偏转账和内部账户周转。")}
    <p class="note-legend">输入形态：交易明细表（18 列）；观察窗口 {ana.WINDOW_DAYS} 天；Swap In {si['n']} 用户 / {int(users[users['group']=='Swap In']['n_txn'].sum()):,} 笔，Swap Out {so['n']} 用户 / {int(users[users['group']=='Swap Out']['n_txn'].sum()):,} 笔；账户数中位 {n(si['accounts'], 1)} vs {n(so['accounts'], 1)}。未提供独立每日余额快照（余额由交易后 balance 推导）；无征信/还款状态、无拒付解决状态。</p>
    <div class="table-card">
      <table>
        <thead><tr><th>指标</th><th>Swap In</th><th>Swap Out</th><th>差异</th></tr></thead>
        <tbody>
          <tr><td>用户数 / 交易笔数中位</td><td>{si['n']} / {n(si['txn'])}</td><td>{so['n']} / {n(so['txn'])}</td><td>—</td></tr>
          <tr><td>中位真实收入</td><td>{m(si['income'])}</td><td>{m(so['income'])}</td><td>{delta(si['income'], so['income'], 'money')}</td></tr>
          <tr><td>工资覆盖率</td><td>{p(si['any_wage'])}</td><td>{p(so['any_wage'])}</td><td>{delta(si['any_wage'], so['any_wage'], 'pct')}</td></tr>
          <tr><td>外部转账 / 全部借记</td><td>{p(si['ext_transfer_share'])}</td><td>{p(so['ext_transfer_share'])}</td><td>{delta(si['ext_transfer_share'], so['ext_transfer_share'], 'pct')}</td></tr>
          <tr><td>内部转账 / 全部借记</td><td>{p(si['internal_transfer_share'])}</td><td>{p(so['internal_transfer_share'])}</td><td>{delta(si['internal_transfer_share'], so['internal_transfer_share'], 'pct')}</td></tr>
          <tr><td>偿债 / 收入</td><td>{p(si['repay_rate'])}</td><td>{p(so['repay_rate'])}</td><td>{delta(si['repay_rate'], so['repay_rate'], 'pct')}</td></tr>
          <tr><td>有拒付记录</td><td>{p(si['any_dishonour'])}</td><td>{p(so['any_dishonour'])}</td><td>{delta(si['any_dishonour'], so['any_dishonour'], 'pct')}</td></tr>
          <tr><td>中位账户余额</td><td>{m(si['balance'])}</td><td>{m(so['balance'])}</td><td>{delta(si['balance'], so['balance'], 'money')}</td></tr>
          <tr><td>缓冲天数中位</td><td>{n(si['buffer'], 2)} 天</td><td>{n(so['buffer'], 2)} 天</td><td>{delta(si['buffer'], so['buffer'], 'days')}</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <section class="section">
    <h2><span class="num">3</span>核心人群分层</h2>
    {conclusion(f"6 类人群中，Swap Out 更集中在“借贷周转”和“工薪高杠杆”；Swap In 更分散在转账过路、现金流脆弱和混合/低活跃。")}
    <div class="table-card">
      <table>
        <thead><tr><th>人群</th><th>Swap In（占比）</th><th>Swap Out（占比）</th><th>合计</th></tr></thead>
        <tbody>{seg_rows}</tbody>
      </table>
    </div>
    <h3>分层画像（中位数）</h3>
    <div class="table-card">
      <table>
        <thead><tr><th>人群</th><th>收入</th><th>工资占比</th><th>偿债/收入</th><th>外部转账</th><th>缓冲天数</th><th>拒付次数</th><th>活跃机构</th><th>贷款周转</th></tr></thead>
        <tbody>{seg_prof_rows}</tbody>
      </table>
    </div>
    <h3>人群定义</h3>
    <ul class="notes">{seg_desc}</ul>
    <p class="note-legend">分层为规则判定，不是统计聚类；阈值用于描述性对比，换阈值会改变人群边界。</p>
  </section>

  <section class="section">
    <h2><span class="num">4</span>人群分化的核心驱动因素</h2>
    {conclusion(f"真正拉开两组的是工资覆盖与收入水平、外部/内部转账、借贷使用与偿债负担；消费差异不大，赌博和拒付更像修饰标签。")}
    <h3>4.1 收入质量</h3>
    {income_rows}
    <h3>4.2 真实收入构成（按金额加总）</h3>
    <div class="table-card">
      <table>
        <thead><tr><th>收入构成</th><th>Swap In</th><th>Swap Out</th><th>差异</th></tr></thead>
        <tbody>{income_mix_rows}</tbody>
      </table>
    </div>
    <p class="note-legend">真实收入 = 工资 + 福利 + 其他入账；贷款放款、外部转账流入、内部转账不计入。工资占真实收入按金额加总：Swap In {p(mi['wage_share'])}，Swap Out {p(mo['wage_share'])}。</p>
    <h3>4.3 资金流向</h3>
    {outflow_rows}
    <h3>4.4 主分层变量 vs 修饰标签</h3>
    <div class="table-card">
      <table>
        <thead><tr><th>变量</th><th>更像</th><th>数据证据</th></tr></thead>
        <tbody>
          <tr><td>工资覆盖与收入水平</td><td>主分层变量</td><td>工资覆盖 {p(si['any_wage'])} → {p(so['any_wage'])}；中位真实收入 {m(si['income'])} → {m(so['income'])}</td></tr>
          <tr><td>外部/内部转账</td><td>主分层变量</td><td>外部转账 {p(si['ext_transfer_share'])} → {p(so['ext_transfer_share'])}；内部转账 {p(si['internal_transfer_share'])} → {p(so['internal_transfer_share'])}</td></tr>
          <tr><td>借贷使用与活跃机构</td><td>主分层变量</td><td>有借贷放款 {p(si['any_loan_draw'])} → {p(so['any_loan_draw'])}；活跃机构 {n(si['active_lender'],0)} → {n(so['active_lender'],0)}</td></tr>
          <tr><td>偿债负担</td><td>主分层变量</td><td>偿债/收入 {p(si['repay_rate'])} → {p(so['repay_rate'])}</td></tr>
          <tr><td>发薪后还贷+转账</td><td>主分层 + 链路</td><td>发薪后 7 天还贷+转账/工资 {p(si['pay_recycle'])} → {p(so['pay_recycle'])}</td></tr>
          <tr><td>拒付、赌博、低余额</td><td>修饰标签</td><td>拒付 {p(si['any_dishonour'])} → {p(so['any_dishonour'])}；高赌博 {p(si['gambling_high'])} → {p(so['gambling_high'])}；缓冲 {n(si['buffer'],2)} → {n(so['buffer'],2)} 天</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <section class="section">
    <h2><span class="num">5</span>横向风险标签分析</h2>
    {conclusion(f"两组都不是“余额缓冲型”人群：缓冲天数中位不足半天，缓冲低于 7 天占比接近全覆盖。Swap Out 的拒付更重；赌博在两组都高度集中，不是主要分化轴。")}
    {risk_rows}
    <div class="table-card" style="margin-top:12px">
      <table>
        <thead><tr><th>风险标签</th><th>Swap In</th><th>Swap Out</th><th>解读</th></tr></thead>
        <tbody>
          <tr><td>赌博发生 / 高暴露</td><td>{p(si['any_gambling'])} / {p(si['gambling_high'])}</td><td>{p(so['any_gambling'])} / {p(so['gambling_high'])}</td><td>发生率高，但金额集中；高暴露定义为赌博占消费 ≥5% 或金额 ≥ $1,000。</td></tr>
          <tr><td>拒付发生 / 次数中位</td><td>{p(si['any_dishonour'])} / {n(si['dishonour_cnt'])}</td><td>{p(so['any_dishonour'])} / {n(so['dishonour_cnt'])}</td><td>拒付是横向增强标签；Swap Out 更重，但无解决状态，需后续验证。</td></tr>
          <tr><td>负余额天数 / 低余额天数</td><td>{p(si['neg_days'])} / {p(si['low_days'])}</td><td>{p(so['neg_days'])} / {p(so['low_days'])}</td><td>低余额是共同底盘，不宜单独作为主分层。</td></tr>
          <tr><td>外部转账 / 内部转账占比</td><td>{p(si['ext_transfer_share'])} / {p(si['internal_transfer_share'])}</td><td>{p(so['ext_transfer_share'])} / {p(so['internal_transfer_share'])}</td><td>Swap Out 更外流，Swap In 更内部周转。</td></tr>
          <tr><td>贷款后 7 天周转倍数</td><td>{ratio(si['loan_recycle'])}</td><td>{ratio(so['loan_recycle'])}</td><td>贷款到账后 7 天内还贷+外部转账 / 贷款金额；Swap Out 更高。</td></tr>
        </tbody>
      </table>
    </div>
    <p class="note-legend">横向标签用于增强主分层，不单独作为主分群；拒付包含冲正/费用，解决状态不可得。</p>
    <h3>风险标签在人群中的分布</h3>
    <div class="table-card">
      <table>
        <thead><tr><th>人群</th><th>拒付发生</th><th>拒付次数中位</th><th>高赌博暴露</th><th>缓冲天数</th></tr></thead>
        <tbody>{risk_seg_rows}</tbody>
      </table>
    </div>
  </section>

  <section class="section">
    <h2><span class="num">6</span>借贷生态与机构层观察</h2>
    {conclusion(f"借贷生态以现金贷/工资预支为主；Swap Out 的 BNPL、个人贷覆盖和活跃机构数更高，更接近多平台周转依赖。")}
    {debt_rows}
    <h3>借贷产品使用率</h3>
    <div class="table-card">
      <table>
        <thead><tr><th>产品类型</th><th>Swap In</th><th>Swap Out</th><th>差异</th></tr></thead>
        <tbody>{product_rows}</tbody>
      </table>
    </div>
    <h3>主要借款机构 / 产品（按覆盖用户数）</h3>
    <div class="table-card">
      <table>
        <thead><tr><th>机构 / 产品</th><th>Swap In 用户</th><th>Swap Out 用户</th><th>合计</th></tr></thead>
        <tbody>{lender_rows}</tbody>
      </table>
    </div>
  </section>

  <section class="section">
    <h2><span class="num">7</span>发薪后行为与借新还旧</h2>
    {conclusion(f"Swap Out 的工资到账后更快被还贷和外部转账吸收；Swap In 的发薪后还贷/转账比例较低，但内部转账更多。")}
    <p class="note-legend">发薪后窗口为工资入账日之后的 7 天；贷款窗口为贷款到账日之后的 7 天。比率为用户内事件中位数，可能超过 100%，因为窗口内还有其它入账。</p>
    {pay_rows}
    <p class="note-legend">发薪覆盖率：Swap In {p(si['pay_cov'])}、Swap Out {p(so['pay_cov'])}；贷款覆盖率：Swap In {p(si['loan_cov'])}、Swap Out {p(so['loan_cov'])}。</p>
  </section>

  <div class="footer">
    <p><b>数据源</b>：data/swap_in.csv、data/swap_out.csv；一行一笔交易，按 user_id 聚合。</p>
    <p><b>指标口径</b>：真实收入 = 工资 + 福利 + 其他入账；可识别消费 = 非结构型类目的借记；高赌博暴露 = 赌博占消费 ≥5% 或赌博金额 ≥ $1,000；发薪后/贷款后窗口 = 事件日之后 7 天。</p>
    <p><b>数据与口径限制</b>：Swap In {si['n']} 人 / {int(users[users['group']=='Swap In']['n_txn'].sum()):,} 笔，Swap Out {so['n']} 人 / {int(users[users['group']=='Swap Out']['n_txn'].sum()):,} 笔；观察窗口 {ana.WINDOW_DAYS} 天。收入 = 工资 + 福利 + 其他入账，已剔除内部转账和贷款放款；借贷还款按贷款机构/产品对手方识别；拒付为事件代理指标；余额按账户 + 日期取最后一条交易余额。分类映射基于当前样本实际取值，跨数据集需重新校准；以上为描述性对比，不做因果推断。</p>
    <p><b>生成时间</b>：{html.escape(generated_at)}</p>
  </div>
</div>
</body>
</html>"""


def main() -> None:
    frames = [ana.load_group(name, path) for name, path in ana.DATA.items()]
    tx = pd.concat(frames, ignore_index=True)
    users = assign_segments(ana.user_metrics(tx))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_html(users, tx, generated_at), encoding="utf-8")
    print(f"报告: {OUT}")
    print(f"用户: {len(users)} | 交易: {len(tx):,}")


if __name__ == "__main__":
    main()
