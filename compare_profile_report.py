#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fundo vs WageGo 人群画像对比报告

用法:
    python compare_profile_report.py
    python compare_profile_report.py --fundo data/fundo_sample_with_feature_all.csv \
        --wagego data/wagego_sample_with_feature_all.csv --out output

输出:
    output/compare_profile_report.html   对比报告（图表内嵌）
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import fundo_profile_report as fpr
import wagego_profile_report as wpr
from fundo_profile_report import (
    BASELINE,
    INK,
    INK2,
    SEQ_BLUE,
    SERIES,
    STATUS,
    SURFACE,
    base_layout,
    fig_html,
    fmt_money,
    fmt_pct,
    get_plotlyjs,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 plotly，请先执行: pip install plotly") from exc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent
DEFAULT_FUNDO = ROOT / "data" / "fundo_sample_with_feature_all.csv"
DEFAULT_WAGEGO = ROOT / "data" / "wagego_sample_with_feature_all.csv"
DEFAULT_OUT = ROOT / "output"

POP_COLORS = {"Fundo": SERIES[0], "WageGo": SERIES[1]}
POP_Y_ORDER = ["WageGo", "Fundo"]  # 水平图中 Fundo 在上
HILITE = "#d9a441"  # 结论对应数据的金色描边
SEG_COLORS = {
    "盈余型": (STATUS["good"], "#ffffff"),
    "平衡型": (STATUS["warning"], INK),
    "赤字型": (STATUS["serious"], "#ffffff"),
    "重压型": (STATUS["critical"], "#ffffff"),
}
CAT_LABELS = fpr.CONSUMPTION_CATS


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
@dataclass
class Profile:
    d: pd.DataFrame
    m: dict
    seg: pd.DataFrame
    missing: list[str]
    n_fields: int


def load_profile(name: str, module, path: Path) -> Profile:
    t0 = time.time()
    df_raw, missing, n_fields = module.load_data(path)
    df_raw, n_apps = module.clean_frame(df_raw)
    df = module.dedupe_users(df_raw)
    d, m, seg = module.build_metrics(df, n_apps)

    bal = d["bal"]
    bucket_labels = ["透支 (<$0)", "$0–100", "$100–500", "$500–1k", "≥$1k"]
    bucket_masks = [
        bal < 0,
        (bal >= 0) & (bal < 100),
        (bal >= 100) & (bal < 500),
        (bal >= 500) & (bal < 1000),
        bal >= 1000,
    ]
    m["balance_buckets"] = [
        (lab, module.cond_share(mask, bal)) for lab, mask in zip(bucket_labels, bucket_masks)
    ]
    print(f"  {name}: 有效样本 {m['n_valid']:,} 人，用时 {time.time() - t0:.1f}s")
    return Profile(d=d, m=m, seg=seg, missing=missing, n_fields=n_fields)


def seg_val(p: Profile, segment: str, col: str) -> float:
    return float(p.seg.loc[segment, col]) if segment in p.seg.index else np.nan


def metric_value(p: Profile, key: str):
    if key.startswith("seg:"):
        _, segment, col = key.split(":", 2)
        return seg_val(p, segment, col)
    if key == "employer_share":
        return p.m["employer_n"] / p.m["n_valid"] if p.m["n_valid"] else np.nan
    return p.m.get(key, np.nan)


# ---------------------------------------------------------------------------
# 数字格式
# ---------------------------------------------------------------------------
def valid(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fmt_metric(value, kind: str) -> str:
    if kind == "text":
        if value is None:
            return "—"
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return "—"
        return html.escape(text)
    if not valid(value):
        return "—"
    x = float(value)
    if kind == "money":
        return fmt_money(x)
    if kind == "pct":
        return fmt_pct(x)
    if kind == "num":
        return f"{x:.2f}"
    if kind == "num0":
        return f"{x:.0f}"
    if kind == "num1":
        return f"{x:.1f}"
    if kind == "int":
        return f"{int(round(x)):,}"
    return html.escape(str(value))


def fmt_delta(fv, wv, kind: str) -> str:
    if kind == "text" or not (valid(fv) and valid(wv)):
        return '<span class="delta muted">—</span>'
    f, w = float(fv), float(wv)
    if kind == "pct":
        diff = (w - f) * 100
        if abs(diff) < 0.05:
            return '<span class="delta muted">基本持平</span>'
        who = "WageGo 高" if diff > 0 else "Fundo 高"
        text = f"{abs(diff):.1f}pp"
    else:
        diff = w - f
        if abs(diff) < 1e-9:
            return '<span class="delta muted">基本持平</span>'
        who = "WageGo 高" if diff > 0 else "Fundo 高"
        if kind == "money":
            text = fmt_money(abs(diff))
            # 基数接近 0 时相对变化会失真，只在基数足够大时展示
            if f > 0 and abs(diff) / f <= 3:
                text += f"（{abs(diff) / f:.0%}）"
        elif kind in {"num", "num0", "num1"}:
            digits = {"num": 2, "num1": 1, "num0": 0}[kind]
            text = f"{abs(diff):.{digits}f}"
        elif kind == "int":
            text = f"{abs(diff):,.0f}"
        else:
            return '<span class="delta muted">—</span>'
    color = POP_COLORS[who.split()[0]]
    return (
        f'<span class="delta"><span class="dot" style="background:{color}"></span>'
        f"{who} {text}</span>"
    )


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# 表格
# ---------------------------------------------------------------------------
def comp_table(rows: list, pf: Profile, pw: Profile) -> str:
    body = []
    for row in rows:
        if isinstance(row, str):
            body.append(f'<tr class="group"><td colspan="4">{html.escape(row)}</td></tr>')
            continue
        label, key, kind = row
        fv, wv = metric_value(pf, key), metric_value(pw, key)
        row_cls = ' class="hl"' if key in KEY_METRICS else ""
        body.append(
            f"<tr{row_cls}>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{fmt_metric(fv, kind)}</td>"
            f"<td>{fmt_metric(wv, kind)}</td>"
            f"<td>{fmt_delta(fv, wv, kind)}</td>"
            "</tr>"
        )
    head = (
        "<tr>"
        "<th>指标</th>"
        f'<th><span class="dot" style="background:{POP_COLORS["Fundo"]}"></span>Fundo</th>'
        f'<th><span class="dot" style="background:{POP_COLORS["WageGo"]}"></span>WageGo</th>'
        "<th>差异</th>"
        "</tr>"
    )
    return f'<table class="cmp"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def status_chip(level: str, label: str) -> str:
    colors = {
        "good": "var(--good)",
        "warning": "var(--warning)",
        "serious": "var(--serious)",
        "critical": "var(--critical)",
    }
    return (
        f'<span class="chip-status"><span class="dot" '
        f'style="background:{colors.get(level, "var(--muted)")}"></span>'
        f"{html.escape(label)}</span>"
    )


def category_table(pf: Profile, pw: Profile) -> str:
    keys = sorted(
        pf.m["cat_shares"],
        key=lambda k: max(
            pf.m["cat_shares"].get(k, 0) if valid(pf.m["cat_shares"].get(k)) else 0,
            pw.m["cat_shares"].get(k, 0) if valid(pw.m["cat_shares"].get(k)) else 0,
        ),
        reverse=True,
    )
    body = []
    for k in keys:
        fv = pf.m["cat_shares"].get(k, np.nan)
        wv = pw.m["cat_shares"].get(k, np.nan)
        f_avg = pf.m["cat_totals"].get(k, np.nan) / pf.m["n_valid"]
        w_avg = pw.m["cat_totals"].get(k, np.nan) / pw.m["n_valid"]
        body.append(
            "<tr>"
            f"<td>{html.escape(CAT_LABELS[k])}</td>"
            f"<td>{fmt_metric(fv, 'pct')}</td>"
            f"<td>{fmt_metric(wv, 'pct')}</td>"
            f"<td>{fmt_delta(fv, wv, 'pct')}</td>"
            f"<td>{fmt_metric(f_avg, 'money')}</td>"
            f"<td>{fmt_metric(w_avg, 'money')}</td>"
            "</tr>"
        )
    head = (
        "<tr><th>消费类目</th><th>Fundo 份额</th><th>WageGo 份额</th><th>差异</th>"
        "<th>Fundo 人均 6 个月支出</th><th>WageGo 人均 6 个月支出</th></tr>"
    )
    return f'<table class="cmp cat"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def balance_bucket_table(pf: Profile, pw: Profile) -> str:
    body = []
    for (label, f_share), (_, w_share) in zip(pf.m["balance_buckets"], pw.m["balance_buckets"]):
        body.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{fmt_pct(f_share)}</td>"
            f"<td>{fmt_pct(w_share)}</td>"
            f"<td>{fmt_delta(f_share, w_share, 'pct')}</td>"
            "</tr>"
        )
    head = (
        "<tr><th>余额区间</th>"
        f'<th><span class="dot" style="background:{POP_COLORS["Fundo"]}"></span>Fundo</th>'
        f'<th><span class="dot" style="background:{POP_COLORS["WageGo"]}"></span>WageGo</th>'
        "<th>差异</th></tr>"
    )
    return f'<table class="cmp"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


def segment_table(pf: Profile, pw: Profile) -> str:
    meta = {
        "盈余型": "good",
        "平衡型": "warning",
        "赤字型": "serious",
        "重压型": "critical",
    }
    body = []
    for name, level in meta.items():
        fs, ws = seg_val(pf, name, "share"), seg_val(pw, name, "share")
        is_key = f"seg:{name}:share" in KEY_METRICS
        row_cls = ' class="hl"' if is_key else ""
        body.append(
            f"<tr{row_cls}>"
            f"<td>{status_chip(level, name)}</td>"
            f"<td>{fmt_pct(fs)}</td>"
            f"<td>{fmt_pct(ws)}</td>"
            f"<td>{fmt_delta(fs, ws, 'pct')}</td>"
            f"<td>{fmt_money(seg_val(pf, name, 'sur_m'))}</td>"
            f"<td>{fmt_money(seg_val(pw, name, 'sur_m'))}</td>"
            f"<td>{fmt_pct(seg_val(pf, name, 'dishon'))}</td>"
            f"<td>{fmt_pct(seg_val(pw, name, 'dishon'))}</td>"
            "</tr>"
        )
    head = (
        "<tr><th>分层</th><th>Fundo 占比</th><th>WageGo 占比</th><th>差异</th>"
        "<th>Fundo 月盈余</th><th>WageGo 月盈余</th>"
        "<th>Fundo 拒付率</th><th>WageGo 拒付率</th></tr>"
    )
    return f'<table class="seg-table"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'


# ---------------------------------------------------------------------------
# 指标清单
# ---------------------------------------------------------------------------
OVERVIEW_ROWS = [
    ("申请数", "n_apps", "int"),
    ("去重用户数", "n_users", "int"),
    ("有效流水用户", "n_valid", "int"),
    ("有效率", "coverage", "pct"),
    ("样本月份", "sample_month", "text"),
    ("用户类型", "user_type_label", "text"),
    ("中位交易笔数（6 个月）", "median_txn_cnt", "num0"),
    ("中位月交易笔数", "median_txn_pm", "num0"),
    ("中位活跃天数", "median_active_days", "num0"),
]

DISTRIBUTION_ROWS = [
    ("月收入 P25", "inc_p25", "money"),
    ("月收入中位", "inc_p50", "money"),
    ("月收入 P75", "inc_p75", "money"),
    ("月支出 P25", "exp_p25", "money"),
    ("月支出中位", "exp_p50", "money"),
    ("月支出 P75", "exp_p75", "money"),
    ("月盈余中位", "sur_p50", "money"),
    ("盈余率中位", "sur_rate_p50", "pct"),
    ("账户余额 P25", "bal_p25", "money"),
    ("账户余额中位", "bal_p50", "money"),
    ("账户余额 P75", "bal_p75", "money"),
]

INCOME_ROWS = [
    ("中位月收入", "inc_p50", "money"),
    ("月收入 P25", "inc_p25", "money"),
    ("月收入 P75", "inc_p75", "money"),
    ("有工资收入占比", "wage_incidence", "pct"),
    ("工资占入账金额", "wage_share_agg", "pct"),
    ("工资中位月收入（有工资）", "wage_median_m", "money"),
    ("有福利收入占比", "cent_incidence", "pct"),
    ("福利占入账金额", "cent_share_agg", "pct"),
    ("其他入账占入账金额", "other_share_agg", "pct"),
    ("入账 CV 中位", "income_cv_p50", "num"),
    ("工资 CV 中位", "wage_cv_p50", "num"),
    ("近 3 月工资较近 6 月下降 >10%", "wage_3m6m_decline", "pct"),
    ("收入来源数中位", "income_source_p50", "num0"),
    ("可识别雇主占比", "employer_share", "pct"),
    ("有雇主用户中周薪制占比", "pay_weekly", "pct"),
    ("有雇主用户中双周薪制占比", "pay_fortnightly", "pct"),
    ("主行业", "industry_top_text", "text"),
]

EXPENSE_ROWS = [
    ("中位月支出", "exp_p50", "money"),
    ("月支出 P25", "exp_p25", "money"),
    ("月支出 P75", "exp_p75", "money"),
    ("可识别消费中位月额", "cons_p50", "money"),
    ("可识别消费 / 总支出（用户中位）", "cons_share_exp_p50", "pct"),
    ("外部转账 / 总支出", "ext_share_agg", "pct"),
    ("外部转账 / 全部借记", "ext_transfer_share_global", "pct"),
    ("内部转账 / 全部借记", "internal_transfer_share_global", "pct"),
    ("必需品占可识别消费", "ess_share_cons", "pct"),
    ("可选消费占可识别消费", "dis_share_cons", "pct"),
    ("前 3 类消费占比中位", "cat_top3_p50", "pct"),
    ("消费集中度 HHI 中位", "cat_hhi_p50", "num"),
    ("债务相关支出占比中位", "debt_share_p50", "pct"),
    ("高承诺支出 / 收入中位", "high_commit_share_p50", "pct"),
]

LIABILITY_ROWS = [
    ("有借贷放款", "any_disburse", "pct"),
    ("有还款", "any_repay", "pct"),
    ("现金贷 / 工资预支使用率", "adv_use", "pct"),
    ("分期贷使用率", "loan_use", "pct"),
    ("BNPL 使用率", "bnpl_use", "pct"),
    ("银行借贷使用率", "bank_use", "pct"),
    ("放款金额结构", "product_mix_text", "text"),
    ("净借款人占比", "net_borrow_share", "pct"),
    ("中位月还款额", "rep_m_p50", "money"),
    ("偿债 / 收入中位", "rep_rate_p50", "pct"),
    ("偿债 / 收入 P90", "rep_rate_p90", "pct"),
    ("中位月新增借款", "draw_m_p50", "money"),
    ("借款 / 收入中位", "draw_rate_p50", "pct"),
    ("有拒付记录", "dish_any", "pct"),
    ("未解决拒付", "dish_unres", "pct"),
    ("拒付解决率", "dish_resolution_rate", "pct"),
    ("催收类拒付发生率", "collections_inc", "pct"),
    ("债务整合发生率", "debt_consolidation_inc", "pct"),
]

CASHFLOW_ROWS = [
    ("偿债前现金流中位（月）", "pre_debt_cf_p50", "money"),
    ("偿债前收不抵支占比", "pre_debt_neg", "pct"),
    ("中位月盈余", "sur_p50", "money"),
    ("负盈余占比", "sur_neg", "pct"),
    ("中位盈余率", "sur_rate_p50", "pct"),
    ("双周盈余为负占比", "fortnight_neg", "pct"),
    ("账户储蓄率为负占比", "savings_neg", "pct"),
    ("中位账户余额", "bal_p50", "money"),
    ("账户余额 P25", "bal_p25", "money"),
    ("账户余额 P75", "bal_p75", "money"),
    ("透支用户占比", "bal_neg", "pct"),
    ("余额 < $100 用户占比", "bal_lt100", "pct"),
    ("缓冲天数中位", "buffer_p50", "num1"),
    ("缓冲 < 7 天占比", "buffer_lt7", "pct"),
    ("缓冲 < 30 天占比", "buffer_lt30", "pct"),
    ("余额正负切换中位（次 / 6 个月）", "sign_change_p50", "num0"),
]

RISK_ROWS = [
    ("赌博交易发生率", "gamb_incidence", "pct"),
    ("赌博占可识别消费", "gamb_share_cons", "pct"),
    ("赌博单笔 P99", "gamb_p99", "money"),
    ("手续费支出发生率", "fees_incidence", "pct"),
    ("手续费 / 透支交易发生率", "fees_ticket_incidence", "pct"),
    ("月大额支出笔数中位", "large_expense_cnt_pm_p50", "num1"),
    ("月交易笔数中位", "median_txn_pm", "num0"),
    ("周末消费占比中位", "weekend_share_p50", "pct"),
]

KPI_SPECS = [
    ("中位月收入", "inc_p50", "money"),
    ("中位月支出", "exp_p50", "money"),
    ("中位月盈余", "sur_p50", "money"),
    ("负盈余用户", "sur_neg", "pct"),
    ("中位账户余额", "bal_p50", "money"),
    ("有借贷放款", "any_disburse", "pct"),
    ("有拒付记录", "dish_any", "pct"),
    ("流动性缓冲 < 7 天", "buffer_lt7", "pct"),
]

# 明细表中高亮这些指标行
KEY_METRICS = {
    "n_valid",
    "inc_p50",
    "exp_p50",
    "wage_incidence",
    "wage_share_agg",
    "ext_transfer_share_global",
    "internal_transfer_share_global",
    "any_disburse",
    "adv_use",
    "rep_rate_p50",
    "dish_any",
    "dish_unres",
    "dish_resolution_rate",
    "sur_neg",
    "bal_p50",
    "buffer_lt7",
    "gamb_incidence",
    "gamb_share_cons",
    "fees_incidence",
    "collections_inc",
    "seg:重压型:share",
    "seg:盈余型:share",
}

DIFF_OVERVIEW = [
    ("负盈余占比", "sur_neg"),
    ("缓冲 < 7 天", "buffer_lt7"),
    ("有借贷放款", "any_disburse"),
    ("现金贷使用率", "adv_use"),
    ("BNPL 使用率", "bnpl_use"),
    ("有拒付记录", "dish_any"),
    ("未解决拒付", "dish_unres"),
    ("赌博交易发生率", "gamb_incidence"),
    ("手续费发生率", "fees_incidence"),
    ("净借款人占比", "net_borrow_share"),
    ("工资占入账金额", "wage_share_agg"),
    ("福利收入发生率", "cent_incidence"),
    ("重压型占比", "seg:重压型:share"),
    ("盈余型占比", "seg:盈余型:share"),
]


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def add_grouped_bars(
    fig,
    labels: list[str],
    f_vals: list[float],
    w_vals: list[float],
    row: int = 1,
    col: int = 1,
    kind: str = "pct",
    digits: int = 0,
    showlegend: bool = True,
    highlight: set[str] | None = None,
) -> None:
    highlight = highlight or set()
    for name, vals in (("Fundo", f_vals), ("WageGo", w_vals)):
        texts = []
        for v in vals:
            if not valid(v):
                texts.append("")
            elif kind == "money":
                texts.append(fmt_money(float(v)))
            else:
                texts.append(f"{float(v):.{digits}f}%")
        if kind == "money":
            hover = f"{name}：%{{x:$,.0f}}<extra>%{{y}}</extra>"
        else:
            hover = f"{name}：%{{x:.{digits}f}}%<extra>%{{y}}</extra>"
        marker = dict(color=POP_COLORS[name], cornerradius=3)
        if highlight:
            marker["line"] = dict(
                color=[HILITE if lab in highlight else "rgba(0,0,0,0)" for lab in labels],
                width=[2.5 if lab in highlight else 0 for lab in labels],
            )
        trace = go.Bar(
            y=labels,
            x=[float(v) if valid(v) else None for v in vals],
            orientation="h",
            name=name,
            marker=marker,
            text=texts,
            textposition="outside",
            cliponaxis=False,
            textfont=dict(color=INK2, size=10.5),
            showlegend=showlegend,
            legendgroup=name,
            hovertemplate=hover,
        )
        if getattr(fig, "_grid_ref", None) is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)


def chart_income_mix(pf: Profile, pw: Profile) -> str:
    labels = ["工资", "福利", "其他入账"]
    fm, wm = pf.m, pw.m
    f_inc = [fm["wage_incidence"] * 100, fm["cent_incidence"] * 100, fm["other_incidence"] * 100]
    w_inc = [wm["wage_incidence"] * 100, wm["cent_incidence"] * 100, wm["other_incidence"] * 100]
    f_amt = [fm["wage_share_agg"] * 100, fm["cent_share_agg"] * 100, fm["other_share_agg"] * 100]
    w_amt = [wm["wage_share_agg"] * 100, wm["cent_share_agg"] * 100, wm["other_share_agg"] * 100]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.16,
        subplot_titles=("有该类收入的用户占比", "收入金额占比"),
    )
    add_grouped_bars(fig, labels, f_inc, w_inc, row=1, col=1, showlegend=True, highlight={"工资"})
    add_grouped_bars(fig, labels, f_amt, w_amt, row=1, col=2, showlegend=False, highlight={"工资"})

    mx = max([v for v in f_inc + w_inc + f_amt + w_amt if valid(v)] + [0]) * 1.22
    fig.update_xaxes(range=[0, mx], ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False)
    base_layout(fig, height=420, margin=dict(l=70, r=22, t=64, b=74), showlegend=True)
    fig.update_layout(
        barmode="group",
        bargap=0.34,
        bargroupgap=0.08,
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.2, yanchor="top",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_annotations(font=dict(size=12, color=INK2))
    return fig_html(fig, 420)


def chart_distributions(pf: Profile, pw: Profile) -> str:
    specs = [
        ("月收入", "inc_m"),
        ("月支出", "exp_m"),
        ("月盈余", "sur_m"),
        ("账户余额", "bal"),
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.14,
        vertical_spacing=0.22,
        subplot_titles=[s[0] for s in specs],
    )
    for i, (_, col) in enumerate(specs):
        r, c = i // 2 + 1, i % 2 + 1
        combined = []
        for name, p in (("Fundo", pf), ("WageGo", pw)):
            vals = pd.to_numeric(p.d[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            combined.append(vals)
            fig.add_trace(
                go.Box(
                    x=vals,
                    y=[name] * len(vals),
                    name=name,
                    orientation="h",
                    marker=dict(color=POP_COLORS[name], size=4),
                    line=dict(color=POP_COLORS[name], width=1.5),
                    fillcolor=rgba(POP_COLORS[name], 0.12),
                    boxpoints=False,
                    whiskerwidth=0.35,
                    showlegend=(i == 0),
                    legendgroup=name,
                    hovertemplate=f"{name}<br>%{{x:$,.0f}}<extra></extra>",
                ),
                row=r,
                col=c,
            )
        allv = pd.concat(combined, ignore_index=True).dropna()
        if len(allv):
            lo, hi = np.nanpercentile(allv, [1, 99])
            if col in {"sur_m", "bal"}:
                lo = min(lo, 0.0)
            pad = (hi - lo) * 0.1 or 1.0
            fig.update_xaxes(
                range=[lo - pad, hi + pad],
                tickprefix="$",
                tickformat=",",
                row=r,
                col=c,
            )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=POP_Y_ORDER,
        showgrid=False,
    )
    base_layout(fig, height=560, margin=dict(l=76, r=26, t=68, b=58), showlegend=True)
    fig.update_layout(
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.12, yanchor="top",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        )
    )
    fig.update_annotations(font=dict(size=12, color=INK2))
    return fig_html(fig, 560)


def chart_expense_structure(pf: Profile, pw: Profile) -> str:
    labels = [
        "可识别消费 / 总支出",
        "必需品 / 可识别消费",
        "可选消费 / 可识别消费",
        "外部转账 / 总支出",
        "外部转账 / 全部借记",
        "内部转账 / 全部借记",
        "债务相关支出占比",
        "高承诺支出 / 收入",
    ]
    keys = [
        "cons_share_exp_p50",
        "ess_share_cons",
        "dis_share_cons",
        "ext_share_agg",
        "ext_transfer_share_global",
        "internal_transfer_share_global",
        "debt_share_p50",
        "high_commit_share_p50",
    ]
    f_vals = [pf.m[k] * 100 for k in keys]
    w_vals = [pw.m[k] * 100 for k in keys]

    fig = go.Figure()
    add_grouped_bars(fig, labels, f_vals, w_vals, kind="pct", digits=0,
                     highlight={"外部转账 / 全部借记", "内部转账 / 全部借记"})
    mx = max([v for v in f_vals + w_vals if valid(v)] + [0]) * 1.24
    fig.update_xaxes(range=[0, mx], ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False)
    base_layout(fig, height=420, margin=dict(l=150, r=24, t=44, b=70), showlegend=True)
    fig.update_layout(
        barmode="group",
        bargap=0.34,
        bargroupgap=0.08,
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.22, yanchor="top",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig_html(fig, 420)


def chart_consumption_dumbbell(pf: Profile, pw: Profile, top_n: int = 12) -> str:
    shares_f, shares_w = pf.m["cat_shares"], pw.m["cat_shares"]
    keys = sorted(
        shares_f,
        key=lambda k: max(
            shares_f.get(k, 0) if valid(shares_f.get(k)) else 0,
            shares_w.get(k, 0) if valid(shares_w.get(k)) else 0,
        ),
        reverse=True,
    )[:top_n]
    labels = [CAT_LABELS[k] for k in keys]
    f_vals = [shares_f[k] * 100 if valid(shares_f.get(k)) else np.nan for k in keys]
    w_vals = [shares_w[k] * 100 if valid(shares_w.get(k)) else np.nan for k in keys]

    fig = go.Figure()
    for label, fv, wv in zip(labels, f_vals, w_vals):
        if valid(fv) and valid(wv):
            fig.add_trace(
                go.Scatter(
                    x=[fv, wv], y=[label, label], mode="lines",
                    line=dict(color=BASELINE, width=2),
                    hoverinfo="skip", showlegend=False,
                )
            )
    for name, vals in (("Fundo", f_vals), ("WageGo", w_vals)):
        fig.add_trace(
            go.Scatter(
                x=vals,
                y=labels,
                mode="markers",
                name=name,
                marker=dict(
                    color=POP_COLORS[name],
                    size=9,
                    line=dict(
                        color=[HILITE if lab == "赌博" else SURFACE for lab in labels],
                        width=[2.8 if lab == "赌博" else 2 for lab in labels],
                    ),
                ),
                hovertemplate=f"{name}：%{{x:.1f}}%<extra>%{{y}}</extra>",
            )
        )

    mx = max([v for v in f_vals + w_vals if valid(v)] + [0])
    for label, fv, wv in zip(labels, f_vals, w_vals):
        if valid(fv) and valid(wv) and mx:
            diff = wv - fv
            fig.add_annotation(
                x=max(fv, wv) + mx * 0.03,
                y=label,
                text=f"{diff:+.1f}pp",
                showarrow=False,
                xanchor="left",
                font=dict(color=INK2, size=10.5),
            )
    fig.update_xaxes(range=[0, mx * 1.24], ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False)
    base_layout(fig, height=520, margin=dict(l=96, r=64, t=40, b=70), showlegend=True)
    fig.update_layout(
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.14, yanchor="top",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        )
    )
    return fig_html(fig, 520)


def chart_liability(pf: Profile, pw: Profile) -> str:
    use_labels = ["有放款", "有还款", "现金贷", "分期贷", "BNPL", "银行"]
    use_keys = ["any_disburse", "any_repay", "adv_use", "loan_use", "bnpl_use", "bank_use"]
    dish_labels = ["任一拒付", "未解决拒付", "现金贷拒付", "分期贷拒付", "BNPL 拒付", "银行拒付", "催收类拒付"]
    dish_keys = [
        "dish_any",
        "dish_unres",
        "dish_advance",
        "dish_personal_loan",
        "dish_bnpl",
        "dish_bank",
        "dish_collection",
    ]
    f_use = [pf.m[k] * 100 for k in use_keys]
    w_use = [pw.m[k] * 100 for k in use_keys]
    f_dish = [pf.m[k] * 100 for k in dish_keys]
    w_dish = [pw.m[k] * 100 for k in dish_keys]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.22,
        subplot_titles=("借贷使用率", "拒付发生率（有对应记录的用户占比）"),
    )
    add_grouped_bars(fig, use_labels, f_use, w_use, row=1, col=1, showlegend=True,
                     highlight={"有放款", "现金贷"})
    add_grouped_bars(fig, dish_labels, f_dish, w_dish, row=1, col=2, showlegend=False,
                     highlight={"任一拒付", "未解决拒付"})

    mx = max([v for v in f_use + w_use + f_dish + w_dish if valid(v)] + [0]) * 1.24
    fig.update_xaxes(range=[0, mx], ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(use_labels)), row=1, col=1, showgrid=False)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(dish_labels)), row=1, col=2, showgrid=False)
    base_layout(fig, height=460, margin=dict(l=86, r=22, t=64, b=74), showlegend=True)
    fig.update_layout(
        barmode="group",
        bargap=0.34,
        bargroupgap=0.08,
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.18, yanchor="top",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_annotations(font=dict(size=12, color=INK2))
    return fig_html(fig, 460)


def chart_cashflow(pf: Profile, pw: Profile) -> str:
    labels = [
        "偿债前收不抵支",
        "负盈余",
        "双周盈余为负",
        "储蓄率为负",
        "余额 < $100",
        "缓冲 < 7 天",
        "缓冲 < 30 天",
    ]
    keys = [
        "pre_debt_neg",
        "sur_neg",
        "fortnight_neg",
        "savings_neg",
        "bal_lt100",
        "buffer_lt7",
        "buffer_lt30",
    ]
    f_vals = [pf.m[k] * 100 for k in keys]
    w_vals = [pw.m[k] * 100 for k in keys]

    fig = go.Figure()
    add_grouped_bars(fig, labels, f_vals, w_vals, highlight={"负盈余", "缓冲 < 7 天"})
    mx = max([v for v in f_vals + w_vals if valid(v)] + [0]) * 1.22
    fig.update_xaxes(range=[0, mx], ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False)
    base_layout(fig, height=400, margin=dict(l=130, r=24, t=44, b=70), showlegend=True)
    fig.update_layout(
        barmode="group",
        bargap=0.34,
        bargroupgap=0.08,
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.2, yanchor="top",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig_html(fig, 400)


def chart_balance_buckets(pf: Profile, pw: Profile) -> str:
    buckets = [lab for lab, _ in pf.m["balance_buckets"]]
    ramp = SEQ_BLUE[: len(buckets)]
    fig = go.Figure()
    for i, bucket in enumerate(buckets):
        f_share = pf.m["balance_buckets"][i][1]
        w_share = pw.m["balance_buckets"][i][1]
        text_color = INK if i < 2 else "#ffffff"
        fig.add_trace(
            go.Bar(
                y=POP_Y_ORDER,
                x=[w_share * 100, f_share * 100],
                orientation="h",
                name=bucket,
                marker=dict(color=ramp[i], line=dict(color=SURFACE, width=2)),
                text=[
                    f"{v * 100:.0f}%" if valid(v) and v * 100 >= 4 else ""
                    for v in (w_share, f_share)
                ],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color=text_color, size=11),
                hovertemplate=f"{bucket}：%{{x:.1f}}%<extra>%{{y}}</extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=POP_Y_ORDER, showgrid=False)
    base_layout(fig, height=280, margin=dict(l=70, r=24, t=40, b=70), showlegend=True)
    fig.update_layout(
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.3, yanchor="top",
            font=dict(size=10.5), bgcolor="rgba(0,0,0,0)",
        )
    )
    return fig_html(fig, 280)


def chart_segments(pf: Profile, pw: Profile) -> str:
    fig = go.Figure()
    for segment, (color, text_color) in SEG_COLORS.items():
        f_share = seg_val(pf, segment, "share")
        w_share = seg_val(pw, segment, "share")
        is_key = segment == "重压型"
        fig.add_trace(
            go.Bar(
                y=POP_Y_ORDER,
                x=[w_share * 100, f_share * 100],
                orientation="h",
                name=segment,
                marker=dict(
                    color=color,
                    line=dict(color=HILITE if is_key else SURFACE, width=2.8 if is_key else 2),
                ),
                text=[
                    f"{v:.0f}%" if valid(v) and v >= 4 else ""
                    for v in (w_share * 100, f_share * 100)
                ],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color=text_color, size=11),
                hovertemplate=f"{segment}：%{{x:.1f}}%<extra>%{{y}}</extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=POP_Y_ORDER, showgrid=False)
    base_layout(fig, height=300, margin=dict(l=70, r=24, t=40, b=78), showlegend=True)
    fig.update_layout(
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.26, yanchor="top",
            font=dict(size=10.5), bgcolor="rgba(0,0,0,0)",
        )
    )
    return fig_html(fig, 300)


def chart_segment_surplus(pf: Profile, pw: Profile) -> str:
    labels = list(SEG_COLORS.keys())
    f_vals = [seg_val(pf, s, "sur_m") for s in labels]
    w_vals = [seg_val(pw, s, "sur_m") for s in labels]

    fig = go.Figure()
    add_grouped_bars(fig, labels, f_vals, w_vals, kind="money", highlight={"重压型"})
    values = [v for v in f_vals + w_vals if valid(v)]
    if values:
        lo, hi = min(values), max(values)
        span = (hi - lo) or 1.0
        fig.update_xaxes(
            range=[lo - span * 0.2, hi + span * 0.28],
            tickprefix="$",
            tickformat=",",
        )
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False)
    base_layout(fig, height=340, margin=dict(l=70, r=24, t=40, b=70), showlegend=True)
    fig.update_layout(
        barmode="group",
        bargap=0.36,
        bargroupgap=0.08,
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.24, yanchor="top",
            font=dict(size=10.5), bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig_html(fig, 340)


def chart_risk(pf: Profile, pw: Profile) -> str:
    labels = [
        "赌博交易",
        "赌博占消费",
        "手续费支出",
        "手续费 / 透支交易",
        "催收类拒付",
        "债务整合",
        "高承诺支出 / 收入",
    ]
    keys = [
        "gamb_incidence",
        "gamb_share_cons",
        "fees_incidence",
        "fees_ticket_incidence",
        "collections_inc",
        "debt_consolidation_inc",
        "high_commit_share_p50",
    ]
    f_vals = [pf.m[k] * 100 for k in keys]
    w_vals = [pw.m[k] * 100 for k in keys]

    fig = go.Figure()
    add_grouped_bars(fig, labels, f_vals, w_vals,
                     highlight={"赌博交易", "赌博占消费", "手续费支出", "催收类拒付"})
    mx = max([v for v in f_vals + w_vals if valid(v)] + [0]) * 1.22
    fig.update_xaxes(range=[0, mx], ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False)
    base_layout(fig, height=380, margin=dict(l=130, r=24, t=44, b=70), showlegend=True)
    fig.update_layout(
        barmode="group",
        bargap=0.34,
        bargroupgap=0.08,
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.2, yanchor="top",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig_html(fig, 380)


def chart_difference_overview(pf: Profile, pw: Profile) -> str:
    rows = []
    for label, key in DIFF_OVERVIEW:
        fv, wv = metric_value(pf, key), metric_value(pw, key)
        if valid(fv) and valid(wv):
            rows.append((label, (float(wv) - float(fv)) * 100))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    labels = [r[0] for r in rows]
    diffs = [r[1] for r in rows]
    pos = [d if d > 0 else None for d in diffs]
    neg = [d if d < 0 else None for d in diffs]
    top = set(labels[:3])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=pos,
            orientation="h",
            name="WageGo 更高",
            marker=dict(
                color=POP_COLORS["WageGo"],
                cornerradius=3,
                line=dict(
                    color=[HILITE if lab in top else "rgba(0,0,0,0)" for lab in labels],
                    width=[2.5 if lab in top else 0 for lab in labels],
                ),
            ),
            text=[f"{d:+.1f}pp" if d is not None else "" for d in pos],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(color=INK2, size=10.5),
            hovertemplate="WageGo 高 %{x:.1f}pp<extra>%{y}</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=neg,
            orientation="h",
            name="Fundo 更高",
            marker=dict(
                color=POP_COLORS["Fundo"],
                cornerradius=3,
                line=dict(
                    color=[HILITE if lab in top else "rgba(0,0,0,0)" for lab in labels],
                    width=[2.5 if lab in top else 0 for lab in labels],
                ),
            ),
            text=[f"{d:+.1f}pp" if d is not None else "" for d in neg],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(color=INK2, size=10.5),
            hovertemplate="Fundo 高 %{x:.1f}pp<extra>%{y}</extra>",
        )
    )
    span = max([abs(d) for d in diffs] + [0]) * 1.32 or 1
    fig.update_xaxes(
        range=[-span, span],
        ticksuffix="pp",
        zeroline=True,
        zerolinecolor=BASELINE,
        zerolinewidth=1,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False)
    base_layout(fig, height=max(420, 30 * len(labels) + 130), margin=dict(l=140, r=30, t=46, b=70), showlegend=True)
    fig.update_layout(
        barmode="relative",
        bargap=0.32,
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.14, yanchor="top",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig_html(fig, max(420, 30 * len(labels) + 130))


# ---------------------------------------------------------------------------
# 主要差异文字
# ---------------------------------------------------------------------------
def build_findings(pf: Profile, pw: Profile) -> list[str]:
    fm, wm = pf.m, pw.m
    parts = []

    parts.append(
        f"<b>收入水平接近，WageGo 的工资覆盖更广。</b> Fundo 中位月收入 {fmt_money(fm['inc_p50'])}，"
        f"WageGo {fmt_money(wm['inc_p50'])}。有工资收入的比例 Fundo {fmt_pct(fm['wage_incidence'])}、"
        f"WageGo {fmt_pct(wm['wage_incidence'])}；工资占入账金额分别为 "
        f"{fmt_pct(fm['wage_share_agg'])} 与 {fmt_pct(wm['wage_share_agg'])}，"
        f"其余主要来自“其他入账”（{fmt_pct(fm['other_share_agg'])} vs {fmt_pct(wm['other_share_agg'])}）。"
    )

    cat_gaps = []
    for k, fv in fm["cat_shares"].items():
        wv = wm["cat_shares"].get(k, np.nan)
        if valid(fv) and valid(wv):
            cat_gaps.append((k, float(fv), float(wv), float(wv) - float(fv)))
    top_cat = ""
    if cat_gaps:
        k, fv, wv, _ = max(cat_gaps, key=lambda x: abs(x[3]))
        top_cat = (
            f"消费类目份额差异最大的是{CAT_LABELS[k]}"
            f"（Fundo {fmt_pct(fv)} vs WageGo {fmt_pct(wv)}）。"
        )
    parts.append(
        f"<b>支出规模接近，转账与消费结构存在差异。</b> 中位月支出 Fundo {fmt_money(fm['exp_p50'])}、"
        f"WageGo {fmt_money(wm['exp_p50'])}；可识别消费占总支出 "
        f"{fmt_pct(fm['cons_share_exp_p50'])} vs {fmt_pct(wm['cons_share_exp_p50'])}。"
        f"外部转账占全部借记 {fmt_pct(fm['ext_transfer_share_global'])} vs "
        f"{fmt_pct(wm['ext_transfer_share_global'])}，内部转账 "
        f"{fmt_pct(fm['internal_transfer_share_global'])} vs "
        f"{fmt_pct(wm['internal_transfer_share_global'])}。{top_cat}"
    )

    parts.append(
        f"<b>两组均以负盈余为主，账户缓冲普遍不足。</b> 中位月盈余 Fundo {fmt_money(fm['sur_p50'])}、"
        f"WageGo {fmt_money(wm['sur_p50'])}；负盈余用户占比 "
        f"{fmt_pct(fm['sur_neg'])} vs {fmt_pct(wm['sur_neg'])}。中位账户余额 "
        f"{fmt_money(fm['bal_p50'])} vs {fmt_money(wm['bal_p50'])}，"
        f"缓冲不足 7 天的用户占比 {fmt_pct(fm['buffer_lt7'])} vs {fmt_pct(wm['buffer_lt7'])}。"
    )

    parts.append(
        f"<b>WageGo 的借贷参与度和拒付暴露更高。</b> 有借贷放款 "
        f"{fmt_pct(wm['any_disburse'])} vs Fundo {fmt_pct(fm['any_disburse'])}，"
        f"现金贷使用率 {fmt_pct(wm['adv_use'])} vs {fmt_pct(fm['adv_use'])}，"
        f"分期贷 {fmt_pct(wm['loan_use'])} vs {fmt_pct(fm['loan_use'])}。"
        f"偿债 / 收入中位 {fmt_pct(wm['rep_rate_p50'])} vs {fmt_pct(fm['rep_rate_p50'])}；"
        f"有拒付记录 {fmt_pct(wm['dish_any'])} vs {fmt_pct(fm['dish_any'])}，"
        f"未解决拒付 {fmt_pct(wm['dish_unres'])} vs {fmt_pct(fm['dish_unres'])}，"
        f"解决率 {fmt_pct(wm['dish_resolution_rate'])} vs {fmt_pct(fm['dish_resolution_rate'])}。"
    )

    parts.append(
        f"<b>WageGo 的赌博、手续费与催收暴露更高。</b> 赌博交易发生率 "
        f"{fmt_pct(wm['gamb_incidence'])} vs {fmt_pct(fm['gamb_incidence'])}，"
        f"赌博占可识别消费 {fmt_pct(wm['gamb_share_cons'])} vs {fmt_pct(fm['gamb_share_cons'])}；"
        f"手续费发生率 {fmt_pct(wm['fees_incidence'])} vs {fmt_pct(fm['fees_incidence'])}，"
        f"手续费 / 透支类交易 {fmt_pct(wm['fees_ticket_incidence'])} vs "
        f"{fmt_pct(fm['fees_ticket_incidence'])}，催收类拒付 "
        f"{fmt_pct(wm['collections_inc'])} vs {fmt_pct(fm['collections_inc'])}。"
    )

    f_heavy, w_heavy = seg_val(pf, "重压型", "share"), seg_val(pw, "重压型", "share")
    f_surplus, w_surplus = seg_val(pf, "盈余型", "share"), seg_val(pw, "盈余型", "share")
    parts.append(
        f"<b>WageGo 的分层结构略好，但重压型仍占两成以上。</b> 盈余型占比 Fundo "
        f"{fmt_pct(f_surplus)}、WageGo {fmt_pct(w_surplus)}；重压型占比 "
        f"{fmt_pct(f_heavy)} vs {fmt_pct(w_heavy)}，重压型中位月盈余 "
        f"{fmt_money(seg_val(pf, '重压型', 'sur_m'))} vs {fmt_money(seg_val(pw, '重压型', 'sur_m'))}。"
    )

    parts.append(
        "<b>比较口径提示。</b> Fundo 样本全部为 2026-09，WageGo 覆盖 2026-08 至 2026-10；"
        "两者均为新客样本且相互独立。以上差异为描述性对比，未做统计显著性检验，也不代表因果。"
    )
    return parts


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = fpr.CSS + """
.kpi-grid.compare { grid-template-columns: repeat(auto-fit, minmax(235px, 1fr)); }
.kpi .cmp { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-top: 6px; }
.kpi .cmp .who { color: var(--ink2); font-size: 11.5px; display: inline-flex; align-items: center; gap: 6px; }
.kpi .cmp .v { font-size: 18px; font-weight: 650; letter-spacing: -.01em; }
.kpi .delta-line { color: var(--muted); font-size: 11.5px; margin-top: 6px; }
.delta { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
.delta.muted { color: var(--muted); }
table.cmp td:nth-child(1) { width: 36%; }
table.cmp tr.group td {
  background: var(--page); color: var(--ink2); font-weight: 650; font-size: 11px;
  letter-spacing: .08em; padding-top: 12px; border-bottom: 1px solid var(--grid);
}
table.cmp tbody tr:hover td { background: #f4f4f1; }
table.seg-table th, table.seg-table td { white-space: nowrap; }
.chart-grid.even { grid-template-columns: 1fr 1fr; }
.star { color: #a16207; font-size: 12px; line-height: 1; margin-right: 4px; }
h2 { flex-wrap: wrap; }
.key-conclusion {
  display: flex; gap: 8px; align-items: flex-start;
  margin: 4px 0 14px; padding: 9px 12px;
  border-left: 3px solid #d9a441; background: #fdf9ec;
  border-radius: 0 8px 8px 0; color: var(--ink);
  font-size: 13.5px; line-height: 1.6;
}
.key-conclusion .star { margin-top: 2px; flex: none; font-size: 14px; }
mark.hl {
  background: #fbeec4; color: inherit; padding: 0 2px;
  border-radius: 3px; font-weight: 650;
}
table.cmp tr.hl td { background: #fdf9ec; }
table.cmp tr.hl td:first-child { font-weight: 650; }
table.cmp tbody tr.hl:hover td { background: #f7efd7; }
@media (max-width: 900px) { .chart-grid.even { grid-template-columns: 1fr; } }
"""


def kpi_html(pf: Profile, pw: Profile) -> str:
    cards = []
    for label, key, kind in KPI_SPECS:
        fv, wv = metric_value(pf, key), metric_value(pw, key)
        cards.append(
            '<div class="kpi">'
            f'<div class="label">{html.escape(label)}</div>'
            f'<div class="cmp"><span class="who"><span class="dot" style="background:{POP_COLORS["Fundo"]}"></span>Fundo</span>'
            f'<span class="v">{fmt_metric(fv, kind)}</span></div>'
            f'<div class="cmp"><span class="who"><span class="dot" style="background:{POP_COLORS["WageGo"]}"></span>WageGo</span>'
            f'<span class="v">{fmt_metric(wv, kind)}</span></div>'
            f'<div class="delta-line">{fmt_delta(fv, wv, kind)}</div>'
            "</div>"
        )
    return "".join(cards)


def build_html(pf: Profile, pw: Profile, figs: dict[str, str], generated_at: str) -> str:
    plotly_js = get_plotlyjs()
    findings = "".join(
        f'<div class="finding"><div class="n">{i + 1:02d}</div><p>{text}</p></div>'
        for i, text in enumerate(build_findings(pf, pw))
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fundo vs WageGo 人群画像对比</title>
<script>{plotly_js}</script>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">FUNDO × WAGEGO · PROFILE COMPARISON</div>
  <h1>Fundo vs WageGo 人群画像对比</h1>
  <p class="subtitle">同一套银行流水特征口径 · {fpr.WINDOW_DAYS} 天观察窗口 · 两组独立新客样本</p>
  <div class="chips">
    <span class="chip"><span class="dot" style="background:{POP_COLORS['Fundo']}"></span>Fundo 申请 <b>{pf.m['n_apps']:,}</b> · 用户 <b>{pf.m['n_users']:,}</b> · 有效 <b>{pf.m['n_valid']:,}</b>（{fmt_pct(pf.m['coverage'])}）</span>
    <span class="chip"><span class="dot" style="background:{POP_COLORS['WageGo']}"></span>WageGo 申请 <b>{pw.m['n_apps']:,}</b> · 用户 <b>{pw.m['n_users']:,}</b> · 有效 <b>{pw.m['n_valid']:,}</b>（{fmt_pct(pw.m['coverage'])}）</span>
    <span class="chip">生成时间 <b>{html.escape(generated_at)}</b></span>
  </div>
  <p class="note">提示：Fundo 样本全部来自 2026-09，WageGo 覆盖 2026-08 至 2026-10，月份构成差异可能影响部分指标；图表可悬停查看明细。<span class="star">★</span>各节开头的金色结论框为该节结论；金色高亮为支撑结论的数据。</p>

  <div class="kpi-grid compare" style="margin-top:18px">{kpi_html(pf, pw)}</div>

  <section class="section">
    <h2><span class="num">0</span>样本概览</h2>
    <div class="key-conclusion"><span class="star">★</span><div>WageGo 有效流水覆盖率 <mark class="hl">{fmt_pct(pw.m['coverage'])}</mark>（Fundo <mark class="hl">{fmt_pct(pf.m['coverage'])}</mark>），有效用户 <mark class="hl">{pw.m['n_valid']:,} vs {pf.m['n_valid']:,}</mark>；两组样本月份不完全重叠，后续差异需结合这一背景解读。</div></div>
    <div class="table-card">{comp_table(OVERVIEW_ROWS, pf, pw)}</div>
    <p class="note">有效样本要求 txn_raw_input_cnt &gt; 0，且收入 / 支出 / 盈余非缺失、收入 &gt; 0；同一 user_id 保留交易笔数最多的申请。</p>
  </section>

  <section class="section">
    <h2><span class="num">1</span>核心财务指标分布</h2>
    <div class="key-conclusion"><span class="star">★</span><div>两组收入与支出分布接近（中位月收入 <mark class="hl">{fmt_money(pw.m['inc_p50'])} vs {fmt_money(pf.m['inc_p50'])}</mark>、月支出 <mark class="hl">{fmt_money(pw.m['exp_p50'])} vs {fmt_money(pf.m['exp_p50'])}</mark>），盈余整体位于零以下，余额集中在低位。</div></div>
    <div class="chart">{figs['distributions']}<p class="note">箱体为 P25–P75，横线为中位数，须线为 1.5×IQR；为便于阅读，横轴裁至两组合并后的 P1–P99。</p></div>
    <details>
      <summary>查看指标明细</summary>
      <div class="table-card">{comp_table(DISTRIBUTION_ROWS, pf, pw)}</div>
    </details>
  </section>

  <section class="section">
    <h2><span class="num">2</span>收入与就业</h2>
    <div class="key-conclusion"><span class="star">★</span><div>收入水平接近，WageGo 工资覆盖更广 <mark class="hl">{fmt_pct(pw.m['wage_incidence'])}</mark>（Fundo <mark class="hl">{fmt_pct(pf.m['wage_incidence'])}</mark>），工资占入账 <mark class="hl">{fmt_pct(pw.m['wage_share_agg'])} vs {fmt_pct(pf.m['wage_share_agg'])}</mark>；其他入账占入账金额 <mark class="hl">{fmt_pct(pw.m['other_share_agg'])} vs {fmt_pct(pf.m['other_share_agg'])}</mark>，收入结构高度依赖转账类流水。</div></div>
    <div class="chart">{figs['income_mix']}<p class="note">n = Fundo {pf.m['n_valid']:,} / WageGo {pw.m['n_valid']:,}；发生率以有效用户为分母，金额占比以分项收入（工资 + 福利 + 其他入账）加总为分母。</p></div>
    <ul class="notes">
      <li>有工资收入：Fundo <b>{fmt_pct(pf.m['wage_incidence'])}</b>、WageGo <b>{fmt_pct(pw.m['wage_incidence'])}</b>；工资占入账金额 <b>{fmt_pct(pf.m['wage_share_agg'])}</b> vs <b>{fmt_pct(pw.m['wage_share_agg'])}</b>。</li>
      <li>有福利收入：Fundo <b>{fmt_pct(pf.m['cent_incidence'])}</b>、WageGo <b>{fmt_pct(pw.m['cent_incidence'])}</b>；其他入账占入账金额 <b>{fmt_pct(pf.m['other_share_agg'])}</b> vs <b>{fmt_pct(pw.m['other_share_agg'])}</b>。</li>
      <li>入账 CV 中位：Fundo <b>{fmt_metric(pf.m['income_cv_p50'], 'num')}</b>、WageGo <b>{fmt_metric(pw.m['income_cv_p50'], 'num')}</b>；工资 CV 中位 <b>{fmt_metric(pf.m['wage_cv_p50'], 'num')}</b> vs <b>{fmt_metric(pw.m['wage_cv_p50'], 'num')}</b>，两组入账波动都偏高。</li>
      <li>可识别雇主占比：Fundo <b>{fmt_pct(metric_value(pf, 'employer_share'))}</b>、WageGo <b>{fmt_pct(metric_value(pw, 'employer_share'))}</b>；主行业分别为 <b>{fmt_metric(pf.m['industry_top_text'], 'text')}</b> 与 <b>{fmt_metric(pw.m['industry_top_text'], 'text')}</b>。</li>
    </ul>
    <details>
      <summary>查看指标明细</summary>
      <div class="table-card">{comp_table(INCOME_ROWS, pf, pw)}</div>
    </details>
  </section>

  <section class="section">
    <h2><span class="num">3</span>支出与消费</h2>
    <div class="key-conclusion"><span class="star">★</span><div>支出规模接近，但资金流向不同：Fundo 外部转账占全部借记 <mark class="hl">{fmt_pct(pf.m['ext_transfer_share_global'])}</mark>（WageGo <mark class="hl">{fmt_pct(pw.m['ext_transfer_share_global'])}</mark>），WageGo 内部转账 <mark class="hl">{fmt_pct(pw.m['internal_transfer_share_global'])}</mark>（Fundo <mark class="hl">{fmt_pct(pf.m['internal_transfer_share_global'])}</mark>）；消费类目中仅赌博差异较大（<mark class="hl">{fmt_pct(pw.m['cat_shares']['gambling'])} vs {fmt_pct(pf.m['cat_shares']['gambling'])}</mark>）。</div></div>
    <div class="chart">{figs['expense_structure']}<p class="note">外部 / 内部转账占比以全部借记流水为分母；其余占比的分母见指标名。</p></div>
    <div class="chart">{figs['consumption']}<p class="note">仅展示份额最高的 12 个消费类目；Δ 为 WageGo − Fundo（百分点）；完整类目见明细表。</p></div>
    <ul class="notes">
      <li>可识别消费 / 总支出（用户中位）：Fundo <b>{fmt_pct(pf.m['cons_share_exp_p50'])}</b>、WageGo <b>{fmt_pct(pw.m['cons_share_exp_p50'])}</b>；必需品占可识别消费 <b>{fmt_pct(pf.m['ess_share_cons'])}</b> vs <b>{fmt_pct(pw.m['ess_share_cons'])}</b>，可选消费 <b>{fmt_pct(pf.m['dis_share_cons'])}</b> vs <b>{fmt_pct(pw.m['dis_share_cons'])}</b>。</li>
      <li>外部转账 / 全部借记：Fundo <b>{fmt_pct(pf.m['ext_transfer_share_global'])}</b>、WageGo <b>{fmt_pct(pw.m['ext_transfer_share_global'])}</b>；内部转账 <b>{fmt_pct(pf.m['internal_transfer_share_global'])}</b> vs <b>{fmt_pct(pw.m['internal_transfer_share_global'])}</b>。</li>
      <li>债务相关支出占比中位 <b>{fmt_pct(pf.m['debt_share_p50'])}</b> vs <b>{fmt_pct(pw.m['debt_share_p50'])}</b>；高承诺支出 / 收入中位 <b>{fmt_pct(pf.m['high_commit_share_p50'])}</b> vs <b>{fmt_pct(pw.m['high_commit_share_p50'])}</b>。</li>
    </ul>
    <details>
      <summary>查看指标明细</summary>
      <div class="table-card">{comp_table(EXPENSE_ROWS, pf, pw)}</div>
      <p class="note" style="margin-top:12px"><b>全部消费类目</b>（按较高一方份额排序）</p>
      <div class="table-card">{category_table(pf, pw)}</div>
      <p class="note">份额以可识别消费为分母；人均 = 类别支出加总 / 全部有效用户数（含无该类别消费的用户）。</p>
    </details>
  </section>

  <section class="section">
    <h2><span class="num">4</span>负债与信用履约</h2>
    <div class="key-conclusion"><span class="star">★</span><div>WageGo 的借贷和偿债负担显著更高：有借贷放款 <mark class="hl">{fmt_pct(pw.m['any_disburse'])}</mark>（Fundo <mark class="hl">{fmt_pct(pf.m['any_disburse'])}</mark>）、现金贷 <mark class="hl">{fmt_pct(pw.m['adv_use'])}</mark>（<mark class="hl">{fmt_pct(pf.m['adv_use'])}</mark>）、偿债/收入 <mark class="hl">{fmt_pct(pw.m['rep_rate_p50'])}</mark>（<mark class="hl">{fmt_pct(pf.m['rep_rate_p50'])}</mark>）；拒付记录 <mark class="hl">{fmt_pct(pw.m['dish_any'])}</mark> vs <mark class="hl">{fmt_pct(pf.m['dish_any'])}</mark>，但 Fundo 解决率更高（<mark class="hl">{fmt_pct(pf.m['dish_resolution_rate'])}</mark> vs <mark class="hl">{fmt_pct(pw.m['dish_resolution_rate'])}</mark>）。</div></div>
    <div class="chart">{figs['liability']}<p class="note">使用率与拒付发生率均以有效用户为分母。</p></div>
    <ul class="notes">
      <li>有借贷放款：Fundo <b>{fmt_pct(pf.m['any_disburse'])}</b>、WageGo <b>{fmt_pct(pw.m['any_disburse'])}</b>；现金贷使用率 <b>{fmt_pct(pf.m['adv_use'])}</b> vs <b>{fmt_pct(pw.m['adv_use'])}</b>，分期贷 <b>{fmt_pct(pf.m['loan_use'])}</b> vs <b>{fmt_pct(pw.m['loan_use'])}</b>，BNPL <b>{fmt_pct(pf.m['bnpl_use'])}</b> vs <b>{fmt_pct(pw.m['bnpl_use'])}</b>。</li>
      <li>偿债 / 收入中位：Fundo <b>{fmt_pct(pf.m['rep_rate_p50'])}</b>、WageGo <b>{fmt_pct(pw.m['rep_rate_p50'])}</b>；月还款中位 <b>{fmt_money(pf.m['rep_m_p50'])}</b> vs <b>{fmt_money(pw.m['rep_m_p50'])}</b>，净借款人占比 <b>{fmt_pct(pf.m['net_borrow_share'])}</b> vs <b>{fmt_pct(pw.m['net_borrow_share'])}</b>。</li>
      <li>有拒付记录：Fundo <b>{fmt_pct(pf.m['dish_any'])}</b>、WageGo <b>{fmt_pct(pw.m['dish_any'])}</b>；未解决拒付 <b>{fmt_pct(pf.m['dish_unres'])}</b> vs <b>{fmt_pct(pw.m['dish_unres'])}</b>，解决率 <b>{fmt_pct(pf.m['dish_resolution_rate'])}</b> vs <b>{fmt_pct(pw.m['dish_resolution_rate'])}</b>。</li>
      <li>放款金额结构：Fundo <b>{fmt_metric(pf.m['product_mix_text'], 'text')}</b>；WageGo <b>{fmt_metric(pw.m['product_mix_text'], 'text')}</b>。</li>
    </ul>
    <details>
      <summary>查看指标明细</summary>
      <div class="table-card">{comp_table(LIABILITY_ROWS, pf, pw)}</div>
    </details>
  </section>

  <section class="section">
    <h2><span class="num">5</span>现金流与流动性</h2>
    <div class="key-conclusion"><span class="star">★</span><div>两组都以负盈余为主，Fundo 负盈余占比 <mark class="hl">{fmt_pct(pf.m['sur_neg'])}</mark>（WageGo <mark class="hl">{fmt_pct(pw.m['sur_neg'])}</mark>）；缓冲不足 7 天占比 <mark class="hl">{fmt_pct(pf.m['buffer_lt7'])} vs {fmt_pct(pw.m['buffer_lt7'])}</mark>，流动性都极薄，WageGo 中位余额更低（<mark class="hl">{fmt_money(pw.m['bal_p50'])} vs {fmt_money(pf.m['bal_p50'])}</mark>）。</div></div>
    <div class="chart">{figs['cashflow']}<p class="note">占比均以有效用户为分母；缓冲天数 = 账户余额 /（月支出 / 30）。</p></div>
    <div class="chart">{figs['balance_buckets']}<p class="note">余额取最近快照；两组各自归一化为 100%。</p></div>
    <ul class="notes">
      <li>中位月盈余：Fundo <b>{fmt_money(pf.m['sur_p50'])}</b>、WageGo <b>{fmt_money(pw.m['sur_p50'])}</b>；负盈余占比 <b>{fmt_pct(pf.m['sur_neg'])}</b> vs <b>{fmt_pct(pw.m['sur_neg'])}</b>，偿债前收不抵支 <b>{fmt_pct(pf.m['pre_debt_neg'])}</b> vs <b>{fmt_pct(pw.m['pre_debt_neg'])}</b>。</li>
      <li>中位账户余额：Fundo <b>{fmt_money(pf.m['bal_p50'])}</b>、WageGo <b>{fmt_money(pw.m['bal_p50'])}</b>；缓冲 &lt; 7 天占比 <b>{fmt_pct(pf.m['buffer_lt7'])}</b> vs <b>{fmt_pct(pw.m['buffer_lt7'])}</b>，透支用户占比 <b>{fmt_pct(pf.m['bal_neg'])}</b> vs <b>{fmt_pct(pw.m['bal_neg'])}</b>。</li>
    </ul>
    <details>
      <summary>查看指标明细</summary>
      <div class="table-card">{comp_table(CASHFLOW_ROWS, pf, pw)}</div>
      <p class="note" style="margin-top:12px"><b>余额分桶</b></p>
      <div class="table-card">{balance_bucket_table(pf, pw)}</div>
    </details>
  </section>

  <section class="section">
    <h2><span class="num">6</span>用户分层</h2>
    <div class="key-conclusion"><span class="star">★</span><div>WageGo 盈余型占比 <mark class="hl">{fmt_pct(seg_val(pw, "盈余型", "share"))}</mark>（Fundo <mark class="hl">{fmt_pct(seg_val(pf, "盈余型", "share"))}</mark>），重压型 <mark class="hl">{fmt_pct(seg_val(pw, "重压型", "share"))}</mark>（<mark class="hl">{fmt_pct(seg_val(pf, "重压型", "share"))}</mark>）；两组的重压型都处于明显负盈余状态。</div></div>
    <div class="chart-grid even">
      <div class="chart">{figs['segments']}<p class="note">各分层占比，组内归一化为 100%。</p></div>
      <div class="chart">{figs['segment_surplus']}<p class="note">各分层中位月盈余。</p></div>
    </div>
    <div class="table-card">{segment_table(pf, pw)}</div>
    <p class="note">盈余率已截尾至 [-100%, +100%]；拒付率 = 拒付笔数 /（拒付笔数 + 还款笔数）。</p>
  </section>

  <section class="section">
    <h2><span class="num">7</span>风险信号</h2>
    <div class="key-conclusion"><span class="star">★</span><div>WageGo 的赌博、手续费与催收暴露更高：赌博交易 <mark class="hl">{fmt_pct(pw.m['gamb_incidence'])}</mark>（Fundo <mark class="hl">{fmt_pct(pf.m['gamb_incidence'])}</mark>）、赌博占消费 <mark class="hl">{fmt_pct(pw.m['gamb_share_cons'])}</mark>（<mark class="hl">{fmt_pct(pf.m['gamb_share_cons'])}</mark>）、手续费 <mark class="hl">{fmt_pct(pw.m['fees_incidence'])}</mark>（<mark class="hl">{fmt_pct(pf.m['fees_incidence'])}</mark>）、催收类拒付 <mark class="hl">{fmt_pct(pw.m['collections_inc'])}</mark>（<mark class="hl">{fmt_pct(pf.m['collections_inc'])}</mark>）。</div></div>
    <div class="chart">{figs['risk']}<p class="note">“赌博占消费”为样本加总口径，其余为有效用户发生率。</p></div>
    <details>
      <summary>查看指标明细</summary>
      <div class="table-card">{comp_table(RISK_ROWS, pf, pw)}</div>
    </details>
  </section>

  <section class="section">
    <h2><span class="num">8</span>主要差异</h2>
    <div class="key-conclusion"><span class="star">★</span><div>WageGo 在借贷、拒付、赌博与手续费上的暴露更高；Fundo 在负盈余与外部转账占比上更高。</div></div>
    <div class="chart">{figs['difference']}<p class="note">差值 = WageGo − Fundo，单位为百分点；仅列可比的比例类指标，按差值绝对值排序。</p></div>
    <div class="findings" style="margin-top:14px">{findings}</div>
  </section>

  <div class="footer">
    <p><b>数据源</b>：fundo_sample_with_feature_all.csv（共 {pf.n_fields:,} 个字段）、wagego_sample_with_feature_all.csv（共 {pw.n_fields:,} 个字段）；两套数据使用同一套字段清单、缺失哨兵与清洗规则。</p>
    <p><b>样本口径</b>：Fundo 为 2026-09 新客样本，WageGo 为 2026-08 至 2026-10 新客样本；同一 user_id 保留交易笔数最多的申请。有效样本要求 txn_raw_input_cnt &gt; 0，且收入 / 支出 / 盈余非缺失、收入 &gt; 0。观察窗口 {fpr.WINDOW_DAYS} 天，月度指标按 6 个月折算。</p>
    <p><b>缺失处理</b>：金额类字段的 −999,999.99999 与 lender 类字段的 −1 视为缺失；未做插补。无有效流水用户未纳入画像指标。</p>
    <p><b>指标定义</b>：盈余 = 收入 − 支出 − 还本付息 − 拒付费 + 新增借款；可识别消费 = {len(fpr.CONSUMPTION_CATS)} 个消费类目的 debit 金额之和；拒付率 = 拒付笔数 /（拒付笔数 + 还款笔数），解决率 = 已解决笔数 /（已解决 + 未解决）；CV = 交易级金额变异系数；HHI = 各消费类目份额平方和；高承诺支出 = 房租 / 保险 / 教育 / 医疗 / 健身类支出加总；缓冲天数 = 账户余额 /（月支出 / 30）。分位数均为各自分布独立计算，不可直接相减。</p>
    <p><b>对比说明</b>：两组为独立样本，未做统一加权或显著性检验；差异仅用于描述性对比，不构成信用评估结论。生成时间：{html.escape(generated_at)}。</p>
  </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Fundo vs WageGo 人群画像对比报告")
    parser.add_argument("--fundo", default=str(DEFAULT_FUNDO), help="Fundo 样本 CSV 路径")
    parser.add_argument("--wagego", default=str(DEFAULT_WAGEGO), help="WageGo 样本 CSV 路径")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("加载 Fundo 样本 ...")
    pf = load_profile("Fundo", fpr, Path(args.fundo))
    print("加载 WageGo 样本 ...")
    pw = load_profile("WageGo", wpr, Path(args.wagego))

    print("生成图表 ...")
    figs = {
        "income_mix": chart_income_mix(pf, pw),
        "distributions": chart_distributions(pf, pw),
        "expense_structure": chart_expense_structure(pf, pw),
        "consumption": chart_consumption_dumbbell(pf, pw),
        "liability": chart_liability(pf, pw),
        "cashflow": chart_cashflow(pf, pw),
        "balance_buckets": chart_balance_buckets(pf, pw),
        "segments": chart_segments(pf, pw),
        "segment_surplus": chart_segment_surplus(pf, pw),
        "risk": chart_risk(pf, pw),
        "difference": chart_difference_overview(pf, pw),
    }

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_path = out / "compare_profile_report.html"
    report_path.write_text(build_html(pf, pw, figs, generated_at), encoding="utf-8")

    print("\n" + "=" * 64)
    print("Fundo vs WageGo 人群画像对比报告已生成")
    print("=" * 64)
    print(
        f"有效样本: Fundo {pf.m['n_valid']:,} 人 ({fmt_pct(pf.m['coverage'])}) | "
        f"WageGo {pw.m['n_valid']:,} 人 ({fmt_pct(pw.m['coverage'])})"
    )
    print(
        f"收入: 中位月收入 {fmt_money(pf.m['inc_p50'])} vs {fmt_money(pw.m['inc_p50'])} | "
        f"负盈余 {fmt_pct(pf.m['sur_neg'])} vs {fmt_pct(pw.m['sur_neg'])}"
    )
    print(
        f"负债: 有借贷 {fmt_pct(pf.m['any_disburse'])} vs {fmt_pct(pw.m['any_disburse'])} | "
        f"有拒付 {fmt_pct(pf.m['dish_any'])} vs {fmt_pct(pw.m['dish_any'])}"
    )
    print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
