#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fundo vs WageGo 人群画像对比数据与图表模块。

供 build_compare_decks.py 生成幻灯片。
"""

from __future__ import annotations

import html
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import fundo_profile_report as fpr
import profile_enrichment as profile_extra
import compare_profile_enrichment as compare_extra
from fundo_profile_report import (
    BASELINE,
    INK,
    INK2,
    SEQ_BLUE,
    STATUS,
    SURFACE,
    base_layout,
    fig_html,
    fmt_money,
    fmt_pct,
)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 plotly，请先执行: pip install plotly") from exc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


POP_COLORS = {"Fundo": "#0E7558", "WageGo": "#6B4FA8"}
POP_Y_ORDER = ["WageGo", "Fundo"]  # 水平图中 Fundo 在上
HILITE = "rgba(0,0,0,0)"  # 不再使用金色描边强调
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
    df_raw, missing, n_fields = module.load_data(path, extra_columns=profile_extra.EXTRA_COLUMNS)
    df_raw, n_apps = module.clean_frame(df_raw)
    df = module.dedupe_users(df_raw)
    d, m, seg = module.build_metrics(df, n_apps)
    profile_extra.analyze(d, m, dataset_key=name.lower(), dataset_name=name)
    import csv
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source))
    profile_extra.export_evidence(Path(__file__).resolve().parent, header, module.ALL_COLUMNS, d, m)

    for key, col in (
        ("inc_mean", "inc_m"),
        ("exp_mean", "exp_m"),
        ("sur_mean", "sur_m"),
        ("bal_mean", "bal"),
        ("rep_mean", "rep_m"),
        ("draw_mean", "draw_m"),
    ):
        m[key] = float(d[col].mean())
    m["dishfee_mean"] = float(
        d["bank_txn_surplus_dishonoursfee_sum_182d"].div(fpr.MONTHS).mean()
    )

    m["inc_cat_mean"] = float(
        d["bank_txn_income_wages_sum_182d"].div(fpr.MONTHS).mean()
    )
    m["exp_cat_mean"] = float(
        d["bank_txn_category_global_debit_amt_182d"].div(fpr.MONTHS).mean()
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
    m["balance_buckets"] = [
        (lab, module.cond_share(mask, bal)) for lab, mask in zip(bucket_labels, bucket_masks)
    ]
    print(f"  {name}: 有效样本 {m['n_valid']:,} 人，用时 {time.time() - t0:.1f}s")
    return Profile(d=d, m=m, seg=seg, missing=missing, n_fields=n_fields)


def seg_val(p: Profile, segment: str, col: str) -> float:
    return float(p.seg.loc[segment, col]) if segment in p.seg.index else np.nan


def metric_value(p: Profile, key: str):
    if key.startswith("enrich:"):
        return p.m.get("profile_enrichment", {}).get(key.split(":",1)[1], np.nan)
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


# 明细表中高亮这些指标行

DIFF_OVERVIEW = [
    ("负盈余占比", "sur_neg"),
    ("缓冲 < 7 天", "buffer_lt7"),
    ("有借贷放款", "any_disburse"),
    ("现金贷使用率", "adv_use"),
    ("历史曾有负余额", "enrich:ever_negative"),
    ("有拒付记录", "dish_any"),
    ("未解决拒付", "dish_unres"),
    ("赌博交易发生率", "gamb_incidence"),
    ("手续费发生率", "fees_incidence"),
    ("工资占入账金额", "wage_share_agg"),
    ("福利收入发生率", "cent_incidence"),
    ("还款/(工资+福利)中位", "enrich:recurring_p50"),
    ("固定账单 / 工资中位", "fixed_wage_p50"),
    ("品质消费 / 工资中位", "prem_wage_p50"),
    ("≥5 家已识别还款机构", "enrich:multi5"),
    ("重压型占比", "seg:重压型:share"),
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


def chart_income_resilience(pf: Profile, pw: Profile) -> str:
    ben_defs = [
        ("pension", "养老金"),
        ("family", "家庭福利"),
        ("jobseeker", "JobSeeker"),
        ("child", "儿童抚养"),
        ("carer", "照顾者福利"),
        ("youth", "青年津贴"),
    ]
    labels = [name for _, name in ben_defs]
    f_ben = [
        float(pf.m["benefit_type_inc"].get(key, np.nan)) * 100
        for key, _ in ben_defs
    ]
    w_ben = [
        float(pw.m["benefit_type_inc"].get(key, np.nan)) * 100
        for key, _ in ben_defs
    ]
    seg_labels = list(SEG_COLORS.keys())
    f_wage = [seg_val(pf, s, "wage_share") * 100 for s in seg_labels]
    w_wage = [seg_val(pw, s, "wage_share") * 100 for s in seg_labels]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.22,
        subplot_titles=("福利类型覆盖率（福利人群，可重叠）", "各分层工资占入账"),
    )
    add_grouped_bars(
        fig, labels, f_ben, w_ben, row=1, col=1, digits=1,
        highlight={"养老金", "JobSeeker"},
    )
    add_grouped_bars(
        fig, seg_labels, f_wage, w_wage, row=1, col=2, digits=1,
        showlegend=False, highlight={"重压型"},
    )
    mx_ben = max([v for v in f_ben + w_ben if valid(v)] + [0]) * 1.24
    mx_wage = max([v for v in f_wage + w_wage if valid(v)] + [0]) * 1.36
    fig.update_xaxes(range=[0, mx_ben], ticksuffix="%", row=1, col=1)
    fig.update_xaxes(range=[0, mx_wage], ticksuffix="%", row=1, col=2)
    fig.update_yaxes(
        categoryorder="array", categoryarray=list(reversed(labels)),
        showgrid=False, row=1, col=1,
    )
    fig.update_yaxes(
        categoryorder="array", categoryarray=list(reversed(seg_labels)),
        showgrid=False, row=1, col=2,
    )
    base_layout(fig, height=450, margin=dict(l=104, r=24, t=64, b=74), showlegend=True)
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
    return fig_html(fig, 450)


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


def chart_consumption_depth(pf: Profile, pw: Profile) -> str:
    picks = [
        ("groceries", "超市"),
        ("dining_out", "餐饮"),
        ("rent", "房租"),
        ("insurance", "保险"),
        ("travel", "旅游"),
        ("automotive", "汽车"),
        ("gambling", "赌博"),
        ("fees", "手续费"),
    ]
    labels = [label for _, label in picks]
    f_pen = [
        float(pf.m["cat_pen"].get(key, np.nan)) * 100 for key, _ in picks
    ]
    w_pen = [
        float(pw.m["cat_pen"].get(key, np.nan)) * 100 for key, _ in picks
    ]
    seg_labels = list(SEG_COLORS.keys())
    f_prem = [seg_val(pf, s, "prem_wage") * 100 for s in seg_labels]
    w_prem = [seg_val(pw, s, "prem_wage") * 100 for s in seg_labels]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.23,
        subplot_titles=("消费类目渗透率", "品质消费 / 工资（各分层中位）"),
    )
    add_grouped_bars(
        fig, labels, f_pen, w_pen, row=1, col=1, digits=1,
        highlight={"赌博", "手续费"},
    )
    add_grouped_bars(
        fig, seg_labels, f_prem, w_prem, row=1, col=2, digits=1,
        showlegend=False, highlight={"重压型"},
    )
    mx_pen = max([v for v in f_pen + w_pen if valid(v)] + [0]) * 1.20
    mx_prem = max([v for v in f_prem + w_prem if valid(v)] + [0]) * 1.38
    fig.update_xaxes(range=[0, mx_pen], ticksuffix="%", row=1, col=1)
    fig.update_xaxes(range=[0, mx_prem], ticksuffix="%", row=1, col=2)
    fig.update_yaxes(
        categoryorder="array", categoryarray=list(reversed(labels)),
        showgrid=False, row=1, col=1,
    )
    fig.update_yaxes(
        categoryorder="array", categoryarray=list(reversed(seg_labels)),
        showgrid=False, row=1, col=2,
    )
    base_layout(fig, height=450, margin=dict(l=92, r=24, t=64, b=74), showlegend=True)
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
    return fig_html(fig, 450)


def chart_fixed_burden(pf: Profile, pw: Profile) -> str:
    seg_labels = list(SEG_COLORS.keys())
    f_fix = [seg_val(pf, s, "fixed_wage") * 100 for s in seg_labels]
    w_fix = [seg_val(pw, s, "fixed_wage") * 100 for s in seg_labels]
    cov_labels = ["固定账单", "房租", "保险"]
    f_cov = [
        pf.m["fixed_bill_incidence"] * 100,
        pf.m["rent_incidence"] * 100,
        pf.m["insurance_incidence"] * 100,
    ]
    w_cov = [
        pw.m["fixed_bill_incidence"] * 100,
        pw.m["rent_incidence"] * 100,
        pw.m["insurance_incidence"] * 100,
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.23,
        subplot_titles=("固定账单 / 工资（各分层中位）", "刚性支出覆盖率"),
    )
    add_grouped_bars(
        fig, seg_labels, f_fix, w_fix, row=1, col=1, digits=1,
        highlight={"重压型"},
    )
    add_grouped_bars(
        fig, cov_labels, f_cov, w_cov, row=1, col=2, digits=1,
        showlegend=False, highlight={"房租"},
    )
    mx_fix = max([v for v in f_fix + w_fix if valid(v)] + [0]) * 1.38
    fig.update_xaxes(range=[0, mx_fix], ticksuffix="%", row=1, col=1)
    fig.update_xaxes(range=[0, 108], ticksuffix="%", row=1, col=2)
    fig.update_yaxes(
        categoryorder="array", categoryarray=list(reversed(seg_labels)),
        showgrid=False, row=1, col=1,
    )
    fig.update_yaxes(
        categoryorder="array", categoryarray=list(reversed(cov_labels)),
        showgrid=False, row=1, col=2,
    )
    base_layout(fig, height=450, margin=dict(l=72, r=24, t=64, b=74), showlegend=True)
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
    return fig_html(fig, 450)


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
        "催收类支出",
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
                     highlight={"赌博交易", "赌博占消费", "手续费支出", "催收类支出"})
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
            textfont=dict(color=INK2, size=11.5),
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
            textfont=dict(color=INK2, size=11.5),
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
    base_layout(fig, height=max(420, 30 * len(labels) + 130), margin=dict(l=140, r=42, t=46, b=88), showlegend=True)
    fig.update_layout(
        barmode="relative",
        bargap=0.32,
        legend=dict(
            orientation="h", x=0, xanchor="left", y=-0.06, yanchor="top",
            font=dict(size=11.5), bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig_html(fig, max(420, 30 * len(labels) + 130))


# ---------------------------------------------------------------------------
# 主要差异文字
# ---------------------------------------------------------------------------
def build_findings(pf: Profile, pw: Profile) -> list[str]:
    return compare_extra.findings(pf, pw)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


