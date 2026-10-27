#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Swap In vs Swap Out 置换画像（临时脚本）。

独立放在 analysis/swap_compare/，不修改现有产品配置与分析引擎：
1. 合并两份 v1_2 特征文件，加 cohort 列；
2. 用 products/fundo.json 的同一套指标/阈值对合并样本统一评分；
3. 按 cohort 聚合 user_scores 与原始金额特征，围绕“置入了谁、为什么可能置入、
   置出了谁、为什么可能置出”生成自包含 HTML。

运行：python analysis/swap_compare/run_swap_comparison.py
"""
from __future__ import annotations

import html
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
MERGED = BASE / "merged_input.csv"
OUT_DIR = BASE / "out"
CACHE_DIR = BASE / "cache"
REPORT = BASE / "swap_in_vs_swap_out_comparison.html"
CSS_SOURCE = ROOT / "fundo_vs_wagego_comparison.html"
IN_PATH = ROOT / "data" / "existing_swap_in_v1_2_with_feature_all.csv"
OUT_PATH = ROOT / "data" / "existing_swap_out_v1_2_with_feature_all.csv"

sys.path.insert(0, str(ROOT))
from profile_analysis import (  # noqa: E402  (只读复用现有清洗口径)
    UNRESOLVED_AMT_COLUMNS,
    RESOLVED_AMT_COLUMNS,
    clean_source_numeric,
)

KEYS = ("swap_in", "swap_out")
NAMES = ("Swap In", "Swap Out")
SEGMENTS = ["优质均衡", "稳健工薪", "均衡普通", "月光消费", "现金流紧张", "高风险承压"]
ARCHETYPES = ["稳定工资", "不稳定工资", "多源/非工资", "福利依赖"]
CROSS_GROUPS = ["稳定工资", "不稳定工资", "多源/非工资"]
LENDER_BEHAVIORS = ["净偿还", "借还平衡", "净融资扩张", "仅新增借款", "仅偿还", "近28天无借贷", "无借贷记录"]
DISHONOUR_STATUS = ["无拒付", "已全部解决", "部分解决", "未解决"]
BURN_BANDS = ["支出低于收入80%", "支出接近收入", "支出达到或超过收入", "数据不足"]
BUFFER_BANDS = ["不足7天", "7至30天", "30至90天", "至少90天"]
DIM_LABELS = {
    "income": "收入稳定性",
    "cashflow": "收支储蓄",
    "liquidity": "流动性缓冲",
    "debt": "负债压力",
    "engagement": "消费活跃度",
}
FLAG_LABELS = {
    "flag_unresolved_dishonour": "存在未结清拒付",
    "flag_high_debt": "负债压力高",
    "flag_multi_lender": "多头借贷（名单内）",
    "flag_payday": "现金贷/工资贷使用",
    "flag_negative_cashflow": "现金净流出",
    "flag_low_buffer": "流动性缓冲不足",
    "flag_gambling": "赌博类支出",
    "flag_income_unstable": "收入不稳定",
    "flag_fees_heavy": "费用/透支负担",
    "flag_welfare_dependent": "福利收入依赖",
    "flag_no_wage": "无工资入账",
}
RAW_COLS = {
    "inc_global_sum": "bank_txn_income_global_sum_182d",
    "inc_wage_sum": "bank_txn_income_wages_sum_182d",
    "cf_expense_sum": "bank_txn_expense_global_sum_182d",
    "cf_surplus_sum": "bank_txn_surplus_surplus_sum_182d",
    "db_disburse_sum": "bank_txn_lender_disburse_amount_sum",
    "db_disburse_count": "bank_txn_lender_disburse_amount_count",
    "db_cash_advance_sum": "bank_txn_lender_disburse_advance_amount_sum",
    "db_repay_sum": "bank_txn_lender_repay_amount_sum",
    "lq_balance_median": "bank_txn_worthiness_balance_median_avg_acct_182d",
    "lq_negative_share": "bank_txn_worthiness_balance_negative_share_avg_acct_182d",
    "lq_reserve_months": "bank_txn_worthiness_reserve_months_182d",
    "cf_saving_rate": "bank_txn_balance_global_savings_rate_182d",
    "en_active_days": "bank_txn_category_active_days_182d",
    "inc_wage_share": "bank_txn_worthiness_wage_income_amt_share_in_income_tag_182d",
    "db_dishonour_ratio": "bank_txn_dishonour_lender_ratio_182d",
}


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

def ensure_merged() -> None:
    if MERGED.exists() and MERGED.stat().st_mtime >= max(IN_PATH.stat().st_mtime, OUT_PATH.stat().st_mtime):
        return
    print("merging cohorts ...", flush=True)
    frames = {}
    for key, path in zip(KEYS, (IN_PATH, OUT_PATH)):
        df = pd.read_csv(path, dtype={"user_id": "string", "application_id": "string"})
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
        frames[key] = df
    if list(frames[KEYS[0]].columns) != list(frames[KEYS[1]].columns):
        raise RuntimeError("两份样本的列顺序不一致，无法直接合并")
    parts = []
    for key in KEYS:
        df = frames[key]
        df.insert(0, "cohort", key)
        parts.append(df)
    pd.concat(parts, ignore_index=True).to_csv(MERGED, index=False)


def ensure_analysis() -> None:
    user_scores = OUT_DIR / "user_scores.csv"
    if user_scores.exists() and user_scores.stat().st_mtime >= MERGED.stat().st_mtime:
        return
    print("running profiling engine on merged sample ...", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-I", str(ROOT / "profile_analysis.py"),
            "--config", str(ROOT / "products" / "fundo.json"),
            "--input", str(MERGED),
            "--output-dir", str(OUT_DIR),
            "--cache-dir", str(CACHE_DIR),
        ],
        check=True,
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(
        MERGED,
        usecols=["cohort", "user_id", "application_id", "sample_datetime", "duedate_1m_5"],
        dtype={"user_id": "string", "application_id": "string"},
    )
    scores = pd.read_csv(OUT_DIR / "user_scores.csv", dtype={"user_id": "string", "application_id": "string"})
    cohort_map = dict(zip(base.application_id, base.cohort))
    scores["cohort"] = scores.application_id.map(cohort_map)
    if scores.cohort.isna().any():
        raise RuntimeError("user_scores 中存在无法匹配 cohort 的 application_id")

    raw_cols = sorted(set(RAW_COLS.values()) | set(UNRESOLVED_AMT_COLUMNS) | set(RESOLVED_AMT_COLUMNS))
    raw = pd.read_csv(MERGED, usecols=["cohort", "application_id"] + raw_cols,
                      dtype={"application_id": "string"})
    unresolved = np.vstack([clean_source_numeric(raw[c], c) for c in UNRESOLVED_AMT_COLUMNS])
    resolved = np.vstack([clean_source_numeric(raw[c], c) for c in RESOLVED_AMT_COLUMNS])
    raw["db_unresolved_amt"] = np.sum(unresolved, axis=0)
    raw["db_resolved_amt"] = np.sum(resolved, axis=0)
    return base, scores, raw


# ---------------------------------------------------------------------------
# 统计与格式
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def rate(series: pd.Series) -> tuple[float, tuple[float, float]]:
    s = series.dropna()
    k, n = int(s.sum()), len(s)
    return (k / n * 100 if n else float("nan")), wilson(k, n)


def median(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.median()) if len(s) else float("nan")


def share(series: pd.Series, value) -> float:
    s = series.dropna()
    return float(s.eq(value).mean() * 100) if len(s) else float("nan")


def raw_median(raw: pd.DataFrame, key: str, cohort: str) -> float:
    col = RAW_COLS.get(key, key)
    arr = clean_source_numeric(raw.loc[raw.cohort.eq(cohort), col], col)
    s = pd.Series(arr).dropna()
    return float(s.median()) if len(s) else float("nan")


def raw_mean(raw: pd.DataFrame, key: str, cohort: str) -> float:
    col = RAW_COLS.get(key, key)
    arr = clean_source_numeric(raw.loc[raw.cohort.eq(cohort), col], col)
    s = pd.Series(arr).dropna()
    return float(s.mean()) if len(s) else float("nan")


def fmt(v: float, kind: str) -> str:
    if not math.isfinite(v):
        return "—"
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "pct2":
        return f"{v:.2f}%"
    if kind == "money":
        return f"A${v:,.0f}"
    if kind == "k":
        return f"A${v / 1000:,.1f}k"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "two":
        return f"{v:.2f}"
    if kind == "days":
        return f"{v:.1f} 天"
    if kind == "months":
        return f"{v:.1f} 个月"
    return f"{v:.1f}"


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def pair(vals) -> list[float]:
    return [float(vals[k]) for k in KEYS]


def delta_text(a: float, b: float, unit: str = "pp") -> str:
    d = b - a
    return f"{d:+.1f}{unit}"


# ---------------------------------------------------------------------------
# 图表组件（复用 Fundo × WageGo 对比页的视觉语言）
# ---------------------------------------------------------------------------

def section(key: str, number: str, title: str, subtitle: str, content: str) -> str:
    return f'''<section id="{key}" class="section"><div class="wrap">
    <div class="section-head"><div><div class="eyebrow">{number}</div><h2>{title}</h2></div>
    <p class="section-sub">{subtitle}</p></div>{content}</div></section>'''


def kpi(label: str, values, kind: str = "pct", sub: str = "", delta: str = "") -> str:
    vals = "".join(
        f'<span class="kv {c}"><span class="product-mini">{n}</span>{fmt(v, kind)}</span>'
        for c, n, v in zip(("f", "w"), NAMES, values)
    )
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-values">{vals}</div><div class="kpi-sub">{sub}</div>{delta_html}</div>')


def grid(*items: str) -> str:
    return '<div class="kpi-grid">' + "".join(items) + "</div>"


def cards(items) -> str:
    return '<div class="two-col">' + "".join(
        f'<div class="card"><h3>{t}</h3><p>{b}</p></div>' for t, b in items) + "</div>"


def chart_value(v: float, kind: str) -> str:
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "score":
        return f"{v:.1f}"
    return fmt(v, kind)


def chart_axis(v: float, kind: str) -> str:
    if kind == "pct":
        return f"{v:g}%"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "score":
        return f"{v:g}"
    return fmt(v, kind)


def chart_delta(d: float, kind: str, diff_unit: str = "") -> str:
    if kind == "pct":
        return f"{d:+.1f}pp"
    if kind == "int":
        return f"{d:+,.0f}{diff_unit}"
    if kind == "score":
        return f"{d:+.1f}{diff_unit or ' 分'}"
    return f"{d:+.1f}{diff_unit}"


def paired(title: str, rows, caption: str = "统一 0–100% 刻度", maximum: float = 100,
           kind: str = "pct", note: str = "", diff_unit: str = "",
           axis_suffix: str | None = None) -> str:
    rows = list(rows)
    if axis_suffix is None:
        axis_suffix = "%" if kind == "pct" else ""
    out = (f'<div class="compare pd-chart"><div class="compare-head"><h3>{title}</h3>'
           f'<span>{caption}</span></div>'
           '<div class="pd-legend"><span><i class="pd-key in"></i>Swap In</span>'
           '<span><i class="pd-key out"></i>Swap Out</span></div>')
    out += (f'<div class="pd-axis-row"><span></span><div class="pd-axis">'
            f'<span>0{axis_suffix}</span><span>{chart_axis(maximum / 2, kind)}</span>'
            f'<span>{chart_axis(maximum, kind)}</span></div><span></span></div>')
    for label, values, detail in rows:
        a, b = float(values[0]), float(values[1])
        pa = max(0.0, min(a / maximum * 100, 100.0))
        pb = max(0.0, min(b / maximum * 100, 100.0))
        left, width = min(pa, pb), abs(pb - pa)
        va, vb = chart_value(a, kind), chart_value(b, kind)
        d = b - a
        detail_html = f'<small>{esc(detail)}</small>' if detail else ""
        out += f'''<div class="pd-row"><div class="compare-label">{esc(label)}{detail_html}</div>
        <div class="pd-track" role="img" aria-label="{esc(label)}：Swap In {va}，Swap Out {vb}，差值 {chart_delta(d, kind, diff_unit)}">
        <span class="pd-mid"></span><span class="pd-link" style="left:{left:.4f}%;width:{width:.4f}%"></span>
        <span class="pd-point in" style="left:{pa:.4f}%" title="Swap In：{va}"></span>
        <span class="pd-point out" style="left:{pb:.4f}%" title="Swap Out：{vb}"></span></div>
        <div class="pd-values"><span class="in">IN {va}</span><span class="out">OUT {vb}</span><b>{chart_delta(d, kind, diff_unit)}</b></div></div>'''
    if not note:
        note = "IN = Swap In（圆点）；OUT = Swap Out（方点）。差值 = Swap Out − Swap In。"
    out += f'<p class="table-note">{note}</p>'
    body = "".join(
        f'<tr><td>{esc(label)}</td><td>{chart_value(float(values[0]), kind)}</td>'
        f'<td>{chart_value(float(values[1]), kind)}</td>'
        f'<td>{chart_delta(float(values[1]) - float(values[0]), kind, diff_unit)}</td></tr>'
        for label, values, _detail in rows
    )
    out += ('<details class="pd-data"><summary>查看数据表</summary>'
            '<table><thead><tr><th>指标</th><th>Swap In</th><th>Swap Out</th><th>差值</th>'
            f'</tr></thead><tbody>{body}</tbody></table></details>')
    return out + "</div>"


def share_compare(title: str, categories, values, contexts, note: str = "") -> str:
    rows = [(cat, [values[0][i], values[1][i]], "") for i, cat in enumerate(categories)]
    ctx = (f"Swap In：{contexts[0]}；Swap Out：{contexts[1]}。"
           "每类占比在各自客群内计算，两组分别合计 100%。")
    if note:
        ctx += " " + note
    return paired(title, rows, caption="统一 0–100% 刻度 · 组内占比",
                  kind="pct", note=ctx)


def range_rows(rows) -> str:
    rows = list(rows)
    out = ('<div class="compare pd-chart"><div class="compare-head"><h3>健康分分布：中位数与中间 50%</h3>'
           '<span>合并样本统一断点 · 同一 0–100 分刻度</span></div>'
           '<div class="pd-legend"><span><i class="pd-key in"></i>Swap In</span>'
           '<span><i class="pd-key out"></i>Swap Out</span></div>'
           '<div class="pd-axis-row"><span></span><div class="pd-axis">'
           '<span>0 分</span><span>50 分</span><span>100 分</span></div><span></span></div>')
    lanes = []
    for _name, c, (p25, p50, p75) in zip(NAMES, ("f", "w"), rows):
        color = "var(--series-in)" if c == "f" else "var(--wagego)"
        top = 2 if c == "f" else 20
        lanes.append(
            f'<span class="pd-range-lane {c}" style="top:{top}px">'
            f'<span class="range-iqr" style="left:{p25:.4f}%;width:{p75 - p25:.4f}%;background:{color}"></span>'
            f'<span class="range-median" style="left:{p50:.4f}%;background:{color}"></span></span>'
        )
    aria = "；".join(
        f"{name} P25 {p25:.1f}，中位数 {p50:.1f}，P75 {p75:.1f}"
        for name, (p25, p50, p75) in zip(NAMES, rows)
    )
    med_in, med_out = rows[0][1], rows[1][1]
    out += (f'<div class="pd-row pd-range-row"><div class="compare-label">健康分'
            '<small>P25–P75 区间；竖线为中位数</small></div>'
            f'<div class="pd-range" role="img" aria-label="{esc(aria)}">{"".join(lanes)}</div>'
            f'<div class="pd-values"><span class="in">IN {med_in:.1f}</span>'
            f'<span class="out">OUT {med_out:.1f}</span>'
            f'<b>{med_out - med_in:+.1f} 分</b></div></div>')
    out += ('<p class="table-note">浅色区间为 P25–P75，竖线为中位数；差值 = Swap Out − Swap In。'
            '区间反映分布，不是置信区间。</p>')
    out += ('<details class="pd-data"><summary>查看数据表</summary><table><thead><tr>'
            '<th>客群</th><th>P25</th><th>中位数</th><th>P75</th></tr></thead><tbody>'
            + "".join(f'<tr><td>{esc(name)}</td><td>{p25:.1f}</td><td>{p50:.1f}</td><td>{p75:.1f}</td></tr>'
                      for name, (p25, p50, p75) in zip(NAMES, rows))
            + '</tbody></table></details>')
    return out + "</div>"


def ci_text(ci: tuple[float, float]) -> str:
    if not all(math.isfinite(x) for x in ci):
        return "—"
    return f"[{ci[0]:.1f}, {ci[1]:.1f}]"


def flag_table(stats: dict) -> str:
    rows = []
    entries = []
    for flag, label in FLAG_LABELS.items():
        a, b = stats["swap_in"]["flags"][flag], stats["swap_out"]["flags"][flag]
        entries.append((label, a, b))
    entries.sort(key=lambda r: abs(r[2][0] - r[1][0]), reverse=True)
    for label, a, b in entries:
        rows.append(
            f"<tr><td>{esc(label)}</td>"
            f"<td>{a[0]:.1f}%</td><td class='ci'>{ci_text(a[1])}</td>"
            f"<td>{b[0]:.1f}%</td><td class='ci'>{ci_text(b[1])}</td>"
            f"<td class='d'>{b[0] - a[0]:+.1f}pp</td></tr>"
        )
    return ('<table class="flag-table"><thead><tr><th>风险标签</th>'
            '<th>Swap In</th><th>95% CI</th><th>Swap Out</th><th>95% CI</th><th>差值</th>'
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ensure_merged()
    ensure_analysis()
    base, scores, raw = load_data()
    obs = scores[scores.flag_no_txn.eq(0)].copy()
    scored = obs[obs.health_score.notna()].copy()

    S = {}
    for key in KEYS:
        b = base[base.cohort.eq(key)]
        g = obs[obs.cohort.eq(key)]
        gs = scored[scored.cohort.eq(key)]
        S[key] = {
            "apps": len(b),
            "users": int(b.user_id.nunique()),
            "obs": len(g),
            "scored": len(gs),
            "period": (pd.to_datetime(b.sample_datetime).min(), pd.to_datetime(b.sample_datetime).max()),
            "health": (float(gs.health_score.quantile(.25)), float(gs.health_score.median()),
                       float(gs.health_score.quantile(.75))),
            "health_median": median(gs.health_score),
            "segments": {s: share(gs.segment, s) for s in SEGMENTS},
            "dims": {d: median(gs[f"score_{d}"]) for d in DIM_LABELS},
            "archetypes": {s: share(g.income_archetype, s) for s in ARCHETYPES},
            "lender_behaviors": {s: share(g.lender_behavior, s) for s in LENDER_BEHAVIORS},
            "dishonour_status": {s: share(g.dishonour_status, s) for s in DISHONOUR_STATUS},
            "burn_bands": {s: share(g.tr_recent_burn_band, s) for s in BURN_BANDS},
            "buffer_bands": {s: share(g.rs_expense_buffer_days_band, s) for s in BUFFER_BANDS},
            "flags": {f: rate(g[f]) for f in FLAG_LABELS},
            "flag_count_mean": float(g.risk_flag_count.mean()),
            "flag_count_ge3": float(g.risk_flag_count.ge(3).mean() * 100),
            "joint": None,
            "joint_n": 0,
            "income_decline": share(g.tr_income_daily_ratio_band, "下降超过20%"),
            "expense_rise": share(g.tr_expense_daily_ratio_band, "上升至少20%"),
            "commit_ge30": share(g.rs_commitment_income_band, "30%至60%") + share(g.rs_commitment_income_band, "至少60%"),
            "fixed_ge20": share(g.rs_fixed_income_band, "20%至40%") + share(g.rs_fixed_income_band, "至少40%"),
            "disburse_rate": float(g.db_disburse_count.gt(0).mean() * 100),
            "active_lender_ge1": float(g.db_active_lender_cnt.ge(1).mean() * 100),
            "median": {
                "balance": median(g.lq_balance_median),
                "negative_share": raw_median(raw, "lq_negative_share", key) * 100,
                "saving_rate": median(g.cf_saving_rate) * 100,
                "unresolved_cnt": median(g.db_unresolved_cnt),
                "disburse_cnt": median(g.db_disburse_count),
                "active_lender": median(g.db_active_lender_cnt),
                "buffer_days": median(g.rs_expense_buffer_days),
                "tr_burn": median(g.tr_recent_burn) * 100,
                "active_days": median(g.en_active_days),
                "wage_share": median(g.inc_wage_share) * 100,
                "discretionary": median(g.cs_discretionary_share) * 100,
            },
            "raw": {
                "inc_global_sum": raw_median(raw, "inc_global_sum", key),
                "inc_wage_sum": raw_median(raw, "inc_wage_sum", key),
                "cf_expense_sum": raw_median(raw, "cf_expense_sum", key),
                "cf_surplus_sum": raw_median(raw, "cf_surplus_sum", key),
                "db_disburse_sum": raw_median(raw, "db_disburse_sum", key),
                "db_cash_advance_sum": raw_median(raw, "db_cash_advance_sum", key),
                "db_repay_sum": raw_median(raw, "db_repay_sum", key),
                "lq_reserve_months": raw_median(raw, "lq_reserve_months", key),
                "db_dishonour_ratio": raw_median(raw, "db_dishonour_ratio", key),
            },
        }
        r = raw[raw.cohort.eq(key)]
        S[key]["raw"]["unresolved_amt_p50"] = float(r.db_unresolved_amt.median())
        S[key]["raw"]["unresolved_amt_mean"] = float(r.db_unresolved_amt.mean())
        affected = r.loc[r.db_unresolved_amt.gt(0), "db_unresolved_amt"]
        S[key]["raw"]["unresolved_amt_affected_mean"] = float(affected.mean()) if len(affected) else float("nan")
        S[key]["raw"]["unresolved_amt_affected_n"] = int(len(affected))
        denom = r.db_unresolved_amt.sum() + r.db_resolved_amt.sum()
        S[key]["raw"]["resolved_amt_ratio"] = (
            float(r.db_resolved_amt.sum() / denom * 100) if denom else float("nan"))
        joint = g[["flag_payday", "flag_unresolved_dishonour", "flag_low_buffer"]].eq(1).all(axis=1)
        S[key]["joint"] = float(joint.mean() * 100)
        S[key]["joint_n"] = int(joint.sum())
        S[key]["joint_rate"] = rate(joint.astype(int))

    # ---- 文本中反复使用的量 ----
    def v(key, path):
        cur = S[key]
        for part in path.split("."):
            cur = cur[int(part)] if part.isdigit() else cur[part]
        return cur

    def pv(path):
        return [v(k, path) for k in KEYS]

    def pct_pair(path, kind="pct"):
        return " / ".join(fmt(x, kind) for x in pv(path))

    def inout(path, kind="pct"):
        a, b = pv(path)
        return f"置入 {fmt(a, kind)} / 置出 {fmt(b, kind)}"

    def pp(path):
        a, b = pv(path)
        return f"{b - a:+.1f}pp"

    # 标签差异排序
    flag_gaps = sorted(((FLAG_LABELS[f], pv(f"flags.{f}.0"), pv(f"flags.{f}.1"),
                         v("swap_out", f"flags.{f}.0") - v("swap_in", f"flags.{f}.0"))
                        for f in FLAG_LABELS), key=lambda x: abs(x[3]), reverse=True)

    # 收入形态内分层与标准化（福利依赖组 Swap Out 为 0 条，不纳入）
    cross = {}
    for flag in ("flag_payday", "flag_unresolved_dishonour"):
        cross[flag] = {}
        for grp in CROSS_GROUPS:
            vals = []
            for key in KEYS:
                g = obs[obs.cohort.eq(key) & obs.income_archetype.eq(grp)]
                vals.append((rate(g[flag])[0], len(g)))
            cross[flag][grp] = vals
    total_cross = sum(len(obs[obs.income_archetype.eq(g)]) for g in CROSS_GROUPS)
    cross_w = {g: len(obs[obs.income_archetype.eq(g)]) / total_cross for g in CROSS_GROUPS}
    cross_std = {
        flag: [sum(cross_w[g] * cross[flag][g][i][0] for g in CROSS_GROUPS) for i in range(2)]
        for flag in cross
    }

    # ---- 报告 ----
    css = CSS_SOURCE.read_text(encoding="utf-8").split("<style>", 1)[1].split("</style>", 1)[0]
    css += """
/* Swap In / Swap Out overrides */
:root{--fundo:#0f766e;--fundo-dark:#115e59;--fundo-soft:#ccfbf1;--series-in:#0d9488;--wagego:#b45309;--wagego-dark:#92400e;--wagego-soft:#fef3c7}
.flag-table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px}
.flag-table th,.flag-table td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
.flag-table th:first-child,.flag-table td:first-child{text-align:left}
.flag-table th{font-size:11.5px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.03em}
.flag-table td.ci{color:var(--muted);font-size:11.5px}
.flag-table td.d{font-weight:700}
.flag-table tr:hover td{background:#f8fafc}
.logic-table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px}
.logic-table th,.logic-table td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.logic-table th{font-size:11.5px;color:var(--muted);font-weight:700;letter-spacing:.03em}
.logic-table td:first-child{font-weight:700;white-space:nowrap}
.logic-table td:nth-child(2),.logic-table td:nth-child(3){font-variant-numeric:tabular-nums}
@media(max-width:640px){.logic-table{font-size:12px}.logic-table th,.logic-table td{padding:7px 6px}.logic-table td:first-child{white-space:normal}}
.cohort-pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700;margin-right:6px}
.cohort-pill.in{background:var(--fundo-soft);color:var(--fundo-dark)}
.cohort-pill.out{background:var(--wagego-soft);color:var(--wagego-dark)}
@media(max-width:640px){.flag-table{font-size:12px}.flag-table th,.flag-table td{padding:7px 5px}}
/* Same-row paired comparison */
.pd-chart .compare-head{margin-bottom:10px}
.pd-legend{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center;font-size:11.5px;color:var(--muted);margin:2px 0 8px}
.pd-legend span{display:inline-flex;align-items:center;gap:6px}
.pd-key{display:inline-block;width:10px;height:10px;border:2px solid #fff;box-shadow:0 0 0 1px currentColor}
.pd-key.in{background:var(--series-in);color:var(--series-in);border-radius:50%}
.pd-key.out{background:var(--wagego);color:var(--wagego);border-radius:2px}
.pd-axis-row{display:grid;grid-template-columns:190px 1fr 184px;gap:18px;align-items:center;margin:8px 0 2px}
.pd-axis{display:flex;justify-content:space-between;padding:0 7px;font-size:11px;color:var(--muted)}
.pd-row{display:grid;grid-template-columns:190px 1fr 184px;gap:18px;align-items:center;padding:13px 0;border-bottom:1px dashed var(--line)}
.pd-row:last-of-type{border-bottom:0}
.pd-track{position:relative;height:34px;margin:0 7px}
.pd-track:before{content:"";position:absolute;left:0;right:0;top:16px;height:2px;background:#e2e8f0}
.pd-mid{position:absolute;left:50%;top:7px;bottom:7px;width:1px;background:#dbe3ec}
.pd-link{position:absolute;top:15px;height:4px;border-radius:999px;background:#94a3b8}
.pd-point{position:absolute;z-index:1;width:11px;height:11px;transform:translateX(-50%);border:2px solid #fff;box-shadow:0 0 0 1px currentColor}
.pd-point.in{top:5px;background:var(--series-in);color:var(--series-in);border-radius:50%}
.pd-point.out{top:18px;background:var(--wagego);color:var(--wagego);border-radius:2px}
.pd-values{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:2px 10px;font-size:11.5px;line-height:1.5;font-variant-numeric:tabular-nums;color:#334155}
.pd-values .in,.pd-values .out{color:#334155}
.pd-values b{min-width:58px;text-align:right;color:#475569}
.pd-range{position:relative;height:36px;margin:0 7px}
.pd-range-lane{position:absolute;left:0;right:0;height:14px;border-radius:999px;background:#f1f5f9}
.pd-range-lane .range-iqr{top:3px;height:8px;border-radius:999px}
.pd-range-lane .range-median{top:0;height:14px;width:3px}
.pd-data{margin-top:12px;padding-top:10px;border-top:1px dashed var(--line)}
.pd-data summary{cursor:pointer;font-size:11.5px;font-weight:700;color:var(--muted)}
.pd-data table{width:100%;margin-top:8px;border-collapse:collapse;font-size:11.5px;font-variant-numeric:tabular-nums}
.pd-data th,.pd-data td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right}
.pd-data th:first-child,.pd-data td:first-child{text-align:left}
.pd-data th{font-size:10.5px;color:var(--muted);font-weight:700}
@media(max-width:960px){.pd-axis-row,.pd-row{grid-template-columns:155px 1fr 160px;gap:12px}}
@media(max-width:640px){
.pd-axis-row{grid-template-columns:1fr;gap:0}
.pd-axis-row>span{display:none}
.pd-axis{margin-top:8px}
.pd-row{grid-template-columns:1fr;gap:6px}
.pd-row .compare-label{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
.pd-row .compare-label small{margin:0}
.pd-values{justify-content:flex-start}
.pd-values b{margin-left:auto}
.pd-data{font-size:11px}
}
"""

    period_start = min(S[k]["period"][0] for k in KEYS)
    period_end = max(S[k]["period"][1] for k in KEYS)
    hero_stats = [
        ("申请数", [S[k]["apps"] for k in KEYS], "int", "置入 / 置出"),
        ("工资收入占比中位数", pv("median.wage_share"), "pct", "每条申请先算占比，再取中位数"),
        ("现金贷/工资贷使用", pv("flags.flag_payday.0"), "pct", "全部有流水申请"),
        ("存在未结清拒付", pv("flags.flag_unresolved_dishonour.0"), "pct", "182 天窗口内识别"),
    ]
    hero = "".join(
        f'<div class="hero-stat"><div class="label">{label}</div>'
        f'<div class="value"><span class="f">{fmt(vals[0], kind)}</span>'
        f'<span class="slash">/</span><span class="w">{fmt(vals[1], kind)}</span></div>'
        f'<div class="sub">{sub}</div></div>'
        for label, vals, kind, sub in hero_stats)

    header = f'''<header class="hero"><div class="wrap">
<div class="eyebrow light">EXISTING CUSTOMER SWAP · WHO MOVED IN / OUT</div>
<h1><span style="color:#0f766e">Swap In</span> <span class="x">×</span> <span style="color:#b45309">Swap Out</span><br>新模型置入了谁，又置出了谁</h1>
<p class="hero-lede"><strong>置入 641 条申请 / 641 人，置出 139 条申请 / 138 人；两组同产品、同版特征，合并统一评分。</strong>置换呈现出清晰的风险类型交换：置入组工资可见性更弱、现金流更薄，但负债与拒付更轻；置出组工资更稳定，但借贷与拒付更重。</p>
<div class="hero-meta">
<span><i class="dot f"></i>Swap In · 置入 · {S["swap_in"]["apps"]:,} 条申请 · {S["swap_in"]["users"]:,} 人</span>
<span><i class="dot w"></i>Swap Out · 置出 · {S["swap_out"]["apps"]:,} 条申请 · {S["swap_out"]["users"]:,} 人</span>
<span>user_type 均为 Old · 样本期 {period_start:%Y-%m-%d} 至 {period_end:%Y-%m-%d}</span>
</div>
<div class="hero-grid">{hero}</div>
<p class="hero-note">所有成对数值按 <strong>Swap In / Swap Out</strong> 排列。健康分与维度分来自合并样本的统一断点，是相对分数，不是违约概率；置换原因结合画像差异与两版入模变量反推，变量只说明模型能看什么，不代表作用方向，也不能视为模型真实决策逻辑。</p>
</div></header>'''

    nav_items = [("basis", "置换结果"), ("profile_in", "置入画像"), ("why_in", "置入原因"),
                 ("profile_out", "置出画像"), ("why_out", "置出原因"),
                 ("logic", "交换逻辑"), ("conclusion", "结论与边界")]
    nav = ('<nav class="nav" aria-label="报告章节"><div class="nav-inner">'
           + "".join(f'<a href="#{k}">{label}</a>' for k, label in nav_items) + "</div></nav>")

    seg_values = [[S[k]["segments"][s] for s in SEGMENTS] for k in KEYS]
    arch_values = [[S[k]["archetypes"][s] for s in ARCHETYPES] for k in KEYS]
    burn_values = [[S[k]["burn_bands"][s] for s in BURN_BANDS] for k in KEYS]
    buf_values = [[S[k]["buffer_bands"][s] for s in BUFFER_BANDS] for k in KEYS]
    lb_values = [[S[k]["lender_behaviors"][s] for s in LENDER_BEHAVIORS] for k in KEYS]
    ds_values = [[S[k]["dishonour_status"][s] for s in DISHONOUR_STATUS] for k in KEYS]

    parts = []

    # 00 置换结果
    content = (f'<p class="lead">新模型在这批存量客户中置入 <strong>{S["swap_in"]["apps"]:,} 条申请 / '
               f'{S["swap_in"]["users"]:,} 人</strong>，置出 <strong>{S["swap_out"]["apps"]:,} 条申请 / '
               f'{S["swap_out"]["users"]:,} 人</strong>。两组来自同一产品、同一版特征，'
               '样本期相同，用户与申请互不重叠；合并 780 条后统一评分，'
               '因此下文的分数与分群使用同一套断点。</p>')
    content += paired(
        "置换规模",
        [("申请数", [S[k]["apps"] for k in KEYS], "合并母本 780 条"),
         ("用户数", [S[k]["users"] for k in KEYS], "Swap Out 有 1 人 2 条申请")],
        caption=f"按较大组定标 · 最大 {max(S[k]['apps'] for k in KEYS):,} 条",
        maximum=max(S[k]["apps"] for k in KEYS), kind="int",
        note="申请数按申请计；用户数按人去重。差值 = Swap Out − Swap In。")
    content += share_compare("置换后的客群结构", SEGMENTS, seg_values,
                             [f'已评分 · n = {S[k]["scored"]:,}' for k in KEYS])
    content += range_rows([S[k]["health"] for k in KEYS])
    content += cards([
        ("Swap In · 置入组",
         f'{S["swap_in"]["apps"]:,} 条申请 / {S["swap_in"]["users"]:,} 人。'
         f'高风险承压占 {fmt(S["swap_in"]["segments"]["高风险承压"], "pct")}，'
         f'现金流紧张占 {fmt(S["swap_in"]["segments"]["现金流紧张"], "pct")}；'
         f'稳健工薪仅 {fmt(S["swap_in"]["segments"]["稳健工薪"], "pct")}。'),
        ("Swap Out · 置出组",
         f'{S["swap_out"]["apps"]:,} 条申请 / {S["swap_out"]["users"]:,} 人。'
         f'高风险承压占 {fmt(S["swap_out"]["segments"]["高风险承压"], "pct")}，'
         f'但稳健工薪占 {fmt(S["swap_out"]["segments"]["稳健工薪"], "pct")}，高于置入组；'
         f'健康分中位数也略高（{fmt(S["swap_out"]["health_median"], "one")}）。'),
    ])
    content += (f'<div class="callout warn"><strong>先看一个反直觉结果：</strong>'
                f'置出组的健康分中位数反而略高（{pct_pair("health_median", "one")}），'
                f'高风险承压占比也更高（{pct_pair("segments.高风险承压")}）。'
                '分群与总分给出的信号并不一致，说明新模型的置换边界不是简单按现有健康分从低到高切；'
                '它可能使用了不同权重，或叠加了负债、拒付等否决项。</div>')
    parts.append(section("basis", "00 · 置换结果", "置换结果：641 条申请置入，139 条置换出",
                         "先看规模、分群与健康分，再分别看置入与置出的画像。", content))

    # 01 置入画像
    content = (f'<p class="lead">Swap In 的画像可以概括为：<strong>收入总量不低，但工资可见性弱；'
               '现金流与流动性很薄；负债和拒付强度低于置出组。</strong>'
               f'182 天总收入中位数 {inout("raw.inc_global_sum", "k")}，但工资收入只有 '
               f'{inout("raw.inc_wage_sum", "k")}，工资占比中位数 {inout("median.wage_share")}；'
               f'多源/非工资收入占 {fmt(S["swap_in"]["archetypes"]["多源/非工资"], "pct")}，'
               f'无工资入账 {fmt(S["swap_in"]["flags"]["flag_no_wage"][0], "pct")}。</p>')
    content += grid(
        kpi("182 天总收入中位数", pv("raw.inc_global_sum"), "k", "原始金额特征"),
        kpi("工资收入中位数", pv("raw.inc_wage_sum"), "k", "原始金额特征"),
        kpi("工资收入占比中位数", pv("median.wage_share"), "pct", "每条申请先算占比，再取中位数"),
        kpi("多源/非工资收入", [S[k]["archetypes"]["多源/非工资"] for k in KEYS], "pct", "有流水申请"),
        kpi("期末余额中位数", pv("median.balance"), "money", "逐申请余额中位数，再取样本中位数"),
        kpi("缓冲天数中位数", pv("median.buffer_days"), "days", "期末余额可覆盖支出天数"),
        kpi("现金贷/工资贷使用", pv("flags.flag_payday.0"), "pct", "全部有流水申请"),
        kpi("存在未结清拒付", pv("flags.flag_unresolved_dishonour.0"), "pct", "182 天窗口内识别"),
    )
    content += share_compare("收入形态：置入组更偏多源/非工资", ARCHETYPES, arch_values,
                             [f'有流水申请 · n = {S[k]["obs"]:,}' for k in KEYS])
    content += share_compare("近 28 天收支位置", BURN_BANDS, burn_values,
                             [f'有流水申请 · n = {S[k]["obs"]:,}' for k in KEYS])
    content += share_compare("期末余额覆盖支出天数", BUFFER_BANDS, buf_values,
                             [f'有流水申请 · n = {S[k]["obs"]:,}' for k in KEYS])
    content += ('<div class="callout"><strong>画像小结：</strong>'
                '这不是“没有收入”的人群，而是“收入来源分散、工资痕迹弱、现金留存极薄”的人群。'
                '置入组的负债和拒付绝对水平低于置出组，但现金流脆弱性更高；'
                '这为下一节的“为什么可能被置入”提供了核心线索。</div>')
    parts.append(section("profile_in", "01 · 置入画像", "置入的人：非工资/多源收入、低负债、薄缓冲",
                         "Swap In 的收入、现金流、流动性与负债特征。", content))

    # 02 置入原因
    content = ('<p class="lead">数据中没有新模型的评分、SHAP 值或拒绝原因，无法直接还原决策逻辑。'
               '下面只能从置入组相对置出组的稳定差异，结合入模变量线索，反推四个可能原因；'
               '它们是<strong>画像假设</strong>，不是模型事实。</p>')
    content += cards([
        ("1. 非工资/多源收入被识别",
         f'多源/非工资收入 {inout("archetypes.多源/非工资")}；'
         f'工资占比 {inout("median.wage_share")}；'
         f'无工资入账 {inout("flags.flag_no_wage.0")}。'
         '模型可能更依赖交易流水识别收入，而不是工资标签。'),
        ("2. 历史负债更轻",
         f'有放款 {inout("disburse_rate")}；'
         f'放款笔数中位 {inout("median.disburse_cnt", "int")}；'
         f'放款金额中位 {inout("raw.db_disburse_sum", "money")}；'
         f'现金贷/工资贷 {inout("flags.flag_payday.0")}；'
         f'高负债 {inout("flags.flag_high_debt.0")}。'),
        ("3. 拒付更轻",
         f'未结清拒付 {inout("flags.flag_unresolved_dishonour.0")}；'
         f'未结清笔数中位 {inout("median.unresolved_cnt", "int")}；'
         f'金额中位 {inout("raw.unresolved_amt_p50", "money")}。'
         '整体比例区间有重叠，但笔数和金额方向一致。'),
        ("4. 用薄缓冲换低负债",
         f'支出≥收入 {inout("burn_bands.支出达到或超过收入")}；'
         f'低缓冲标签 {inout("flags.flag_low_buffer.0")}；'
         f'负现金流 {inout("flags.flag_negative_cashflow.0")}。'
         '这更像一次主动取舍：用更弱的工资可见性和更薄的缓冲，换更低的负债/拒付。'),
    ])
    content += ('<p><strong>从入模变量看：</strong>新模型并没有把“工资稳定”当成唯一主轴。'
                '两版模型的前两个变量都是债务类交易占比和拒付类入账笔数，但新模型把拒付窗口从 168 天缩短到 56 天，'
                '并加入余额波动、内部/外部转账、工资消费率、非 SACC 贷款和 BNPL 等近期信号。'
                '置入组的非工资流水、余额波动和现金流缺口，正好落在这些变量的观察范围内；'
                '工资痕迹弱，不再等于收入不可见。这只是变量层面的相容解释，不能证明模型实际按这些方向打分。</p>')
    content += (f'<div class="callout warn"><strong>排除几个直观但证据不足的解释：</strong>'
                '收入规模接近（'
                f'{pct_pair("raw.inc_global_sum", "k")}），不足以解释置换；'
                f'交易活跃度也不更高（活跃天数中位 {pct_pair("median.active_days", "int")}）；'
                f'健康分也不是更高（{pct_pair("health_median", "one")}）。'
                '最可能的解释是：新模型把负债/拒付强度放在工资稳定性之前，'
                '用“低负债 + 多源流水”换“工资可见性”。</div>')
    parts.append(section("why_in", "02 · 置入原因", "为什么新模型可能让这批人进来",
                         "结合画像差异与入模变量线索反推的四个可能驱动，不等于模型真实决策逻辑。", content))

    # 03 置出画像
    content = (f'<p class="lead">Swap Out 的画像与 Swap In 形成镜像：<strong>工资更稳定、现金流压力反而略低，'
               '但历史借贷和拒付更重。</strong>'
               f'工资收入中位数 {inout("raw.inc_wage_sum", "k")}，工资占比 {inout("median.wage_share")}，'
               f'稳定工资形态占 {fmt(S["swap_out"]["archetypes"]["稳定工资"], "pct")}；'
               f'但放款比例 {fmt(S["swap_out"]["disburse_rate"], "pct")}，'
               f'现金贷/工资贷使用 {fmt(S["swap_out"]["flags"]["flag_payday"][0], "pct")}，'
               f'未结清拒付 {fmt(S["swap_out"]["flags"]["flag_unresolved_dishonour"][0], "pct")}。</p>')
    content += grid(
        kpi("182 天工资收入中位数", pv("raw.inc_wage_sum"), "k", "原始金额特征"),
        kpi("工资收入占比中位数", pv("median.wage_share"), "pct", "每条申请先算占比，再取中位数"),
        kpi("稳定工资形态", [S[k]["archetypes"]["稳定工资"] for k in KEYS], "pct", "有流水申请"),
        kpi("有放款记录", pv("disburse_rate"), "pct", "历史 182 天 · 有流水申请"),
        kpi("放款笔数中位数", pv("median.disburse_cnt"), "int", "全部有流水申请，含 0"),
        kpi("放款金额中位数", pv("raw.db_disburse_sum"), "money", "历史 182 天 · 原始金额特征"),
        kpi("现金贷/工资贷使用", pv("flags.flag_payday.0"), "pct", "全部有流水申请"),
        kpi("存在未结清拒付", pv("flags.flag_unresolved_dishonour.0"), "pct", "182 天窗口内识别"),
    )
    content += share_compare("借贷行为：置出组历史借得更重，但窗口内更多净偿还", LENDER_BEHAVIORS, lb_values,
                             [f'有流水申请 · n = {S[k]["obs"]:,}' for k in KEYS])
    content += share_compare("拒付恢复状态：置出组更多未解决/部分解决", DISHONOUR_STATUS, ds_values,
                             [f'有流水申请 · n = {S[k]["obs"]:,}' for k in KEYS])
    content += paired("收入与现金流：置出组并不更差", [
        ("工资收入占比", pv("median.wage_share"), "每条申请先算占比，再取中位数"),
        ("近 28 天支出 ≥ 收入", [S[k]["burn_bands"]["支出达到或超过收入"] for k in KEYS], "有流水申请"),
        ("流动性缓冲不足", pv("flags.flag_low_buffer.0"), "负余额占比高或余额中位数为负"),
    ], caption="统一 0–100% 刻度", kind="pct")
    content += (f'<div class="callout"><strong>画像小结：</strong>置出组不是“现金流最差”的人——'
                f'他们的支出≥收入比例（{fmt(S["swap_out"]["burn_bands"]["支出达到或超过收入"], "pct")}）'
                f'和低缓冲标签（{fmt(S["swap_out"]["flags"]["flag_low_buffer"][0], "pct")}）都优于置入组；'
                '真正把他们与置入组区分开的是更重的历史借贷和拒付。'
                '工资更稳定，但没有抵消这两项。</div>')
    parts.append(section("profile_out", "03 · 置出画像", "置出的人：工资更稳定，但借得更多、拒付更重",
                         "Swap Out 的收入、负债、拒付与现金流特征。", content))

    # 04 置出原因
    content = ('<p class="lead">如果只看现金流，置出组并不比置入组差；真正拉开差距的是借贷与拒付。'
               '以下同样是画像与变量线索的反推，不是模型事实。</p>')
    content += cards([
        ("1. 历史借贷更重",
         f'有放款 {inout("disburse_rate")}；'
         f'放款笔数中位 {inout("median.disburse_cnt", "int")}；'
         f'金额中位 {inout("raw.db_disburse_sum", "money")}；'
         f'现金贷/工资贷 {inout("flags.flag_payday.0")}；'
         f'高负债 {inout("flags.flag_high_debt.0")}。'),
        ("2. 拒付恢复更差",
         f'未结清 {inout("flags.flag_unresolved_dishonour.0")}；'
         f'未结清笔数中位 {inout("median.unresolved_cnt", "int")}；'
         f'金额中位 {inout("raw.unresolved_amt_p50", "money")}；'
         f'部分解决 {inout("dishonour_status.部分解决")}。'
         '整体比例区间有重叠，但笔数、金额和恢复状态方向一致。'),
        ("3. 高风险分群更集中",
         f'高风险承压 {inout("segments.高风险承压")}（风险标签≥3）。'
         f'但三项压力交集置出组反而更低（{inout("joint")}），'
         '说明模型不是简单按复合标签排序。'),
        ("4. 工资稳定没有抵消上述信号",
         f'收入稳定性得分 {inout("dims.income", "one")}；'
         f'健康分中位数 {inout("health_median", "one")}。'
         '置出组靠收入维度把总分拉高；如果按现有健康分排序，置出组反而更高，'
         '新模型可能用了不同权重或额外否决项。'),
    ])
    content += ('<p><strong>从入模变量看：</strong>两版模型都看债务和拒付，差别更可能在“看多近、看多细”。'
                '新模型把拒付窗口缩短到 56 天，并加入非 SACC 贷款、BNPL 和 MyPayNow 竞品使用等更近期的负债信号；'
                '旧模型保留更多福利金、公用事业、医疗、金额总量等长期生活结构变量。'
                '置出组“工资更稳定、但近期借贷和拒付更重”的画像，与这种变量重心变化相容；'
                '它可能解释为什么这类人在两版模型下得到不同边界，但不能证明模型的实际方向。</p>')
    content += ('<div class="callout warn"><strong>最可能的解释：</strong>'
                '新模型把“历史借贷强度 + 拒付恢复”当成更强的否决项；'
                '工资稳定、现金流尚可，但不足以抵消这两项。'
                '另一个不能排除的解释是策略/政策规则而非纯模型分数——仅凭画像无法区分。</div>')
    parts.append(section("why_out", "04 · 置出原因", "为什么新模型可能把这批人换出",
                         "结合画像差异与入模变量线索反推的可能原因，以及不能排除的替代解释。", content))

    # 05 交换逻辑
    content = ('<p class="lead">把两边放在一起，置换不是“好客户 vs 坏客户”，而是一次<strong>风险类型交换</strong>：'
               '用工资稳定性换更低的负债/拒付，同时承担更高的现金流波动。</p>')
    dim_pairs = [(DIM_LABELS[d], [S[k]["dims"][d] for k in KEYS], "") for d in DIM_LABELS]
    content += paired("五维评分中位数：置出组收入更高，置入组负债得分更高", dim_pairs,
                      "合并样本统一断点 · 0–100 分，越高越好；负债压力维度得分越高表示压力越低",
                      kind="score",
                      note="IN = Swap In（圆点）；OUT = Swap Out（方点）。差值 = Swap Out − Swap In，单位：分。")
    logic_rows = [
        ("收入形态",
         f'多源/非工资 {fmt(S["swap_in"]["archetypes"]["多源/非工资"], "pct")}；'
         f'工资占比 {fmt(S["swap_in"]["median"]["wage_share"], "pct")}',
         f'多源/非工资 {fmt(S["swap_out"]["archetypes"]["多源/非工资"], "pct")}；'
         f'工资占比 {fmt(S["swap_out"]["median"]["wage_share"], "pct")}',
         "置入更依赖非工资流水"),
        ("负债强度",
         f'放款 {fmt(S["swap_in"]["disburse_rate"], "pct")}；'
         f'{fmt(S["swap_in"]["median"]["disburse_cnt"], "int")} 笔；'
         f'{fmt(S["swap_in"]["raw"]["db_disburse_sum"], "money")}；'
         f'现金贷 {fmt(S["swap_in"]["flags"]["flag_payday"][0], "pct")}',
         f'放款 {fmt(S["swap_out"]["disburse_rate"], "pct")}；'
         f'{fmt(S["swap_out"]["median"]["disburse_cnt"], "int")} 笔；'
         f'{fmt(S["swap_out"]["raw"]["db_disburse_sum"], "money")}；'
         f'现金贷 {fmt(S["swap_out"]["flags"]["flag_payday"][0], "pct")}',
         "置出借得更多"),
        ("拒付",
         f'未结清 {fmt(S["swap_in"]["flags"]["flag_unresolved_dishonour"][0], "pct")}；'
         f'{fmt(S["swap_in"]["median"]["unresolved_cnt"], "int")} 笔；'
         f'{fmt(S["swap_in"]["raw"]["unresolved_amt_p50"], "money")}',
         f'未结清 {fmt(S["swap_out"]["flags"]["flag_unresolved_dishonour"][0], "pct")}；'
         f'{fmt(S["swap_out"]["median"]["unresolved_cnt"], "int")} 笔；'
         f'{fmt(S["swap_out"]["raw"]["unresolved_amt_p50"], "money")}',
         "置出拒付更重"),
        ("现金流",
         f'支出≥收入 {fmt(S["swap_in"]["burn_bands"]["支出达到或超过收入"], "pct")}；'
         f'负现金流 {fmt(S["swap_in"]["flags"]["flag_negative_cashflow"][0], "pct")}',
         f'支出≥收入 {fmt(S["swap_out"]["burn_bands"]["支出达到或超过收入"], "pct")}；'
         f'负现金流 {fmt(S["swap_out"]["flags"]["flag_negative_cashflow"][0], "pct")}',
         "置入现金流更紧"),
        ("流动性",
         f'低缓冲 {fmt(S["swap_in"]["flags"]["flag_low_buffer"][0], "pct")}；'
         f'余额 {fmt(S["swap_in"]["median"]["balance"], "money")}',
         f'低缓冲 {fmt(S["swap_out"]["flags"]["flag_low_buffer"][0], "pct")}；'
         f'余额 {fmt(S["swap_out"]["median"]["balance"], "money")}',
         "置入缓冲更薄"),
        ("分群与健康分",
         f'高风险 {fmt(S["swap_in"]["segments"]["高风险承压"], "pct")}；'
         f'健康分 {fmt(S["swap_in"]["health_median"], "one")}',
         f'高风险 {fmt(S["swap_out"]["segments"]["高风险承压"], "pct")}；'
         f'健康分 {fmt(S["swap_out"]["health_median"], "one")}',
         "置出健康分反而更高"),
    ]
    content += ('<table class="logic-table"><thead><tr><th>维度</th><th>Swap In · 置入</th>'
                '<th>Swap Out · 置出</th><th>交换方向</th></tr></thead><tbody>'
                + "".join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>'
                          for a, b, c, d in logic_rows)
                + "</tbody></table>")
    content += paired("风险标签差值排序：Swap Out − Swap In",
                      [(label, vals, "") for label, vals, _ci, _gap in flag_gaps[:8]],
                      caption="按绝对差排序 · 统一 0–100% 刻度", kind="pct",
                      note="按两组发生率的绝对差排序；圆点为 Swap In，方点为 Swap Out，右侧为 Swap Out − Swap In。")
    content += (f'<p class="table-note">收入形态标准化：未结清拒付原始差 '
                f'{pp("flags.flag_unresolved_dishonour.0")}，按合并样本三类收入形态加权后差 '
                f'{cross_std["flag_unresolved_dishonour"][1] - cross_std["flag_unresolved_dishonour"][0]:+.1f}pp；'
                f'现金贷/工资贷原始差 {pp("flags.flag_payday.0")}，标准化后仍 '
                f'{cross_std["flag_payday"][1] - cross_std["flag_payday"][0]:+.1f}pp。'
                '收入结构解释了一部分拒付差，但解释不了借贷差。</p>')
    content += ('<div class="callout"><strong>关键结论：</strong>'
                '新模型的置换边界与现有健康分不完全一致。'
                '它更像是在以“负债与拒付”为主要区分轴，而不是按综合健康分排序；'
                '两版模型的变量窗口与重心差异，为这种边界变化提供了相容解释，但不能证明方向；'
                '代价是置入组的现金流更脆弱。</div>')
    content += cards([
        ("如果目标是降低坏账",
         '需要验证低负债/低拒付能否抵消薄缓冲带来的逾期风险；'
         '重点跟踪 Swap In 的现金流断裂与小额拒付滚动。'),
        ("如果目标是扩大覆盖",
         '置入组确实覆盖了传统工资模型不容易识别的非工资/多源收入人群；'
         '价值在覆盖，而不是风险更低。'),
    ])
    parts.append(section("logic", "05 · 交换逻辑", "模型在用什么换什么",
                         "把两边放在一起看：这不是“好客户 vs 坏客户”，而是一次风险类型交换。", content))

    # 06 结论
    content = ('<p class="lead">一句话总结：新模型置入的是<strong>“负债更轻、工资更不可见、现金流更紧”</strong>的人；'
               '置出的是<strong>“工资更稳定、但借贷与拒付更重”</strong>的人。</p>')
    content += cards([
        ("置入侧 · 为什么可能进",
         f'多源/非工资 {fmt(S["swap_in"]["archetypes"]["多源/非工资"], "pct")}；'
         f'负债轻（放款 {fmt(S["swap_in"]["disburse_rate"], "pct")}、'
         f'现金贷 {fmt(S["swap_in"]["flags"]["flag_payday"][0], "pct")}）；'
         f'拒付轻（{fmt(S["swap_in"]["flags"]["flag_unresolved_dishonour"][0], "pct")}）；'
         f'代价是现金流薄（支出≥收入 {fmt(S["swap_in"]["burn_bands"]["支出达到或超过收入"], "pct")}、'
         f'低缓冲 {fmt(S["swap_in"]["flags"]["flag_low_buffer"][0], "pct")}）。'),
        ("置出侧 · 为什么可能出",
         f'工资更稳定（工资占比 {fmt(S["swap_out"]["median"]["wage_share"], "pct")}）；'
         f'但借贷重（放款 {fmt(S["swap_out"]["disburse_rate"], "pct")}、'
         f'{fmt(S["swap_out"]["median"]["disburse_cnt"], "int")} 笔、'
         f'{fmt(S["swap_out"]["raw"]["db_disburse_sum"], "money")}）；'
         f'拒付重（{fmt(S["swap_out"]["flags"]["flag_unresolved_dishonour"][0], "pct")}、'
         f'{fmt(S["swap_out"]["median"]["unresolved_cnt"], "int")} 笔、'
         f'{fmt(S["swap_out"]["raw"]["unresolved_amt_p50"], "money")}）。'),
    ])
    content += cards([
        ("验证模型原因",
         '画像差异与两版入模变量的窗口/重心变化相容：新模型更偏近期余额、转账和负债行为；'
         '下一步拉取模型评分、SHAP/特征重要度与拒绝原因码，核对方向与强度。'),
        ("验证置换效果",
         '等 duedate_1m_5 等表现标签成熟，比较 Swap In / Swap Out 的逾期与还款；'
         '当前标签仅 83/641 与 31/139 有值，阳性 11/5，不能做结果对比。'),
        ("验证策略属性",
         '确认置出是否包含政策、名单或人工规则；仅凭画像无法区分“模型分数”与“策略规则”。'),
    ])
    content += ('<div class="callout warn"><strong>结论适用边界：</strong>'
                '（1）两组按置换结果划分，不是随机实验，差异不能归因于置换动作本身；'
                '（2）Swap Out 仅 139 条，拒付、流动性、近期缺口等指标区间重叠；'
                '（3）健康分是合并样本内的相对分数，不等同于违约概率；'
                '（4）画像差异只能支持“可能原因”，不能证明模型真实决策逻辑。'
                '</div>')
    content += ('<details class="details"><summary>数据来源与口径</summary><div class="details-body">'
                '<p>输入：existing_swap_in_v1_2_with_feature_all.csv、existing_swap_out_v1_2_with_feature_all.csv；'
                '合并后 780 条申请。输出：analysis/swap_compare/out/。'
                '评分沿用 products/fundo.json 的 122 个注册指标与 46 个计分指标，断点由合并样本统一计算。</p>'
                '<p>金额类指标取自原始特征并对哨兵值做同一清洗；未结清/已结清金额为六类拒付金额加总。'
                '分群、维度分与标签均直接读取 user_scores.csv，不重新定义底层指标。'
                '收入形态标准化以稳定工资、不稳定工资、多源/非工资三类在合并有流水样本中的占比为权重；'
                '福利依赖组 Swap Out 为 0 条，未纳入。</p>'
                '<p>两组 sample_datetime 覆盖 2026-09-14 至 2026-10-13，均为 Old 用户；'
                'Swap Out 的 139 条申请来自 138 个 user_id，另 1 人为 2 条申请。</p>'
                '</div></details>')
    parts.append(section("conclusion", "06 · 结论与边界", "结论：置入了谁，为什么；置出了谁，为什么",
                         "一页总结，以及需要后续验证的边界。", content))

    footer = ('<footer class="footer"><div class="wrap footer-inner">'
              f'<span>Swap In × Swap Out 置换画像 · 合并统一评分</span>'
              f'<span>样本 780 条 · 生成 {pd.Timestamp.now():%Y-%m-%d %H:%M} · '
              '<a href="#basis">数据与口径</a></span></div></footer>')

    output = ('<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width, initial-scale=1">'
              '<meta name="description" content="新模型置入了谁、置出了谁：Swap In / Swap Out 的画像、可能原因与交换逻辑。">'
              f'<title>新模型置入了谁，又置出了谁 · Swap In × Swap Out</title><style>{css}</style></head>'
              f'<body>{header}{nav}<main>{"".join(parts)}</main>{footer}</body></html>\n')
    REPORT.write_text(output, encoding="utf-8")
    print(f"report: {REPORT} ({REPORT.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
