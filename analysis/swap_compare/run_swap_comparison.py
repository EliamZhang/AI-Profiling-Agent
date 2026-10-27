#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Swap In vs Swap Out 客群画像对比（临时脚本）。

独立放在 analysis/swap_compare/，不修改现有产品配置与分析引擎：
1. 合并两份 v1_2 特征文件，加 cohort 列；
2. 用 products/fundo.json 的同一套指标/阈值对合并样本统一评分；
3. 按 cohort 聚合 user_scores 与原始金额特征，生成自包含 HTML 对比页。

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
COLORS = ["#0e7490", "#1e40af", "#64748b", "#b45309", "#dc2626", "#7c3aed", "#0891b2"]
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


def compare(title: str, rows, caption: str = "统一 0–100% 刻度", maximum: float = 100) -> str:
    out = (f'<div class="compare"><div class="compare-head"><h3>{title}</h3>'
           f'<span>{caption}</span></div>')
    out += ('<div class="chart-axis"><span></span><div class="bar-axis"><span></span><div>'
            f'<span>0</span><span>{maximum / 2:g}</span><span>{maximum:g}</span></div>'
            f'<span>{NAMES[0]} / {NAMES[1]}</span></div></div>')
    for label, values, kind, detail in rows:
        lines = "".join(
            f'<div class="bar-line"><span class="bar-tag {c}">{tag}</span>'
            f'<div class="track"><span class="fill {c}" style="width:{max(0, min(v / maximum * 100, 100)):.4f}%"></span></div>'
            f'<span class="bar-val">{fmt(v, kind)}</span></div>'
            for c, tag, v in zip(("f", "w"), ("I", "O"), values)
        )
        out += (f'<div class="compare-row"><div class="compare-label">{label}'
                f'<small>{detail}</small></div><div class="bar-stack">{lines}</div></div>')
    return out + "</div>"


def dumbbell(title: str, rows, caption: str, maximum: float = 100, suffix: str = "%") -> str:
    out = (f'<div class="compare"><div class="compare-head"><h3>{title}</h3>'
           f'<span>{caption}</span></div>')
    out += (f'<div class="point-axis"><span>0{suffix}</span><span>{maximum / 2:g}{suffix}</span>'
            f'<span>{maximum:g}{suffix}</span></div>')
    for label, values, detail in rows:
        a, b = values
        out += f'''<div class="point-row"><div class="compare-label">{label}<small>{detail}</small></div>
        <div class="point-track" role="img" aria-label="{esc(label)}：Swap In {a:.1f}{suffix}，Swap Out {b:.1f}{suffix}">
        <span class="point-link" style="left:{min(a, b) / maximum * 100:.4f}%;width:{abs(b - a) / maximum * 100:.4f}%"></span>
        <span class="point f" style="left:{a / maximum * 100:.4f}%"></span><span class="point w" style="left:{b / maximum * 100:.4f}%"></span></div>
        <div class="point-values"><span class="f">IN {a:.1f}{suffix}</span><span class="w">OUT {b:.1f}{suffix}</span><b>{b - a:+.1f}{suffix.replace('%', 'pp')}</b></div></div>'''
    return out + f'<p class="table-note">IN = Swap In（圆点）；OUT = Swap Out（方点）。差值均为 Swap Out − Swap In。</p></div>'


def stack(title: str, categories, values, contexts, colors=None) -> str:
    colors = colors or COLORS[: len(categories)]
    out = f'<div class="stack-card"><h3>{title}</h3>'
    for name, c, vals, ctx in zip(NAMES, ("f", "w"), values, contexts):
        aria = "；".join(f"{cat} {fmt(v, 'pct')}" for cat, v in zip(categories, vals))
        out += (f'<div class="stack-row"><div class="stack-meta"><b class="{c}">{name}</b>'
                f'<span>{ctx}</span></div><div class="stack" role="img" aria-label="{esc(name + "：" + aria)}">')
        for cat, v, color in zip(categories, vals, colors):
            out += (f'<span class="seg" style="width:{v:.6f}%;background:{color}" '
                    f'title="{esc(cat)}：{fmt(v, "pct")}">{fmt(v, "pct") if v >= 9 else ""}</span>')
        out += "</div></div>"
    out += '<div class="legend">' + "".join(
        f'<span><i class="dot" style="background:{color}"></i>{esc(cat)}</span>'
        for cat, color in zip(categories, colors)) + "</div>"
    out += ('<p class="table-note">完整占比：' + "；".join(
        f'{esc(cat)} {fmt(values[0][i], "pct")} / {fmt(values[1][i], "pct")}'
        for i, cat in enumerate(categories)) + "（IN / OUT）。</p>")
    return out + "</div>"


def diffchart(title: str, rows, maximum: float, caption: str) -> str:
    out = (f'<div class="diff-chart"><div class="diff-head"><h3>{title}</h3><span>{caption}</span></div>'
           f'<div class="diff-legend"><span class="f">← Swap In 更高</span>'
           f'<span class="w">Swap Out 更高 →</span></div>')
    out += f'<div class="diff-axis"><span>−{maximum:g}pp</span><span>0</span><span>+{maximum:g}pp</span></div>'
    for label, values in rows:
        a, b = values
        d = b - a
        cls = "pos" if d >= 0 else "neg"
        out += (f'<div class="diff-row"><div class="diff-label">{label}'
                f'<small>IN {a:.1f}% · OUT {b:.1f}%</small></div>'
                f'<div class="diff-track"><span class="diff-fill {cls}" '
                f'style="width:{min(abs(d) / maximum * 50, 50):.4f}%"></span></div>'
                f'<div class="diff-val {cls}">{d:+.1f}pp</div></div>')
    return out + "</div>"


def range_rows(rows) -> str:
    out = ('<div class="compare"><div class="compare-head"><h3>健康分分布：中位数与中间 50%</h3>'
           '<span>合并样本统一断点 · 同一 0–100 分刻度</span></div>'
           '<div class="range-axis"><span>0 分</span><span>50 分</span><span>100 分</span></div>')
    for name, c, (p25, p50, p75) in zip(NAMES, ("f", "w"), rows):
        color = "var(--fundo)" if c == "f" else "var(--wagego)"
        out += (f'<div class="range-row"><div class="range-label">{name}</div>'
                f'<div class="range-track" role="img" aria-label="P25 {p25:.1f}，中位数 {p50:.1f}，P75 {p75:.1f}">'
                f'<span class="range-iqr" style="left:{p25:.4f}%;width:{p75 - p25:.4f}%;background:{color}"></span>'
                f'<span class="range-median" style="left:{p50:.4f}%;background:{color}"></span></div>'
                f'<div class="range-value" style="color:{color}">{p50:.1f}</div></div>')
    return out + '<p class="table-note">浅色区间为 P25–P75，竖线为中位数。区间反映分布，不是置信区间。</p></div>'


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
:root{--fundo:#0f766e;--fundo-dark:#115e59;--fundo-soft:#ccfbf1;--wagego:#b45309;--wagego-dark:#92400e;--wagego-soft:#fef3c7}
.flag-table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px}
.flag-table th,.flag-table td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums}
.flag-table th:first-child,.flag-table td:first-child{text-align:left}
.flag-table th{font-size:11.5px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.03em}
.flag-table td.ci{color:var(--muted);font-size:11.5px}
.flag-table td.d{font-weight:700}
.flag-table tr:hover td{background:#f8fafc}
.cohort-pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700;margin-right:6px}
.cohort-pill.in{background:var(--fundo-soft);color:var(--fundo-dark)}
.cohort-pill.out{background:var(--wagego-soft);color:var(--wagego-dark)}
@media(max-width:640px){.flag-table{font-size:12px}.flag-table th,.flag-table td{padding:7px 5px}}
"""

    in_p, out_p = S["swap_in"]["period"], S["swap_out"]["period"]
    hero_stats = [
        ("健康分中位数", pv("health_median"), "two", "统一断点 · 0–100 分"),
        ("高风险承压占比", [S[k]["segments"]["高风险承压"] for k in KEYS], "pct", "已评分申请"),
        ("存在未结清拒付", [S[k]["flags"]["flag_unresolved_dishonour"][0] for k in KEYS], "pct", "全部有流水申请"),
        ("三项压力交集", [S[k]["joint"] for k in KEYS], "pct", "现金贷 × 未结清 × 薄缓冲"),
    ]
    hero = "".join(
        f'<div class="hero-stat"><div class="label">{label}</div>'
        f'<div class="value"><span class="f">{fmt(vals[0], kind)}</span>'
        f'<span class="slash">/</span><span class="w">{fmt(vals[1], kind)}</span></div>'
        f'<div class="sub">{sub}</div></div>'
        for label, vals, kind, sub in hero_stats)

    header = f'''<header class="hero"><div class="wrap">
<div class="eyebrow light">COHORT PROFILE COMPARISON · EXISTING CUSTOMER SWAP</div>
<h1><span style="color:#0f766e">Swap In</span> <span class="x">×</span> <span style="color:#b45309">Swap Out</span><br>同一产品、同一版特征下的两组存量客群画像</h1>
<p class="hero-lede"><strong>两组均为存量客户（user_type 全部为 Old），且均具备可用银行流水。</strong>合并 780 条申请后以统一断点评分：Swap Out 的工资型收入更集中，借贷与拒付活动更重；Swap In 的非工资/多源收入更多，近期收支缺口与薄缓冲更常见。</p>
<div class="hero-meta">
<span><i class="dot f"></i>Swap In · {S["swap_in"]["apps"]:,} 条申请 · {S["swap_in"]["users"]:,} 人</span>
<span><i class="dot w"></i>Swap Out · {S["swap_out"]["apps"]:,} 条申请 · {S["swap_out"]["users"]:,} 人</span>
<span>user_type 均为 Old · 样本期 {in_p[0]:%Y-%m-%d} 至 {out_p[1]:%Y-%m-%d}</span>
</div>
<div class="hero-grid">{hero}</div>
<p class="hero-note">所有成对数值按 <strong>Swap In / Swap Out</strong> 排列。两组用户与申请均不重叠。健康分与维度分来自合并样本的统一断点，但仍是相对分数，不是违约概率。金额沿用 A$ 展示。</p>
</div></header>'''

    nav_items = [("basis", "样本与口径"), ("segments", "客群结构"), ("income", "收入"),
                 ("lender", "负债"), ("dishonour", "拒付"), ("liquidity", "流动性"),
                 ("trend", "近期变化"), ("flags", "风险标签"), ("summary", "结论")]
    nav = ('<nav class="nav" aria-label="报告章节"><div class="nav-inner">'
           + "".join(f'<a href="#{k}">{label}</a>' for k, label in nav_items) + "</div></nav>")

    parts = []

    # 00 样本与口径
    content = ('<p class="lead">两份样本均来自同一产品的存量客群（<strong>user_type 全部为 Old</strong>），'
               '用户与申请互不重叠，且均具备可用银行流水。合并后统一扫描 780 条申请，'
               '断点、评分与分群只计算一次，因此下文的分数和分群占比基于<strong>同一套评分基准</strong>，可直接横向比较。</p>')
    content += compare(
        "样本规模（申请数）",
        [("申请数", [S[k]["apps"] for k in KEYS], "int", "合并母本 780 条"),
         ("用户数", [S[k]["users"] for k in KEYS], "int", "Swap Out 有 1 人 2 条申请")],
        caption=f"按较大组定标 · 最大 {max(S[k]['apps'] for k in KEYS):,} 条",
        maximum=max(S[k]["apps"] for k in KEYS))
    content += compare(
        "评分覆盖",
        [("有流水申请占比", [S[k]["obs"] / S[k]["apps"] * 100 for k in KEYS], "pct", "两组均为 100%"),
         ("已评分占比", [S[k]["scored"] / S[k]["obs"] * 100 for k in KEYS], "pct", "健康分可用")],
        caption="统一 0–100% 刻度")
    content += cards([
        ("Swap In · 进入组",
         f'{in_p[0]:%Y-%m-%d} 至 {in_p[1]:%Y-%m-%d}，<strong>{S["swap_in"]["apps"]:,} 条申请 / {S["swap_in"]["users"]:,} 人</strong>，'
         f'全部有流水。数据中不含分组规则，本报告仅作描述性对比。'),
        ("Swap Out · 退出组",
         f'{out_p[0]:%Y-%m-%d} 至 {out_p[1]:%Y-%m-%d}，<strong>{S["swap_out"]["apps"]:,} 条申请 / {S["swap_out"]["users"]:,} 人</strong>，'
         f'全部有流水；样本量约为 Swap In 的 22%。'),
    ])
    content += ('<div class="callout warn"><strong>比较边界：</strong>两组是按置换结果划分的存量客群，不是随机分组；'
                '差异可能来自客群本身、置换规则或两者共同作用，不能解释为因果。Swap Out 仅 139 条，'
                '比例类指标的 95% Wilson 区间明显更宽，以下章节对区间重叠的差异仅作方向性参考。</div>')
    content += ('<details class="details"><summary>合并与评分方式</summary><div class="details-body">'
                '<p>两份文件列结构一致（13,363 个特征列，另有一列导出索引），合并后加上 cohort 标记。'
                '特征清洗沿用现有引擎：数值哨兵 -999999.99999 视为缺失，lender 模块 -1 视为无匹配记录，'
                '可加总金额/笔数/机构数的 -1 还原为 0。</p>'
                '<p>统一评分只改变断点与百分位的参照人群，不改变指标定义、维度和权重；'
                '因此本报告的健康分与分群可与合并样本内部互相对照，但不能与之前 Fundo 全量报告中的绝对分数直接对齐。</p>'
                '</div></details>')
    parts.append(section("basis", "00 · 样本与口径", "样本范围与评分口径", "合并 780 条申请统一评分；两组均为存量客户，且均有可用流水。", content))

    # 01 客群结构
    seg_values = [[S[k]["segments"][s] for s in SEGMENTS] for k in KEYS]
    content = (f'<p class="lead">统一评分后，Swap Out 的<strong>高风险承压</strong>占比更高'
               f'（{fmt(S["swap_in"]["segments"]["高风险承压"], "pct")} / {fmt(S["swap_out"]["segments"]["高风险承压"], "pct")}，'
               f'{pp("segments.高风险承压")}），Swap In 更偏<strong>现金流紧张</strong>与<strong>月光消费</strong>。'
               f'两组的健康分中位数接近（{fmt(S["swap_in"]["health_median"], "two")} / {fmt(S["swap_out"]["health_median"], "two")}），'
               '说明总分接近并不等于压力类型相同。</p>')
    content += stack("已评分申请的客群结构", SEGMENTS, seg_values,
                     [f'已评分 · n = {S[k]["scored"]:,}' for k in KEYS])
    content += range_rows([S[k]["health"] for k in KEYS])
    dim_pairs = [(DIM_LABELS[d], [S[k]["dims"][d] for k in KEYS],
                  "得分越高越好；负债压力维度得分越高表示压力越低") for d in DIM_LABELS]
    content += dumbbell("五维评分中位数：差异主要在收入与负债", dim_pairs,
                        "合并样本统一断点 · 同一 0–100 分刻度")
    content += ('<div class="callout"><strong>结构解读：</strong>分群同时受评分和风险标签数影响，'
                '与五维得分不是独立证据。Swap Out 的负债与拒付标签更多，但收入稳定性得分更高，'
                '两个方向共同把健康分拉得接近。</div>')
    parts.append(section("segments", "01 · 客群结构", "客群结构：高风险承压与现金流紧张的分布差异",
                         "统一评分后的分群结构、健康分分布与五维得分。", content))

    # 02 收入
    content = (f'<p class="lead">总收入规模接近（182 天中位数 {pct_pair("raw.inc_global_sum", "k")}），'
               f'但收入来源结构差异很大：Swap Out 的工资收入中位数是 Swap In 的 '
               f'{S["swap_out"]["raw"]["inc_wage_sum"] / S["swap_in"]["raw"]["inc_wage_sum"]:.2f} 倍，'
               f'工资占比中位数 {pct_pair("median.wage_share")}。Swap In 的多源/非工资收入占 '
               f'{fmt(S["swap_in"]["archetypes"]["多源/非工资"], "pct")}，Swap Out 为 '
               f'{fmt(S["swap_out"]["archetypes"]["多源/非工资"], "pct")}。</p>')
    content += grid(
        kpi("182 天总收入中位数", pv("raw.inc_global_sum"), "k", "原始金额特征"),
        kpi("182 天工资收入中位数", pv("raw.inc_wage_sum"), "k", "原始金额特征"),
        kpi("工资收入占比中位数", pv("median.wage_share"), "pct", "每条申请先算占比，再取中位数",
            delta=pp("median.wage_share")),
    )
    arch_values = [[S[k]["archetypes"][s] for s in ARCHETYPES] for k in KEYS]
    content += stack("收入形态：工资型 vs 多源/非工资", ARCHETYPES, arch_values,
                     [f'有流水申请 · n = {S[k]["obs"]:,}' for k in KEYS],
                     ["#0e7490", "#1e40af", "#64748b", "#b45309"])
    content += dumbbell("收入相关标签", [
        ("无工资入账", pv("flags.flag_no_wage.0"), "观察性标签，不计入分群"),
        ("福利收入依赖", pv("flags.flag_welfare_dependent.0"), "观察性标签，不计入分群"),
        ("收入不稳定", pv("flags.flag_income_unstable.0"), "计入分群的风险标签"),
    ], "分母：全部有流水申请 · 统一 0–100% 刻度")
    content += ('<div class="callout"><strong>收入规模接近，差异集中在结构：</strong>两组总收入中位数只差约 3.5%；'
                '真正的差异是工资的可见性和来源集中度。Swap In 有更多福利依赖和无工资入账，'
                '这不等于收入更低，但会影响收入稳定性的识别与后续额度判断。'
                '收入形态内部的标签对比与标准化见「07 · 风险标签」。</div>')
    parts.append(section("income", "02 · 收入", "收入：总量接近，工资可见性与来源结构差异明显",
                         "收入规模、来源结构、工资占比与相关标签。", content))

    # 03 负债
    content = (f'<p class="lead">Swap Out 的借贷活动明显更重：有放款记录的比例 '
               f'{pct_pair("disburse_rate")}，放款笔数中位数 {pct_pair("median.disburse_cnt", "int")}，'
               f'放款金额中位数 {pct_pair("raw.db_disburse_sum", "money")}；'
               f'现金贷/工资贷使用标签为 {pct_pair("flags.flag_payday.0")}（{pp("flags.flag_payday.0")}）。</p>')
    content += grid(
        kpi("有放款记录", pv("disburse_rate"), "pct", "历史 182 天 · 有流水申请", delta=pp("disburse_rate")),
        kpi("放款笔数中位数", pv("median.disburse_cnt"), "int", "全部有流水申请，含 0", delta=pp("median.disburse_cnt")),
        kpi("放款金额中位数", pv("raw.db_disburse_sum"), "money", "历史 182 天 · 原始金额特征"),
        kpi("现金贷/工资贷放款中位数", pv("raw.db_cash_advance_sum"), "money", "历史 182 天 · 原始金额特征"),
        kpi("还款金额中位数", pv("raw.db_repay_sum"), "money", "历史 182 天 · 原始金额特征"),
        kpi("高负债压力标签", pv("flags.flag_high_debt.0"), "pct", "按合并样本评分触发", delta=pp("flags.flag_high_debt.0")),
    )
    lb_values = [[S[k]["lender_behaviors"][s] for s in LENDER_BEHAVIORS] for k in KEYS]
    content += stack("借贷行为：Swap Out 更偏净借款", LENDER_BEHAVIORS, lb_values,
                     [f'有流水申请 · n = {S[k]["obs"]:,}' for k in KEYS])
    content += dumbbell("借款广度", [
        ("至少 1 家活跃借贷机构", pv("active_lender_ge1"), "活跃机构按去重口径；全部有流水申请"),
        ("名单内多头借贷", pv("flags.flag_multi_lender.0"), "仅竞品名单口径，触发率很低"),
    ], "统一 0–100% 刻度")
    content += ('<p class="table-note">“净借款/净融资扩张”描述窗口内借入与偿还的相对关系，不等于当前负债余额；'
                '机构数只统计可识别的放款记录，未识别机构不会进入分子。</p>')
    parts.append(section("lender", "03 · 负债", "负债：Swap Out 借贷参与更深，现金贷使用更集中",
                         "借贷参与、借款广度、金额规模、近期行为与负债标签。", content))

    # 04 拒付
    content = (f'<p class="lead">未结清拒付比例 {pct_pair("flags.flag_unresolved_dishonour.0")}'
               f'（{pp("flags.flag_unresolved_dishonour.0")}），但两组 95% Wilson 区间有重叠。'
               f'Swap Out 的未结清笔数中位数为 {pct_pair("median.unresolved_cnt", "int")}，'
               f'未结清金额中位数 {pct_pair("raw.unresolved_amt_p50", "money")}；'
               f'按全部有流水申请折算的未结清金额均值 {pct_pair("raw.unresolved_amt_mean", "money")}。</p>')
    content += grid(
        kpi("存在未结清拒付", pv("flags.flag_unresolved_dishonour.0"), "pct",
            "182 天窗口内未匹配到结清", delta=pp("flags.flag_unresolved_dishonour.0")),
        kpi("未结清笔数中位数", pv("median.unresolved_cnt"), "int", "六类拒付加总"),
        kpi("未结清金额中位数", pv("raw.unresolved_amt_p50"), "money", "每条申请口径"),
        kpi("已结清金额占比", pv("raw.resolved_amt_ratio"), "pct",
            "已结清金额 /（已结清 + 未结清）", delta=pp("raw.resolved_amt_ratio")),
    )
    ds_values = [[S[k]["dishonour_status"][s] for s in DISHONOUR_STATUS] for k in KEYS]
    content += stack("拒付恢复状态", DISHONOUR_STATUS, ds_values,
                     [f'有流水申请 · n = {S[k]["obs"]:,}' for k in KEYS],
                     ["#0e7490", "#1e40af", "#b45309", "#dc2626"])
    content += compare("未结清金额：仅受影响申请", [
        ("未结清金额均值",
         [S[k]["raw"]["unresolved_amt_affected_mean"] for k in KEYS],
         "money",
         f'仅存在未结清拒付的申请：IN {S["swap_in"]["raw"]["unresolved_amt_affected_n"]} / OUT {S["swap_out"]["raw"]["unresolved_amt_affected_n"]} 条'),
    ], caption="金额均值 · 统一 0–A$3,000 刻度", maximum=3000)
    content += ('<div class="callout warn"><strong>“未结清”是窗口内识别状态：</strong>'
                '表示回溯窗口内没有匹配到补扣结清，不等同于当前违约。两组观察窗口相同（182 天），'
                '但样本日期不同，后续表现仍需用同一追踪期限验证。</div>')
    parts.append(section("dishonour", "04 · 拒付", "拒付：Swap Out 未结清比例更高，但差异区间有重叠",
                         "未结清比例、笔数、金额、恢复状态与受影响申请强度。", content))

    # 05 流动性
    content = (f'<p class="lead">两组余额都很薄：余额中位数 {pct_pair("median.balance", "money")}，'
               f'负余额天数占比中位数 {pct_pair("median.negative_share")}。'
               f'Swap In 的<strong>流动性缓冲不足</strong>标签更高（{pct_pair("flags.flag_low_buffer.0")}，'
               f'{pp("flags.flag_low_buffer.0")}），但两组 95% 区间同样有重叠；'
               f'期末余额覆盖不足 7 天的比例都超过 96%。</p>')
    content += grid(
        kpi("余额中位数", pv("median.balance"), "money", "逐申请余额中位数，再取样本中位数"),
        kpi("负余额天数占比中位数", pv("median.negative_share"), "pct", "182 天窗口"),
        kpi("流动性缓冲不足", pv("flags.flag_low_buffer.0"), "pct",
            "负余额占比高或余额中位数为负", delta=pp("flags.flag_low_buffer.0")),
        kpi("固定支出备用月数中位数", pv("raw.lq_reserve_months"), "months", "原始特征有效值"),
    )
    buf_values = [[S[k]["buffer_bands"][s] for s in BUFFER_BANDS] for k in KEYS]
    content += stack("期末余额覆盖支出天数", BUFFER_BANDS, buf_values,
                     [f'有流水申请 · n = {S[k]["obs"]:,}' for k in KEYS],
                     ["#dc2626", "#b45309", "#1e40af", "#0e7490"])
    content += ('<div class="callout"><strong>共同短板：</strong>'
                '“不足 7 天”在两组都接近或超过 99%，更适合作为共同跟踪项；'
                '“流动性缓冲不足”标签的组间差为 '
                f'{pp("flags.flag_low_buffer.0")}，方向明确但区间重叠，不宜当作稳定分层依据。</div>')
    parts.append(section("liquidity", "05 · 流动性", "流动性：两组缓冲均薄，Swap In 压力略高",
                         "余额水平、持续透支、缓冲标签与支出覆盖天数。", content))

    # 06 近期变化
    content = (f'<p class="lead">近 28 天<strong>支出达到或超过收入</strong>的比例为 '
               f'{pct_pair("burn_bands.支出达到或超过收入")}（{pp("burn_bands.支出达到或超过收入")}），'
               f'Swap In 更突出；收入下降超过 20% 的比例为 {pct_pair("income_decline")}，'
               f'Swap Out 更高。两组的刚性支出占比都很低（高承诺支出/收入 ≥30%：{pct_pair("commit_ge30")}）。</p>')
    content += grid(
        kpi("近 28 天支出 ≥ 收入", [S[k]["burn_bands"]["支出达到或超过收入"] for k in KEYS], "pct",
            "收入为零等缺失不计为正常", delta=pp("burn_bands.支出达到或超过收入")),
        kpi("日均收入下降 >20%", pv("income_decline"), "pct", "近 28 天 vs 此前 154 天", delta=pp("income_decline")),
        kpi("日均支出上升 ≥20%", pv("expense_rise"), "pct", "近 28 天 vs 此前 154 天", delta=pp("expense_rise")),
        kpi("高承诺支出/收入 ≥30%", pv("commit_ge30"), "pct", "182 天 · 同一资质口径", delta=pp("commit_ge30")),
    )
    burn_values = [[S[k]["burn_bands"][s] for s in BURN_BANDS] for k in KEYS]
    content += stack("近 28 天收支位置", BURN_BANDS, burn_values,
                     [f'有流水申请 · n = {S[k]["obs"]:,}' for k in KEYS],
                     ["#0e7490", "#1e40af", "#dc2626", "#94a3b8"])
    content += ('<p class="table-note">近期比值是同一条申请内两个不重叠窗口的比较，不是逐月趋势；'
                '一次性收入、窗口内新增账户和流水历史不足都会影响解释。'
                f'支出上升 ≥20% 为 {pct_pair("expense_rise")}，'
                f'固定账单/收入 ≥20% 为 {pct_pair("fixed_ge20")}。</p>')
    parts.append(section("trend", "06 · 近期变化", "近期变化：Swap In 的收支缺口更常见",
                         "近 28 天与此前 154 天的日均比较，以及刚性支出负担。", content))

    # 07 风险标签
    content = (f'<p class="lead">单项标签中差异最大的是<strong>现金贷/工资贷使用</strong>'
               f'（{pct_pair("flags.flag_payday.0")}，{pp("flags.flag_payday.0")}）和'
               f'<strong>无工资入账</strong>（{pct_pair("flags.flag_no_wage.0")}，{pp("flags.flag_no_wage.0")}）；'
               f'与借贷相关的负债压力高、收入不稳定也更偏向 Swap Out。'
               f'三项压力交集为 {pct_pair("joint")}（{pp("joint")}），方向与单项标签并不一致。</p>')
    content += flag_table(S)
    content += ('<p class="table-note">分母均为该组全部有流水申请。CI 为 95% Wilson 区间，'
                '把当前 cohort 视为样本时的粗略不确定性参考；区间重叠的组间差仅作方向性参考。</p>')
    for flag, title in (("flag_payday", "按收入形态分层：现金贷/工资贷使用"),
                        ("flag_unresolved_dishonour", "按收入形态分层：未结清拒付")):
        content += dumbbell(title, [
            (grp, [cross[flag][grp][0][0], cross[flag][grp][1][0]],
             f'组内有流水：IN {cross[flag][grp][0][1]} / OUT {cross[flag][grp][1][1]}')
            for grp in CROSS_GROUPS], "各收入形态内分别比较 · 统一 0–100% 刻度")
    content += (f'<div class="callout"><strong>收入形态标准化：</strong>'
                f'未结清拒付原始差 {pp("flags.flag_unresolved_dishonour.0")}，'
                f'按合并样本三类收入形态加权后为 {cross_std["flag_unresolved_dishonour"][0]:.1f}% / '
                f'{cross_std["flag_unresolved_dishonour"][1]:.1f}%'
                f'（差 {cross_std["flag_unresolved_dishonour"][1] - cross_std["flag_unresolved_dishonour"][0]:+.1f}pp）；'
                f'现金贷/工资贷使用原始差 {pp("flags.flag_payday.0")}，标准化后为 '
                f'{cross_std["flag_payday"][0]:.1f}% / {cross_std["flag_payday"][1]:.1f}%'
                f'（差 {cross_std["flag_payday"][1] - cross_std["flag_payday"][0]:+.1f}pp）。'
                '收入构成差异解释了部分拒付差，但解释不了现金贷使用差；'
                '这是描述性标准化，不控制其他变量；福利依赖组 Swap Out 无样本，未纳入加权。</div>')
    content += dumbbell("三项压力同时出现", [
        ("现金贷/工资贷使用 × 未结清拒付 × 流动性缓冲不足",
         pv("joint"),
         f'交集计数：IN {S["swap_in"]["joint_n"]} / OUT {S["swap_out"]["joint_n"]}'),
    ], "统一 0–100% 刻度")
    content += (f'<p class="table-note">计入分群的风险标签数均值 {pct_pair("flag_count_mean", "two")}，'
                f'3 个及以上为 {pct_pair("flag_count_ge3")}。标签密度与“高风险承压”分群高度重叠，'
                '不作为独立证据。</p>')
    parts.append(section("flags", "07 · 风险标签", "风险标签：债务型与收入可见性的分化",
                         "单项标签、复合压力、标签密度与收入形态分层。", content))

    # 08 结论
    content = ('<p class="lead">综合各章节指标，两组差异可归纳为两类：'
               '<strong>Swap Out 呈现为“借贷与拒付更重、工资更稳定”的债务型画像；'
               'Swap In 呈现为“非工资收入更多、近期收支缺口更明显”的现金流型画像。</strong></p>')
    content += cards([
        ("Swap Out · 债务与拒付更集中",
         f'现金贷/工资贷使用 {fmt(S["swap_out"]["flags"]["flag_payday"][0], "pct")}（{pp("flags.flag_payday.0")}）；'
         f'高负债压力 {fmt(S["swap_out"]["flags"]["flag_high_debt"][0], "pct")}（{pp("flags.flag_high_debt.0")}）；'
         f'放款笔数中位数 {fmt(S["swap_out"]["median"]["disburse_cnt"], "int")}；'
         f'未结清拒付 {fmt(S["swap_out"]["flags"]["flag_unresolved_dishonour"][0], "pct")}。'),
        ("Swap In · 收入可见性与近期缺口",
         f'多源/非工资收入 {fmt(S["swap_in"]["archetypes"]["多源/非工资"], "pct")}（Swap Out 为 '
         f'{fmt(S["swap_out"]["archetypes"]["多源/非工资"], "pct")}）；'
         f'无工资入账 {fmt(S["swap_in"]["flags"]["flag_no_wage"][0], "pct")}；'
         f'近 28 天支出 ≥ 收入 {fmt(S["swap_in"]["burn_bands"]["支出达到或超过收入"], "pct")}（{pp("burn_bands.支出达到或超过收入")}）；'
         f'流动性缓冲不足 {fmt(S["swap_in"]["flags"]["flag_low_buffer"][0], "pct")}。'),
    ])
    content += diffchart("风险标签差值排序：Swap Out − Swap In",
                         [(label, vals) for label, vals, _ci, _gap in flag_gaps[:8]], 25,
                         "按绝对差排序 · 统一 ±25pp")
    content += ('<div class="callout warn"><strong>结论适用边界：</strong>'
                '（1）两组按置换结果划分，不是随机实验，差异不能归因于置换动作本身；'
                '（2）Swap Out 仅 139 条，拒付、流动性、近期缺口等指标区间重叠；'
                '（3）健康分是合并样本内的相对分数，不等同于违约概率；'
                '（4）逾期标签仍不成熟：Swap In 仅 83/641、Swap Out 仅 31/139 有 duedate_1m_5，'
                '阳性分别 11 与 5，不能做结果对比。'
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
    parts.append(section("summary", "08 · 结论", "核心差异与结论",
                         "两组客群的主要差异、可解释范围与后续验证方向。", content))

    footer = ('<footer class="footer"><div class="wrap footer-inner">'
              f'<span>Swap In × Swap Out 客群画像对比 · 合并统一评分</span>'
              f'<span>样本 780 条 · 生成 {pd.Timestamp.now():%Y-%m-%d %H:%M} · '
              '<a href="#basis">数据与口径</a></span></div></footer>')

    output = ('<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
              '<meta name="viewport" content="width=device-width, initial-scale=1">'
              '<meta name="description" content="Swap In 与 Swap Out 存量客群画像对比：收入、负债、拒付、流动性与风险标签。">'
              f'<title>Swap In × Swap Out 客群画像对比</title><style>{css}</style></head>'
              f'<body>{header}{nav}<main>{"".join(parts)}</main>{footer}</body></html>\n')
    REPORT.write_text(output, encoding="utf-8")
    print(f"report: {REPORT} ({REPORT.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
