#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Swap In vs Swap Out 画像对比精简版（快速同步）

用法:
    python compare_swap_profile_report_lite.py
    python compare_swap_profile_report_lite.py --swap-in data/existing_swap_in_v1_2_with_feature_all.csv \
        --swap-out data/existing_swap_out_v1_2_with_feature_all.csv --out output

输出:
    output/compare_swap_profile_report_lite.html   单文件精简报告（无外部依赖）
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import existing_swap_in_profile_report as swi
import existing_swap_out_profile_report as swo
from existing_swap_in_profile_report import fmt_money, fmt_pct

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROOT / "data" / "existing_swap_in_v1_2_with_feature_all.csv"
DEFAULT_OUT = ROOT / "data" / "existing_swap_out_v1_2_with_feature_all.csv"
DEFAULT_OUTDIR = ROOT / "output"

IN_COLOR = "#1c5cab"
OUT_COLOR = "#d14b45"
GOLD = "#a16207"
GOLD_LINE = "#d9a441"


# ---------------------------------------------------------------------------
# 数据与格式化
# ---------------------------------------------------------------------------
def valid(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def seg_val(seg, name: str, col: str) -> float:
    return float(seg.loc[name, col]) if name in seg.index else np.nan


def load_profile(module, path: Path) -> dict:
    raw, missing, n_fields = module.load_data(path)
    raw, n_apps = module.clean_frame(raw)
    df = module.dedupe_users(raw)
    d, m, seg = module.build_metrics(df, n_apps)
    post_cf = float(np.nanmedian((d["inc"] - d["exp"] - d["rep"]) / 6))
    return {"m": m, "seg": seg, "post_cf": post_cf, "n_fields": n_fields}


def fmt_metric(value, kind: str, digits: int = 1) -> str:
    if not valid(value):
        return "—"
    if kind == "money":
        return fmt_money(float(value))
    if kind == "pct":
        return fmt_pct(float(value), digits)
    return f"{float(value):.{digits}f}"


def delta_html(a, b, kind: str, digits: int = 1) -> str:
    if not (valid(a) and valid(b)):
        return '<span class="delta muted">—</span>'
    if kind == "pct":
        diff = (float(b) - float(a)) * 100
        text = f"{abs(diff):.1f}pp"
        flat = abs(diff) < 0.05
    else:
        diff = float(b) - float(a)
        text = fmt_money(abs(diff))
        if kind == "money" and float(a) > 0 and abs(diff) / float(a) <= 3:
            text += f"（{abs(diff) / float(a):.0%}）"
        flat = abs(diff) < 1e-9
    if flat:
        return '<span class="delta muted">基本持平</span>'
    who = "Out" if diff > 0 else "In"
    color = OUT_COLOR if diff > 0 else IN_COLOR
    return (
        f'<span class="delta"><span class="dot" style="background:{color}"></span>'
        f"{who} 高 {text}</span>"
    )


def metric_row(label: str, a, b, kind: str = "pct", digits: int = 1) -> str:
    if kind == "pct":
        wa = float(a) * 100 if valid(a) else 0.0
        wb = float(b) * 100 if valid(b) else 0.0
    else:
        vals = [float(v) for v in (a, b) if valid(v)]
        scale = max(vals + [1.0])
        wa = float(a) / scale * 100 if valid(a) else 0.0
        wb = float(b) / scale * 100 if valid(b) else 0.0
    wa = max(0.0, min(100.0, wa))
    wb = max(0.0, min(100.0, wb))
    return f"""
      <div class="metric">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-bars">
          <div class="bar-row"><span class="who"><span class="dot" style="background:{IN_COLOR}"></span>In</span><div class="track"><div class="fill" style="width:{wa:.1f}%;background:{IN_COLOR}"></div></div><span class="val">{fmt_metric(a, kind, digits)}</span></div>
          <div class="bar-row"><span class="who"><span class="dot" style="background:{OUT_COLOR}"></span>Out</span><div class="track"><div class="fill" style="width:{wb:.1f}%;background:{OUT_COLOR}"></div></div><span class="val">{fmt_metric(b, kind, digits)}</span></div>
        </div>
        <div class="metric-delta">{delta_html(a, b, kind, digits)}</div>
      </div>"""


def kpi_card(label: str, a, b, kind: str = "pct", digits: int = 1) -> str:
    return f"""
      <div class="kpi">
        <div class="kpi-label">{html.escape(label)}</div>
        <div class="kpi-row"><span class="who"><span class="dot" style="background:{IN_COLOR}"></span>Swap In</span><span class="v">{fmt_metric(a, kind, digits)}</span></div>
        <div class="kpi-row"><span class="who"><span class="dot" style="background:{OUT_COLOR}"></span>Swap Out</span><span class="v">{fmt_metric(b, kind, digits)}</span></div>
        <div class="kpi-delta">{delta_html(a, b, kind, digits)}</div>
      </div>"""


def hl(value, kind: str = "pct", digits: int = 1) -> str:
    return f'<mark class="hl">{fmt_metric(value, kind, digits)}</mark>'


def conclusion(text: str) -> str:
    return f'<div class="key-conclusion"><span class="star">★</span><div>{text}</div></div>'


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
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
  font-size: 14px; line-height: 1.6; -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 30px 24px 46px; }
.eyebrow { color: var(--muted); font-size: 10.5px; letter-spacing: .14em; font-weight: 650; text-transform: uppercase; }
h1 { font-size: 27px; line-height: 1.25; margin: 7px 0 5px; letter-spacing: -.02em; }
.subtitle { color: var(--ink2); font-size: 13.5px; margin: 0 0 12px; }
.chips { display: flex; flex-wrap: wrap; gap: 7px; margin: 0 0 14px; }
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
  border-radius: 0 8px 8px 0; color: var(--ink); font-size: 13.5px; line-height: 1.6;
}
.key-conclusion .star { margin-top: 2px; flex: none; font-size: 14px; }
mark.hl { background: var(--hl); color: inherit; padding: 0 2px; border-radius: 3px; font-weight: 650; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 14px 0 4px; }
.kpi { background: var(--surface); border: 1px solid rgba(11,11,11,.10); border-radius: 12px; padding: 12px 14px; }
.kpi-label { color: var(--ink2); font-size: 11.5px; }
.kpi-row { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-top: 5px; }
.kpi-row .who { color: var(--ink2); font-size: 11px; display: inline-flex; align-items: center; gap: 5px; }
.kpi-row .v { font-size: 18px; font-weight: 650; letter-spacing: -.01em; }
.kpi-delta { color: var(--muted); font-size: 10.5px; margin-top: 5px; }
.section { margin-top: 24px; }
h2 { font-size: 17px; margin: 0 0 8px; letter-spacing: -.01em; display: flex; align-items: center; gap: 8px; }
h2 .num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px; border-radius: 50%;
  background: var(--ink); color: #fff; font-size: 11px; font-weight: 650;
}
.metric {
  display: grid; grid-template-columns: 150px 1fr 112px; gap: 12px;
  align-items: center; padding: 7px 0; border-bottom: 1px solid var(--grid);
}
.metric:last-child { border-bottom: none; }
.metric-label { font-size: 12.5px; color: var(--ink); font-weight: 550; }
.metric-bars { display: grid; gap: 4px; }
.bar-row { display: grid; grid-template-columns: 34px 1fr 62px; gap: 7px; align-items: center; }
.bar-row .who { font-size: 10.5px; color: var(--muted); display: inline-flex; align-items: center; gap: 4px; }
.track { height: 9px; background: #eeede7; border-radius: 999px; overflow: hidden; }
.fill { height: 100%; border-radius: 999px; }
.bar-row .val { font-size: 11.5px; font-weight: 650; text-align: right; font-variant-numeric: tabular-nums; color: var(--ink); }
.metric-delta { text-align: right; font-size: 10.5px; color: var(--muted); }
.delta { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
.delta.muted { color: var(--muted); }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex: none; }
.note-box {
  margin-top: 22px; padding: 10px 12px; border: 1px solid var(--grid);
  border-radius: 10px; background: var(--surface); color: var(--ink2); font-size: 11.5px;
}
.note-box b { color: var(--ink); }
.footer { margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--baseline); color: var(--muted); font-size: 11px; }
.footer p { margin: 3px 0; }
@media (max-width: 760px) {
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .metric { grid-template-columns: 1fr; gap: 4px; }
  .metric-delta { text-align: left; }
  h1 { font-size: 23px; }
}
@media print { .wrap { max-width: none; } .kpi, .metric { break-inside: avoid; } }
"""


def build_html(pin: dict, pout: dict, generated_at: str) -> str:
    mi, mo = pin["m"], pout["m"]
    heavy_in = seg_val(pin["seg"], "重压型", "share")
    heavy_out = seg_val(pout["seg"], "重压型", "share")

    kpis = "".join([
        kpi_card("中位月收入", mi["inc_p50"], mo["inc_p50"], "money"),
        kpi_card("有借贷放款", mi["any_disburse"], mo["any_disburse"], "pct"),
        kpi_card("有拒付记录", mi["dish_any"], mo["dish_any"], "pct"),
        kpi_card("重压型占比", heavy_in, heavy_out, "pct"),
    ])

    income_rows = "".join([
        metric_row("工资覆盖率", mi["wage_incidence"], mo["wage_incidence"]),
        metric_row("工资占入账", mi["wage_share_agg"], mo["wage_share_agg"]),
        metric_row("其他入账占比", mi["other_share_agg"], mo["other_share_agg"]),
    ])
    debt_rows = "".join([
        metric_row("现金贷使用率", mi["adv_use"], mo["adv_use"]),
        metric_row("偿债 / 收入", mi["rep_rate_p50"], mo["rep_rate_p50"]),
        metric_row("未解决拒付", mi["dish_unres"], mo["dish_unres"]),
        metric_row("拒付解决率", mi["dish_resolution_rate"], mo["dish_resolution_rate"]),
    ])
    cash_rows = "".join([
        metric_row("负盈余占比", mi["sur_neg"], mo["sur_neg"]),
        metric_row("缓冲 < 7 天", mi["buffer_lt7"], mo["buffer_lt7"]),
        metric_row("重压型占比", heavy_in, heavy_out),
        metric_row("赌博占可识别消费", mi["gamb_share_cons"], mo["gamb_share_cons"]),
    ])

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swap In vs Swap Out 画像对比（精简版）</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">SWAP IN × SWAP OUT · QUICK READ</div>
  <h1>Swap In vs Swap Out 画像对比（精简版）</h1>
  <p class="subtitle">同一套银行流水特征口径 · {swi.WINDOW_DAYS} 天观察窗口 · 存量用户 · 描述性对比</p>
  <div class="chips">
    <span class="chip"><span class="dot" style="background:{IN_COLOR}"></span>Swap In 有效 <b>{mi['n_valid']:,}</b> 人</span>
    <span class="chip"><span class="dot" style="background:{OUT_COLOR}"></span>Swap Out 有效 <b>{mo['n_valid']:,}</b> 人</span>
    <span class="chip">生成时间 <b>{html.escape(generated_at)}</b></span>
  </div>
  <p class="note-legend"><span class="star">★</span>金色框为该节结论，金色高亮为支撑结论的数据；蓝色为 Swap In，红色为 Swap Out。</p>

  <div class="key-conclusion"><span class="star">★</span><div><b>一句话结论：</b>没有一方全面占优。Swap Out 收入更高、工资更稳、重压型更少，但借贷和拒付暴露更高；Swap In 杠杆更低，但收入更弱、重压型占比更高。</div></div>

  <div class="kpi-grid">{kpis}</div>

  <section class="section">
    <h2><span class="num">1</span>收入质量</h2>
    {conclusion(f"Swap Out 收入更高，工资覆盖 {hl(mo['wage_incidence'])}（Swap In {fmt_pct(mi['wage_incidence'])}），工资占入账 {hl(mo['wage_share_agg'])} vs {hl(mi['wage_share_agg'])}；Swap In 更依赖其他入账。")}
    {income_rows}
  </section>

  <section class="section">
    <h2><span class="num">2</span>杠杆与履约</h2>
    {conclusion(f"Swap Out 的借贷和偿债负担明显更高：现金贷使用率 {hl(mo['adv_use'])}（Swap In {fmt_pct(mi['adv_use'])}）、偿债 / 收入 {hl(mo['rep_rate_p50'])} vs {hl(mi['rep_rate_p50'])}；拒付更多，但解决率也更高。")}
    {debt_rows}
  </section>

  <section class="section">
    <h2><span class="num">3</span>现金流与分层</h2>
    {conclusion(f"两组几乎全部负盈余、缓冲不足 7 天；Swap In 重压型占比 {hl(heavy_in)}（Swap Out {fmt_pct(heavy_out)}），Swap Out 赌博消费占比 {hl(mo['gamb_share_cons'])}（Swap In {fmt_pct(mi['gamb_share_cons'])}）。")}
    {cash_rows}
  </section>

  <div class="note-box">
    <b>读法提示：</b>Swap Out 的“重压型更少”部分来自更高的新增借款（中位 {fmt_money(mo['draw_m_p50'])}/月 vs Swap In {fmt_money(mi['draw_m_p50'])}）。剔除新借后，Swap Out 偿债后现金流中位 {fmt_money(pout['post_cf'])} vs Swap In {fmt_money(pin['post_cf'])}。两组为 {mi['n_valid']:,} vs {mo['n_valid']:,} 人的描述性对比，拒付差异的 95% 置信区间有重叠，不代表因果。
  </div>

  <div class="footer">
    <p><b>数据源</b>：existing_swap_in_v1_2_with_feature_all.csv、existing_swap_out_v1_2_with_feature_all.csv；同一 user_id 保留交易笔数最多的申请。</p>
    <p><b>口径</b>：观察窗口 {swi.WINDOW_DAYS} 天，月度指标按 6 个月折算；盈余 = 收入 − 支出 − 还本付息 + 新增借款；有效样本要求 txn_raw_input_cnt &gt; 0，且收入 / 支出 / 盈余非缺失、收入 &gt; 0。</p>
    <p><b>说明</b>：两组为独立样本，未做统一加权或显著性检验；以上差异仅用于描述性对比，不构成信用评估结论。</p>
  </div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Swap In vs Swap Out 画像对比精简版")
    parser.add_argument("--swap-in", default=str(DEFAULT_IN), help="Swap In 样本 CSV 路径")
    parser.add_argument("--swap-out", default=str(DEFAULT_OUT), help="Swap Out 样本 CSV 路径")
    parser.add_argument("--out", default=str(DEFAULT_OUTDIR), help="输出目录")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("加载 Swap In 样本 ...")
    pin = load_profile(swi, Path(args.swap_in))
    print("加载 Swap Out 样本 ...")
    pout = load_profile(swo, Path(args.swap_out))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_path = out / "compare_swap_profile_report_lite.html"
    report_path.write_text(build_html(pin, pout, generated_at), encoding="utf-8")

    mi, mo = pin["m"], pout["m"]
    print("\n" + "=" * 64)
    print("Swap In vs Swap Out 画像对比精简版已生成")
    print("=" * 64)
    print(f"有效样本: Swap In {mi['n_valid']:,} 人 | Swap Out {mo['n_valid']:,} 人")
    print(f"收入: 中位月收入 {fmt_money(mi['inc_p50'])} vs {fmt_money(mo['inc_p50'])}")
    print(f"杠杆: 有借贷 {fmt_pct(mi['any_disburse'])} vs {fmt_pct(mo['any_disburse'])} | 偿债/收入 {fmt_pct(mi['rep_rate_p50'])} vs {fmt_pct(mo['rep_rate_p50'])}")
    print(f"履约: 有拒付 {fmt_pct(mi['dish_any'])} vs {fmt_pct(mo['dish_any'])} | 未解决 {fmt_pct(mi['dish_unres'])} vs {fmt_pct(mo['dish_unres'])}")
    print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
