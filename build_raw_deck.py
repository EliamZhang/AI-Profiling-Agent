#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the PPT-style slide deck for the raw-transaction Swap In/Out report.

Usage:
    python build_raw_deck.py

Output:
    output/html/swap_in_out_profile_report_slides.html
"""

from __future__ import annotations

import html
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

import compare_swap_profile_report as chart_ui
import deck_framework as ui
import swap_in_out_analysis as ana
import swap_in_out_profile_report as rep

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "html" / "swap_in_out_profile_report_slides.html"
BRAND = "SWAP IN <em>×</em> SWAP OUT"
IN_COLOR = "#1c5cab"
OUT_COLOR = "#d14b45"


def fmt_pair(a, b, kind: str) -> str:
    if kind == "pct":
        return f"{rep.p(a)}<small> vs </small>{rep.p(b)}"
    if kind == "money":
        return f"{rep.m(a)}<small> vs </small>{rep.m(b)}"
    if kind == "days":
        return f"{rep.n(a, 2)}<small> vs </small>{rep.n(b, 2)} 天"
    if kind == "ratio":
        return f"{rep.ratio(a)}<small> vs </small>{rep.ratio(b)}"
    return f"{rep.n(a)}<small> vs </small>{rep.n(b)}"


def mrow(label: str, a, b, kind: str = "pct") -> tuple[str, str, str]:
    return (label, fmt_pair(a, b, kind), rep.delta(a, b, kind))


def build_deck(users: pd.DataFrame, tx: pd.DataFrame, generated_at: str) -> str:
    si = rep.summary(users[users["group"] == "Swap In"])
    so = rep.summary(users[users["group"] == "Swap Out"])
    mi = rep.income_mix(users[users["group"] == "Swap In"])
    mo = rep.income_mix(users[users["group"] == "Swap Out"])
    oi = rep.outflow_mix(users[users["group"] == "Swap In"])
    oo = rep.outflow_mix(users[users["group"] == "Swap Out"])
    n_txn_in = int(users[users["group"] == "Swap In"]["n_txn"].sum())
    n_txn_out = int(users[users["group"] == "Swap Out"]["n_txn"].sum())

    seg_ct = pd.crosstab(users["seg"], users["group"]).reindex(rep.SEG_ORDER, fill_value=0)
    for col in ("Swap In", "Swap Out"):
        if col not in seg_ct:
            seg_ct[col] = 0
    seg_ct["total"] = seg_ct["Swap In"] + seg_ct["Swap Out"]
    seg_ct["In%"] = seg_ct["Swap In"] / si["n"]
    seg_ct["Out%"] = seg_ct["Swap Out"] / so["n"]

    seg_cols = [
        "income", "wage_ratio", "repay_rate", "ext_transfer_share",
        "buffer_days", "dishonour_cnt", "active_lender_cnt", "loan_recycle_7d",
    ]
    seg_prof = users.groupby("seg")[seg_cols].median().reindex(rep.SEG_ORDER)

    users = users.copy()
    users["gambling_high_flag"] = (
        (users["gambling_share"] >= 0.05) | (users["gambling"] >= 1000)
    ).astype(float)
    risk_seg = users.groupby("seg").agg(
        dishon_any=("any_dishonour", "mean"),
        dishon_cnt=("dishonour_cnt", "median"),
        gamb_high=("gambling_high_flag", "mean"),
        buffer=("buffer_days", "median"),
    ).reindex(rep.SEG_ORDER)

    loan = tx[tx["is_loan_in"] | tx["is_debt_repay"]].copy()
    loan["third_party"] = loan["third_party"].fillna("(未标注)")
    top = (
        loan.groupby("third_party")["user_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )

    kpi_html = ui.cmp_kpi_cards(
        [
            {
                "label": "中位真实收入",
                "left_name": "Swap In",
                "left_value": rep.m(si["income"]),
                "right_name": "Swap Out",
                "right_value": rep.m(so["income"]),
                "delta": rep.delta(si["income"], so["income"], "money"),
                "left_color": IN_COLOR,
                "right_color": OUT_COLOR,
            },
            {
                "label": "工资覆盖率",
                "left_name": "Swap In",
                "left_value": rep.p(si["any_wage"]),
                "right_name": "Swap Out",
                "right_value": rep.p(so["any_wage"]),
                "delta": rep.delta(si["any_wage"], so["any_wage"], "pct"),
                "left_color": IN_COLOR,
                "right_color": OUT_COLOR,
            },
            {
                "label": "偿债 / 收入",
                "left_name": "Swap In",
                "left_value": rep.p(si["repay_rate"]),
                "right_name": "Swap Out",
                "right_value": rep.p(so["repay_rate"]),
                "delta": rep.delta(si["repay_rate"], so["repay_rate"], "pct"),
                "left_color": IN_COLOR,
                "right_color": OUT_COLOR,
            },
            {
                "label": "拒付次数中位",
                "left_name": "Swap In",
                "left_value": rep.n(si["dishonour_cnt"]),
                "right_name": "Swap Out",
                "right_value": rep.n(so["dishonour_cnt"]),
                "delta": rep.delta(si["dishonour_cnt"], so["dishonour_cnt"], "count"),
                "left_color": IN_COLOR,
                "right_color": OUT_COLOR,
            },
        ]
    )

    slides: list[str] = []

    # 01 Cover
    slides.append(
        ui.slide_cover(
            brand=BRAND,
            tag="RAW TRANSACTION PROFILE",
            eyebrow=f"原始交易明细画像 · {generated_at}",
            title="Swap In vs Swap Out<br>人群画像对比分析",
            desc=(
                "基于原始银行流水交易明细（一行一笔交易）聚合用户画像，"
                f"观察窗口 {ana.WINDOW_DAYS} 天，对比两组存量用户的收入、资金流向、杠杆、履约与风险标签。"
            ),
            statuses=[
                ("green", f"Swap In {si['n']} 用户"),
                ("blue", f"Swap Out {so['n']} 用户"),
            ],
            metrics=[
                ("Swap In", f"{si['n']}", f"用户 · {n_txn_in:,} 笔"),
                ("Swap Out", f"{so['n']}", f"用户 · {n_txn_out:,} 笔"),
                ("Period", "2026-03 – 10", "交易日期"),
                ("Window", f"{ana.WINDOW_DAYS} 天", "观察窗口"),
            ],
        )
    )

    # 02 Executive summary
    exec_body = (
        kpi_html
        + """
        <div class="duo wide-right">
          <div class="statement reveal d2">
            <span class="status green"><i></i>执行摘要</span>
            <h3>Swap Out 不是“更优质”，而是“更高收入 + 更高杠杆”</h3>
            <p>Swap In 不是“更差”，而是“更低杠杆 + 更弱收入”。两组共同点是账户缓冲极薄、资金高度依赖转账与借贷周转。</p>
          </div>
        """
        + ui.bullet_list(
            [
                f"<b>收入：</b>Swap Out 中位真实收入 <mark>{rep.m(so['income'])}</mark>，Swap In <mark>{rep.m(si['income'])}</mark>；工资覆盖率 <mark>{rep.p(so['any_wage'])}</mark> vs <mark>{rep.p(si['any_wage'])}</mark>。",
                f"<b>资金流向：</b>Swap Out 外部转账 / 借记 <mark>{rep.p(so['ext_transfer_share'])}</mark>，Swap In <mark>{rep.p(si['ext_transfer_share'])}</mark>；Swap In 内部转账更高。",
                f"<b>杠杆：</b>Swap Out 有借贷放款 <mark>{rep.p(so['any_loan_draw'])}</mark> vs <mark>{rep.p(si['any_loan_draw'])}</mark>，活跃机构中位 <mark>{rep.n(so['active_lender'], 0)}</mark> vs <mark>{rep.n(si['active_lender'], 0)}</mark>。",
                f"<b>流动性：</b>中位余额仅 <mark>{rep.m(si['balance'])}</mark> / <mark>{rep.m(so['balance'])}</mark>，缓冲中位 <mark>{rep.n(si['buffer'], 2)}</mark> / <mark>{rep.n(so['buffer'], 2)}</mark> 天。",
            ]
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="Executive Summary",
            kicker="执行摘要",
            title="两组人群，一种底盘",
            sub="Swap Out 收入更高但杠杆与拒付更重；Swap In 收入更弱、内部转账更多；两组缓冲都极薄。",
            body=exec_body,
        )
    )

    # 03 Divider 01
    slides.append(
        ui.slide_divider(
            brand=BRAND,
            section="01 · 人群与底盘",
            num="01",
            kicker="Population & Baseline",
            title="人群与底盘",
            desc="先确认样本形态与整体判断，再看六类人群如何分布、各自的特征是什么。",
            status_html='<span class="status green"><i></i>交易明细直接聚合</span>',
            meta=[
                (f"{si['n']} / {so['n']}", "用户数"),
                (f"{n_txn_in:,} / {n_txn_out:,}", "交易笔数"),
                (f"{ana.WINDOW_DAYS} 天", "观察窗口"),
            ],
        )
    )

    # 04 Baseline table
    baseline_rows = [
        ["用户数 / 交易笔数中位", f"{si['n']} / {rep.n(si['txn'])}", f"{so['n']} / {rep.n(so['txn'])}", "—"],
        ["中位真实收入", rep.m(si["income"]), rep.m(so["income"]), rep.delta(si["income"], so["income"], "money")],
        ["工资覆盖率", rep.p(si["any_wage"]), rep.p(so["any_wage"]), rep.delta(si["any_wage"], so["any_wage"], "pct")],
        ["外部转账 / 全部借记", rep.p(si["ext_transfer_share"]), rep.p(so["ext_transfer_share"]), rep.delta(si["ext_transfer_share"], so["ext_transfer_share"], "pct")],
        ["内部转账 / 全部借记", rep.p(si["internal_transfer_share"]), rep.p(so["internal_transfer_share"]), rep.delta(si["internal_transfer_share"], so["internal_transfer_share"], "pct")],
        ["偿债 / 收入", rep.p(si["repay_rate"]), rep.p(so["repay_rate"]), rep.delta(si["repay_rate"], so["repay_rate"], "pct")],
        ["有拒付记录", rep.p(si["any_dishonour"]), rep.p(so["any_dishonour"]), rep.delta(si["any_dishonour"], so["any_dishonour"], "pct")],
        ["中位账户余额", rep.m(si["balance"]), rep.m(so["balance"]), rep.delta(si["balance"], so["balance"], "money")],
        ["缓冲天数中位", f"{rep.n(si['buffer'], 2)} 天", f"{rep.n(so['buffer'], 2)} 天", rep.delta(si["buffer"], so["buffer"], "days")],
    ]
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="01 · 人群与底盘",
            kicker="1.1 · Sample Baseline",
            title="样本整体底盘",
            sub="输入为交易明细表（18 列），按 user_id 聚合；余额由交易后 balance 推导，无独立每日余额快照。",
            body=ui.table_box(
                ["指标", "Swap In", "Swap Out", "差异"],
                baseline_rows,
                note=f"Swap In {si['n']} 用户 / {n_txn_in:,} 笔，Swap Out {so['n']} 用户 / {n_txn_out:,} 笔；账户数中位 {rep.n(si['accounts'], 1)} vs {rep.n(so['accounts'], 1)}；无征信 / 还款状态与拒付解决状态。",
                num_cols=(1, 2, 3),
            ),
        )
    )

    # 05 Segment distribution
    seg_rows = [
        [
            html.escape(seg),
            f"{int(seg_ct.loc[seg, 'Swap In'])}（{rep.p(seg_ct.loc[seg, 'In%'])}）",
            f"{int(seg_ct.loc[seg, 'Swap Out'])}（{rep.p(seg_ct.loc[seg, 'Out%'])}）",
            f"{int(seg_ct.loc[seg, 'total'])}",
        ]
        for seg in rep.SEG_ORDER
    ]
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="01 · 人群与底盘",
            kicker="1.2 · Segments",
            title="核心人群分层",
            sub="Swap Out 更集中在“借贷周转”和“工薪高杠杆”；Swap In 更分散在转账过路、现金流脆弱和混合/低活跃。",
            body=ui.table_box(
                ["人群", "Swap In（占比）", "Swap Out（占比）", "合计"],
                seg_rows,
                note="分层为规则判定，不是统计聚类；阈值用于描述性对比，换阈值会改变人群边界。",
                num_cols=(3,),
            ),
        )
    )

    # 06 Segment profile
    seg_prof_rows = []
    for seg in rep.SEG_ORDER:
        r = seg_prof.loc[seg]
        seg_prof_rows.append(
            [
                html.escape(seg),
                rep.m(r["income"]),
                rep.p(r["wage_ratio"]),
                rep.p(r["repay_rate"]),
                rep.p(r["ext_transfer_share"]),
                f"{rep.n(r['buffer_days'], 2)} 天",
                rep.n(r["dishonour_cnt"]),
                rep.n(r["active_lender_cnt"], 1),
                rep.ratio(r["loan_recycle_7d"]),
            ]
        )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="01 · 人群与底盘",
            kicker="1.3 · Segment Profile",
            title="分层画像（中位数）",
            sub="六类人群的收入、工资占比、偿债负担、转账结构、缓冲与借贷周转特征。",
            body=ui.table_box(
                ["人群", "收入", "工资占比", "偿债/收入", "外部转账", "缓冲天数", "拒付次数", "活跃机构", "贷款周转"],
                seg_prof_rows,
                note="比率为组内用户中位数；贷款周转 = 贷款到账后 7 天还贷 + 外部转账 / 贷款金额。",
                num_cols=(1, 2, 3, 4, 5, 6, 7, 8),
            ),
        )
    )

    # 07 Divider 02
    slides.append(
        ui.slide_divider(
            brand=BRAND,
            section="02 · 收入、资金流向与风险",
            num="02",
            kicker="Income, Flow & Risk",
            title="收入、资金流向与风险",
            desc="真正拉开两组的是工资覆盖与收入水平、外部/内部转账、借贷使用与偿债负担；消费差异不大。",
            status_html='<span class="status amber"><i></i>风险标签用于增强分层</span>',
            meta=[
                (rep.p(si["any_wage"]) + " / " + rep.p(so["any_wage"]), "工资覆盖率"),
                (rep.p(si["ext_transfer_share"]) + " / " + rep.p(so["ext_transfer_share"]), "外部转账 / 借记"),
                (rep.n(si["buffer"], 2) + " / " + rep.n(so["buffer"], 2) + " 天", "缓冲天数中位"),
            ],
        )
    )

    # 08 Income quality
    income_rows = [
        mrow("工资覆盖率", si["any_wage"], so["any_wage"]),
        mrow("工资占真实收入（用户中位）", si["wage_ratio"], so["wage_ratio"]),
        mrow("真实收入 / 总入账（剔除内部转账）", si["true_income_ratio"], so["true_income_ratio"]),
        mrow("外部转账流入 / 总入账", si["transfer_in_ratio"], so["transfer_in_ratio"]),
        mrow("贷款流入 / 总入账", si["loan_inflow_ratio"], so["loan_inflow_ratio"]),
    ]
    mix_rows = [
        ["工资", rep.p(mi["wage_share"]), rep.p(mo["wage_share"]), rep.delta(mi["wage_share"], mo["wage_share"], "pct")],
        ["福利", rep.p(mi["welfare_share"]), rep.p(mo["welfare_share"]), rep.delta(mi["welfare_share"], mo["welfare_share"], "pct")],
        ["其他入账", rep.p(mi["other_share"]), rep.p(mo["other_share"]), rep.delta(mi["other_share"], mo["other_share"], "pct")],
    ]
    income_body = (
        '<div class="duo wide-right">'
        + ui.metric_rows(income_rows)
        + ui.table_box(
            ["收入构成（按金额加总）", "Swap In", "Swap Out", "差异"],
            mix_rows,
            note="真实收入 = 工资 + 福利 + 其他入账；贷款放款、外部转账流入、内部转账不计入。",
            num_cols=(1, 2, 3),
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="02 · 收入、资金流向与风险",
            kicker="2.1 · Income Quality",
            title="收入质量",
            sub="Swap Out 收入更高、工资更稳；Swap In 更依赖其他入账与外部转账流入。",
            body=income_body,
        )
    )

    # 09 Outflow & debt
    outflow_rows = [
        mrow("可识别消费 / 全部借记", si["consumption_share"], so["consumption_share"]),
        mrow("外部转账 / 全部借记", si["ext_transfer_share"], so["ext_transfer_share"]),
        mrow("内部转账 / 全部借记", si["internal_transfer_share"], so["internal_transfer_share"]),
        mrow("债务还款 / 全部借记", si["debt_share"], so["debt_share"]),
        mrow("赌博 / 全部借记", oi["gambling"], oo["gambling"]),
    ]
    debt_rows = [
        mrow("有借贷放款", si["any_loan_draw"], so["any_loan_draw"]),
        mrow("有还款", si["any_repay"], so["any_repay"]),
        mrow("活跃借款机构数（中位）", si["active_lender"], so["active_lender"], "count"),
    ]
    flow_body = (
        '<div class="duo wide-right">'
        + ui.metric_rows(outflow_rows)
        + ui.metric_rows(debt_rows)
        + "</div>"
        + ui.note_strip(
            "口径提示",
            "构成为按金额加总；外部/内部转账占比以全部借记为分母；Swap In 的赌博金额集中在少数用户。",
        )
    )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="02 · 收入、资金流向与风险",
            kicker="2.2 · Outflow & Leverage",
            title="资金流向与债务结构",
            sub="Swap Out 更外流、借贷覆盖更广；Swap In 内部账户周转更多，杠杆相对较低。",
            body=flow_body,
        )
    )

    # 10 Risk tags
    risk_rows = [
        mrow("中位账户余额", si["balance"], so["balance"], "money"),
        mrow("缓冲天数（中位）", si["buffer"], so["buffer"], "days"),
        mrow("负余额天数占比", si["neg_days"], so["neg_days"]),
        mrow("余额 < $100 天数占比", si["low_days"], so["low_days"]),
        mrow("有拒付记录", si["any_dishonour"], so["any_dishonour"]),
        mrow("拒付次数（中位）", si["dishonour_cnt"], so["dishonour_cnt"], "count"),
        mrow("高赌博暴露占比", si["gambling_high"], so["gambling_high"]),
    ]
    risk_tag_rows = [
        ["赌博发生 / 高暴露", f"{rep.p(si['any_gambling'])} / {rep.p(si['gambling_high'])}", f"{rep.p(so['any_gambling'])} / {rep.p(so['gambling_high'])}", "发生率高，但金额集中；高暴露 = 赌博占消费 ≥5% 或金额 ≥ $1,000。"],
        ["拒付发生 / 次数中位", f"{rep.p(si['any_dishonour'])} / {rep.n(si['dishonour_cnt'])}", f"{rep.p(so['any_dishonour'])} / {rep.n(so['dishonour_cnt'])}", "拒付是横向增强标签；Swap Out 更重，但无解决状态。"],
        ["负余额 / 低余额天数", f"{rep.p(si['neg_days'])} / {rep.p(si['low_days'])}", f"{rep.p(so['neg_days'])} / {rep.p(so['low_days'])}", "低余额是共同底盘，不宜单独作为主分层。"],
        ["外部 / 内部转账", f"{rep.p(si['ext_transfer_share'])} / {rep.p(si['internal_transfer_share'])}", f"{rep.p(so['ext_transfer_share'])} / {rep.p(so['internal_transfer_share'])}", "Swap Out 更外流，Swap In 更内部周转。"],
        ["贷款后 7 天周转倍数", rep.ratio(si["loan_recycle"]), rep.ratio(so["loan_recycle"]), "贷款到账后 7 天还贷 + 外部转账 / 贷款金额。"],
    ]
    risk_body = (
        '<div class="duo wide-right">'
        + ui.metric_rows(risk_rows)
        + ui.table_box(
            ["风险标签", "Swap In", "Swap Out", "解读"],
            risk_tag_rows,
            note="横向标签用于增强主分层，不单独作为主分群；拒付包含冲正/费用，解决状态不可得。",
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="02 · 收入、资金流向与风险",
            kicker="2.3 · Risk Labels",
            title="横向风险标签",
            sub="两组都不是“余额缓冲型”人群：缓冲天数中位不足半天，缓冲低于 7 天接近全覆盖。",
            body=risk_body,
        )
    )

    # 11 Lending ecosystem
    product_rows = [
        ["现金贷 / 工资预支", rep.p(si["cash_advance"]), rep.p(so["cash_advance"]), rep.delta(si["cash_advance"], so["cash_advance"], "pct")],
        ["BNPL", rep.p(si["bnpl"]), rep.p(so["bnpl"]), rep.delta(si["bnpl"], so["bnpl"], "pct")],
        ["个人贷", rep.p(si["personal_loan"]), rep.p(so["personal_loan"]), rep.delta(si["personal_loan"], so["personal_loan"], "pct")],
        ["银行贷", rep.p(si["bank_loan"]), rep.p(so["bank_loan"]), rep.delta(si["bank_loan"], so["bank_loan"], "pct")],
    ]
    lender_rows = [
        [
            html.escape(str(provider)),
            str(loan[(loan["third_party"] == provider) & (loan["group"] == "Swap In")]["user_id"].nunique()),
            str(loan[(loan["third_party"] == provider) & (loan["group"] == "Swap Out")]["user_id"].nunique()),
            str(loan[loan["third_party"] == provider]["user_id"].nunique()),
        ]
        for provider in top.index
    ]
    lending_body = (
        '<div class="duo wide-right">'
        + ui.table_box(
            ["借贷产品", "Swap In", "Swap Out", "差异"],
            product_rows,
            note="使用率以组内用户为分母。",
            num_cols=(1, 2, 3),
        )
        + ui.table_box(
            ["主要借款机构 / 产品（按覆盖用户数）", "Swap In", "Swap Out", "合计"],
            lender_rows,
            note="仅列覆盖用户数最高的 10 个机构/产品；对手方由交易描述识别。",
            num_cols=(1, 2, 3),
        )
        + "</div>"
    )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="02 · 收入、资金流向与风险",
            kicker="2.4 · Lending Ecosystem",
            title="借贷生态与机构层观察",
            sub="借贷生态以现金贷/工资预支为主；Swap Out 的 BNPL、个人贷覆盖和活跃机构数更高。",
            body=lending_body,
        )
    )

    # 12 Payday behaviour + risk by segment
    pay_rows = [
        mrow("发薪后 7 天还贷 / 工资", si["pay_repay"], so["pay_repay"]),
        mrow("发薪后 7 天外部转账 / 工资", si["pay_ext"], so["pay_ext"]),
        mrow("发薪后 7 天还贷+转账 / 工资", si["pay_recycle"], so["pay_recycle"]),
        mrow("发薪后 7 天总流出 / 工资", si["pay_outflow"], so["pay_outflow"]),
        mrow("贷款到账后 7 天还贷 / 贷款", si["loan_repay"], so["loan_repay"], "ratio"),
        mrow("贷款到账后 7 天还贷+外部转账 / 贷款", si["loan_recycle"], so["loan_recycle"], "ratio"),
    ]
    risk_seg_rows = [
        [
            html.escape(seg),
            rep.p(risk_seg.loc[seg, "dishon_any"]),
            rep.n(risk_seg.loc[seg, "dishon_cnt"]),
            rep.p(risk_seg.loc[seg, "gamb_high"]),
            f"{rep.n(risk_seg.loc[seg, 'buffer'], 2)} 天",
        ]
        for seg in rep.SEG_ORDER
    ]
    pay_body = (
        '<div class="duo wide-right">'
        + ui.metric_rows(pay_rows)
        + ui.table_box(
            ["人群", "拒付发生", "拒付次数中位", "高赌博暴露", "缓冲天数"],
            risk_seg_rows,
            note=f"发薪覆盖率：Swap In {rep.p(si['pay_cov'])}、Swap Out {rep.p(so['pay_cov'])}；贷款覆盖率：Swap In {rep.p(si['loan_cov'])}、Swap Out {rep.p(so['loan_cov'])}。",
            num_cols=(1, 2, 3, 4),
        )
        + "</div>"
        + ui.note_strip(
            "口径提示",
            "发薪后窗口为工资入账日之后的 7 天；贷款窗口为贷款到账日之后的 7 天。比率为用户内事件中位数，可能超过 100%，因为窗口内还有其它入账。",
        )
    )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="02 · 收入、资金流向与风险",
            kicker="2.5 · Payday Behaviour",
            title="发薪后行为与借新还旧",
            sub="Swap Out 的工资到账后更快被还贷和外部转账吸收；Swap In 内部转账更多。",
            body=pay_body,
        )
    )

    # 13 Closing
    slides.append(
        ui.slide_closing(
            brand=BRAND,
            kicker="结论 · Takeaway",
            title="对比结论",
            sub="Swap Out 更高收入、更高杠杆、更高拒付暴露；Swap In 收入更弱、内部周转更多；两组共同点是账户缓冲极薄。",
            takeaways=[
                ("收入", f"Swap Out 中位真实收入 <b>{rep.m(so['income'])}</b>，工资覆盖 <b>{rep.p(so['any_wage'])}</b>；Swap In 分别为 <b>{rep.m(si['income'])}</b> 与 <b>{rep.p(si['any_wage'])}</b>。"),
                ("杠杆", f"Swap Out 有借贷放款 <b>{rep.p(so['any_loan_draw'])}</b>、偿债/收入 <b>{rep.p(so['repay_rate'])}</b>、活跃机构中位 <b>{rep.n(so['active_lender'], 0)}</b>；Swap In 均更低。"),
                ("流动性", f"两组缓冲天数中位仅 <b>{rep.n(si['buffer'], 2)}</b> / <b>{rep.n(so['buffer'], 2)}</b> 天，缓冲不足 7 天占比 <b>{rep.p(si['buffer_lt7'])}</b> / <b>{rep.p(so['buffer_lt7'])}</b>。"),
            ],
            source=(
                "<b>数据源</b>：data/swap_in.csv、data/swap_out.csv；一行一笔交易，按 user_id 聚合。"
                "分类映射基于当前样本实际取值，跨数据集需重新校准；以上为描述性对比，不做因果推断。"
            ),
            date=f"生成时间 {generated_at}",
        )
    )

    return ui.deck_document(
        title="Swap In vs Swap Out 人群画像对比分析 · 演示文稿",
        slides=slides,
    )


def build_concise_deck(users: pd.DataFrame, tx: pd.DataFrame, generated_at: str) -> str:
    """Four-page deck: income, liability, spending, liquidity."""
    grp = {name: users[users["group"] == name] for name in ("Swap In", "Swap Out")}
    si, so = (rep.summary(grp[name]) for name in ("Swap In", "Swap Out"))
    mi, mo = (rep.income_mix(grp[name]) for name in ("Swap In", "Swap Out"))
    oi, oo = (rep.outflow_mix(grp[name]) for name in ("Swap In", "Swap Out"))
    n_note = f"n = Swap In {si['n']:,} / Swap Out {so['n']:,}"

    def kpi_row(cards):
        inner = ""
        for label, value, note, var in cards:
            inner += (
                '<div class="cmp-kpi solo">'
                f'<div class="label"><span>{label}</span><code class="var">{var}</code></div>'
                f'<div class="pair"><span class="val">{value}</span></div>'
                f'<div class="delta">{note}</div>'
                "</div>"
            )
        return f'<div class="cmp-kpi-row reveal d1">{inner}</div>'

    def cmp_cards(cards):
        inner = ""
        for label, iv, ov, note, var in cards:
            inner += (
                '<div class="side-card cmp">'
                f'<div class="k"><span>{label}</span><code class="var">{var}</code></div>'
                f'<div class="prow"><span class="who"><i style="background:{IN_COLOR}"></i>Swap In</span>'
                f"<b>{iv}</b></div>"
                f'<div class="prow"><span class="who"><i style="background:{OUT_COLOR}"></i>Swap Out</span>'
                f"<b>{ov}</b></div>"
                f"<p>{note}</p></div>"
            )
        return f'<div class="side-stack reveal d2">{inner}</div>'

    def pct_chart(labels, in_vals, out_vals, digits=1):
        fig = go.Figure()
        chart_ui.add_grouped_bars(
            fig,
            labels,
            [v * 100 for v in in_vals],
            [v * 100 for v in out_vals],
            kind="pct",
            digits=digits,
        )
        vals = [v * 100 for v in in_vals + out_vals if v == v]
        mx = max(vals + [1]) * 1.24
        fig.update_xaxes(range=[0, mx], ticksuffix="%")
        fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)), showgrid=False)
        chart_ui.base_layout(fig, height=420, margin=dict(l=150, r=24, t=44, b=70), showlegend=True)
        fig.update_layout(
            barmode="group",
            bargap=0.34,
            bargroupgap=0.08,
            legend=dict(
                orientation="h", x=0, xanchor="left", y=-0.22, yanchor="top",
                font=dict(size=11), bgcolor="rgba(0,0,0,0)",
            ),
        )
        return chart_ui.fig_html(fig, 420)

    slides: list[str] = []

    # 01 收入
    income_chart = pct_chart(
        ["工资 / 入账", "福利 / 入账", "其他 / 入账", "转账入账 / 外部入账", "借贷入账 / 外部入账"],
        [mi["wage_share"], mi["welfare_share"], mi["other_share"], mi["transfer_in"], mi["loan_in_ratio"]],
        [mo["wage_share"], mo["welfare_share"], mo["other_share"], mo["transfer_in"], mo["loan_in_ratio"]],
    )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="01 · 收入",
            kicker="Income · 01",
            title="Swap Out 收入更高、工资占比更高，Swap In 更依赖福利与其他入账",
            body=kpi_row(
                [
                    ("中位真实收入", rep.m(si["income"]), f"Swap Out {rep.m(so['income'])}", "income"),
                    ("工资覆盖率", rep.p(si["any_wage"]), f"Swap Out {rep.p(so['any_wage'])}", "any_wage"),
                    ("工资 / 入账中位", rep.p(si["wage_ratio"]), f"Swap Out {rep.p(so['wage_ratio'])}", "wage_ratio"),
                    ("收入波动 CV 中位", rep.n(si["income_cv"], 2), f"Swap Out {rep.n(so['income_cv'], 2)}", "income_cv"),
                ]
            )
            + '<div class="duo">'
            + ui.chart_box(
                income_chart,
                f"{n_note}；工资、福利、其他占比以分项入账为分母，"
                "转账与借贷占比以外部入账为分母。变量：income_mix().wage_share / welfare_share / "
                "other_share / transfer_in / loan_in_ratio。",
            )
            + cmp_cards(
                [
                    (
                        "真实收入 / 外部入账",
                        rep.p(si["true_income_ratio"]),
                        rep.p(so["true_income_ratio"]),
                        "真实收入定义为工资 + 福利 + 其他入账",
                        "true_income_ratio",
                    ),
                    (
                        "借贷入账 / 外部入账",
                        rep.p(mi["loan_in_ratio"]),
                        rep.p(mo["loan_in_ratio"]),
                        "外部入账含转账、借贷与真实收入",
                        "income_mix().loan_in_ratio",
                    ),
                    (
                        "中位交易笔数",
                        rep.n(si["txn"]),
                        rep.n(so["txn"]),
                        "观察窗口内每人借记与贷记合计",
                        "txn",
                    ),
                    (
                        "中位账户数",
                        rep.n(si["accounts"]),
                        rep.n(so["accounts"]),
                        "可识别银行账户数量",
                        "accounts",
                    ),
                ]
            )
            + "</div>",
        )
    )

    # 02 负债
    liability_chart = pct_chart(
        ["有借贷放款", "现金贷", "BNPL", "个人贷", "银行贷款", "有拒付"],
        [si["any_loan_draw"], si["cash_advance"], si["bnpl"], si["personal_loan"], si["bank_loan"], si["any_dishonour"]],
        [so["any_loan_draw"], so["cash_advance"], so["bnpl"], so["personal_loan"], so["bank_loan"], so["any_dishonour"]],
    )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="02 · 负债",
            kicker="Liability · 02",
            title="Swap Out 借贷渗透更高、拒付笔数接近三倍，还款负担超过两成",
            body=kpi_row(
                [
                    ("有借贷放款", rep.p(si["any_loan_draw"]), f"Swap Out {rep.p(so['any_loan_draw'])}", "any_loan_draw"),
                    ("现金贷使用率", rep.p(si["cash_advance"]), f"Swap Out {rep.p(so['cash_advance'])}", "cash_advance"),
                    ("偿债 / 收入中位", rep.p(si["repay_rate"]), f"Swap Out {rep.p(so['repay_rate'])}", "repay_rate"),
                    ("有拒付记录", rep.p(si["any_dishonour"]), f"Swap Out {rep.p(so['any_dishonour'])}", "any_dishonour"),
                ]
            )
            + '<div class="duo">'
            + ui.chart_box(
                liability_chart,
                f"{n_note}；发生率以有效用户为分母。"
                "变量：any_loan_draw / cash_advance / bnpl / personal_loan / bank_loan / any_dishonour。",
            )
            + cmp_cards(
                [
                    (
                        "活跃借贷机构中位",
                        rep.n(si["active_lender"], 0),
                        rep.n(so["active_lender"], 0),
                        "观察窗口内发生放款或还款的机构数",
                        "active_lender",
                    ),
                    (
                        "拒付笔数中位",
                        rep.n(si["dishonour_cnt"], 0),
                        rep.n(so["dishonour_cnt"], 0),
                        "Swap Out 接近 Swap In 的三倍",
                        "dishonour_cnt",
                    ),
                    (
                        "拒付金额中位",
                        rep.m(si["dishonour_amt"]),
                        rep.m(so["dishonour_amt"]),
                        "按用户汇总的拒付金额中位",
                        "dishonour_amt",
                    ),
                    (
                        "个人贷使用率",
                        rep.p(si["personal_loan"]),
                        rep.p(so["personal_loan"]),
                        f"BNPL {rep.p(si['bnpl'])} / {rep.p(so['bnpl'])}",
                        "personal_loan",
                    ),
                ]
            )
            + "</div>",
        )
    )

    # 03 支出
    spend_labels = ["消费", "外部转账", "内部转账", "还债", "赌博", "费用", "取现", "其他"]
    spend_keys = ["consumption", "ext_transfer_out", "internal_transfer_out", "debt_repay", "gambling", "fees", "cash", "other_debit"]
    spend_chart = pct_chart(
        spend_labels,
        [oi[k] for k in spend_keys],
        [oo[k] for k in spend_keys],
        digits=1,
    )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="03 · 支出",
            kicker="Spending · 03",
            title="Swap Out 外部转账与还债占比更高，Swap In 内部周转更重",
            body=kpi_row(
                [
                    ("消费 / 总支出", rep.p(si["consumption_share"]), f"Swap Out {rep.p(so['consumption_share'])}", "consumption_share"),
                    ("外部转账 / 总支出", rep.p(si["ext_transfer_share"]), f"Swap Out {rep.p(so['ext_transfer_share'])}", "ext_transfer_share"),
                    ("内部转账 / 总支出", rep.p(si["internal_transfer_share"]), f"Swap Out {rep.p(so['internal_transfer_share'])}", "internal_transfer_share"),
                    ("还债 / 总支出", rep.p(si["debt_share"]), f"Swap Out {rep.p(so['debt_share'])}", "debt_share"),
                ]
            )
            + '<div class="duo">'
            + ui.chart_box(
                spend_chart,
                f"{n_note}；金额占比以全部借记流水为分母，非用户中位。"
                "变量：outflow_mix().consumption / ext_transfer_out / internal_transfer_out / "
                "debt_repay / gambling / fees / cash / other_debit。",
            )
            + cmp_cards(
                [
                    (
                        "赌博 / 支出中位",
                        rep.p(si["gambling_share"]),
                        rep.p(so["gambling_share"]),
                        f"高赌博标识 {rep.p(si['gambling_high'])} / {rep.p(so['gambling_high'])}",
                        "gambling_share",
                    ),
                    (
                        "取现 / 总支出",
                        rep.p(oi["cash"]),
                        rep.p(oo["cash"]),
                        f"其他支出 {rep.p(oi['other_debit'])} / {rep.p(oo['other_debit'])}",
                        "outflow_mix().cash",
                    ),
                    (
                        "高赌博标识",
                        rep.p(si["gambling_high"]),
                        rep.p(so["gambling_high"]),
                        "赌博占支出 ≥5% 或金额 ≥$1,000",
                        "gambling_high",
                    ),
                    (
                        "中位交易笔数",
                        rep.n(si["txn"]),
                        rep.n(so["txn"]),
                        "流水越活跃，分类覆盖率越高",
                        "txn",
                    ),
                ]
            )
            + "</div>",
        )
    )

    # 04 流动性
    liquidity_chart = pct_chart(
        ["缓冲 < 7 天", "缓冲 < 1 天", "负余额天数占比", "低余额天数占比"],
        [si["buffer_lt7"], si["buffer_lt1"], si["neg_days"], si["low_days"]],
        [so["buffer_lt7"], so["buffer_lt1"], so["neg_days"], so["low_days"]],
    )
    slides.append(
        ui.slide_content(
            brand=BRAND,
            section="04 · 流动性",
            kicker="Liquidity · 04",
            title="账户缓冲不足半天，Swap Out 的负余额与拒付压力更高",
            body=kpi_row(
                [
                    ("中位账户余额", rep.m(si["balance"]), f"Swap Out {rep.m(so['balance'])}", "balance"),
                    ("中位缓冲天数", f"{rep.n(si['buffer'], 2)} 天", f"Swap Out {rep.n(so['buffer'], 2)} 天", "buffer"),
                    ("缓冲 < 7 天", rep.p(si["buffer_lt7"]), f"Swap Out {rep.p(so['buffer_lt7'])}", "buffer_lt7"),
                    ("负余额天数占比", rep.p(si["neg_days"]), f"Swap Out {rep.p(so['neg_days'])}", "neg_days_share"),
                ]
            )
            + '<div class="duo">'
            + ui.chart_box(
                liquidity_chart,
                f"{n_note}；缓冲天数 = 余额 /（日均支出）；"
                "负余额与低余额为观察窗口内天数占比。变量：buffer_lt7 / buffer_lt1 / "
                "neg_days_share / low_balance_days_share。",
            )
            + cmp_cards(
                [
                    (
                        "缓冲 < 1 天",
                        rep.p(si["buffer_lt1"]),
                        rep.p(so["buffer_lt1"]),
                        "账户余额不足一天支出",
                        "buffer_lt1",
                    ),
                    (
                        "低余额天数占比",
                        rep.p(si["low_days"]),
                        rep.p(so["low_days"]),
                        "观察窗口内余额处于低位的天数占比",
                        "low_balance_days_share",
                    ),
                    (
                        "拒付笔数中位",
                        rep.n(si["dishonour_cnt"], 0),
                        rep.n(so["dishonour_cnt"], 0),
                        "与薄缓冲同时出现，履约压力更直接",
                        "dishonour_cnt",
                    ),
                    (
                        "拒付金额中位",
                        rep.m(si["dishonour_amt"]),
                        rep.m(so["dishonour_amt"]),
                        "按用户汇总的拒付金额中位",
                        "dishonour_amt",
                    ),
                ]
            )
            + "</div>",
        )
    )

    return ui.deck_document(
        title="Swap In vs Swap Out 人群画像对比 · 4 页版",
        slides=slides,
        plotly_js=get_plotlyjs(),
        extra_css=ui.CONCISE_CSS,
    )


def main() -> None:
    frames = [ana.load_group(name, path) for name, path in ana.DATA.items()]
    tx = pd.concat(frames, ignore_index=True)
    users = rep.assign_segments(ana.user_metrics(tx))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = build_concise_deck(users, tx, generated_at)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(report, encoding="utf-8")
    print(f"deck -> {OUT} ({len(report)/1e6:.1f} MB)")
    print(f"users: {len(users):,} | tx: {len(tx):,}")


if __name__ == "__main__":
    main()
