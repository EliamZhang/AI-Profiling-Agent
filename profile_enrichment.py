"""Shared Fundo / WageGo variable review, cross-analysis and slide additions.

All calculations use the same deduplicated, valid sample as the original deck.
No outcome label or externally sourced customer data is used.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import deck_framework as ui

S = "bank_txn_surplus_"
W = "bank_txn_worthiness_"
L = "bank_txn_lender_"
G = "bank_txn_category_gambling_"
EXTRA_COLUMNS = [
    W + "income_tag_amt_sum_182d",
    W + "expense_tag_amt_sum_182d",
    L + "disburse_jgs_l182d", L + "repay_jgs_l182d",
    L + "disburse_jgs_l28d", L + "repay_jgs_l28d",
    W + "liability_repayment_active_month_cnt_182d",
    W + "balance_above_1x_monthly_fixed_spend_days_share_182d",
    W + "balance_obs_cnt_182d", W + "balance_account_covered_cnt_182d",
    G + "credit_amt_182d",
] + [S + k + "_sum_28d" for k in [
    "income", "expense", "surplus", "liabilityrepayment", "liabilitydrawdown"
]]

GROUPS = ["无已识别机构", "1 家", "2 家", "3–4 家", "≥5 家"]
BURDEN_GROUPS = ["≤10%", "10%–30%", "30%–50%", "50%–100%", ">100%"]
BAL_GROUPS = ["未见负余额", "负余额占比 ≤10%", "负余额占比 10%–25%", "负余额占比 >25%"]
GAMBLE_GROUPS = ["无净流出", "净流出 ≤5%", "净流出 >5%"]
EXTRA_CSS = """
  .fundo-evidence .deck-table th{font-size:11px;padding:12px 16px}
  .fundo-evidence .deck-table td{font-size:14px;line-height:1.6;padding:18px 16px}
  .fundo-audit .deck-table td{padding:20px 16px}
  .fundo-evidence .table-note{font-size:11px;line-height:1.6}
  .fundo-scope .kpi-value{font-size:30px}
"""


def ratio(a, b):
    return (a / b.where(b > 0)).replace([np.inf, -np.inf], np.nan)


def rate(s):
    return float(s.dropna().mean()) if s.notna().any() else float("nan")


def med(s):
    return float(s.median()) if s.notna().any() else float("nan")


def outcome(d):
    s = d["bank_txn_dishonour_lender_unresolved_cnt_3gp_182d"]
    return s.gt(0).astype(float).where(s.notna())


def group_rows(d, groups, order):
    rows = []
    for label in order:
        z = d.loc[groups == label]
        y = z["e_unresolved"].dropna()
        n = len(y)
        p = rate(y)
        # Wilson 95% intervals describe sampling variability, not model accuracy.
        if n:
            den = 1 + 1.96**2 / n
            center = (p + 1.96**2 / (2*n)) / den
            half = 1.96 * np.sqrt(p*(1-p)/n + 1.96**2/(4*n*n)) / den
        else:
            center = half = np.nan
        rows.append(dict(group=label, n=len(z), outcome_n=n,
                         unresolved= p, low=center-half, high=center+half,
                         repayment_total_p50=med(z["rep_rate"]),
                         repayment_recurring_p50=med(z["e_recurring_burden"]),
                         balance_p50=med(z["bal"]),
                         active_months_p50=med(z[W+"liability_repayment_active_month_cnt_182d"])))
    return rows


def analyze(d, m, *, dataset_key="fundo", dataset_name=None):
    """Add derived fields and aggregate evidence; missing values stay explicit."""
    missing = sorted(set(EXTRA_COLUMNS) - set(d.columns))
    if missing:
        raise ValueError(f"{dataset_name or dataset_key} 补充分析缺少字段: {missing}")
    e = {"n": len(d), "new_columns": len(EXTRA_COLUMNS),
         "dataset_key": dataset_key, "dataset_name": dataset_name or ("Fundo" if dataset_key == "fundo" else "WageGo")}
    d["e_unresolved"] = outcome(d)
    wage = d["bank_txn_income_wages_sum_182d"]
    benefit = d["bank_txn_income_centrelink_sum_182d"]
    # Both components must be observable. An observed zero is not missing.
    d["e_recurring"] = wage + benefit
    d["e_recurring_burden"] = ratio(d["rep"], d["e_recurring"])
    e["recurring_n"] = int(d.e_recurring_burden.notna().sum())
    e["recurring_absent_n"] = int(d.e_recurring.eq(0).sum())
    e["recurring_absent_repay_n"] = int((d.e_recurring.eq(0) & d.rep.gt(0)).sum())
    e["recurring_p50"] = med(d.e_recurring_burden)
    e["recurring_p90"] = float(d.e_recurring_burden.quantile(.9))
    eligible = d.e_recurring_burden.notna()
    e["same_total_p50"] = med(d.loc[eligible, "rep_rate"])
    e["burden_gt30_n"] = int(d.e_recurring_burden.gt(.3).sum())
    e["burden_gt30"] = e["burden_gt30_n"] / e["recurring_n"]
    groups = pd.cut(d.e_recurring_burden, [-.001,.1,.3,.5,1,np.inf], labels=BURDEN_GROUPS)
    e["burden_rows"] = group_rows(d, groups, BURDEN_GROUPS)

    # jgs counts identified institutions with transactions, not outstanding loans.
    jgs = d[L+"repay_jgs_l182d"]
    groups = pd.cut(jgs.fillna(0), [-1,0,1,2,4,np.inf], labels=GROUPS)
    d["e_lender_group"] = groups
    e["lender_rows"] = group_rows(d, groups, GROUPS)
    e["multi5_n"] = int(jgs.ge(5).sum())
    e["multi5"] = e["multi5_n"] / len(d)
    e["recent_multi3_n"] = int(d[L+"repay_jgs_l28d"].ge(3).sum())
    e["recent_multi3"] = e["recent_multi3_n"] / len(d)
    e["disburse_multi3"] = float(d[L+"disburse_jgs_l182d"].ge(3).mean())
    e["recent_disburse_multi3"] = float(d[L+"disburse_jgs_l28d"].ge(3).mean())
    e["jgs_zero_with_unresolved"] = int((jgs.fillna(0).eq(0) & d.e_unresolved.eq(1)).sum())
    e["jgs_zero_with_repay"] = int((jgs.fillna(0).eq(0) & d.rep.gt(0)).sum())

    tag_inc = d[W+"income_tag_amt_sum_182d"]
    tag_exp = d[W+"expense_tag_amt_sum_182d"]
    component_cols = ["bank_txn_income_"+k+"_sum_182d" for k in ["wages","centrelink","otherincome"]]
    components = d[component_cols].sum(axis=1, min_count=3)
    formula = d.inc - d.exp - d.rep - d[S+"dishonoursfee_sum_182d"] + d.draw
    checks = [
        ("盈余恒等式", formula, d.sur, .02),
        ("工资+福利+其他入账 = income_tag", components, tag_inc, .02),
        ("预计算收支比 = income_tag / expense_tag", ratio(tag_inc,tag_exp), d[W+"income_to_expense_ratio_182d"], .00002),
    ]
    e["identity_checks"] = []
    for name, a, b, tol in checks:
        delta = (a-b).dropna().abs()
        e["identity_checks"].append(dict(check=name,n=len(delta),pass_n=int(delta.le(tol).sum()),tolerance=tol,max_abs_error=float(delta.max())))
        if len(delta) == 0 or not delta.le(tol).all():
            raise ValueError(f"口径核验失败: {name}")
    e["surplus_ratio_p50"] = med(ratio(d.inc,d.exp))
    e["tag_ratio_p50"] = med(ratio(tag_inc,tag_exp))
    e["tag_ratio_n"] = int(ratio(tag_inc,tag_exp).notna().sum())
    e["tag_deficit"] = rate((tag_inc-tag_exp).lt(0).astype(float).where(ratio(tag_inc,tag_exp).notna()))
    e["tag_income_month_p50"] = med(tag_inc/6)
    e["tag_expense_month_p50"] = med(tag_exp/6)
    e["income_difference_month_p50"] = med((tag_inc-d.inc)/6)
    e["same_income_n"] = int((tag_inc-d.inc).abs().le(.02).sum())

    # Compare two non-overlapping windows: latest 28 days vs previous 154.
    e["recent_rows"] = []
    for key,label in [("income","现金流收入"),("expense","现金流支出"),("liabilityrepayment","还本付息"),("liabilitydrawdown","新增借款"),("surplus","含借款净流量")]:
        recent = d[S+key+"_sum_28d"]
        previous = d[S+key+"_sum_182d"]-recent
        if key != "surplus" and (previous.dropna() < -.02).any():
            raise ValueError(f"{key}: 28天金额大于182天金额，不能构造互斥窗口")
        d["e_recent_"+key] = ratio(recent/28,previous/154)
        e["recent_rows"].append(dict(metric=label,previous_month_p50=med(previous/154*30),recent_month_p50=med(recent/28*30),paired_ratio_p50=med(d["e_recent_"+key]) if key != "surplus" else None,paired_n=int(d["e_recent_"+key].notna().sum()) if key != "surplus" else int((recent.notna()&previous.notna()).sum())))
    comparable = d.e_recent_income.notna() & d.e_recent_liabilityrepayment.notna()
    double = comparable & d.e_recent_income.lt(.8) & d.e_recent_liabilityrepayment.gt(1.2)
    d["e_recent_double"] = double
    e["recent_comparable_n"] = int(comparable.sum())
    e["double_n"] = int(double.sum())
    e["double_all_share"] = e["double_n"] / len(d)
    e["double_comparable_share"] = e["double_n"] / e["recent_comparable_n"]
    e["double_rate"] = rate(d.loc[double,"e_unresolved"])
    e["other_comparable_rate"] = rate(d.loc[comparable & ~double,"e_unresolved"])
    e["double_rate_difference_pp"] = (e["double_rate"]-e["other_comparable_rate"])*100
    e["recent_repayment_growth"] = med(d.e_recent_liabilityrepayment)-1
    e["recent_disbursement_growth"] = med(d.e_recent_liabilitydrawdown)-1

    neg = d[W+"balance_negative_share_avg_acct_182d"]
    if not neg.dropna().between(0,1).all():
        raise ValueError("负余额占比超出 [0,1]")
    groups = pd.cut(neg, [-.001,0,.1,.25,1], labels=BAL_GROUPS)
    e["balance_rows"] = group_rows(d, groups, BAL_GROUPS)
    e["balance_n"] = int(neg.notna().sum())
    e["ever_negative"] = rate(neg.gt(0).astype(float).where(neg.notna()))
    positive = d.bal.ge(0) & neg.notna()
    hidden = positive & neg.ge(.1)
    e["hidden_n"] = int(hidden.sum())
    e["positive_snapshot_n"] = int(positive.sum())
    e["hidden_share_positive"] = e["hidden_n"] / e["positive_snapshot_n"]
    fixedcover = d[W+"balance_above_1x_monthly_fixed_spend_days_share_182d"]
    e["fixedcover_n"] = int(fixedcover.notna().sum())
    e["fixedcover_p50"] = med(fixedcover)
    e["balance_obs_p50"] = med(d[W+"balance_obs_cnt_182d"])
    e["balance_accounts_p50"] = med(d[W+"balance_account_covered_cnt_182d"])

    debit = d[G+"debit_amt_182d"]
    credit = d[G+"credit_amt_182d"]
    net = debit-credit
    d["e_gamble_burden"] = ratio(net.clip(lower=0),d.e_recurring)
    groups = pd.cut(d.e_gamble_burden, [-.001,0,.05,np.inf], labels=GAMBLE_GROUPS)
    e["gamble_rows"] = group_rows(d, groups, GAMBLE_GROUPS)
    cells = []
    for lender_group in GROUPS:
        for gamble_group in GAMBLE_GROUPS:
            z = d.loc[(d.e_lender_group == lender_group)&(groups==gamble_group),"e_unresolved"].dropna()
            cells.append(dict(lenders=lender_group,gambling=gamble_group,n=len(z),unresolved=rate(z)))
    e["gamble_cells"] = cells
    e["gamble_n"] = int(d.e_gamble_burden.notna().sum())
    matched = debit.notna() & credit.notna()
    e["gamble_matched_n"] = int(matched.sum())
    e["gamble_debit_total"] = float(debit[matched].sum())
    e["gamble_credit_total"] = float(credit[matched].sum())
    e["gamble_net_total"] = float(net[matched].sum())
    e["gamble_gt5_n"] = int(d.e_gamble_burden.gt(.05).sum())
    e["gamble_gt5"] = e["gamble_gt5_n"] / e["gamble_n"]
    m["profile_enrichment"] = e
    if dataset_key == "fundo":
        m["fundo_enrichment"] = e  # Backward-compatible Fundo evidence consumers.
    return e


def rate_chart(mod, rows):
    p = [r["unresolved"] for r in rows]
    fig = go.Figure(go.Bar(
        x=p, y=[r["group"] for r in rows], orientation="h",
        marker_color=["#5E8BB4","#2C6CB0","#1D4E79","#C98A2E","#B95F5F"][:len(rows)],
        text=[f"{r['unresolved']:.1%} · n={r['outcome_n']:,}" for r in rows],
        textposition="outside", cliponaxis=False,
        error_x=dict(type="data",symmetric=False,array=[r['high']-r['unresolved'] for r in rows],arrayminus=[r['unresolved']-r['low'] for r in rows],color="#66788A",thickness=1),
        hovertemplate="%{y}<br>未解决拒付用户占比 %{x:.1%}<br>%{text}<extra></extra>",
    ))
    mod.base_layout(fig, margin=dict(l=155,r=155,t=35,b=55))
    fig.update_xaxes(title_text="未解决拒付用户占比",tickformat=".0%",range=[0,max(.7,max(p)*1.3)])
    fig.update_yaxes(autorange="reversed",showgrid=False)
    return mod.fig_html(fig)


def gamble_chart(mod, e):
    rates=[]; labels=[]
    for g in GROUPS:
        cells=[c for c in e["gamble_cells"] if c["lenders"]==g]
        rates.append([c["unresolved"] if c["n"]>=30 else None for c in cells])
        labels.append([f"{c['unresolved']:.1%}<br>n={c['n']:,}" if c['n']>=30 else f"n={c['n']}<br>样本不足" for c in cells])
    fig=go.Figure(go.Heatmap(z=rates,x=GAMBLE_GROUPS,y=GROUPS,text=labels,texttemplate="%{text}",textfont=dict(size=12),colorscale=[[0,"#EDF4FB"],[.5,"#8AAECC"],[1,"#1D4E79"]],zmin=0,zmax=1,showscale=False,hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>",xgap=4,ygap=4))
    mod.base_layout(fig,margin=dict(l=125,r=18,t=30,b=65))
    fig.update_xaxes(title_text="赌博正净流出 / 工资与福利",showgrid=False)
    fig.update_yaxes(title_text="还款机构数",autorange="reversed",showgrid=False)
    return mod.fig_html(fig)


def extra_slides(mod, ds, m):
    """Return named slides for insertion next to their relevant original analysis."""
    e=m["profile_enrichment"]; fp=mod.fmt_pct; fm=mod.fmt_money
    brand=ds["brand"]
    def slide(title,kicker,sub,body,section="补充分析"):
        return ui.slide_content(brand=brand,section=section,kicker=kicker,title=title,sub=sub,body=body)
    def chart_body(chart,caption,cards,facts):
        return '<div class="duo">'+ui.chart_box(chart,caption)+ui.side_cards(cards)+"</div>"+ui.fact_strip(facts)
    def evidence_table(headers,rows,**kwargs):
        return ui.table_box(headers,rows,**kwargs).replace('class="table-box"','class="table-box fundo-evidence"',1)
    slides={}
    scope_rows=[
        ["月收入中位",fm(m['inc_p50']),fm(e['tag_income_month_p50']),"统计范围不同，不可互换"],
        ["月支出中位",fm(m['exp_p50']),fm(e['tag_expense_month_p50']),"同名支出也需区分体系"],
        ["收入 / 支出中位",f"{e['surplus_ratio_p50']:.3f}",f"{e['tag_ratio_p50']:.3f}","两个比值不能混用于同一结论"],
        ["偿债前负差额占比",fp(m['pre_debt_neg']),fp(e['tag_deficit']),"按各自收入减支出逐户计算"],
        ["本报告用途","收支、净流量与分层","来源构成、标签支出与负担比","分位数独立计算，不能直接相减"],
    ]
    slides["scope"]=slide("先统一口径，再解释缺口","Analysis Basis · 数据口径", "两套特征体系的收入与支出范围不同；入账来源构成与现金流盈余需分别解读。",
        evidence_table(["比较指标","现金流体系 · surplus","标签体系 · income/worthiness","解读"],scope_rows,
            note=f"现金流体系 n={e['n']:,}；标签体系收支比 n={e['tag_ratio_n']:,}。金额月化按 6 个月折算。")
        +ui.kpi_cards([("现金流盈余公式", "核验通过",f"{e['n']:,} 人误差均 ≤$0.02"),("来源分项之和", "等于标签收入", "工资 + 福利 + 其他入账"),("标签收入−现金流收入",fm(e['income_difference_month_p50']),"逐户差额月化中位；差异原因待释义确认"),("两套收入一致的用户",f"{e['same_income_n']:,} 人",f"仅占 {e['same_income_n']/e['n']:.2%}；相差不超过 $0.02")]).replace('class="kpi-row reveal d1"','class="kpi-row reveal d1 fundo-scope"')
        +ui.note_strip("解释边界","盈余 = 现金流收入 − 现金流支出 − 还本付息 − 拒付费 + 新增借款。含借款的净流量不等于自主还款能力；转账也不直接等于可持续收入。固定账单、房租、日常生活及高承诺支出可能重叠，不重复相加。"))

    slides["recurring"]=slide("收入来源与偿债承受力","1.3 · Repayment Capacity", "改用工资与福利作为可识别经常性入账的代理，观察被总入账稀释的偿债负担。",
        chart_body(rate_chart(mod,e['burden_rows']),f"按还本付息/(工资+福利)分组；可计算 n={e['recurring_n']:,}；横线为比例的 Wilson 95% 区间。",
        [("同一批用户 · 两种分母",f"{fp(e['same_total_p50'])} → {fp(e['recurring_p50'])}","现金流总收入 → 工资+福利；各自计算中位数"),
         ("工资+福利口径负担 >30%",fp(e['burden_gt30']),f"{e['burden_gt30_n']:,}/{e['recurring_n']:,} 人；P90 {fp(e['recurring_p90'])}"),
         ("未识别工资与福利入账",f"{e['recurring_absent_n']:,} 人",f"其中 {e['recurring_absent_repay_n']:,} 人有还款；不能据此判断无收入")],
        [(f"{e['recurring_n']:,}","工资+福利为正且还款可观测"),(fp(e['burden_rows'][0]['unresolved']),"负担 ≤10% 组未解决拒付率"),(fp(e['burden_rows'][2]['unresolved']),"负担 30%–50% 组未解决拒付率"),("代理口径","不含核实收入与可支配收入认定")]),"01 · 收入与支出")

    slides["multi"]=slide("多头借贷：还款关系比新增借款更广","2.3 · Multi-lender Exposure", "机构数揭示持续还款关系；这里衡量发生过交易的已识别机构，不代表当前在贷机构或合同数。",
        chart_body(rate_chart(mod,e['lender_rows']),f"182 天已识别还款机构数（repay_jgs）；n={e['n']:,}；横线为 Wilson 95% 区间。无已识别机构组仍有 {e['jgs_zero_with_repay']:,} 人存在还款，不能视为真实无债。",
        [("≥5 家还款机构",fp(e['multi5']),f"{e['multi5_n']:,} 人；该组未解决拒付率 {fp(e['lender_rows'][-1]['unresolved'])}"),
         ("近 28 天仍向 ≥3 家机构还款",fp(e['recent_multi3']),f"{e['recent_multi3_n']:,} 人 / 全部有效用户"),
         ("新借款涉及 ≥3 家机构",fp(e['disburse_multi3']),f"182 天；近 28 天 {fp(e['recent_disburse_multi3'])}")],
        [(fp(e['lender_rows'][1]['unresolved']),"1 家组未解决拒付率"),(fp(e['lender_rows'][-1]['repayment_recurring_p50']),"≥5 家组还款/工资福利中位"),(f"{e['lender_rows'][-1]['active_months_p50']:.0f}","≥5 家组有还款自然月份数中位"),("不作因果判断","机构数与拒付同期存在关联")]),"02 · 负债、流动性与风险")

    recent_rows=[[r['metric'],fm(r['previous_month_p50']),fm(r['recent_month_p50']),f"{r['paired_ratio_p50']:.3f}" if r['paired_ratio_p50'] is not None else "不对负值计算增长率",f"{r['paired_n']:,}"] for r in e['recent_rows']]
    slides["recent"]=slide("近期压力：收入回落与还款上升叠加","2.5 · Recent Cashflow Pressure", "用最近 28 天与此前 154 天的日均金额比较，避免重叠窗口掩盖变化；仅构造同一用户可比的增长比。",
        evidence_table(["指标","此前154天 · 月化中位","最近28天 · 月化中位","逐户日均比中位","比值有效人数"],recent_rows,
            note="日均比=(28天金额/28)÷[(182天金额−28天金额)/154]，历史基数须为正；月化仅为日均×30。两列中位数不可相除得到逐户比。",num_cols=(1,2,3,4))
        +ui.kpi_cards([("收入下降 >20% 且还款上升 >20%",f"{e['double_n']:,} 人",f"占全部用户 {fp(e['double_all_share'])}；占成对可比样本 {fp(e['double_comparable_share'])}"),
        ("双重压力组 · 未解决拒付",fp(e['double_rate']),f"其余可比用户 {fp(e['other_comparable_rate'])}"),
        ("收入与还款均可比较",f"{e['recent_comparable_n']:,} 人","历史基数为零或缺失者不计算增长比"),
        ("同期拒付率差异",f"{e['double_rate_difference_pp']:.1f} 个百分点","可比组表现接近；新增价值在近期资金变化")])
        +ui.note_strip("用途与边界","用于定位近期资金变化；短窗口可能受发薪日、到账周期和流水覆盖影响。双重压力组与其余可比用户的同期未解决拒付率接近，不能称为明显更高风险；也不能仅凭总体趋势中位数判断借新还旧。"),"02 · 负债、流动性与风险")

    slides["balance"]=slide("余额为正，也可能存在持续资金压力","2.6 · Liquidity Persistence", "从最近快照扩展到账户历史负余额占比，区分一时回补与长期可用缓冲。",
        chart_body(rate_chart(mod,e['balance_rows']),f"账户等权平均的负余额观测占比分组；n={e['balance_n']:,}。不同账户时点可能不同，不解释为用户精确透支天数。",
        [("历史曾出现负余额",fp(e['ever_negative']),f"最近快照为负仅 {fp(m['bal_neg'])}；两个口径反映不同时间尺度"),
         ("非负快照仍有负余额历史",f"{e['hidden_n']:,} 人",f"历史负余额占比 ≥10%；占非负快照且历史有效的 {e['positive_snapshot_n']:,} 人中 {fp(e['hidden_share_positive'])}"),
         ("余额可覆盖一月固定支出的占比",fp(e['fixedcover_p50']),f"原始预计算占比字段中位；有效 n={e['fixedcover_n']:,}")],
        [(fp(e['balance_rows'][0]['unresolved']),"未见负余额组未解决拒付率"),(fp(e['balance_rows'][-1]['unresolved']),"负余额占比 >25% 组未解决拒付率"),(f"{e['balance_accounts_p50']:.0f} 个","余额覆盖账户数中位"),(f"{e['balance_obs_p50']:.0f} 条","余额观测数中位 · 非自然日数")]),"02 · 负债、流动性与风险")

    slides["gamble"]=slide("消费行为：看净流出，并控制多头暴露","1.6 · Gambling Cashflow", "赌博支出不等于净资金流失；补入同类贷记，并按还款机构数分层查看与未解决拒付的关系。",
        chart_body(gamble_chart(mod,e),f"单元格为未解决拒付用户占比及人数；工资+福利>0且赌博借贷记均可观测 n={e['gamble_n']:,}；n<30 不显示比例。",
        [("赌博借记 → 带符号净流出",f"{fm(e['gamble_debit_total'],compact=True)} → {fm(e['gamble_net_total'],compact=True)}",f"同样本贷记 {fm(e['gamble_credit_total'],compact=True)}；借贷记均有效 n={e['gamble_matched_n']:,}"),
         ("正净流出超过工资+福利 5%",fp(e['gamble_gt5']),f"{e['gamble_gt5_n']:,}/{e['gamble_n']:,} 人；净流入在负担指标中记 0"),
         ("关联需结合借贷关系解释","分层后再判断","同一机构数组内，净流出负担与拒付并非一致单调上升；不能直接归因或设拒绝阈值")],
        [("借记−贷记","带符号净流出 · 不等于博彩损失"),("同类标签","借贷记仅作净额，不重复加总其他入账"),(fm(m['gamb_p99']),"6个月赌博累计借记 P99"),("同期观察","不能推断行为导致拒付")]),"01 · 收入与支出")

    slides["audit"]=slide("变量审查：补充关系，保留边界","Methodology · 变量利用与限制", "完整字段清单按家族、时间窗口与用途归档；扩展只选能解释差异的变量。",
        evidence_table(["审查对象","处理","原因 / 业务用途"],[
            [f"{e.get('all_variables',13363):,} 个完整字段","逐项归档","以 CSV 表头为完整变量清单；项目未附独立业务释义字典"],
            [f"原报告 {e.get('baseline_columns',172)} 个加载字段",f"补入 {e['new_columns']} 个字段","补足标签口径、多头、近期压力、历史余额与消费净流出；复用原有工资、福利、还款与拒付变量"],
            ["7/14/56/84/168 天重复窗口","不逐个展示","仅增 28 天构造近期与历史互斥窗口，避免指标堆砌"],
            ["固定账单 / 房租 / 日常生活 / 高承诺","不机械相加","存在类目交集；缺少底层交易映射，不能证明互斥"],
            ["due date 标签、技术统计量与全量竞品","不直接纳入结论","缺少标签定义、成熟期或明确业务解释；仅有机构使用不等于存在未偿贷款"],
            ["-999999.99999 与 lender -1","按原模块清洗","后者仅代表该模块未识别出记录；无已识别机构不等于真实无负债"],
            ["机构识别的覆盖边界",f"{e['jgs_zero_with_repay']:,} 人有还款但无已识别机构","交易标签与机构识别存在覆盖差异，机构数为可识别关系的下限"],
        ],note=f"可复核文件：{e['dataset_key']}_variable_audit.csv（完整清单）、{e['dataset_key']}_enrichment_metrics.json（指标与核验）、{e['dataset_key']}_cross_analysis.csv（各组样本量与比例）。").replace('class="table-box fundo-evidence"','class="table-box fundo-evidence fundo-audit"')
        +ui.note_strip("结论适用范围",f"有效样本 {e['n']:,} 人，样本月份 {m['sample_month']}；按交易数最多申请去重，不是全体客户的随机样本。金额沿用原报告的 $ 记号，币种需由业务确认。阈值为描述性切分；同期拒付既是比较指标，也与机构识别存在共同覆盖，不作为模型性能。"))
    return slides


def revised_findings(mod,m,seg):
    e=m['profile_enrichment']; fp=mod.fmt_pct; fm=mod.fmt_money
    return [
        f"<b>收支缺口广泛，但要先区分两套口径。</b> 现金流体系偿债前负差额占 {fp(m['pre_debt_neg'])}，含新增借款的净流量为负占 {fp(m['sur_neg'])}。来源分项之和等于标签收入，不能与现金流收入直接互换；标签收支比 {e['tag_ratio_p50']:.3f} 不能解释现金流体系的缺口。",
        f"<b>工资覆盖面与入账金额结构回答不同问题。</b> {fp(m['wage_incidence'])} 有工资，工资仅占标签入账金额 {fp(m['wage_share_agg'])}；转账占来源分项金额 {fp(m['transfer_inc_share'])}。转账用途与持续性未核实，不能直接视为稳定收入。",
        f"<b>工资与福利口径更能暴露偿债负担差异。</b> 在同一批 {e['recurring_n']:,} 名可计算用户中，偿债/现金流收入中位 {fp(e['same_total_p50'])}，偿债/(工资+福利)中位 {fp(e['recurring_p50'])}；{fp(e['burden_gt30'])} 超过 30%。该口径仅为收入来源代理。",
        f"<b>无工资福利入账，也可能仍承担还款。</b> {e['recurring_absent_n']:,} 人未识别工资与福利，其中 {e['recurring_absent_repay_n']:,} 人有还款。这部分不能强行计算工资福利负担比，应复核其他收入的来源与持续性。",
        f"<b>多头还款关系与未解决拒付呈明显梯度。</b> ≥5 家组占 {fp(e['multi5'])}，未解决拒付率 {fp(e['lender_rows'][-1]['unresolved'])}，1 家组为 {fp(e['lender_rows'][1]['unresolved'])}；近 28 天仍向 ≥3 家机构还款占 {fp(e['recent_multi3'])}。机构数不是在贷合同数。",
        f"<b>近期双重压力提供资金变化信息。</b> {e['double_n']:,} 人最近 28 天收入日均较此前 154 天下滑 >20%，且还款日均上升 >20%；未解决拒付率 {fp(e['double_rate'])}，其余可比用户 {fp(e['other_comparable_rate'])}。两者接近，不能据此认为拒付风险明显更高。",
        f"<b>总体趋势中位数不能证明借新还旧。</b> 在历史基数为正的各自样本中，最近 28 天还款日均比中位 {1+e['recent_repayment_growth']:.2f}、新增借款日均比中位 {1+e['recent_disbursement_growth']:.2f}；二者样本不同，需逐户核验资金用途。",
        f"<b>最近余额会掩盖历史资金压力。</b> {fp(e['ever_negative'])} 的用户历史观测曾出现负余额；非负快照且历史有效的 {e['positive_snapshot_n']:,} 人中，{e['hidden_n']:,} 人的账户平均负余额占比仍 ≥10%（{fp(e['hidden_share_positive'])}）。",
        f"<b>刚性负担应按可观测范围解释。</b> 固定账单覆盖 {fp(m['fixed_bill_incidence'])}、房租覆盖 {fp(m['rent_incidence'])}，固定账单/工资中位 {fp(m['fixed_wage_p50'])}。固定账单、房租与高承诺类目可能交叠，不能相加后再次扣减。",
        f"<b>赌博交易要看净额与共同借贷暴露。</b> 同一有效样本赌博借记合计 {fm(e['gamble_debit_total'],compact=True)}，抵减同类贷记后带符号净流出 {fm(e['gamble_net_total'],compact=True)}。按机构数分层后关联并非一致单调上升，不能直接视为独立风险因素。",
        f"<b>拒付是观察到的履约信号。</b> {fp(m['dish_unres'])} 有未解决拒付，未解决金额合计 {fm(m['dish_unres_amt_total'],compact=True)}；按事件数计算解决率 {fp(m['dish_resolution_rate'])}。拒付与最终逾期/违约不是同一标签。",
        f"<b>策略优先级应来自交叉关系。</b> 优先复核近期双重压力、多头还款与持续负余额的叠加，再核验收入来源和资金用途；阈值用于描述分组，未经结果标签、时间外样本和成本收益验证，不转为审批规则。",
    ]


def export_evidence(root, header, base_columns, d, m):
    out=root/'output'/'data'; out.mkdir(parents=True,exist_ok=True)
    e=m['profile_enrichment']; e['baseline_columns']=len(base_columns)
    key=e['dataset_key']
    e.update(source=f"{key}_sample_with_feature_all.csv",sample_month=m["sample_month"],applications=m["n_apps"],deduplicated_users=m["n_users"],all_variables=len(header),loaded_variables=len(set(base_columns)|set(EXTRA_COLUMNS)),definition_status="项目未提供独立释义字典；字段名解释与数值恒等式验证并用",dedupe="同一用户保留交易笔数最多的申请",recent_window="最新28天与此前154天；日均比；历史基数须为正")
    # Read every header, but do not confuse window variants with new dimensions.
    audit=[]
    for name in header:
        window=re.search(r'_(l?\d+d|\d+[wm]_vs_\d+[wm])$',name)
        family='_'.join(name.split('_')[:3]) if name.startswith('bank_txn_') else 'metadata'
        if name in EXTRA_COLUMNS:
            status='本次新增'; reason='补足口径核验或交叉分析'
        elif name in base_columns:
            status='原报告已加载'; reason='沿用；加载不保证每一字段都有独立图表'
        elif name.startswith('duedate_'):
            status='未使用'; reason='结果标签未定义，缺少成熟期，避免误解释与信息泄漏'
        elif window and window.group(1) not in ['182d','l182d']:
            status='未使用'; reason='同一统计的其他窗口；仅保留有业务目的的近期窗口'
        elif any(k in name for k in ['pca_','kgram','kmeans','ridge','hurst','theilsen','ewma','ar1_','poly2','entropy','gini','autocorr']):
            status='未使用'; reason='技术统计或高阶变换；缺少直观增量业务解释'
        elif 'competitor_' in name:
            status='未使用'; reason='机构细项；用总体机构数衡量多头，避免长尾竞品堆砌'
        elif family in ['bank_txn_worthiness','bank_txn_category']:
            status='未使用'; reason='类目金额/次数/比率及层级细分，与既有分析重叠或释义待确认'
        else:
            status='未使用'; reason='候选统计/别名；无证据支持单独展示的增量价值'
        series=d[name] if name in d and pd.api.types.is_numeric_dtype(d[name]) else None
        logical_feature=re.sub(r'_(?:l?\d+d|\d+[wm]_vs_\d+[wm])$','',name)
        audit.append(dict(variable=name,family=family,logical_feature=logical_feature,window=window.group(1) if window else '未标注',status=status,reason=reason,valid_n=int(series.notna().sum()) if series is not None else '',p50=med(series) if series is not None else ''))
    pd.DataFrame(audit).to_csv(out/f'{key}_variable_audit.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for analysis_key in ['lender_rows','burden_rows','balance_rows','gamble_rows','gamble_cells','recent_rows','identity_checks']:
        rows.extend(dict(analysis=analysis_key,**r) for r in e[analysis_key])
    pd.DataFrame(rows).to_csv(out/f'{key}_cross_analysis.csv',index=False,encoding='utf-8-sig')
    def native(obj):
        if isinstance(obj,dict): return {str(k):native(v) for k,v in obj.items()}
        if isinstance(obj,list): return [native(v) for v in obj]
        if isinstance(obj,(np.integer,)): return int(obj)
        if isinstance(obj,(float,np.floating)): return float(obj) if np.isfinite(obj) else None
        return obj
    (out/f'{key}_enrichment_metrics.json').write_text(json.dumps(native(e),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    seg_cols=['segment','inc_m','exp_m','sur_m','bal','rep_rate']
    # Export aggregates only. No identifiers or row-level financial data.
    d.groupby('segment',observed=True)[seg_cols[1:]].median().to_csv(out/f'{key}_segment_summary.csv',encoding='utf-8-sig')
