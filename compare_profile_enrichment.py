"""Aligned Fundo / WageGo comparisons, including cohort sensitivity checks."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import deck_framework as ui
import profile_enrichment as ext
import fundo_profile_report as style

COLORS={"Fundo":"#0E7558","WageGo":"#6B4FA8"}
EXTRA_CSS="""
 .comparison-evidence .deck-table td{padding:14px 13px;font-size:13px;line-height:1.5}
 .comparison-evidence .deck-table td small{font-size:10.5px;color:var(--faint)}
 .comparison-evidence .table-note{font-size:11px}
"""
COHORT_SPECS=[
    ("现金流月收入中位","income","money"),
    ("还款/(工资+福利)中位","burden","pct"),
    ("≥5 家已识别还款机构","multi5","pct"),
    ("未解决拒付用户占比","unresolved","pct"),
    ("历史曾有负余额","negative_history","pct"),
    ("赌博正净流出/工资福利 >5%","gambling5","pct"),
]


def month_keys(d):
    if "sample_month" in d:
        months=d.sample_month.astype("string")
        good=months.str.fullmatch(r"\d{4}-\d{2}",na=False)
        if good.all():
            return months
    return pd.to_datetime(d.sample_datetime,format="mixed",errors="coerce").dt.strftime("%Y-%m")


def cohort_summary(d):
    recurring=d.e_recurring_burden.dropna()
    unresolved=d.e_unresolved.dropna()
    negative=d[ext.W+"balance_negative_share_avg_acct_182d"].dropna()
    gambling=d.e_gamble_burden.dropna()
    return dict(n=len(d),income=ext.med(d.inc_m),income_n=int(d.inc_m.notna().sum()),
                burden=ext.med(recurring),burden_n=len(recurring),
                multi5=float(d[ext.L+"repay_jgs_l182d"].ge(5).mean()),multi5_n=len(d),
                unresolved=ext.rate(unresolved),unresolved_n=len(unresolved),
                negative_history=ext.rate(negative.gt(0)),negative_history_n=len(negative),
                gambling5=ext.rate(gambling.gt(.05)),gambling5_n=len(gambling))


def prepare(pf,pw):
    if "pair_enrichment" in pf.m:
        return pf.m["pair_enrichment"]
    fm,wm=month_keys(pf.d),month_keys(pw.d)
    common=sorted(set(fm.dropna())&set(wm.dropna()))
    fs=pf.d.loc[fm.isin(common)]; ws=pw.d.loc[wm.isin(common)]
    # Report key overlap; matching keys alone do not establish identity namespaces.
    fkeys=set(pf.d.user_id.dropna().astype(str).str.strip())
    wkeys=set(pw.d.user_id.dropna().astype(str).str.strip())
    c=dict(common_months=common,
           month_counts={"Fundo":{str(k):int(v) for k,v in fm.value_counts().sort_index().items()},
                         "WageGo":{str(k):int(v) for k,v in wm.value_counts().sort_index().items()}},
           missing_months={"Fundo":int(fm.isna().sum()),"WageGo":int(wm.isna().sum())},
           user_key_overlap=len(fkeys&wkeys),overlap_scope="两组有效用户的原始 user_id 键；命名空间是否一致未核实",
           all={"Fundo":cohort_summary(pf.d),"WageGo":cohort_summary(pw.d)},
           common={"Fundo":cohort_summary(fs),"WageGo":cohort_summary(ws)},
           cohorts="各品牌先按原规则独立去重和筛选，再限制共同月份；不重选申请，不做加权")
    pf.m["pair_enrichment"]=c
    return c


def export(root,pf,pw):
    c=prepare(pf,pw)
    rows=[]
    for period in ["all","common"]:
        for name in COLORS:
            summary=c[period][name]
            for label,key,kind in COHORT_SPECS:
                rows.append(dict(cohort=period,population=name,metric=label,key=key,
                                 value=summary[key],valid_n=summary[key+"_n"],
                                 cohort_n=summary["n"],format=kind))
    target=Path(root)/"output/data"
    pd.DataFrame(rows).to_csv(target/"fundo_wagego_cohort_comparison.csv",index=False,encoding="utf-8-sig")
    def clean(x):
        if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list): return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
        if isinstance(x,np.integer): return int(x)
        return x
    (target/"fundo_wagego_comparison_checks.json").write_text(json.dumps(clean(c),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def rate_chart(pf,pw,key):
    fig=go.Figure()
    for name,p in [("Fundo",pf),("WageGo",pw)]:
        rows=p.m["profile_enrichment"][key]
        fig.add_trace(go.Bar(y=[r["group"] for r in rows],x=[r["unresolved"] for r in rows],
            name=name,orientation="h",marker_color=COLORS[name],
            text=[f"{r['unresolved']:.1%} · n={r['outcome_n']:,}" for r in rows],textposition="outside",cliponaxis=False,
            error_x=dict(type="data",symmetric=False,array=[r["high"]-r["unresolved"] for r in rows],arrayminus=[r["unresolved"]-r["low"] for r in rows],thickness=1,color="#66788A"),
            hovertemplate=name+" · %{y}<br>%{text}<extra></extra>"))
    style.base_layout(fig,margin=dict(l=170,r=155,t=25,b=75),showlegend=True)
    fig.update_layout(barmode="group",bargap=.23,legend=dict(orientation="h",x=0,y=-.13))
    fig.update_xaxes(range=[0,1],tickformat=".0%",title="未解决拒付用户占比")
    fig.update_yaxes(autorange="reversed",showgrid=False)
    return style.fig_html(fig)


def gambling_chart(pf,pw):
    fig=make_subplots(rows=1,cols=2,shared_yaxes=True,horizontal_spacing=.07,subplot_titles=["Fundo","WageGo"])
    for col,p in enumerate([pf,pw],1):
        cells=p.m["profile_enrichment"]["gamble_cells"]
        z=[]; texts=[]
        for group in ext.GROUPS:
            row=[r for r in cells if r['lenders']==group]
            z.append([r['unresolved'] if r['n']>=30 else None for r in row])
            texts.append([f"{r['unresolved']:.1%}<br>n={r['n']:,}" if r['n']>=30 else f"n={r['n']}<br>样本不足" for r in row])
        fig.add_trace(go.Heatmap(z=z,x=ext.GAMBLE_GROUPS,y=ext.GROUPS,text=texts,texttemplate="%{text}",textfont=dict(size=12),zmin=0,zmax=1,colorscale=[[0,"#EDF4FB"],[.5,"#8AAECC"],[1,"#1D4E79"]],showscale=False,xgap=3,ygap=3,hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>"),row=1,col=col)
    style.base_layout(fig,margin=dict(l=130,r=25,t=45,b=50))
    fig.update_yaxes(autorange="reversed",showgrid=False)
    fig.update_xaxes(showgrid=False)
    return style.fig_html(fig)


def employment_chart(pf,pw):
    fi=pf.m["profile_enrichment"]["industry"]; wi=pw.m["profile_enrichment"]["industry"]
    labels=[r["label"] for r in fi["pay_cycle_rows"]]
    fig=make_subplots(rows=1,cols=2,shared_yaxes=True,horizontal_spacing=.14,
                      subplot_titles=["发薪周期构成","偿债负担中位"],column_widths=[.5,.5])
    for name,ind in [("Fundo",fi),("WageGo",wi)]:
        rows=ind["pay_cycle_rows"]
        fig.add_trace(go.Bar(y=labels,x=[r["share"] for r in rows],name=name,orientation="h",
            marker_color=COLORS[name],text=[f"{r['share']:.1%}" for r in rows],
            textposition="outside",cliponaxis=False,textfont=dict(size=10.5),
            hovertemplate=name+" · %{y}<br>占比 %{x:.1%}<br>n=%{customdata:,}<extra></extra>",
            customdata=[r["n"] for r in rows]),row=1,col=1)
        fig.add_trace(go.Bar(y=labels,x=[r["burden_p50"] for r in rows],name=name,orientation="h",
            marker_color=COLORS[name],text=[f"{r['burden_p50']:.1%}" if np.isfinite(r["burden_p50"]) else "—" for r in rows],
            textposition="outside",cliponaxis=False,textfont=dict(size=10.5),
            hovertemplate=name+" · %{y}<br>负担中位 %{x:.1%}<br>n=%{customdata[0]:,}<br>"
                          "≥5 家机构 %{customdata[1]:.1%}<extra></extra>",
            customdata=[[r["n"],r["multi5"]] for r in rows]),row=1,col=2)
    style.base_layout(fig,height=430,margin=dict(l=82,r=112,t=46,b=72),showlegend=True)
    fig.update_layout(barmode="group",bargap=.24,legend=dict(orientation="h",x=0,y=-.17))
    fig.update_xaxes(range=[0,.64],tickformat=".0%",title_text="占可识别雇主用户",row=1,col=1)
    fig.update_xaxes(range=[0,.64],tickformat=".0%",title_text="还款 / (工资+福利) 中位",row=1,col=2)
    fig.update_yaxes(autorange="reversed",showgrid=False)
    return style.fig_html(fig,height=430)


def slides(pair,pf,pw):
    c=prepare(pf,pw); f=pf.m["profile_enrichment"]; w=pw.m["profile_enrichment"]
    fi=f["industry"]; wi=w["industry"]
    fp=style.fmt_pct; fm=style.fmt_money; brand=pair["brand"]
    common="、".join(c["common_months"]) or "无"
    def pair_text(a,b,kind="pct"):
        fmt=fp if kind=="pct" else fm if kind=="money" else lambda x:f"{x:,}"
        return f"{fmt(a)}<small> / </small>{fmt(b)}"
    def slide(title,sub,body,section="补充 · 同口径交叉分析",kicker="Aligned Comparison"):
        body=body.replace('class="table-box"','class="table-box comparison-evidence"')
        return ui.slide_content(brand=brand,section=section,kicker=kicker,title=title,sub=sub,body=body)
    def duo(chart,caption,cards,facts):
        return '<div class="duo">'+ui.chart_box(chart,caption)+ui.side_cards(cards)+"</div>"+ui.fact_strip(facts)
    rows=[]
    for label,key,kind in COHORT_SPECS:
        fmt=fm if kind=="money" else fp
        values=[]
        for period,name in [("all","Fundo"),("all","WageGo"),("common","Fundo"),("common","WageGo")]:
            r=c[period][name]
            values.append(f"{fmt(r[key])}<br><small>n={r[key+'_n']:,}</small>")
        rows.append([label]+values)
    out={}
    out["scope"]=slide("先校准口径，再比较品牌差异",f"共同月份为 {common}；保留全期结果，并补充共同月份的敏感性检查。",
        ui.table_box(["指标","Fundo · 全期","WageGo · 全期","Fundo · 共同月份","WageGo · 共同月份"],rows,note="每格 n 为该指标有效人数。工资福利比要求工资+福利>0；机构数为识别关系；历史余额为账户平均观测占比。共同月份比较未做其他特征匹配。",num_cols=(1,2,3,4))
        +ui.note_strip("两项必要边界",f"现金流收入与标签收入分别核验，不混用收支比。两组有效样本中 user_id 键重合 {c['user_key_overlap']:,} 个；身份命名空间未核实，不预设样本独立。原始申请月份分布不同，以上仍为描述性对比。"))
    fc={r["cycle"]:r for r in fi["pay_cycle_rows"]}; wc={r["cycle"]:r for r in wi["pay_cycle_rows"]}
    out["employment"]=slide("就业形态：覆盖率与发薪周期", "先比较雇主识别覆盖，再在各自可识别雇主用户内比较发薪周期构成与偿债负担；行业码不直接对比。",
        duo(employment_chart(pf,pw),"发薪周期为互斥标记；负担为还款/(工资+福利)中位；n 为各周期可计算人数。",
            [("可识别雇主",pair_text(fi['employer_share'],wi['employer_share']),f"人数 {fi['employer_n']:,} / {wi['employer_n']:,}；单一雇主 {fp(fi['single_employer_share'])} / {fp(wi['single_employer_share'])}"),
             ("周薪占比",pair_text(fc['weekly']['share'],wc['weekly']['share']),f"双周薪 {fp(fc['fortnightly']['share'])} / {fp(wc['fortnightly']['share'])}；分母为各自可识别雇主用户"),
             ("月薪偿债负担中位",pair_text(fc['monthly']['burden_p50'],wc['monthly']['burden_p50']),f"月薪 n={fc['monthly']['n']:,} / {wc['monthly']['n']:,}；周薪 {fp(fc['weekly']['burden_p50'])} / {fp(wc['weekly']['burden_p50'])}")],
            [(pair_text(fi['top_industry_share'],wi['top_industry_share']),"最大行业码占比"),(fi['top_industry'],"Fundo 最大行业码"),(wi['top_industry'],"WageGo 最大行业码"),("不可直接比较","行业码覆盖与集中度不同")])
        +ui.note_strip("口径与边界",f"最大行业码占比以各自可识别雇主用户为分母；WageGo 行业码 {fp(wi['top_industry_share'])} 集中于{wi['top_industry']}，行业结构不具备与 Fundo 对比的分层意义；Fundo 的行业码还混合银行代发渠道。发薪周期为描述性分层，月薪组样本较小；未解决拒付未控制收入与客群差异，不能作为独立风险因子或审批规则。"),section="01 · 样本与收入",kicker="1.4 · Employment Structure")
    out["capacity"]=slide("收入分母统一后，偿债负担如何变化", "每个品牌均在同一批可计算用户内更换收入分母；随后对齐负担分组比较未解决拒付。",
        duo(rate_chart(pf,pw,"burden_rows"),"分组依据：还款/(工资+福利)。标签仅为收入代理；横线为 Wilson 95% 区间。",
            [("Fundo · 同用户更换分母",f"{fp(f['same_total_p50'])} → {fp(f['recurring_p50'])}",f"现金流收入 → 工资+福利；n={f['recurring_n']:,}"),
             ("WageGo · 同用户更换分母",f"{fp(w['same_total_p50'])} → {fp(w['recurring_p50'])}",f"现金流收入 → 工资+福利；n={w['recurring_n']:,}"),
             ("负担 >30% · Fundo / WageGo",pair_text(f['burden_gt30'],w['burden_gt30']),"分母为各自工资福利为正且还款可观测用户")],
            [(pair_text(f['recurring_absent_n'],w['recurring_absent_n'],'int'),"未识别工资福利人数"),(pair_text(f['recurring_absent_repay_n'],w['recurring_absent_repay_n'],'int'),"其中有还款的人数"),("同口径分组","每根柱标出有效人数"),("代理指标","不能直接称为可支配收入")]))
    out["multi"]=slide("多头还款：区分关系数量与组内表现", "同时对比多头关系的覆盖率，以及同样机构数组内的未解决拒付率，避免只比较总体拒付。",
        duo(rate_chart(pf,pw,"lender_rows"),"182 天识别出还款交易的机构数；不等于当前在贷机构数。无已识别机构不等于真实无债。",
            [("≥5 家还款机构 · Fundo / WageGo",pair_text(f['multi5'],w['multi5']),f"人数 {f['multi5_n']:,} / {w['multi5_n']:,}"),
             ("近28天仍向 ≥3 家机构还款",pair_text(f['recent_multi3'],w['recent_multi3']),"以各自全部有效用户为分母"),
             ("有还款但未识别机构",pair_text(f['jgs_zero_with_repay'],w['jgs_zero_with_repay'],'int'),"交易标签与机构识别存在覆盖差异")],
            [(pair_text(f['lender_rows'][-1]['unresolved'],w['lender_rows'][-1]['unresolved']),"≥5 家组未解决拒付率"),(pair_text(f['lender_rows'][1]['unresolved'],w['lender_rows'][1]['unresolved']),"1 家组未解决拒付率"),("同期关联","不直接认定机构数导致拒付"),("键与识别口径","不能据此计算未偿本金")]))
    recent=[]
    for fr,wr in zip(f['recent_rows'][:4],w['recent_rows'][:4]):
        recent.append([fr['metric'],f"{fr['paired_ratio_p50']:.3f}",f"{fr['paired_n']:,}",f"{wr['paired_ratio_p50']:.3f}",f"{wr['paired_n']:,}"])
    cards=[]
    for label,key in [("双重压力 / 全部用户","double_all_share"),("双重压力 / 可比用户","double_comparable_share"),("双重压力组 · 未解决拒付","double_rate"),("其余可比组 · 未解决拒付","other_comparable_rate")]:
        cards.append(dict(label=label,left_name="Fundo",right_name="WageGo",left_value=fp(f[key]),right_value=fp(w[key]),left_color=COLORS['Fundo'],right_color=COLORS['WageGo'],delta="同品牌组内比较，保留分母差异"))
    out["recent"]=slide("近期变化：用互斥窗口检验资金压力", "最近 28 天与此前 154 天分别换算日均；只在历史基数为正的用户中计算逐户比值。",
        ui.table_box(["指标","Fundo · 日均比中位","Fundo · n","WageGo · 日均比中位","WageGo · n"],recent,note="日均比=(28天金额/28)÷[(182天金额−28天金额)/154]；每项独立筛选有效基数。比值>1代表上升，不能以两个总体中位数判断借新还旧。",num_cols=(1,2,3,4))
        +ui.cmp_kpi_cards(cards)+ui.note_strip("双重压力定义",f"收入日均下降>20%且还款日均上升>20%：Fundo {f['double_n']:,} 人 / 可比 {f['recent_comparable_n']:,} 人，WageGo {w['double_n']:,} 人 / 可比 {w['recent_comparable_n']:,} 人。与其余可比用户的同期拒付率并未形成一致的更高梯度，不解释为未来风险预测。"))
    out["balance"]=slide("流动性：快照之外的历史压力", "使用相同的账户平均负余额占比分组；将曾经负余额与最近快照分开比较。",
        duo(rate_chart(pf,pw,"balance_rows"),"账户等权平均的负余额观测占比；不是用户精确透支天数。每根柱标出拒付标签有效人数。",
            [("历史曾有负余额 · Fundo / WageGo",pair_text(f['ever_negative'],w['ever_negative']),"分母为历史占比字段可观测用户"),
             ("快照非负仍存在历史压力",pair_text(f['hidden_share_positive'],w['hidden_share_positive']),f"历史负余额占比≥10%；人数 {f['hidden_n']:,}/{f['positive_snapshot_n']:,} 与 {w['hidden_n']:,}/{w['positive_snapshot_n']:,}"),
             ("覆盖一月固定支出的观测占比",pair_text(f['fixedcover_p50'],w['fixedcover_p50']),f"预计算字段中位；n={f['fixedcover_n']:,} / {w['fixedcover_n']:,}")],
            [(pair_text(f['balance_rows'][0]['unresolved'],w['balance_rows'][0]['unresolved']),"未见负余额组未解决拒付率"),(pair_text(f['balance_rows'][-1]['unresolved'],w['balance_rows'][-1]['unresolved']),"负余额占比>25%组未解决拒付率"),("账户平均","不等同于全账户合并余额"),("静态快照","不能代表持续可用资金")]))
    out["gambling"]=slide("行为净流出：在同样借贷关系下比较", "两侧采用同一 0–100% 色阶；各格为未解决拒付率和人数，按机构数控制共同借贷暴露。",
        ui.chart_box(gambling_chart(pf,pw),f"正净流出=max(赌博借记−同类贷记,0)，再除以工资+福利；分母为正且借贷记可观测 n={f['gamble_n']:,} / {w['gamble_n']:,}。n<30不显示比例。净额不是博彩损失；不能据此作因果或独立风险判断。")
        +ui.fact_strip([(pair_text(f['gamble_gt5'],w['gamble_gt5']),"正净流出/工资福利>5%"),(f"{f['gamble_n']:,} / {w['gamble_n']:,}","行为负担比有效用户"),("同机构数组","仍可能存在收入与金额等混杂"),("借记与贷记净额","不与其他入账重复相加")]))
    return out


def findings(pf,pw):
    c=prepare(pf,pw); f=pf.m['profile_enrichment']; w=pw.m['profile_enrichment']
    fm=style.fmt_money; fp=style.fmt_pct
    cf,cw=c['common']['Fundo'],c['common']['WageGo']
    common='、'.join(c['common_months']) or '无'
    return [
        f"<b>两套体系分别比较，避免混用收支口径。</b> 现金流收入/支出中位比 Fundo {f['surplus_ratio_p50']:.3f}、WageGo {w['surplus_ratio_p50']:.3f}；标签体系为 {f['tag_ratio_p50']:.3f}、{w['tag_ratio_p50']:.3f}。两组各自通过金额恒等式核验，不能把标签比率解释为现金流缺口。",
        f"<b>工资与福利口径揭示被总入账稀释的还款负担。</b> 同品牌同一批用户，Fundo 偿债/现金流收入 {fp(f['same_total_p50'])}→偿债/(工资+福利) {fp(f['recurring_p50'])}；WageGo {fp(w['same_total_p50'])}→{fp(w['recurring_p50'])}。两种分母的比值分别计算中位数。",
        f"<b>高负担覆盖率应在可计算人群内比较。</b> 偿债/(工资+福利)>30%：Fundo {fp(f['burden_gt30'])}、WageGo {fp(w['burden_gt30'])}；有效人数 {f['recurring_n']:,} 与 {w['recurring_n']:,}。没有识别工资福利不代表无收入，也不强行计算分母。",
        f"<b>多头覆盖与同组表现是两类差异。</b> ≥5 家还款机构覆盖 Fundo {fp(f['multi5'])}、WageGo {fp(w['multi5'])}；该组未解决拒付率 {fp(f['lender_rows'][-1]['unresolved'])}、{fp(w['lender_rows'][-1]['unresolved'])}。机构数仅代表观测期可识别交易关系。",
        f"<b>快照为正不能排除历史压力。</b> 最近快照非负且历史有效的用户中，负余额历史占比仍≥10%的比例 Fundo {fp(f['hidden_share_positive'])}、WageGo {fp(w['hidden_share_positive'])}；人数 {f['hidden_n']:,} 与 {w['hidden_n']:,}，分母分别为 {f['positive_snapshot_n']:,} 与 {w['positive_snapshot_n']:,}。",
        f"<b>双重压力反映近期资金变化。</b> 收入日均下降>20%且还款上升>20%：Fundo {f['double_n']:,} 人、WageGo {w['double_n']:,} 人；该组未解决拒付率 {fp(f['double_rate'])}、{fp(w['double_rate'])}，各自其余可比组 {fp(f['other_comparable_rate'])}、{fp(w['other_comparable_rate'])}，不能称为明显更高风险。",
        f"<b>赌博借记和净流出不能等同。</b> 正净流出/工资福利>5%占比 Fundo {fp(f['gamble_gt5'])}、WageGo {fp(w['gamble_gt5'])}。扣除同类贷记后再按机构数分层；同组内并非一致单调上升，不把行为直接解释为拒付原因。",
        f"<b>月份构成需要单独检验。</b> 共同月份 {common}，有效人数 Fundo {cf['n']:,}、WageGo {cw['n']:,}；共同月份的还款/工资福利中位 {fp(cf['burden'])}、{fp(cw['burden'])}，≥5 家机构覆盖 {fp(cf['multi5'])}、{fp(cw['multi5'])}。月份对齐仍未控制其他客群差异。",
        f"<b>不预设两组用户独立。</b> 有效样本中的 user_id 键重合 {c['user_key_overlap']:,} 个；身份命名空间尚未核实，不据键值重合直接认定为同一自然人，也不据两个品牌名称认定独立样本。",
        f"<b>相对工资的账单负担不是实际可支配收入。</b> 固定账单/工资中位 Fundo {fp(pf.m['fixed_wage_p50'])}、WageGo {fp(pw.m['fixed_wage_p50'])}。固定账单、房租与高承诺支出可能交叠；工资分母差异也会影响比率，不重复扣减。",
        f"<b>含借款净流量分层不等于信用等级。</b> 负净流量用户 Fundo {fp(pf.m['sur_neg'])}、WageGo {fp(pw.m['sur_neg'])}；新增借款进入盈余公式，分层受融资流入影响，不能直接把分层结构称为信用质量更好或更差。",
        f"<b>更正统计对象后再使用行为信号。</b> collections_inc 衡量催收类支出覆盖，不是催收拒付；分层表的拒付占比是有拒付记录的用户占比，不是拒付笔数占比。所有差异为同期描述，未经审批成本收益和时间外验证。",
    ]
