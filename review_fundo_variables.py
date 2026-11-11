"""Reproduce the aggregate review of canonical Fundo feature candidates.

Reads the source CSV directly and exports field statistics, never row-level data.
Usage: python review_fundo_variables.py
"""
from pathlib import Path
import csv
import re

import numpy as np
import pandas as pd

import fundo_profile_report as base


def main():
    root=Path(__file__).resolve().parent
    source=root/'data/fundo_sample_with_feature_all.csv'
    with source.open(encoding='utf-8-sig',newline='') as f:
        header=next(csv.reader(f))
    selected=[c for c in header if c in base.ALL_COLUMNS or (
        c.startswith('bank_txn_') and (
            c.endswith(('_182d','_l182d')) or
            (not re.search(r'_(?:l?\d+d|\d+[wm]_vs_\d+[wm])$',c) and
             not c.startswith(('bank_txn_lender_','bank_txn_dishonour_')))
        )) or c in ['transaction_date_max','balance_date_max']]
    recent={f'bank_txn_surplus_{k}_sum_{w}d'
            for k in ['income','expense','surplus','liabilityrepayment',
                      'liabilitydrawdown','dishonoursfee','refundorreversal']
            for w in [28,84]}
    recent.update(f'bank_txn_lender_{k}_jgs_l28d' for k in ['disburse','repay'])
    selected=list(dict.fromkeys(selected+[c for c in header if c in recent]))
    print(f'Reading {len(selected):,} / {len(header):,} fields...',flush=True)
    raw=pd.read_csv(source,usecols=selected,low_memory=False)
    raw,_=base.clean_frame(raw)
    df=base.dedupe_users(raw)
    core=['bank_txn_surplus_income_sum_182d',
          'bank_txn_surplus_expense_sum_182d',
          'bank_txn_surplus_surplus_sum_182d']
    valid=(df.txn_raw_input_cnt>0)&df[core].notna().all(axis=1)&(df[core[0]]>0)
    records=[]
    for name in selected:
        if not name.startswith('bank_txn_'):
            continue
        values=df.loc[valid,name].dropna()
        records.append(dict(variable=name,current=name in base.ALL_COLUMNS,
                            n=len(values),coverage=len(values)/int(valid.sum()),
                            positive=float((values>0).mean()) if len(values) else np.nan,
                            negative=float((values<0).mean()) if len(values) else np.nan,
                            p10=values.quantile(.1),p50=values.quantile(.5),
                            p90=values.quantile(.9),max=values.max()))
    target=root/'output/data/fundo_candidate_profile.csv'
    target.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(target,index=False,encoding='utf-8-sig')
    print(f'{len(records):,} field profiles; {int(valid.sum()):,} valid users -> {target}')


if __name__=='__main__':
    main()
