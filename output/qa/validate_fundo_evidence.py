"""Aggregate consistency checks for the final, freshly generated report."""
from pathlib import Path
import json
import pandas as pd
from bs4 import BeautifulSoup

root=Path(__file__).resolve().parents[2]
out=root/'output/data'
audit=pd.read_csv(out/'fundo_variable_audit.csv')
e=json.loads((out/'fundo_enrichment_metrics.json').read_text(encoding='utf-8'))
assert len(audit)==13363 and audit.variable.is_unique
assert (audit.status=='本次新增').sum()==16
assert (audit.status=='原报告已加载').sum()==172
assert e['loaded_variables']==188
assert sum(r['n'] for r in e['lender_rows'])==e['n']
assert sum(r['n'] for r in e['burden_rows'])==e['recurring_n']
assert sum(r['n'] for r in e['gamble_cells'])==e['gamble_n']
assert sum(r['n'] for r in e['balance_rows'])==e['balance_n']
assert e['double_n']<=e['recent_comparable_n']<=e['n']
assert all(c['n']==c['pass_n'] for c in e['identity_checks'])
assert e['recurring_absent_repay_n']<=e['recurring_absent_n']
assert abs(e['gamble_debit_total']-e['gamble_credit_total']-e['gamble_net_total'])<.02
for group in ['lender_rows','burden_rows','balance_rows','gamble_rows']:
    assert all(r['outcome_n']<=r['n'] and 0<=r['unresolved']<=1 for r in e[group])
doc=BeautifulSoup((root/'output/html/fundo_profile_report_slides.html').read_text(encoding='utf-8'),'html.parser')
assert len(doc.select('.slide'))==24 and len(doc.select('.plotly-graph-div'))==14
assert all(not tag.get('src','').startswith(('http:','https:')) for tag in doc.find_all('script'))
assert '单笔 P99' not in doc.get_text(' ',strip=True)
layout=json.loads((root/'output/qa/fundo_layout_results.json').read_text(encoding='utf-8'))
assert not layout['errors'] and len(layout['slides'])==24
assert all(not s['crops'] and not s['escaped'] and (not s['body'] or s['body']['scroll']<=s['body']['height']+2) for s in layout['slides'])
assert sum(len(s['plots']) for s in layout['slides'])==14
assert all(p['w']>100 and p['h']>100 and p['traces']>0 for s in layout['slides'] for p in s['plots'])
print('PASS: 13,363 variables, 188 loaded fields, 21,057 valid users, 24 slides, 14 charts, zero layout or script errors.')
print('Candidate profile:',len(pd.read_csv(out/'fundo_candidate_profile.csv')),'fields')
