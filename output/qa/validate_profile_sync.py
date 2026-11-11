"""Check lineage and cross-report agreement in the generated artifacts."""
from pathlib import Path
import hashlib
import json
import math
import pandas as pd
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'output/data'


def main():
    metrics={}
    for key,fields,loaded in [('fundo',13363,188),('wagego',13422,187)]:
        e=json.loads((DATA/f'{key}_enrichment_metrics.json').read_text(encoding='utf-8'))
        metrics[key]=e
        assert e['dataset_key']==key and e['loaded_variables']==loaded
        audit=pd.read_csv(DATA/f'{key}_variable_audit.csv')
        assert len(audit)==fields and audit.variable.is_unique
        assert (audit.status=='本次新增').sum()==16
        assert all(c['n']==c['pass_n'] for c in e['identity_checks'])
        assert sum(r['n'] for r in e['lender_rows'])==e['n']
        assert sum(r['n'] for r in e['burden_rows'])==e['recurring_n']
        assert sum(r['n'] for r in e['gamble_cells'])==e['gamble_n']
        assert e['double_n']<=e['recent_comparable_n']<=e['n']
    pair=json.loads((DATA/'fundo_wagego_comparison_checks.json').read_text(encoding='utf-8'))
    for name,key in [('Fundo','fundo'),('WageGo','wagego')]:
        e=metrics[key]; c=pair['all'][name]
        assert c['n']==e['n']
        assert math.isclose(c['burden'],e['recurring_p50'])
        assert math.isclose(c['multi5'],e['multi5'])
        assert math.isclose(c['negative_history'],e['ever_negative'])
        assert math.isclose(c['gambling5'],e['gamble_gt5'])
        assert sum(pair['month_counts'][name].values())+pair['missing_months'][name]==e['n']
        assert pair['common'][name]['n']<=e['n']
    cohort=pd.read_csv(DATA/'fundo_wagego_cohort_comparison.csv')
    assert len(cohort)==24 and (cohort.valid_n<=cohort.cohort_n).all()
    for key,filename,n_slides,n_charts in [
        ('fundo','fundo_profile_report_slides.html',25,15),
        ('wagego','wagego_profile_report_slides.html',25,15),
        ('compare','compare_profile_report_slides.html',26,19),
    ]:
        source=ROOT/'output/html'/filename
        doc=BeautifulSoup(source.read_text(encoding='utf-8'),'html.parser')
        assert len(doc.select('.slide'))==n_slides
        assert len(doc.select('.plotly-graph-div'))==n_charts
        assert '单笔 P99' not in doc.get_text(' ',strip=True)
        layout=json.loads((ROOT/f'output/qa/{key}_sync_layout_results.json').read_text(encoding='utf-8'))
        assert layout['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        assert not layout['errors'] and len(layout['slides'])==n_slides
        assert all(not s['crops'] and not s['escaped'] and (not s['body'] or s['body']['scroll']<=s['body']['height']+2) for s in layout['slides'])
        assert sum(len(s['plots']) for s in layout['slides'])==n_charts
        assert all(p['w']>100 and p['h']>100 and p['traces']>0 for s in layout['slides'] for p in s['plots'])
        print('PASS',key,n_slides,'slides',n_charts,'charts')
    comparison=BeautifulSoup((ROOT/'output/html/compare_profile_report_slides.html').read_text(encoding='utf-8'),'html.parser')
    seg=next(s for s in comparison.select('.slide') if s.get('aria-label')=='重压型拒付最高，盈余型并非最低')
    assert '拒付用户占比' in seg.get_text(' ',strip=True)
    assert '不是按交易笔数' in seg.get_text(' ',strip=True)
    print('PASS shared formulas, full/cohort denominators, output lineage and renderer checks.')


if __name__=='__main__':
    main()
