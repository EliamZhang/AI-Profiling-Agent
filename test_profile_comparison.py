"""Cross-report checks: shared calculations, cohort boundaries and denominators."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import compare_profile_enrichment as pair
import compare_profile_report as comparison
import fundo_profile_enrichment as compatibility
import fundo_profile_report as fundo
import profile_enrichment as shared
import wagego_profile_report as wagego
import test_fundo_profile_enrichment as fixtures


def frame(months,ids,burdens,outcomes):
    n=len(ids)
    return pd.DataFrame(dict(user_id=ids,sample_datetime=[m+'-15' for m in months],
        inc_m=np.arange(n,dtype=float)+100,e_recurring_burden=burdens,
        e_unresolved=outcomes,e_gamble_burden=[.1]*n,
        **{shared.L+'repay_jgs_l182d':[5]*n,
           shared.W+'balance_negative_share_avg_acct_182d':[.2]*n}))


class ProfileComparisonTests(unittest.TestCase):
    def test_both_loaders_accept_the_same_extra_fields(self):
        name=shared.EXTRA_COLUMNS[0]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'small.csv'
            pd.DataFrame({'user_id':[1],'txn_raw_input_cnt':[1],name:[25.]}).to_csv(path,index=False)
            for module in [fundo,wagego]:
                with self.subTest(module=module.__name__):
                    d,missing,n=module.load_data(path,log=lambda _:None,extra_columns=[name,'unavailable_extra'])
                    self.assertEqual(d[name].iloc[0],25.)
                    self.assertEqual(n,3)
                    self.assertIn('unavailable_extra',missing)
                    self.assertNotIn(name,missing)

    def test_common_months_restrict_both_populations_and_preserve_full_sample(self):
        f=SimpleNamespace(d=frame(['2026-09']*2,['10','11'],[.1,.2],[0.,1.]),m={})
        w=SimpleNamespace(d=frame(['2026-08','2026-09','2026-09','2026-10'],['11','20','21','22'],[.1,.6,.9,.2],[1.,0.,0.,1.]),m={})
        c=pair.prepare(f,w)
        self.assertEqual(c['common_months'],['2026-09'])
        self.assertEqual(c['all']['WageGo']['n'],4)
        self.assertEqual(c['common']['WageGo']['n'],2)
        self.assertAlmostEqual(c['common']['WageGo']['burden'],.75)
        self.assertEqual(c['common']['WageGo']['unresolved'],0.)
        self.assertEqual(c['all']['WageGo']['unresolved'],.5)
        self.assertEqual(c['user_key_overlap'],1)
        self.assertEqual(len(w.d),4)

    def test_missing_outcomes_and_income_ratios_keep_their_own_denominators(self):
        d=frame(['2026-09']*3,['1','2','3'],[.1,np.nan,.5],[0.,np.nan,1.])
        c=pair.cohort_summary(d)
        self.assertEqual(c['n'],3)
        self.assertEqual(c['burden_n'],2)
        self.assertEqual(c['unresolved_n'],2)
        self.assertEqual(c['unresolved'],.5)
        self.assertAlmostEqual(c['burden'],.3)

    def test_wagego_uses_shared_formulas_without_becoming_fundo(self):
        self.assertIs(compatibility.analyze,shared.analyze)
        d=fixtures.FundoEnrichmentTests().sample(); m={}
        e=shared.analyze(d,m,dataset_key='wagego',dataset_name='WageGo')
        self.assertEqual(e['dataset_key'],'wagego')
        self.assertIs(m['profile_enrichment'],e)
        self.assertNotIn('fundo_enrichment',m)
        p=SimpleNamespace(m=m)
        self.assertEqual(comparison.metric_value(p,'enrich:multi5'),e['multi5'])
        self.assertTrue(np.isnan(comparison.metric_value(p,'enrich:not_present')))

    def test_exports_keep_each_brand_prefix_after_aggregating_analysis_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for key,name in [('fundo','Fundo'),('wagego','WageGo')]:
                d=fixtures.FundoEnrichmentTests().sample()
                d['segment']='sample'
                for name_out,name_in in [('inc_m','inc'),('exp_m','exp'),('sur_m','sur')]:
                    d[name_out]=d[name_in]/6
                m=dict(sample_month='2026-09',n_apps=4,n_users=4)
                shared.analyze(d,m,dataset_key=key,dataset_name=name)
                shared.export_evidence(root,list(d.columns),[],d,m)
                for suffix in ['variable_audit.csv','cross_analysis.csv','enrichment_metrics.json','segment_summary.csv']:
                    self.assertTrue((root/'output/data'/f'{key}_{suffix}').exists())
            self.assertFalse(list((root/'output/data').glob('identity_checks_*')))


if __name__=='__main__':
    unittest.main()
