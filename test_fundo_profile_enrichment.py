"""Checks for financial denominator, window and missing-value edge cases."""
import unittest

import numpy as np
import pandas as pd

import fundo_profile_enrichment as extra


class FundoEnrichmentTests(unittest.TestCase):
    def test_ratios_require_positive_observed_denominator(self):
        got=extra.ratio(pd.Series([6.,6.,6.,6.,np.inf]),pd.Series([3.,0.,-3.,np.nan,2.]))
        self.assertEqual(got.iloc[0],2.)
        self.assertTrue(got.iloc[1:].isna().all())

    def sample(self):
        d=pd.DataFrame(0.,index=range(4),columns=extra.EXTRA_COLUMNS)
        d['inc']=100.; d['exp']=80.; d['rep']=[10.,5.,20.,10.]
        d['draw']=0.; d['sur']=d.inc-d.exp-d.rep
        d['bal']=[10.,1.,10.,-5.]
        d['rep_rate']=d.rep/d.inc
        d['bank_txn_income_wages_sum_182d']=[60.,0.,80.,60.]
        d['bank_txn_income_centrelink_sum_182d']=[40.,0.,20.,40.]
        d['bank_txn_income_otherincome_sum_182d']=[20.,100.,10.,0.]
        d[extra.W+'income_tag_amt_sum_182d']=[120.,100.,110.,100.]
        d[extra.W+'expense_tag_amt_sum_182d']=80.
        d[extra.W+'income_to_expense_ratio_182d']=d[extra.W+'income_tag_amt_sum_182d']/80
        d[extra.L+'repay_jgs_l182d']=[1.,0.,5.,np.nan]
        d['bank_txn_dishonour_lender_unresolved_cnt_3gp_182d']=[0.,0.,1.,np.nan]
        d[extra.W+'balance_negative_share_avg_acct_182d']=[0.,.1,.5,1.]
        d[extra.G+'debit_amt_182d']=[20.,10.,0.,np.nan]
        d[extra.G+'credit_amt_182d']=[5.,10.,5.,0.]
        d[extra.S+'income_sum_182d']=d.inc
        d[extra.S+'expense_sum_182d']=d.exp
        d[extra.S+'liabilityrepayment_sum_182d']=d.rep
        d[extra.S+'liabilitydrawdown_sum_182d']=d.draw
        d[extra.S+'surplus_sum_182d']=d.sur
        d[extra.S+'dishonoursfee_sum_182d']=0.
        d[extra.S+'income_sum_28d']=[10.,0.,28.,10.]
        d[extra.S+'liabilityrepayment_sum_28d']=[4.,0.,4.,0.]
        return d

    def test_nonoverlapping_windows_and_zero_income_are_explicit(self):
        d=self.sample(); e=extra.analyze(d,{})
        # Exactly one user: income 10/28 versus 90/154, repayment 4/28 versus 6/154.
        self.assertEqual(e['double_n'],1)
        self.assertAlmostEqual(d.loc[0,'e_recent_income'],(10/28)/(90/154))
        self.assertEqual(e['recurring_n'],3)
        self.assertEqual(e['recurring_absent_repay_n'],1)
        self.assertTrue(pd.isna(d.loc[1,'e_recurring_burden']))
        self.assertEqual(e['hidden_n'],2)

    def test_net_gambling_and_missing_outcomes_are_not_zero_imputed(self):
        d=self.sample(); e=extra.analyze(d,{})
        self.assertEqual(e['gamble_net_total'],10.)  # 15 + 0 - 5; missing debit excluded.
        self.assertEqual(e['gamble_n'],2)
        self.assertEqual(e['gamble_gt5_n'],1)
        self.assertEqual(sum(r['outcome_n'] for r in e['lender_rows']),3)
        self.assertEqual(sum(r['n'] for r in e['lender_rows']),4)

    def test_industry_coverage_and_pay_cycle_are_explicit(self):
        d=self.sample()
        for col in extra.INDUSTRY_COLUMNS:
            d[col]=np.nan
        d[extra.INDUSTRY_COLUMNS[0]]=[2.,1.,1.,np.nan]
        d['bank_txn_industry_primary_industry_code']=[12.,12.,17.,np.nan]
        d['bank_txn_industry_pattern_weekly']=[1.,0.,0.,0.]
        d['bank_txn_industry_pattern_fortnightly']=[0.,1.,0.,0.]
        d['bank_txn_industry_pattern_monthly']=0.
        d['bank_txn_industry_pattern_irregular']=0.
        m={}
        e=extra.analyze(d,m)['industry']
        self.assertEqual(e['employer_n'],3)
        self.assertEqual(e['known_n'],2)
        self.assertEqual(e['unknown_n'],1)
        self.assertEqual(e['no_employer_n'],1)
        self.assertEqual(e['industry_rows'][0]['industry'],'零售')
        self.assertEqual(e['top_industry'],'零售')
        self.assertAlmostEqual(e['top_industry_share'],2/3)
        self.assertFalse(e['industry_concentrated'])
        cycles={r['cycle']:r for r in e['pay_cycle_rows']}
        self.assertAlmostEqual(cycles['weekly']['share'],1/3)
        self.assertAlmostEqual(cycles['fortnightly']['share'],1/3)
        self.assertAlmostEqual(cycles['unidentified']['share'],1/3)
        self.assertIs(m['profile_enrichment']['industry'],e)

    def test_concentrated_industry_is_flagged(self):
        d=self.sample()
        for col in extra.INDUSTRY_COLUMNS:
            d[col]=np.nan
        d[extra.INDUSTRY_COLUMNS[0]]=[1.,1.,1.,1.]
        d['bank_txn_industry_primary_industry_code']=[12.,12.,12.,12.]
        e=extra.analyze(d,{})['industry']
        self.assertTrue(e['industry_concentrated'])
        self.assertEqual(e['top_industry'],'零售')
        self.assertEqual(e['top_industry_share'],1.)

    def test_inconsistent_window_or_cashflow_fails_loudly(self):
        d=self.sample(); d.loc[0,extra.S+'income_sum_28d']=101.
        with self.assertRaisesRegex(ValueError,'28天金额大于182天金额'):
            extra.analyze(d,{})
        d=self.sample(); d.loc[0,'sur']+=1
        with self.assertRaisesRegex(ValueError,'盈余恒等式'):
            extra.analyze(d,{})


if __name__=='__main__':
    unittest.main()
