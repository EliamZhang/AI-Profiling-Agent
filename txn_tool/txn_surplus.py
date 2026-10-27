
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import LinearRegression
from datetime import timedelta


class SingleApplicationSurplusFeatureEngineer:
    def __init__(self,
                 df: pd.DataFrame,
                 time_windows: List[int] = None):
        """
        初始化特征工程类
        """

        self.time_windows = sorted(time_windows) if time_windows else [7, 14, 28, 56, 84, 168, 182]
        self.centrelink_list = [
            'Centrelink Pension', 'Disability Pension', 'JobSeeker', 'Youth Allowance',
            'Parenting Payment', 'Carers Benefits', 'National Disability Insurance Scheme', 'Vet Affairs',
            'Family Benefits', 'Child Support', 'Parental Leave Pay',
            'Mobility Allowance'
        ]

        self.features = {}
        self.df = self._map(df)
        self._prepare_data()
        self.raw_df = self.df.copy()

        # 预计算窗口数据索引
        self._precompute_window_indices()

    def _map(self, df: pd.DataFrame, mapping_file: str = None) -> pd.DataFrame:
        """根据交易映射表进行映射"""
        import os
        df = df.copy()

        cols = ['account_type', 'category', 'dr_cr', 'third_party']
        for c in cols:
            if c not in df.columns:
                df[c] = None

        df[cols] = df[cols].astype(str).apply(lambda x: x.str.strip()).replace({'': None, 'nan': None, 'None': None})
        df.fillna({'account_type': 'Unlabeled', 'category': 'Unlabeled', 'dr_cr': 'Unlabeled'}, inplace=True)

        if mapping_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mapping_file = os.path.join(current_dir, 'reference', 'finv_level', 'finv交易类别映射矩阵_0303.csv')

        if not os.path.exists(mapping_file):
            print(f"警告: 映射文件 {mapping_file} 不存在")
            return df

        file_ext = os.path.splitext(mapping_file)[1].lower()

        try:
            mapping_df = (
                pd.read_csv(mapping_file) if file_ext == '.csv'
                else pd.read_excel(mapping_file) if file_ext in ['.xlsx', '.xls']
                else None
            )
            if mapping_df is None:
                print(f"警告: 不支持的文件格式 {file_ext}")
                return df
        except Exception as e:
            print(f"读取映射文件失败: {e}")
            return df

        need_cols = ['dr_cr', 'category', 'account_type', 'tag_level1', 'tag_level2']
        missing_cols = [c for c in need_cols if c not in mapping_df.columns]
        if missing_cols:
            print(f"警告: 映射文件缺少字段: {missing_cols}")
            return df

        df = df.merge(mapping_df[need_cols], on=['dr_cr', 'category', 'account_type'], how='left')
        return df

    def _prepare_data(self):
        """预处理数据"""

        if 'sample_datetime' in self.df.columns:
            self.df["sample_datetime"] = pd.to_datetime(self.df["sample_datetime"])
            self.sample_datetime = self.df["sample_datetime"].iloc[0]
        else:
            self.sample_datetime = None

        if 'user_id' in self.df.columns:
            self.user_id = self.df['user_id'].iloc[0]
        else:
            self.user_id = None

        # ===== 调整代码：补充 bank_account_id 兜底，确保多卡场景下可先按卡计算再聚合 =====
        if 'bank_account_id' not in self.df.columns:
            self.df['bank_account_id'] = '__MISSING_ACCOUNT__'
        self.df['bank_account_id'] = (
            self.df['bank_account_id']
            .astype(str)
            .replace({'': '__MISSING_ACCOUNT__', 'nan': '__MISSING_ACCOUNT__', 'None': '__MISSING_ACCOUNT__'})
            .fillna('__MISSING_ACCOUNT__')
        )

        if 'transaction_date' in self.df.columns:
            self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
            self.df['month_num'] = self.df['transaction_date'].dt.year * 12 + self.df['transaction_date'].dt.month

        if 'transaction_date' in self.df.columns and 'sample_datetime' in self.df.columns:
            self.df["trac_days"] = (self.df["sample_datetime"].dt.floor('D') - self.df["transaction_date"]).dt.days

        if 'amount' in self.df.columns:
            self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce').abs().fillna(0.0)

        if 'trac_days' in self.df.columns:
            self.df = self.df[self.df['trac_days'] >= 0].copy()
            self.df.sort_values(by=["bank_account_id", "trac_days"], ascending=[True, True], inplace=True)
            self.df.reset_index(drop=True, inplace=True)

        for c in ['tag_level1', 'tag_level2', 'category', 'third_party']:
            if c not in self.df.columns:
                self.df[c] = np.nan

    def _precompute_window_indices(self):
        """预计算每个窗口的数据索引"""
        if 'trac_days' not in self.df.columns:
            self.window_masks = {}
            return

        self.window_masks = {}
        for window in self.time_windows:
            self.window_masks[window] = self.df['trac_days'] <= window

    def _clean_tp_name(self, tp: str) -> str:
        """清洗交易对手名称，用于特征键名"""
        if not isinstance(tp, str):
            tp = str(tp)

        import re
        cleaned = tp.lower().replace(' ', '').replace('/', '').replace('-', '')
        cleaned = cleaned.replace('.', '').replace('&', '').replace('@', '')
        cleaned = cleaned.replace('$', '').replace('%', '').replace('_', '')
        cleaned = re.sub(r'[^\w]', '', cleaned)

        return cleaned

    # ===== 调整代码：新增多卡辅助函数，统一做“先按卡算、再聚合” =====
    def _iter_account_dfs(self):
        if self.df.empty:
            return []
        return [(acc_id, g.copy()) for acc_id, g in self.df.groupby('bank_account_id', dropna=False, sort=False)]

    def _apply_income_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['income_factor'] = 0.0

        wage_mask = (df['tag_level2'] == 'Wages') & (df['category'] == 'Wages')
        centrelink_mask = (df['tag_level2'] == 'Centrelink') & (df['third_party'].isin(self.centrelink_list))
        otherincome_mask = (df['tag_level2'] == 'Other Income') & (df['category'] != 'Internal Transfer')

        df.loc[wage_mask | centrelink_mask, 'income_factor'] = 1.0
        df.loc[otherincome_mask, 'income_factor'] = 0.8
        return df

    def _sum_amount_over_windows(self, df: pd.DataFrame, value_col: str) -> Dict[int, float]:
        out = {}
        if df.empty:
            for window in self.time_windows:
                out[window] = 0.0
            return out

        for window in self.time_windows:
            mask = df['trac_days'] <= window
            out[window] = float(df.loc[mask, value_col].sum()) if mask.any() else 0.0
        return out

    def _calc_fortnightly_by_account(self, account_df: pd.DataFrame, window: int) -> Dict[str, float]:
        """
        多卡口径：
        每张卡各自估算周期强度，再在 application 维度求和。
        这样不会因为 A 卡很新、B 卡很老，混在一起把 span 拉长。
        """
        window_df = account_df[account_df['trac_days'] <= window].copy()

        if window_df.empty:
            return {
                'income': 0.0,
                'expense': 0.0,
                'liabilityrepayment': 0.0,
                'liabilitydrawdown': 0.0,
                'dishonoursfee': 0.0,
                'refundorreversal': 0.0,
                'surplus': 0.0,
            }

        window_df = self._apply_income_factor(window_df)
        window_df['cal_income'] = window_df['amount'] * window_df['income_factor']

        categories = {
            'income': (window_df['tag_level1'] == 'INCOME'),
            'expense': (window_df['tag_level1'] == 'EXPENSE') & (window_df['category'] != 'Internal Transfer'),
            'liabilityrepayment': (window_df['tag_level1'] == 'LIABILITY') & (window_df['tag_level2'] == 'Liability_Repayment'),
            'liabilitydrawdown': (window_df['tag_level1'] == 'LIABILITY') & (window_df['tag_level2'] == 'Liability_Drawdown'),
            'dishonoursfee': (window_df['tag_level1'] == 'LIABILITY') & (window_df['tag_level2'] == 'Dishonours Fee'),
            'refundorreversal': (window_df['tag_level1'] == 'LIABILITY') & (window_df['tag_level2'] == 'Refund_Or_Reversal'),
        }

        raw_values = {}
        raw_values['income'] = float(window_df.loc[categories['income'], 'cal_income'].sum())
        raw_values['expense'] = float(window_df.loc[categories['expense'], 'amount'].sum())
        raw_values['liabilityrepayment'] = float(window_df.loc[categories['liabilityrepayment'], 'amount'].sum())
        raw_values['liabilitydrawdown'] = float(window_df.loc[categories['liabilitydrawdown'], 'amount'].sum())
        raw_values['dishonoursfee'] = float(window_df.loc[categories['dishonoursfee'], 'amount'].sum())
        raw_values['refundorreversal'] = float(window_df.loc[categories['refundorreversal'], 'amount'].sum())

        out = {}
        for cat, mask in categories.items():
            cat_df = window_df[mask]
            if cat_df.empty:
                out[cat] = 0.0
            else:
                first_day = float(cat_df['trac_days'].min())
                last_day = float(cat_df['trac_days'].max())
                day_span = max(last_day - first_day, 14.0)
                factor = 14.0 / day_span
                out[cat] = raw_values[cat] * factor

        out['surplus'] = (
            out['income']
            - out['expense']
            - out['liabilityrepayment']
            - out['dishonoursfee']
            + out['liabilitydrawdown']
        )
        return out

    def cal_income(self) -> Dict:
        """收入识别规则"""
        features = {}

        df = self.df.copy()
        df = df[df['tag_level1'] == 'INCOME'].copy()
        df = self._apply_income_factor(df)
        df['cal_income'] = df['amount'] * df['income_factor']

        # ===== 调整代码：按卡汇总后再求 application 总和，数值与单纯合并求和一致，但口径更清晰 =====
        features_by_window = {window: 0.0 for window in self.time_windows}
        for _, acc_df in df.groupby('bank_account_id', dropna=False, sort=False):
            acc_sums = self._sum_amount_over_windows(acc_df, 'cal_income')
            for window in self.time_windows:
                features_by_window[window] += acc_sums[window]

        for window in self.time_windows:
            features[f'bank_txn_surplus_income_sum_{window}d'] = float(features_by_window[window])

        self.features.update(features)
        return features

    def cal_expense(self) -> Dict:
        """支出识别规则"""
        features = {}
        df = self.df.copy()
        df = df[(df['tag_level1'] == 'EXPENSE') & (df['category'] != 'Internal Transfer')].copy()

        features_by_window = {window: 0.0 for window in self.time_windows}
        for _, acc_df in df.groupby('bank_account_id', dropna=False, sort=False):
            acc_sums = self._sum_amount_over_windows(acc_df, 'amount')
            for window in self.time_windows:
                features_by_window[window] += acc_sums[window]

        for window in self.time_windows:
            features[f'bank_txn_surplus_expense_sum_{window}d'] = float(features_by_window[window])

        self.features.update(features)
        return features

    def cal_liability_repayment(self) -> Dict:
        """计算负债还款总额"""
        features = {}
        df = self.df.copy()
        df = df[(df['tag_level1'] == 'LIABILITY') & (df['tag_level2'] == 'Liability_Repayment')].copy()

        features_by_window = {window: 0.0 for window in self.time_windows}
        for _, acc_df in df.groupby('bank_account_id', dropna=False, sort=False):
            acc_sums = self._sum_amount_over_windows(acc_df, 'amount')
            for window in self.time_windows:
                features_by_window[window] += acc_sums[window]

        for window in self.time_windows:
            features[f'bank_txn_surplus_liabilityrepayment_sum_{window}d'] = float(features_by_window[window])

        self.features.update(features)
        return features

    def cal_dishonours_fee(self) -> Dict:
        """计算拒付费总额"""
        features = {}
        df = self.df.copy()
        df = df[(df['tag_level1'] == 'LIABILITY') & (df['tag_level2'] == 'Dishonours Fee')].copy()

        features_by_window = {window: 0.0 for window in self.time_windows}
        for _, acc_df in df.groupby('bank_account_id', dropna=False, sort=False):
            acc_sums = self._sum_amount_over_windows(acc_df, 'amount')
            for window in self.time_windows:
                features_by_window[window] += acc_sums[window]

        for window in self.time_windows:
            features[f'bank_txn_surplus_dishonoursfee_sum_{window}d'] = float(features_by_window[window])

        self.features.update(features)
        return features

    def cal_liability_drawdown(self) -> Dict:
        """计算负债提取总额"""
        features = {}
        df = self.df.copy()
        df = df[(df['tag_level1'] == 'LIABILITY') & (df['tag_level2'] == 'Liability_Drawdown')].copy()

        features_by_window = {window: 0.0 for window in self.time_windows}
        for _, acc_df in df.groupby('bank_account_id', dropna=False, sort=False):
            acc_sums = self._sum_amount_over_windows(acc_df, 'amount')
            for window in self.time_windows:
                features_by_window[window] += acc_sums[window]

        for window in self.time_windows:
            features[f'bank_txn_surplus_liabilitydrawdown_sum_{window}d'] = float(features_by_window[window])

        self.features.update(features)
        return features

    def cal_refund_or_reversal(self) -> Dict:
        """计算refund总额"""
        features = {}
        df = self.df.copy()
        df = df[(df['tag_level1'] == 'LIABILITY') & (df['tag_level2'] == 'Refund_Or_Reversal')].copy()

        features_by_window = {window: 0.0 for window in self.time_windows}
        for _, acc_df in df.groupby('bank_account_id', dropna=False, sort=False):
            acc_sums = self._sum_amount_over_windows(acc_df, 'amount')
            for window in self.time_windows:
                features_by_window[window] += acc_sums[window]

        for window in self.time_windows:
            features[f'bank_txn_surplus_refundorreversal_sum_{window}d'] = float(features_by_window[window])

        self.features.update(features)
        return features

    def cal_surplus(self) -> Dict:
        """盈余识别规则"""
        features = {}

        if not any(k.startswith('bank_txn_surplus_income_sum_') for k in self.features):
            self.cal_income()
        if not any(k.startswith('bank_txn_surplus_expense_sum_') for k in self.features):
            self.cal_expense()
        if not any(k.startswith('bank_txn_surplus_liabilityrepayment_sum_') for k in self.features):
            self.cal_liability_repayment()
        if not any(k.startswith('bank_txn_surplus_dishonoursfee_sum_') for k in self.features):
            self.cal_dishonours_fee()
        if not any(k.startswith('bank_txn_surplus_liabilitydrawdown_sum_') for k in self.features):
            self.cal_liability_drawdown()
        if not any(k.startswith('bank_txn_surplus_refundorreversal_sum_') for k in self.features):
            self.cal_refund_or_reversal()

        for window in self.time_windows:
            income = self.features.get(f'bank_txn_surplus_income_sum_{window}d', 0)
            expense = self.features.get(f'bank_txn_surplus_expense_sum_{window}d', 0)
            liability_repayment = self.features.get(f'bank_txn_surplus_liabilityrepayment_sum_{window}d', 0)
            dishonours_fee = self.features.get(f'bank_txn_surplus_dishonoursfee_sum_{window}d', 0)
            liability_drawdown = self.features.get(f'bank_txn_surplus_liabilitydrawdown_sum_{window}d', 0)

            if pd.isna(income) or pd.isna(expense) or pd.isna(liability_repayment) or pd.isna(dishonours_fee) or pd.isna(liability_drawdown):
                features[f'bank_txn_surplus_surplus_sum_{window}d'] = 0.0
            else:
                features[f'bank_txn_surplus_surplus_sum_{window}d'] = float(
                    income - expense - liability_repayment - dishonours_fee + liability_drawdown
                )

        self.features.update(features)
        return features

    def cal_fortnightly_metrics(self) -> Dict:
        """计算双周化口径的财务指标"""
        features = {}

        if not any(k.startswith('bank_txn_surplus_income_sum_') for k in self.features):
            self.cal_income()
        if not any(k.startswith('bank_txn_surplus_expense_sum_') for k in self.features):
            self.cal_expense()
        if not any(k.startswith('bank_txn_surplus_liabilityrepayment_sum_') for k in self.features):
            self.cal_liability_repayment()
        if not any(k.startswith('bank_txn_surplus_dishonoursfee_sum_') for k in self.features):
            self.cal_dishonours_fee()
        if not any(k.startswith('bank_txn_surplus_liabilitydrawdown_sum_') for k in self.features):
            self.cal_liability_drawdown()
        if not any(k.startswith('bank_txn_surplus_refundorreversal_sum_') for k in self.features):
            self.cal_refund_or_reversal()

        # ===== 调整代码：双周化改为“每张卡独立估 span，再汇总”，避免多卡混合导致频率被拉歪 =====
        for window in self.time_windows:
            agg_vals = {
                'income': 0.0,
                'expense': 0.0,
                'liabilityrepayment': 0.0,
                'liabilitydrawdown': 0.0,
                'dishonoursfee': 0.0,
                'refundorreversal': 0.0,
                'surplus': 0.0,
            }

            for _, acc_df in self._iter_account_dfs():
                acc_vals = self._calc_fortnightly_by_account(acc_df, window)
                for k in agg_vals:
                    agg_vals[k] += acc_vals[k]

            features[f'bank_txn_surplus_fortnightly_income_{window}d'] = float(agg_vals['income'])
            features[f'bank_txn_surplus_fortnightly_expense_{window}d'] = float(agg_vals['expense'])
            features[f'bank_txn_surplus_fortnightly_liabilityrepayment_{window}d'] = float(agg_vals['liabilityrepayment'])
            features[f'bank_txn_surplus_fortnightly_liabilitydrawdown_{window}d'] = float(agg_vals['liabilitydrawdown'])
            features[f'bank_txn_surplus_fortnightly_dishonoursfee_{window}d'] = float(agg_vals['dishonoursfee'])
            features[f'bank_txn_surplus_fortnightly_refundorreversal_{window}d'] = float(agg_vals['refundorreversal'])
            features[f'bank_txn_surplus_fortnightly_surplus_{window}d'] = float(agg_vals['surplus'])

        self.features.update(features)
        return features

    def amount_comparison(self, windows: List[int] = [7, 30, 90, 180]) -> Dict:
        """计算收入、支出、负债、盈余的折算后趋势特征"""
        features = {}

        if not hasattr(self, 'df') or self.df is None:
            return features

        comparisons = [
            (7, 30, '1w_vs_1m'),
            (30, 90, '1m_vs_3m'),
            (30, 180, '1m_vs_6m'),
            (90, 180, '3m_vs_6m')
        ]

        all_windows = sorted(set([w for pair in comparisons for w in pair[:2]]))

        # ===== 调整代码：按卡先算窗口值，再在 application 层汇总；最终变量名和数量不变 =====
        window_data = {
            w: {
                'income': 0.0,
                'expense': 0.0,
                'surplus': 0.0,
                'liabilityrepayment': 0.0,
                'dishonoursfee': 0.0,
                'liabilitydrawdown': 0.0,
                'refundorreversal': 0.0,
            }
            for w in all_windows
        }

        for _, acc_df in self._iter_account_dfs():
            acc_df = self._apply_income_factor(acc_df.copy())
            acc_df['cal_income'] = acc_df['amount'] * acc_df['income_factor']

            for window in all_windows:
                window_df = acc_df[acc_df['trac_days'] <= window]

                income = float(window_df.loc[window_df['tag_level1'] == 'INCOME', 'cal_income'].sum())
                expense = float(window_df.loc[
                    (window_df['tag_level1'] == 'EXPENSE') & (window_df['category'] != 'Internal Transfer'),
                    'amount'
                ].sum())
                liability_repayment = float(window_df.loc[window_df['tag_level2'] == 'Liability_Repayment', 'amount'].sum())
                dishonours_fee = float(window_df.loc[window_df['tag_level2'] == 'Dishonours Fee', 'amount'].sum())
                liability_drawdown = float(window_df.loc[window_df['tag_level2'] == 'Liability_Drawdown', 'amount'].sum())
                refund_reversal = float(window_df.loc[window_df['tag_level2'] == 'Refund_Or_Reversal', 'amount'].sum())
                surplus = income - expense - liability_repayment - dishonours_fee + liability_drawdown

                window_data[window]['income'] += income
                window_data[window]['expense'] += expense
                window_data[window]['liabilityrepayment'] += liability_repayment
                window_data[window]['dishonoursfee'] += dishonours_fee
                window_data[window]['liabilitydrawdown'] += liability_drawdown
                window_data[window]['refundorreversal'] += refund_reversal
                window_data[window]['surplus'] += surplus

        metrics = ['income', 'expense', 'liabilityrepayment', 'dishonoursfee', 'liabilitydrawdown', 'refundorreversal', 'surplus']

        for metric_name in metrics:
            for w1, w2, comp_name in comparisons:
                feature_key = f'bank_txn_surplus_{metric_name}_{comp_name}'

                val1 = window_data[w1][metric_name] / w1
                val2 = window_data[w2][metric_name] / w2

                features[feature_key] = (val1 / val2) if val2 != 0 else np.nan

        self.features.update(features)
        return features

    def amount_by_type(self) -> Dict:
        """按负债类型的统计特征"""
        features = {}
        df = self.df.copy()
        df = df[df['tag_level1'] == 'LIABILITY'].copy()

        liability_types = ['Liability_Repayment', 'Liability_Drawdown', 'Refund_Or_Reversal', 'Dishonours Fee']
        cleaned_types = {lt: self._clean_tp_name(lt) for lt in liability_types}

        # 这里保留“交易级统计”口径：
        # sum/count/mean/median/max/min/std/cv 都基于 application 下全部相关交易；
        # 该类统计本身就是 transaction-level 分布，直接合并多卡并不会改变量名语义。
        for window in self.time_windows:
            window_data = df[df['trac_days'] <= window]

            if len(window_data) == 0:
                for liability_type in liability_types:
                    cleaned_type = cleaned_types[liability_type]
                    for stat in ['mean', 'median', 'max', 'min', 'std', 'cv']:
                        features[f'bank_txn_surplus_{cleaned_type}_{stat}_{window}d'] = np.nan
                    for stat in ['sum', 'count']:
                        features[f'bank_txn_surplus_{cleaned_type}_{stat}_{window}d'] = 0.0
                continue

            for liability_type in liability_types:
                cleaned_type = cleaned_types[liability_type]
                type_data = window_data[window_data['tag_level2'] == liability_type]

                if len(type_data) > 0:
                    amounts = type_data['amount'].values

                    total_sum = float(np.sum(amounts))
                    total_count = int(len(amounts))
                    total_mean = float(np.mean(amounts))
                    total_median = float(np.median(amounts))
                    total_max = float(np.max(amounts))
                    total_min = float(np.min(amounts))
                    total_std = float(np.std(amounts, ddof=1)) if total_count > 1 else np.nan
                    total_cv = (total_std / total_mean) if total_mean > 0 and not np.isnan(total_std) else np.nan

                    stats = {
                        'sum': total_sum,
                        'count': total_count,
                        'mean': total_mean,
                        'median': total_median,
                        'max': total_max,
                        'min': total_min,
                        'std': total_std,
                        'cv': total_cv
                    }

                    for stat_name, stat_val in stats.items():
                        features[f'bank_txn_surplus_{cleaned_type}_{stat_name}_{window}d'] = stat_val
                else:
                    for stat in ['mean', 'median', 'max', 'min', 'std', 'cv']:
                        features[f'bank_txn_surplus_{cleaned_type}_{stat}_{window}d'] = np.nan
                    for stat in ['sum', 'count']:
                        features[f'bank_txn_surplus_{cleaned_type}_{stat}_{window}d'] = 0.0

        self.features.update(features)
        return features

    def generate_all_features(self, feature_groups: List[str] = None, include_metadata: bool = True) -> Dict:
        """生成所有特征"""

        if feature_groups is None:
            feature_groups = [
                'cal_income', 'cal_expense', 'cal_surplus',
                'cal_fortnightly_metrics', 'amount_comparison', 'amount_by_type'
            ]

        self.features = {}

        for feature in feature_groups:
            method = getattr(self, feature)
            method()

        self.features['user_id'] = self.user_id
        self.features['sample_datetime'] = self.sample_datetime

        df = pd.DataFrame([self.features])
        base_columns = ['user_id', 'sample_datetime']
        existing_base = [col for col in base_columns if col in df.columns]
        feature_cols = [col for col in df.columns if col not in existing_base]

        df = df[existing_base + feature_cols]
        df.columns = df.columns.str.strip().str.replace(' ', '').str.lower()

        return df


# ==========================================================================================
def generate_surplus_features(df: pd.DataFrame = None, feature_groups: List[str] = None, include_metadata: bool = True):
    engineer = SingleApplicationSurplusFeatureEngineer(
        df=df,
        time_windows=[7, 14, 28, 56, 84, 168, 182]
    )
    result_df = engineer.generate_all_features(feature_groups=feature_groups)
    return result_df
