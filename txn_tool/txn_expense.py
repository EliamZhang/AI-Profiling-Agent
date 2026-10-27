import os
import re
from functools import lru_cache
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


class SingleApplicationExpenseFeatureEngineer:
    def __init__(
        self,
        df: pd.DataFrame,
        time_windows: List[int] = None,
        expense_type_tp_pairs: List[List] = None,
        tag_level1: str = "EXPENSE"
    ):
        """
        初始化特征工程类
        """
        self.original_df = df.copy()

        # 交易类别映射表
        self.df = self._map(df)

        if self.df is None:
            self.df = pd.DataFrame()

        if 'transaction_date' in self.df.columns and 'date' not in self.df.columns:
            self.df.rename(columns={'transaction_date': 'date'}, inplace=True)

        self.raw_df = self.df.copy()

        self.time_windows = sorted(time_windows) if time_windows else [7, 14, 28, 56, 84, 168, 182]

        # 支出类型和交易对手
        self.expense_type_tp_pairs = expense_type_tp_pairs if expense_type_tp_pairs is not None else []
        self.expense_types = [pair[0] for pair in self.expense_type_tp_pairs]
        self.third_parties = list(set(tp for pair in self.expense_type_tp_pairs for tp in pair[1] if tp))

        self.third_party_col = 'third_party'
        self.tag_level1 = tag_level1
        self.features = {}

        # 为了加速：预构建 type -> cleaned tp 映射
        self._cleaned_tp_map = {}
        for expense_type, tps in self.expense_type_tp_pairs:
            self._cleaned_tp_map[expense_type] = {tp: self._clean_tp_name(tp) for tp in tps}

        self._prepare_data()

    def __call__(
        self,
        df: pd.DataFrame,
        feature_groups: List[str] = None,
        include_metadata: bool = True
    ) -> pd.DataFrame:
        return self.generate_all_features(
            df=df,
            feature_groups=feature_groups,
            include_metadata=include_metadata
        )

    # ==================== 映射文件缓存 ====================

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_mapping(mapping_file: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(mapping_file):
            print(f"警告: 映射文件 {mapping_file} 不存在")
            return None

        file_ext = os.path.splitext(mapping_file)[1].lower()
        try:
            if file_ext == '.csv':
                mapping_df = pd.read_csv(mapping_file)
            elif file_ext in ['.xlsx', '.xls']:
                mapping_df = pd.read_excel(mapping_file)
            else:
                print(f"警告: 不支持的文件格式 {file_ext}")
                return None
        except Exception as e:
            print(f"读取映射文件失败: {e}")
            return None

        required_cols = ['dr_cr', 'category', 'account_type', 'tag_level1', 'tag_level2']
        missing_cols = [c for c in required_cols if c not in mapping_df.columns]
        if missing_cols:
            print(f"警告: 映射文件缺少必要列: {missing_cols}")
            return None

        return mapping_df[required_cols].copy()

    def _map(
        self,
        df: pd.DataFrame,
        mapping_file: str = None
    ) -> pd.DataFrame:
        """根据交易映射表进行映射"""
        df = df.copy()

        cols = ['account_type', 'category', 'dr_cr', 'third_party']
        exist_cols = [c for c in cols if c in df.columns]
        if exist_cols:
            df[exist_cols] = (
                df[exist_cols]
                .astype(str)
                .apply(lambda x: x.str.strip())
                .replace({'': None, 'nan': None, 'None': None})
            )

        fill_dict = {}
        for c in ['account_type', 'category', 'dr_cr']:
            if c in df.columns:
                fill_dict[c] = 'Unlabeled'
        if fill_dict:
            df.fillna(fill_dict, inplace=True)

        if mapping_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mapping_file = os.path.join(current_dir, 'reference', 'finv_level', 'finv交易类别映射矩阵_0303.csv')

        if not os.path.exists(mapping_file):
            print(f"警告: 映射文件 {mapping_file} 不存在")
            return df

        mapping_df = self._load_mapping(mapping_file)
        if mapping_df is None:
            return df

        merge_keys = ['dr_cr', 'category', 'account_type']
        missing_cols = [c for c in merge_keys if c not in df.columns]
        if missing_cols:
            print(f"警告: 输入数据缺少映射所需列: {missing_cols}")
            return df

        df = df.merge(mapping_df, on=merge_keys, how='left')
        return df

    def _clean_tp_name(self, tp: str) -> str:
        """清洗交易对手名称，用于特征键名"""
        if not isinstance(tp, str):
            tp = str(tp)

        cleaned = tp.lower()
        cleaned = cleaned.replace(' ', '')
        cleaned = cleaned.replace('/', '')
        cleaned = cleaned.replace('-', '')
        cleaned = cleaned.replace('.', '')
        cleaned = cleaned.replace('&', '')
        cleaned = cleaned.replace('@', '')
        cleaned = cleaned.replace('$', '')
        cleaned = cleaned.replace('%', '')
        cleaned = re.sub(r'[^\w_]', '', cleaned)
        return cleaned

    @staticmethod
    def _safe_cv(std_val, mean_val):
        if pd.isna(std_val) or pd.isna(mean_val) or mean_val == 0:
            return np.nan
        return std_val / mean_val

    @staticmethod
    def _calc_slope_from_daily(daily_df: pd.DataFrame) -> float:
        """
        用最小二乘直接计算斜率，等价于 LinearRegression().fit(X, y).coef_[0]
        """
        if daily_df is None or len(daily_df) < 2:
            return np.nan

        daily_df = daily_df.sort_values('date', ascending=True).reset_index(drop=True)
        x = (daily_df['date'] - daily_df['date'].min()).dt.days.to_numpy(dtype=float)
        y = daily_df['amount'].to_numpy(dtype=float)

        if len(x) < 2:
            return np.nan

        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return np.nan
        slope = np.sum((x - x_mean) * (y - y_mean)) / denom
        return float(slope)

    @staticmethod
    def _max_consecutive_ints(vals: np.ndarray) -> float:
        if vals is None or len(vals) == 0:
            return np.nan
        vals = np.sort(np.unique(vals))
        if len(vals) == 0:
            return np.nan
        max_len = 1
        cur_len = 1
        for i in range(1, len(vals)):
            if vals[i] - vals[i - 1] == 1:
                cur_len += 1
                max_len = max(max_len, cur_len)
            else:
                cur_len = 1
        return max_len

    @staticmethod
    def _date_to_day_ordinal(dates: pd.Series) -> np.ndarray:
        """
        Convert datetimes to day ordinals without assuming a fixed datetime unit.
        """
        if dates is None or len(dates) == 0:
            return np.array([], dtype=np.int64)
        normalized = pd.to_datetime(dates).dt.normalize()
        return normalized.to_numpy(dtype='datetime64[D]').astype(np.int64)

    # ==================== 【调整】多卡聚合辅助函数 ====================

    @staticmethod
    def _weighted_nanmean(values, weights=None):
        vals = np.asarray(values, dtype=float)
        if vals.size == 0:
            return np.nan

        valid = np.isfinite(vals)
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.size != vals.size:
                w = np.ones_like(vals, dtype=float)
            valid = valid & np.isfinite(w) & (w > 0)
            if not valid.any():
                valid = np.isfinite(vals)
                return float(np.mean(vals[valid])) if valid.any() else np.nan
            return float(np.average(vals[valid], weights=w[valid]))

        return float(np.mean(vals[valid])) if valid.any() else np.nan

    @staticmethod
    def _nanmax_or_nan(values):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.max(vals)) if vals.size > 0 else np.nan

    @staticmethod
    def _nanmin_or_nan(values):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        return float(np.min(vals)) if vals.size > 0 else np.nan

    def _build_window_cache_from_df(self, wdf: pd.DataFrame) -> Dict:
        """【调整】给申请级和单卡级共用的窗口缓存构建函数"""
        cache = {}

        if wdf is None or len(wdf) == 0:
            cache['global_total_amount'] = 0.0
            cache['type_stats'] = pd.DataFrame()
            cache['type_daily'] = pd.DataFrame()
            cache['type_daily_stats'] = pd.DataFrame()
            cache['tp_stats'] = pd.DataFrame()
            cache['tp_daily'] = pd.DataFrame()
            cache['tp_daily_stats'] = pd.DataFrame()
            return cache

        cache['global_total_amount'] = float(wdf['amount'].sum())

        # type 统计
        if 'tag_level2' in wdf.columns:
            type_stats = (
                wdf.groupby('tag_level2')['amount']
                .agg(['sum', 'count', 'mean', 'median', 'max', 'min', 'std'])
                .reset_index()
            )
            type_stats['std'] = type_stats['std'].fillna(np.nan)
            type_stats['cv'] = np.where(
                type_stats['mean'].notna() & (type_stats['mean'] != 0),
                type_stats['std'] / type_stats['mean'],
                np.nan
            )
            cache['type_stats'] = type_stats

            type_daily = (
                wdf.groupby(['tag_level2', 'date'], as_index=False)['amount']
                .sum()
                .sort_values(['tag_level2', 'date'])
            )
            cache['type_daily'] = type_daily

            if len(type_daily) > 0:
                type_daily_stats = (
                    type_daily.groupby('tag_level2')['amount']
                    .agg(daily_max='max', daily_min='min')
                    .reset_index()
                )
            else:
                type_daily_stats = pd.DataFrame()
            cache['type_daily_stats'] = type_daily_stats
        else:
            cache['type_stats'] = pd.DataFrame()
            cache['type_daily'] = pd.DataFrame()
            cache['type_daily_stats'] = pd.DataFrame()

        # type + tp 统计
        if 'tag_level2' in wdf.columns and 'third_party' in wdf.columns:
            tp_stats = (
                wdf.groupby(['tag_level2', 'third_party'])['amount']
                .agg(['sum', 'count', 'mean', 'median', 'max', 'min', 'std'])
                .reset_index()
            )
            tp_stats['std'] = tp_stats['std'].fillna(np.nan)
            tp_stats['cv'] = np.where(
                tp_stats['mean'].notna() & (tp_stats['mean'] != 0),
                tp_stats['std'] / tp_stats['mean'],
                np.nan
            )
            cache['tp_stats'] = tp_stats

            tp_daily = (
                wdf.groupby(['tag_level2', 'third_party', 'date'], as_index=False)['amount']
                .sum()
                .sort_values(['tag_level2', 'third_party', 'date'])
            )
            cache['tp_daily'] = tp_daily

            if len(tp_daily) > 0:
                tp_daily_stats = (
                    tp_daily.groupby(['tag_level2', 'third_party'])['amount']
                    .agg(daily_max='max', daily_min='min')
                    .reset_index()
                )
            else:
                tp_daily_stats = pd.DataFrame()
            cache['tp_daily_stats'] = tp_daily_stats
        else:
            cache['tp_stats'] = pd.DataFrame()
            cache['tp_daily'] = pd.DataFrame()
            cache['tp_daily_stats'] = pd.DataFrame()

        return cache

    def _prepare_data(self):
        """预处理数据 + 构建缓存"""
        df = self.df.copy()
        raw_df = self.raw_df.copy()

        # ==================== 兼容 transaction_date -> date ====================
        if 'date' not in df.columns:
            if 'transaction_date' in df.columns:
                df['date'] = pd.to_datetime(df['transaction_date']).dt.floor('D')
            else:
                raise KeyError("缺少日期列：需要 'date' 或 'transaction_date' 其中之一")
        else:
            df['date'] = pd.to_datetime(df['date']).dt.floor('D')

        if 'date' not in raw_df.columns:
            if 'transaction_date' in raw_df.columns:
                raw_df['date'] = pd.to_datetime(raw_df['transaction_date']).dt.floor('D')
            else:
                raw_df['date'] = df['date']
        else:
            raw_df['date'] = pd.to_datetime(raw_df['date']).dt.floor('D')
        # ================================================================

        if 'sample_datetime' in df.columns:
            df["sample_datetime"] = pd.to_datetime(df["sample_datetime"])
            self.sample_datetime = df["sample_datetime"].iloc[0]
        else:
            self.sample_datetime = None

        if 'user_id' in df.columns:
            self.user_id = df['user_id'].iloc[0]
        else:
            self.user_id = None

        df['month_num'] = df['date'].dt.year * 12 + df['date'].dt.month

        if 'sample_datetime' in df.columns:
            df["trac_days"] = (df["sample_datetime"].dt.floor('D') - df["date"]).dt.days
        else:
            df["trac_days"] = np.nan

        if 'tag_level1' in df.columns:
            df = df[df['tag_level1'] == self.tag_level1].copy()
        else:
            print("警告: 数据中不存在'tag_level1'列，跳过筛选")

        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').abs()

        if 'trac_days' in df.columns:
            df = df[df['trac_days'] >= 0].copy()

        # ==================== 【调整】多卡兜底处理 ====================
        self.account_col = 'bank_account_id'
        if self.account_col not in df.columns:
            df[self.account_col] = '__single_account__'
        df[self.account_col] = df[self.account_col].astype(str).replace({'nan': '__single_account__', 'None': '__single_account__'})
        df[self.account_col] = df[self.account_col].fillna('__single_account__')

        if self.account_col not in raw_df.columns:
            raw_df[self.account_col] = '__single_account__'
        raw_df[self.account_col] = raw_df[self.account_col].astype(str).replace({'nan': '__single_account__', 'None': '__single_account__'})
        raw_df[self.account_col] = raw_df[self.account_col].fillna('__single_account__')
        # ============================================================

        if 'trac_days' in df.columns:
            df = df.sort_values(by=[self.account_col, "trac_days", "date"], ascending=[True, True, True]).reset_index(drop=True)

        self.df = df
        self.raw_df = raw_df

        # ==================== 申请级基础缓存（保留总量/结构视角） ====================
        self._type_data = {}
        for expense_type in self.expense_types:
            if 'tag_level2' in df.columns:
                td = df[df['tag_level2'] == expense_type]
                if len(td) > 0:
                    self._type_data[expense_type] = td

        self._type_tp_data = {}
        self._tp_data = {}

        for expense_type, tps in self.expense_type_tp_pairs:
            self._type_tp_data[expense_type] = {}
            type_data = self._type_data.get(expense_type, pd.DataFrame())

            if not tps:
                continue

            for tp in tps:
                if 'third_party' in type_data.columns:
                    tp_data = type_data[type_data['third_party'] == tp]
                    self._type_tp_data[expense_type][tp] = tp_data
                    if tp not in self._tp_data:
                        self._tp_data[tp] = tp_data.copy()
                    else:
                        self._tp_data[tp] = pd.concat([self._tp_data[tp], tp_data], axis=0)

        self._window_data = {}
        for window in self.time_windows:
            self._window_data[window] = df[df['trac_days'] <= window]

        self._window_cache = {}
        for window, wdf in self._window_data.items():
            self._window_cache[window] = self._build_window_cache_from_df(wdf)

        self._type_daily_map = {}
        self._type_monthly_map = {}
        for expense_type in self.expense_types:
            td = self._type_data.get(expense_type, pd.DataFrame())
            if len(td) == 0:
                continue

            daily = (
                td.groupby('date', as_index=False)['amount']
                .sum()
                .sort_values('date')
                .reset_index(drop=True)
            )
            self._type_daily_map[expense_type] = daily

            monthly = (
                td.groupby(td['date'].dt.to_period('M'))['amount']
                .sum()
                .reset_index()
            )
            monthly.columns = ['year_month', 'amount']
            monthly = monthly[monthly['amount'] > 0].copy()
            if len(monthly) > 0:
                monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
                monthly['month_num'] = monthly['year_month'].dt.year * 12 + monthly['year_month'].dt.month
            self._type_monthly_map[expense_type] = monthly

        self._type_tp_daily_map = {}
        self._type_tp_monthly_map = {}
        self._type_tp_tracday_map = {}

        for expense_type, tps in self.expense_type_tp_pairs:
            self._type_tp_daily_map[expense_type] = {}
            self._type_tp_monthly_map[expense_type] = {}
            self._type_tp_tracday_map[expense_type] = {}

            td = self._type_data.get(expense_type, pd.DataFrame())
            if len(td) == 0 or 'third_party' not in td.columns or not tps:
                continue

            for tp in tps:
                tpd = td[td['third_party'] == tp]
                self._type_tp_tracday_map[expense_type][tp] = tpd

                if len(tpd) == 0:
                    continue

                daily = (
                    tpd.groupby('date', as_index=False)['amount']
                    .sum()
                    .sort_values('date')
                    .reset_index(drop=True)
                )
                self._type_tp_daily_map[expense_type][tp] = daily

                monthly = (
                    tpd.groupby(tpd['date'].dt.to_period('M'))['amount']
                    .sum()
                    .reset_index()
                )
                monthly.columns = ['year_month', 'amount']
                monthly = monthly[monthly['amount'] > 0].copy()
                if len(monthly) > 0:
                    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
                    monthly['month_num'] = monthly['year_month'].dt.year * 12 + monthly['year_month'].dt.month
                self._type_tp_monthly_map[expense_type][tp] = monthly

        # ==================== 【调整】单卡级缓存（用于时序类特征） ====================
        self.account_ids = sorted(df[self.account_col].dropna().astype(str).unique().tolist())
        self._account_dfs = {}
        self._account_type_data = {}
        self._account_window_cache = {}
        self._account_type_daily_map = {}
        self._account_type_monthly_map = {}
        self._account_type_tp_daily_map = {}
        self._account_type_tp_monthly_map = {}

        for acc in self.account_ids:
            adf = df[df[self.account_col] == acc].copy()
            self._account_dfs[acc] = adf
            self._account_type_data[acc] = {}
            self._account_window_cache[acc] = {}
            self._account_type_daily_map[acc] = {}
            self._account_type_monthly_map[acc] = {}
            self._account_type_tp_daily_map[acc] = {}
            self._account_type_tp_monthly_map[acc] = {}

            for expense_type in self.expense_types:
                tdf = adf[adf['tag_level2'] == expense_type].copy() if 'tag_level2' in adf.columns else pd.DataFrame()
                self._account_type_data[acc][expense_type] = tdf

            for window in self.time_windows:
                awdf = adf[adf['trac_days'] <= window].copy()
                self._account_window_cache[acc][window] = self._build_window_cache_from_df(awdf)

            for expense_type in self.expense_types:
                td = self._account_type_data[acc].get(expense_type, pd.DataFrame())

                if len(td) > 0:
                    daily = (
                        td.groupby('date', as_index=False)['amount']
                        .sum()
                        .sort_values('date')
                        .reset_index(drop=True)
                    )
                    self._account_type_daily_map[acc][expense_type] = daily

                    monthly = (
                        td.groupby(td['date'].dt.to_period('M'))['amount']
                        .sum()
                        .reset_index()
                    )
                    monthly.columns = ['year_month', 'amount']
                    monthly = monthly[monthly['amount'] > 0].copy()
                    if len(monthly) > 0:
                        monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
                        monthly['month_num'] = monthly['year_month'].dt.year * 12 + monthly['year_month'].dt.month
                    self._account_type_monthly_map[acc][expense_type] = monthly
                else:
                    self._account_type_daily_map[acc][expense_type] = pd.DataFrame()
                    self._account_type_monthly_map[acc][expense_type] = pd.DataFrame()

                self._account_type_tp_daily_map[acc][expense_type] = {}
                self._account_type_tp_monthly_map[acc][expense_type] = {}

                tps = []
                for et, tp_list in self.expense_type_tp_pairs:
                    if et == expense_type:
                        tps = tp_list
                        break

                if len(td) > 0 and 'third_party' in td.columns and tps:
                    for tp in tps:
                        tpd = td[td['third_party'] == tp].copy()
                        if len(tpd) > 0:
                            daily = (
                                tpd.groupby('date', as_index=False)['amount']
                                .sum()
                                .sort_values('date')
                                .reset_index(drop=True)
                            )
                            self._account_type_tp_daily_map[acc][expense_type][tp] = daily

                            monthly = (
                                tpd.groupby(tpd['date'].dt.to_period('M'))['amount']
                                .sum()
                                .reset_index()
                            )
                            monthly.columns = ['year_month', 'amount']
                            monthly = monthly[monthly['amount'] > 0].copy()
                            if len(monthly) > 0:
                                monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
                                monthly['month_num'] = monthly['year_month'].dt.year * 12 + monthly['year_month'].dt.month
                            self._account_type_tp_monthly_map[acc][expense_type][tp] = monthly
                        else:
                            self._account_type_tp_daily_map[acc][expense_type][tp] = pd.DataFrame()
                            self._account_type_tp_monthly_map[acc][expense_type][tp] = pd.DataFrame()

    # ==================== amount 相关特征 ====================

    def amount_global(self) -> Dict:
        """全局支出统计特征"""
        features = {}

        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())

            if len(window_data) > 0:
                amounts = window_data['amount'].to_numpy()

                total_sum = np.sum(amounts)
                total_count = len(amounts)
                total_mean = np.mean(amounts)
                total_median = np.median(amounts)
                total_max = np.max(amounts)
                total_min = np.min(amounts)
                total_std = np.std(amounts)
                total_cv = total_std / total_mean if total_mean != 0 else np.nan

                features[f'bank_txn_expense_global_sum_{window}d'] = total_sum
                features[f'bank_txn_expense_global_count_{window}d'] = total_count
                features[f'bank_txn_expense_global_mean_{window}d'] = total_mean
                features[f'bank_txn_expense_global_median_{window}d'] = total_median
                features[f'bank_txn_expense_global_max_{window}d'] = total_max
                features[f'bank_txn_expense_global_min_{window}d'] = total_min
                features[f'bank_txn_expense_global_std_{window}d'] = total_std
                features[f'bank_txn_expense_global_cv_{window}d'] = total_cv
            else:
                features[f'bank_txn_expense_global_sum_{window}d'] = 0.0
                features[f'bank_txn_expense_global_count_{window}d'] = 0.0
                features[f'bank_txn_expense_global_mean_{window}d'] = np.nan
                features[f'bank_txn_expense_global_median_{window}d'] = np.nan
                features[f'bank_txn_expense_global_max_{window}d'] = np.nan
                features[f'bank_txn_expense_global_min_{window}d'] = np.nan
                features[f'bank_txn_expense_global_std_{window}d'] = np.nan
                features[f'bank_txn_expense_global_cv_{window}d'] = np.nan

        self.features.update(features)
        return features

    def amount_by_type(self) -> Dict:
        """按支出类型的统计特征"""
        features = {}

        for window in self.time_windows:
            type_stats = self._window_cache[window]['type_stats']

            if len(type_stats) == 0:
                for expense_type in self.expense_types:
                    for stat in ['sum', 'count']:
                        features[f'bank_txn_expense_{expense_type}_{stat}_{window}d'] = 0.0
                    for stat in ['mean', 'median', 'max', 'min', 'std', 'cv']:
                        features[f'bank_txn_expense_{expense_type}_{stat}_{window}d'] = np.nan
                continue

            type_stats_map = type_stats.set_index('tag_level2').to_dict('index')

            for expense_type in self.expense_types:
                row = type_stats_map.get(expense_type)
                if row:
                    features[f'bank_txn_expense_{expense_type}_sum_{window}d'] = row['sum']
                    features[f'bank_txn_expense_{expense_type}_count_{window}d'] = row['count']
                    features[f'bank_txn_expense_{expense_type}_mean_{window}d'] = row['mean']
                    features[f'bank_txn_expense_{expense_type}_median_{window}d'] = row['median']
                    features[f'bank_txn_expense_{expense_type}_max_{window}d'] = row['max']
                    features[f'bank_txn_expense_{expense_type}_min_{window}d'] = row['min']
                    features[f'bank_txn_expense_{expense_type}_std_{window}d'] = row['std']
                    features[f'bank_txn_expense_{expense_type}_cv_{window}d'] = row['cv']
                else:
                    features[f'bank_txn_expense_{expense_type}_sum_{window}d'] = 0.0
                    features[f'bank_txn_expense_{expense_type}_count_{window}d'] = 0.0
                    features[f'bank_txn_expense_{expense_type}_mean_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_median_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_max_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_min_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_std_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_cv_{window}d'] = np.nan

        self.features.update(features)
        return features

    def amount_daily(self) -> Dict:
        """【调整】单日最大最小支出额度：先单卡计算，再聚合"""
        features = {}

        for window in self.time_windows:
            for expense_type in self.expense_types:
                daily_max_vals = []
                daily_min_vals = []

                for acc in self.account_ids:
                    daily_stats = self._account_window_cache[acc][window]['type_daily_stats']
                    if len(daily_stats) == 0:
                        continue

                    row = daily_stats[daily_stats['tag_level2'] == expense_type]
                    if len(row) == 0:
                        continue

                    daily_max_vals.append(float(row['daily_max'].iloc[0]))
                    daily_min_vals.append(float(row['daily_min'].iloc[0]))

                features[f'bank_txn_expense_{expense_type}_daily_max_{window}d'] = self._nanmax_or_nan(daily_max_vals)
                features[f'bank_txn_expense_{expense_type}_daily_min_{window}d'] = self._nanmin_or_nan(daily_min_vals)

        self.features.update(features)
        return features

    def amount_comparison(self, windows: List[int] = [7, 30, 90, 180]) -> Dict:
        """【调整】支出金额对比特征：先单卡算 ratio，再聚合"""
        features = {}

        available_windows = [w for w in [7, 30, 90, 180] if w in windows]
        if not available_windows:
            return features

        comparisons = [
            ('1w_vs_1m', 7, 30),
            ('1m_vs_3m', 30, 90),
            ('1m_vs_6m', 30, 180),
            ('3m_vs_6m', 90, 180)
        ]

        for expense_type in self.expense_types:
            ratio_map = {suffix: [] for suffix, _, _ in comparisons}
            weight_map = {suffix: [] for suffix, _, _ in comparisons}

            for acc in self.account_ids:
                type_data = self._account_type_data.get(acc, {}).get(expense_type, pd.DataFrame())
                if type_data is None or len(type_data) == 0:
                    continue

                window_avgs = {}
                for window in available_windows:
                    wd = type_data[type_data['trac_days'] <= window]
                    window_avgs[window] = np.mean(wd['amount'].to_numpy()) if len(wd) > 0 else 0

                acc_weight = float(type_data['amount'].sum()) if len(type_data) > 0 else np.nan

                for suffix, num_window, denom_window in comparisons:
                    if (
                        num_window in window_avgs and
                        denom_window in window_avgs and
                        window_avgs[denom_window] is not None and
                        window_avgs[denom_window] > 0
                    ):
                        numerator = window_avgs[num_window]
                        denominator = window_avgs[denom_window]
                        ratio_val = 0.0 if numerator == 0 else numerator / denominator
                        ratio_map[suffix].append(ratio_val)
                        weight_map[suffix].append(acc_weight)

            for suffix, _, _ in comparisons:
                features[f'bank_txn_expense_{expense_type}_{suffix}'] = self._weighted_nanmean(
                    ratio_map[suffix],
                    weight_map[suffix]
                )

        self.features.update(features)
        return features

    def growth_rates(self) -> Dict:
        """
        【调整】最大支出涨跌幅：先单卡计算，再聚合
        """
        features = {}
        for expense_type in self.expense_types:
            max_vals = []
            min_vals = []

            for acc in self.account_ids:
                daily_data = self._account_type_daily_map.get(acc, {}).get(expense_type, pd.DataFrame())

                if daily_data is None or len(daily_data) <= 1:
                    continue

                amounts = daily_data['amount'].to_numpy()
                prev_amounts = amounts[:-1]
                curr_amounts = amounts[1:]
                growth_rates = np.where(prev_amounts != 0, (curr_amounts - prev_amounts) / prev_amounts, 0)

                max_vals.append(float(np.max(growth_rates)))
                min_vals.append(float(np.min(growth_rates)))

            features[f'bank_txn_expense_{expense_type}_growth_rate_max'] = self._nanmax_or_nan(max_vals)
            features[f'bank_txn_expense_{expense_type}_growth_rate_min'] = self._nanmin_or_nan(min_vals)

        self.features.update(features)
        return features

    # ==================== ratio 相关特征 ====================

    def ratio(self) -> Dict:
        """支出占比特征"""
        features = {}

        for window in self.time_windows:
            type_stats = self._window_cache[window]['type_stats']
            total_amount = self._window_cache[window]['global_total_amount']

            if len(type_stats) == 0:
                for expense_type in self.expense_types:
                    features[f'bank_txn_expense_{expense_type}_ratio_{window}d'] = np.nan
                continue

            type_sum_map = type_stats.set_index('tag_level2')['sum'].to_dict()

            if total_amount is None or total_amount <= 0:
                for expense_type in self.expense_types:
                    features[f'bank_txn_expense_{expense_type}_ratio_{window}d'] = np.nan
            else:
                for expense_type in self.expense_types:
                    type_sum = type_sum_map.get(expense_type, 0)
                    if type_sum == 0:
                        features[f'bank_txn_expense_{expense_type}_ratio_{window}d'] = 0
                    else:
                        features[f'bank_txn_expense_{expense_type}_ratio_{window}d'] = type_sum / total_amount

        self.features.update(features)
        return features

    def ratio_by_type(self) -> Dict:
        """
        最大单笔支出占支出类型总支出比
        """
        features = {}

        for window in self.time_windows:
            type_stats = self._window_cache[window]['type_stats']

            if len(type_stats) == 0:
                for expense_type in self.expense_types:
                    features[f'bank_txn_expense_{expense_type}_max_ratio_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_min_ratio_{window}d'] = np.nan
                continue

            type_stats_map = type_stats.set_index('tag_level2').to_dict('index')

            for expense_type in self.expense_types:
                row = type_stats_map.get(expense_type)
                if row and row['sum'] > 0:
                    features[f'bank_txn_expense_{expense_type}_max_ratio_{window}d'] = row['max'] / row['sum']
                    features[f'bank_txn_expense_{expense_type}_min_ratio_{window}d'] = row['min'] / row['sum']
                else:
                    features[f'bank_txn_expense_{expense_type}_max_ratio_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_min_ratio_{window}d'] = np.nan

        self.features.update(features)
        return features

    # ==================== stability 相关特征 ====================

    def max_consecutive_months(self) -> Dict:
        """
        【调整】连续支出月份：先单卡，再取最大
        """
        features = {}
        for expense_type in self.expense_types:
            vals = []
            for acc in self.account_ids:
                monthly_data = self._account_type_monthly_map.get(acc, {}).get(expense_type, pd.DataFrame())
                if monthly_data is None or len(monthly_data) == 0:
                    continue
                months = monthly_data['month_num'].to_numpy()
                vals.append(self._max_consecutive_ints(months))

            features[f'bank_txn_expense_{expense_type}_max_consecutive_months'] = self._nanmax_or_nan(vals)

        self.features.update(features)
        return features

    def max_consecutive_days(self) -> Dict:
        """
        【调整】连续支出天数：先单卡，再取最大
        """
        features = {}
        for expense_type in self.expense_types:
            vals = []
            for acc in self.account_ids:
                daily_data = self._account_type_daily_map.get(acc, {}).get(expense_type, pd.DataFrame())
                if daily_data is None or len(daily_data) == 0:
                    continue
                date_ord = self._date_to_day_ordinal(daily_data['date'])
                vals.append(self._max_consecutive_ints(date_ord))

            features[f'bank_txn_expense_{expense_type}_max_consecutive_days'] = self._nanmax_or_nan(vals)

        self.features.update(features)
        return features

    def trend_slope(self) -> Dict:
        """【调整】支出趋势斜率：先单卡计算，再按该类型金额加权聚合"""
        features = {}

        for window in self.time_windows:
            for expense_type in self.expense_types:
                slope_vals = []
                slope_weights = []

                for acc in self.account_ids:
                    type_daily = self._account_window_cache[acc][window]['type_daily']
                    if len(type_daily) == 0:
                        continue

                    td = type_daily[type_daily['tag_level2'] == expense_type]
                    if len(td) >= 2:
                        slope_vals.append(self._calc_slope_from_daily(td[['date', 'amount']]))
                        acc_type_df = self._account_type_data.get(acc, {}).get(expense_type, pd.DataFrame())
                        slope_weights.append(float(acc_type_df['amount'].sum()) if len(acc_type_df) > 0 else 1.0)

                features[f'bank_txn_expense_{expense_type}_trend_slope_{window}d'] = self._weighted_nanmean(
                    slope_vals,
                    slope_weights
                )

        self.features.update(features)
        return features

    # ==================== third_party 相关特征 ====================

    def third_party_count(self) -> Dict:
        """交易对手特征"""
        features = {}

        if self.third_party_col not in self.df.columns:
            return features

        for expense_type, tps in self.expense_type_tp_pairs:
            type_data = self._type_data.get(expense_type)

            if type_data is None or len(type_data) == 0:
                features[f'bank_txn_expense_{expense_type}_3rdparty_count'] = 0
                continue

            unique_counts = type_data['third_party'].nunique()
            features[f'bank_txn_expense_{expense_type}_3rdparty_count'] = unique_counts

        self.features.update(features)
        return features

    def third_party_consecutive(self) -> Dict:
        """
        【调整】同一支出交易对手最长持续时间：先单卡，再取最大
        """
        features = {}

        for expense_type in self.expense_types:
            vals = []

            for acc in self.account_ids:
                type_data = self._account_type_data.get(acc, {}).get(expense_type, pd.DataFrame())

                if type_data is not None and len(type_data) > 0 and 'third_party' in type_data.columns:
                    grouped = (
                        type_data.groupby(['third_party', 'trac_days'], as_index=False)['amount']
                        .sum()
                        .sort_values(['third_party', 'trac_days'])
                    )

                    max_lens = []
                    for _, g in grouped.groupby('third_party'):
                        days = g['trac_days'].to_numpy()
                        if len(days) > 0:
                            max_len = self._max_consecutive_ints(days)
                            if not pd.isna(max_len):
                                max_lens.append(max_len)

                    if max_lens:
                        vals.append(np.max(max_lens))
                    else:
                        vals.append(1)

            features[f'bank_txn_expense_{expense_type}_3rdparty_max_consecutive'] = self._nanmax_or_nan(vals)

        self.features.update(features)
        return features

    def third_party_amount(self) -> Dict:
        """按支出类型-交易对手组合的金额统计特征"""
        features = {}

        for window in self.time_windows:
            tp_stats = self._window_cache[window]['tp_stats']

            if len(tp_stats) == 0:
                for expense_type, tps in self.expense_type_tp_pairs:
                    if not tps:
                        continue
                    for tp in tps:
                        cleaned_tp = self._cleaned_tp_map[expense_type][tp]
                        for stat in ['sum', 'count']:
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_{stat}_{window}d'] = 0.0
                        for stat in ['mean', 'median', 'max', 'min', 'std', 'cv']:
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_{stat}_{window}d'] = np.nan
                continue

            tp_stats_map = {
                (r['tag_level2'], r['third_party']): r
                for _, r in tp_stats.iterrows()
            }

            for expense_type, tps in self.expense_type_tp_pairs:
                if not tps:
                    continue
                for tp in tps:
                    cleaned_tp = self._cleaned_tp_map[expense_type][tp]
                    row = tp_stats_map.get((expense_type, tp))

                    if row is not None:
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_sum_{window}d'] = row['sum']
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_count_{window}d'] = row['count']
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_mean_{window}d'] = row['mean']
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_median_{window}d'] = row['median']
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_{window}d'] = row['max']
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_min_{window}d'] = row['min']
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_std_{window}d'] = row['std']
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_cv_{window}d'] = row['cv']
                    else:
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_sum_{window}d'] = 0.0
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_count_{window}d'] = 0.0
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_mean_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_median_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_min_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_std_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_cv_{window}d'] = np.nan

        self.features.update(features)
        return features

    def third_party_ratio(self) -> Dict:
        """按支出类型-交易对手组合的占比特征"""
        features = {}

        for window in self.time_windows:
            cache = self._window_cache[window]
            tp_stats = cache['tp_stats']
            type_stats = cache['type_stats']
            total_amount = cache['global_total_amount']

            denominator_type_valid = len(type_stats) > 0
            denominator_total_valid = total_amount is not None and total_amount > 0

            tp_sum_map = {}
            if len(tp_stats) > 0:
                tp_sum_map = {(r['tag_level2'], r['third_party']): r['sum'] for _, r in tp_stats.iterrows()}

            type_sum_map = {}
            if len(type_stats) > 0:
                type_sum_map = type_stats.set_index('tag_level2')['sum'].to_dict()

            for expense_type, tps in self.expense_type_tp_pairs:
                if not tps:
                    continue

                type_sum = type_sum_map.get(expense_type, 0)

                for tp in tps:
                    cleaned_tp = self._cleaned_tp_map[expense_type][tp]
                    tp_sum = tp_sum_map.get((expense_type, tp), 0)

                    if not denominator_type_valid or type_sum == 0:
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_{window}d'] = np.nan
                    else:
                        if tp_sum == 0:
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_{window}d'] = 0
                        else:
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_{window}d'] = tp_sum / type_sum

                    if not denominator_total_valid:
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_total_{window}d'] = np.nan
                    else:
                        if tp_sum == 0:
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_total_{window}d'] = 0
                        else:
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_total_{window}d'] = tp_sum / total_amount

        self.features.update(features)
        return features

    def third_party_amount_daily(self) -> Dict:
        """【调整】交易对手单日最大最小支出额度：先单卡，再聚合"""
        features = {}

        for window in self.time_windows:
            for expense_type, tps in self.expense_type_tp_pairs:
                if not tps:
                    continue

                for tp in tps:
                    cleaned_tp = self._cleaned_tp_map[expense_type][tp]
                    daily_max_vals = []
                    daily_min_vals = []

                    for acc in self.account_ids:
                        daily_stats = self._account_window_cache[acc][window]['tp_daily_stats']
                        if len(daily_stats) == 0:
                            continue

                        row = daily_stats[
                            (daily_stats['tag_level2'] == expense_type) &
                            (daily_stats['third_party'] == tp)
                        ]
                        if len(row) == 0:
                            continue

                        daily_max_vals.append(float(row['daily_max'].iloc[0]))
                        daily_min_vals.append(float(row['daily_min'].iloc[0]))

                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_daily_max_{window}d'] = self._nanmax_or_nan(daily_max_vals)
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_daily_min_{window}d'] = self._nanmin_or_nan(daily_min_vals)

        self.features.update(features)
        return features

    def third_party_amount_comparison(self, windows: List[int] = [30, 90, 180]) -> Dict:
        """【调整】按支出类型-交易对手组合的金额对比特征：先单卡算 ratio，再聚合"""
        features = {}
        available_windows = [w for w in [30, 90, 180] if w in windows]

        if not available_windows:
            return features

        comparisons = [
            ('latest_vs_3m', 'latest', 90),
            ('1m_vs_3m', 30, 90),
            ('3m_vs_6m', 90, 180)
        ]

        for expense_type, tps in self.expense_type_tp_pairs:
            if not tps:
                continue

            for tp in tps:
                cleaned_tp = self._cleaned_tp_map[expense_type][tp]
                ratio_map = {suffix: [] for suffix, _, _ in comparisons}
                weight_map = {suffix: [] for suffix, _, _ in comparisons}

                for acc in self.account_ids:
                    type_data = self._account_type_data.get(acc, {}).get(expense_type, pd.DataFrame())
                    if type_data is None or len(type_data) == 0:
                        continue

                    tp_data = type_data[type_data['third_party'] == tp]
                    if len(tp_data) == 0:
                        continue

                    window_avgs = {}
                    for window in available_windows:
                        wd = tp_data[tp_data['trac_days'] <= window]
                        window_avgs[window] = np.mean(wd['amount'].to_numpy()) if len(wd) > 0 else 0

                    latest = tp_data['amount'].iloc[0] if len(tp_data) > 0 else 0
                    acc_weight = float(tp_data['amount'].sum()) if len(tp_data) > 0 else np.nan

                    for suffix, num, denom in comparisons:
                        if suffix == 'latest_vs_3m':
                            numerator = latest
                            denominator = window_avgs.get(denom, 0)
                        else:
                            numerator = window_avgs.get(num, 0)
                            denominator = window_avgs.get(denom, 0)

                        if denominator == 0:
                            ratio_val = np.nan
                        elif numerator == 0:
                            ratio_val = 0.0
                        else:
                            ratio_val = numerator / denominator

                        if pd.notna(ratio_val):
                            ratio_map[suffix].append(ratio_val)
                            weight_map[suffix].append(acc_weight)

                for suffix, _, _ in comparisons:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_{suffix}'] = self._weighted_nanmean(
                        ratio_map[suffix],
                        weight_map[suffix]
                    )

        self.features.update(features)
        return features

    def third_party_growth_rates(self) -> Dict:
        """【调整】按支出类型-交易对手组合的支出增长率特征：先单卡，再聚合"""
        features = {}

        for expense_type, tps in self.expense_type_tp_pairs:
            if not tps:
                continue

            for tp in tps:
                cleaned_tp = self._cleaned_tp_map[expense_type][tp]
                max_vals = []
                min_vals = []

                for acc in self.account_ids:
                    daily_data = self._account_type_tp_daily_map.get(acc, {}).get(expense_type, {}).get(tp, pd.DataFrame())

                    if daily_data is None or len(daily_data) <= 1:
                        continue

                    amounts = daily_data['amount'].to_numpy()
                    prev_amounts = amounts[:-1]
                    curr_amounts = amounts[1:]
                    growth_rates = np.where(prev_amounts != 0, (curr_amounts - prev_amounts) / prev_amounts, 0)

                    max_vals.append(float(np.max(growth_rates)))
                    min_vals.append(float(np.min(growth_rates)))

                features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_growth_rate_max'] = self._nanmax_or_nan(max_vals)
                features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_growth_rate_min'] = self._nanmin_or_nan(min_vals)

        self.features.update(features)
        return features

    def third_party_max_consecutive_months(self) -> Dict:
        """
        【调整】支出交易对手的最长持续月份：先单卡，再取最大
        """
        features = {}
        for expense_type, tps in self.expense_type_tp_pairs:
            if not tps:
                continue

            for tp in tps:
                cleaned_tp = self._cleaned_tp_map[expense_type][tp]
                vals = []

                for acc in self.account_ids:
                    monthly_data = self._account_type_tp_monthly_map.get(acc, {}).get(expense_type, {}).get(tp, pd.DataFrame())

                    if monthly_data is None or len(monthly_data) == 0:
                        continue

                    months = monthly_data['month_num'].to_numpy()
                    vals.append(self._max_consecutive_ints(months))

                features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_consecutive_months'] = self._nanmax_or_nan(vals)

        self.features.update(features)
        return features

    def third_party_max_consecutive_days(self) -> Dict:
        """
        【调整】支出交易对手的最长持续天数：先单卡，再取最大
        """
        features = {}
        for expense_type, tps in self.expense_type_tp_pairs:
            if not tps:
                continue

            for tp in tps:
                cleaned_tp = self._cleaned_tp_map[expense_type][tp]
                vals = []

                for acc in self.account_ids:
                    daily_data = self._account_type_tp_daily_map.get(acc, {}).get(expense_type, {}).get(tp, pd.DataFrame())

                    if daily_data is None or len(daily_data) == 0:
                        continue

                    date_ord = self._date_to_day_ordinal(daily_data['date'])
                    vals.append(self._max_consecutive_ints(date_ord))

                features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_consecutive_days'] = self._nanmax_or_nan(vals)

        self.features.update(features)
        return features

    def third_party_trend_slope(self) -> Dict:
        """
        【调整】支出交易对手的趋势：先单卡，再按该 tp 金额加权聚合
        """
        features = {}

        for window in self.time_windows:
            for expense_type, tps in self.expense_type_tp_pairs:
                if not tps:
                    continue

                for tp in tps:
                    cleaned_tp = self._cleaned_tp_map[expense_type][tp]
                    slope_vals = []
                    slope_weights = []

                    for acc in self.account_ids:
                        tp_daily = self._account_window_cache[acc][window]['tp_daily']
                        if len(tp_daily) == 0:
                            continue

                        tpd = tp_daily[
                            (tp_daily['tag_level2'] == expense_type) &
                            (tp_daily['third_party'] == tp)
                        ]

                        if len(tpd) >= 2:
                            slope_vals.append(self._calc_slope_from_daily(tpd[['date', 'amount']]))
                            tp_df = self._account_type_data.get(acc, {}).get(expense_type, pd.DataFrame())
                            tp_df = tp_df[tp_df['third_party'] == tp] if len(tp_df) > 0 else tp_df
                            slope_weights.append(float(tp_df['amount'].sum()) if len(tp_df) > 0 else 1.0)

                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_trend_slope_{window}d'] = self._weighted_nanmean(
                        slope_vals,
                        slope_weights
                    )

        self.features.update(features)
        return features

    # ==================== 数据整合 ====================

    def get_metadata(self) -> Dict:
        """获取元数据信息"""
        metadata = {
            'user_id': self.user_id,
            'sample_datetime': self.sample_datetime
        }
        return metadata

    def generate_all_features(
        self,
        df: pd.DataFrame = None,
        feature_groups: List[str] = None,
        include_metadata: bool = True
    ) -> pd.DataFrame:
        """
        生成所有特征
        返回特征DataFrame
        """
        if df is not None:
            self.original_df = df.copy()
            self.df = self._map(df)
            if self.df is None:
                self.df = pd.DataFrame()
            if 'transaction_date' in self.df.columns and 'date' not in self.df.columns:
                self.df.rename(columns={'transaction_date': 'date'}, inplace=True)
            self.raw_df = self.df.copy()
            self._prepare_data()

        if feature_groups is None:
            feature_groups = [
                'amount_global', 'amount_by_type',
                'ratio', 'ratio_by_type',
                'amount_daily', 'amount_comparison', 'growth_rates',
                'max_consecutive_months', 'max_consecutive_days', 'trend_slope',
                'third_party_count', 'third_party_consecutive',
                'third_party_amount',
                'third_party_ratio',
                'third_party_amount_daily', 'third_party_amount_comparison',
                'third_party_growth_rates', 'third_party_max_consecutive_months',
                'third_party_max_consecutive_days', 'third_party_trend_slope'
            ]

        self.features = {}

        for feature in feature_groups:
            method = getattr(self, feature)
            method()

        self.features['user_id'] = self.user_id
        self.features['sample_datetime'] = self.sample_datetime

        if include_metadata:
            metadata = self.get_metadata()
            for key, value in metadata.items():
                if key not in ['user_id', 'sample_datetime']:
                    self.features[f'metadata_{key}'] = value

        df_out = pd.DataFrame([self.features])

        base_columns = ['user_id', 'sample_datetime']
        existing_base = [col for col in base_columns if col in df_out.columns]
        feature_cols = [col for col in df_out.columns if col not in existing_base]

        df_out = df_out[existing_base + feature_cols]
        df_out.columns = df_out.columns.str.strip().str.replace(' ', '').str.lower()

        return df_out


expense_type_tp_pairs = [
    ['Dining Out', []],
    ['External Transfers', []],
    ['Internal Transfer', []],
    ['Groceries', []],
    ['Retail', []],
    ['Department Stores', []],
    ['Automotive', []],
    ['Rent', []],
    # ['Health', []],
    ['Utilities', []],
    ['Education', []],
    ['Telecommunications', []],
    # ['Donations', []],
    ['Centrelink', []],
    ['Fees', []],
    # ['Entertainment', []],
    ['Home Improvement', []],
    # ['Transport', []],
    # ['Subscription TV', []],
    ['Pet Care', []],
    ['Gambling', []],
    # ['Gyms and other memberships', []],
    ['Travel', []],
    # ['Personal Care', []],
    ['Insurance', []],
    # ['Overdrawn', []],
]


def generate_expense_feature(df: pd.DataFrame, feature_groups: List[str] = None):
    expense_engineer = SingleApplicationExpenseFeatureEngineer(
        df=df,
        time_windows=[7, 14, 28, 56, 84, 168, 182],
        expense_type_tp_pairs=expense_type_tp_pairs
    )
    return expense_engineer.generate_all_features(df=df, feature_groups=feature_groups)
