import os
import re
from datetime import timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class SingleApplicationIncomeFeatureEngineer:
    """
    优化版：
    - 尽量不改输出字段名
    - 尽量不改核心业务逻辑
    - 重点优化重复过滤、重复 groupby、重复 copy、重复线性回归拟合
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time_windows: List[int] = None,
        income_type_tp_pairs: List[List] = None,
        income_type_category_pairs: List[List] = None,
        tag_level1: str = "INCOME",
        already_mapped: bool = False,  # ===== [多卡调整] =====
        enable_multi_account_agg: bool = True,  # ===== [多卡调整] =====
    ):
        self.original_df = df.copy()

        self.time_windows = sorted(time_windows) if time_windows else [7, 14, 28, 56, 84, 168, 182]
        self.income_type_tp_pairs = income_type_tp_pairs or []
        self.income_type_category_pairs = income_type_category_pairs or []
        self.tag_level1 = tag_level1

        self.income_types = list(
            set(
                [pair[0] for pair in self.income_type_tp_pairs]
                + [pair[0] for pair in self.income_type_category_pairs]
            )
        )

        self.third_parties = list(
            set(tp for pair in self.income_type_tp_pairs for tp in pair[1] if tp)
        )
        self.categories = list(
            set(cat for pair in self.income_type_category_pairs for cat in pair[1] if cat)
        )

        self.third_party_col = "third_party"
        self.category_col = "category"
        self.features = {}

        # ===== [多卡调整] =====
        self.already_mapped = already_mapped
        self.enable_multi_account_agg = enable_multi_account_agg
        self.account_col = "bank_account_id"
        self.has_multi_accounts = False
        self._account_engines = {}

        self.df = df.copy() if self.already_mapped else self._map(df)
        self.raw_df = self.df.copy()

        self._prepare_data()
        self._init_multi_account_engines()  # ===== [多卡调整] =====

    # =========================================================
    # 基础处理
    # =========================================================
    def _map(
        self,
        df: pd.DataFrame,
        mapping_file: str = None,
    ) -> pd.DataFrame:
        """根据交易映射表进行映射"""
        df = df.copy()

        # 统一空值
        cols = ['account_type','category','dr_cr','third_party']

        df[cols] = df[cols].astype(str).apply(lambda x: x.str.strip()).replace({'': None, 'nan': None, 'None': None})

        df.fillna({'account_type': 'Unlabeled', 'category': 'Unlabeled', 'dr_cr': 'Unlabeled'}, inplace=True)

        # 使用当前文件路径
        if mapping_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mapping_file = os.path.join(current_dir,'reference','finv_level','finv交易类别映射矩阵_0303.csv')

        #print(f"映射文件路径: {mapping_file}")
        if not os.path.exists(mapping_file):
            print(f"警告: 映射文件 {mapping_file} 不存在")
            return df

        # for col in ["account_type", "category", "dr_cr"]:
        #     if col not in df.columns:
        #         df[col] = "Unlabeled"
        #     df[col] = df[col].fillna("Unlabeled")

        # if not os.path.isabs(mapping_file):
        #     try:
        #         base_dir = os.path.dirname(os.path.realpath(__file__))
        #     except Exception:
        #         base_dir = os.getcwd()
        #     mapping_file = os.path.join(base_dir, mapping_file.lstrip("./"))

        # if not os.path.exists(mapping_file):
        #     print(f"警告: 映射文件 {mapping_file} 不存在")
        #     return df

        file_ext = os.path.splitext(mapping_file)[1].lower()
        try:
            if file_ext == ".csv":
                mapping_df = pd.read_csv(mapping_file)
            elif file_ext in [".xlsx", ".xls"]:
                mapping_df = pd.read_excel(mapping_file)
            else:
                print(f"警告: 不支持的文件格式 {file_ext}")
                return df
        except Exception as e:
            print(f"读取映射文件失败: {e}")
            return df

        need_cols = ["dr_cr", "category", "account_type", "tag_level1", "tag_level2"]
        missing = [c for c in need_cols if c not in mapping_df.columns]
        if missing:
            print(f"警告: 映射文件缺少字段: {missing}")
            return df

        df = df.merge(
            mapping_df[need_cols],
            on=["dr_cr", "category", "account_type"],
            how="left",
        )
        return df

    def _clean_entity_name(self, name: str) -> str:
        """清洗实体名称，用于特征键名"""
        if not isinstance(name, str):
            name = str(name)

        cleaned = name.lower()
        cleaned = cleaned.replace(" ", "")
        cleaned = cleaned.replace("/", "_")
        cleaned = cleaned.replace("-", "_")
        cleaned = cleaned.replace(".", "_")
        cleaned = cleaned.replace("&", "and")
        cleaned = cleaned.replace("@", "at")
        cleaned = cleaned.replace("$", "dollar")
        cleaned = cleaned.replace("%", "percent")
        cleaned = re.sub(r"[^\w_]", "", cleaned)
        return cleaned

    def _safe_stats(self, arr: np.ndarray) -> Dict[str, float]:
        """统一金额统计"""
        if arr is None or len(arr) == 0:
            return {
                "sum": 0,
                "count": 0,
                "mean": np.nan,
                "median": np.nan,
                "max": np.nan,
                "min": np.nan,
                "std": np.nan,
                "cv": np.nan,
            }

        total_sum = float(np.sum(arr))
        total_count = int(len(arr))
        total_mean = float(np.mean(arr))
        total_median = float(np.median(arr))
        total_max = float(np.max(arr))
        total_min = float(np.min(arr))
        total_std = float(np.std(arr)) if total_count > 1 else np.nan
        total_cv = total_std / total_mean if (pd.notna(total_std) and total_mean > 0) else np.nan

        return {
            "sum": total_sum,
            "count": total_count,
            "mean": total_mean,
            "median": total_median,
            "max": total_max,
            "min": total_min,
            "std": total_std,
            "cv": total_cv,
        }

    def _calc_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        用闭式解代替 LinearRegression().fit()
        slope = cov(x, y) / var(x)
        """
        if x is None or y is None or len(x) < 2 or len(y) < 2:
            return np.nan

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(np.unique(x)) < 2:
            return np.nan

        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            return np.nan
        slope = np.sum((x - x_mean) * (y - y_mean)) / denom
        return float(slope)

    def _max_consecutive(self, values: np.ndarray, diff_target: int = 1) -> float:
        if values is None or len(values) == 0:
            return np.nan
        if len(values) == 1:
            return 1
        max_len = 1
        cur = 1
        for i in range(1, len(values)):
            if values[i] - values[i - 1] == diff_target:
                cur += 1
                if cur > max_len:
                    max_len = cur
            else:
                cur = 1
        return max_len

    # =========================================================
    # ===== [多卡调整] 多银行卡聚合辅助函数 ======================
    # =========================================================
    def _init_multi_account_engines(self):
        """
        多卡场景下，只对“时间序列/节奏类”特征做按卡计算后再聚合，
        避免不同银行卡的独立收支节奏被直接拼接后造成失真。
        """
        self.has_multi_accounts = False
        self._account_engines = {}

        if not self.enable_multi_account_agg:
            return
        if self.account_col not in self.raw_df.columns:
            return
        if len(self.raw_df) == 0:
            return

        raw_df = self.raw_df.copy()
        raw_df[self.account_col] = raw_df[self.account_col].astype(str).replace({"": np.nan, "nan": np.nan, "None": np.nan})
        raw_df[self.account_col] = raw_df[self.account_col].fillna("__MISSING_BANK_ACCOUNT__")

        account_ids = [acc for acc in raw_df[self.account_col].dropna().unique().tolist() if acc != ""]
        if len(account_ids) <= 1:
            return

        self.has_multi_accounts = True

        for account_id in account_ids:
            account_df = raw_df[raw_df[self.account_col] == account_id].copy()
            if len(account_df) == 0:
                continue
            self._account_engines[account_id] = SingleApplicationIncomeFeatureEngineer(
                df=account_df,
                time_windows=self.time_windows,
                income_type_tp_pairs=self.income_type_tp_pairs,
                income_type_category_pairs=self.income_type_category_pairs,
                tag_level1=self.tag_level1,
                already_mapped=True,
                enable_multi_account_agg=False,
            )

    def _aggregate_account_feature_value(self, feature_name: str, values: List[float]):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan

        # 计数类：按卡求和更符合 application 口径
        if feature_name.endswith("_interval_gap_30_count") or feature_name.endswith("_decrease_count"):
            return int(np.nansum(valid))
        if re.search(r"_(sum|count)_\d+d$", feature_name):
            return float(np.nansum(valid))

        # 明确的最大值类
        if (
            feature_name.endswith("_latest_amount")
            or feature_name.endswith("_max_consecutive_months")
            or feature_name.endswith("_max_consecutive_days")
            or feature_name.endswith("_interval_max")
            or feature_name.endswith("_growth_rate_max")
            or feature_name.endswith("_max_daily_consumption_7d")
            or "_max_consumption_rate_" in feature_name
        ):
            return float(np.nanmax(valid))

        # 明确的最小值类
        if feature_name.endswith("_growth_rate_min") or "_min_consumption_rate_" in feature_name:
            return float(np.nanmin(valid))

        # 比值/波动/趋势类：按卡均值，避免某张卡时间序列主导整体
        if (
            feature_name.endswith("_latest_vs_3m")
            or feature_name.endswith("_1m_vs_3m")
            or feature_name.endswith("_3m_vs_6m")
            or feature_name.endswith("_interval_std")
            or "_trend_slope_" in feature_name
            or "_mean_consumption_rate_" in feature_name
        ):
            return float(np.nanmean(valid))

        # 兜底：其他这批按卡聚合的时间序列特征统一取均值
        return float(np.nanmean(valid))

    def _aggregate_account_feature_dicts(self, feature_dicts: List[Dict]) -> Dict:
        if not feature_dicts:
            return {}

        keys = []
        for d in feature_dicts:
            keys.extend(d.keys())
        keys = list(dict.fromkeys(keys))

        out = {}
        for key in keys:
            vals = [d.get(key, np.nan) for d in feature_dicts]
            out[key] = self._aggregate_account_feature_value(key, vals)
        return out

    def _aggregate_account_method(self, method_name: str, *args, **kwargs) -> Dict:
        feature_dicts = []
        for engine in self._account_engines.values():
            method = getattr(engine, method_name)
            feature_dicts.append(method(*args, **kwargs))
        return self._aggregate_account_feature_dicts(feature_dicts)

    def _prepare_data(self):
        """预处理数据 + 预聚合缓存"""
        df = self.df.copy()

        if "transaction_date" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"transaction_date": "date"})

        if "sample_datetime" in df.columns:
            df["sample_datetime"] = pd.to_datetime(df["sample_datetime"], errors="coerce")
            self.sample_datetime = df["sample_datetime"].iloc[0] if len(df) > 0 else None
        else:
            self.sample_datetime = None

        if "user_id" in df.columns and len(df) > 0:
            self.user_id = df["user_id"].iloc[0]
        else:
            self.user_id = None

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")

        if "date" in df.columns and "sample_datetime" in df.columns:
            df["trac_days"] = (df["sample_datetime"].dt.floor("D") - df["date"]).dt.days
        else:
            df["trac_days"] = np.nan

        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()

        if "date" in df.columns:
            df["month_num"] = df["date"].dt.year * 12 + df["date"].dt.month

        # 保留 raw_df（含所有 tag_level1），后面消费率要用 EXPENSE
        self.raw_df = df.copy()

        # 只保留当前收入层级
        if "tag_level1" in df.columns:
            df = df[df["tag_level1"] == self.tag_level1].copy()
        else:
            print("警告: 数据中不存在'tag_level1'列，跳过筛选")

        # 去掉未来交易
        df = df[df["trac_days"].notna()].copy()
        df = df[df["trac_days"] >= 0].copy()

        # 排序：越近越靠前
        df = df.sort_values(["trac_days", "date"], ascending=[True, False]).reset_index(drop=True)
        self.df = df

        # -------------------------
        # type 级缓存
        # -------------------------
        self._type_data: Dict[str, pd.DataFrame] = {}
        if len(df) > 0 and "tag_level2" in df.columns:
            for income_type, g in df.groupby("tag_level2", sort=False):
                self._type_data[income_type] = g.copy()

        # 每个 window 的数据
        self._window_data: Dict[int, pd.DataFrame] = {}
        self._window_type_data: Dict[int, Dict[str, pd.DataFrame]] = {}

        for window in self.time_windows:
            wdf = df[df["trac_days"] <= window]
            self._window_data[window] = wdf
            if len(wdf) == 0:
                self._window_type_data[window] = {}
            else:
                self._window_type_data[window] = {
                    k: g.copy() for k, g in wdf.groupby("tag_level2", sort=False)
                }

        # -------------------------
        # raw type-entity 缓存
        # -------------------------
        self._type_tp_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._type_category_data: Dict[str, Dict[str, pd.DataFrame]] = {}

        for income_type, tps in self.income_type_tp_pairs:
            self._type_tp_data[income_type] = {}
            type_df = self._type_data.get(income_type, pd.DataFrame())
            if len(type_df) == 0 or not tps or "third_party" not in type_df.columns:
                continue
            grouped = {k: g.copy() for k, g in type_df.groupby("third_party", sort=False)}
            for tp in tps:
                self._type_tp_data[income_type][tp] = grouped.get(tp, pd.DataFrame())

        for income_type, cats in self.income_type_category_pairs:
            self._type_category_data[income_type] = {}
            type_df = self._type_data.get(income_type, pd.DataFrame())
            if len(type_df) == 0 or not cats or "category" not in type_df.columns:
                continue
            grouped = {k: g.copy() for k, g in type_df.groupby("category", sort=False)}
            for cat in cats:
                self._type_category_data[income_type][cat] = grouped.get(cat, pd.DataFrame())

        # -------------------------
        # 每种类型按日聚合
        # -------------------------
        self._type_daily: Dict[str, pd.DataFrame] = {}
        self._type_daily_desc: Dict[str, pd.DataFrame] = {}

        for income_type, tdf in self._type_data.items():
            daily = (
                tdf.groupby("date", as_index=False)["amount"]
                .sum()
                .sort_values("date", ascending=True)
                .reset_index(drop=True)
            )
            self._type_daily[income_type] = daily
            self._type_daily_desc[income_type] = daily.sort_values("date", ascending=False).reset_index(drop=True)

        # -------------------------
        # type + entity 按日聚合
        # -------------------------
        self._type_tp_daily: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._type_category_daily: Dict[Tuple[str, str], pd.DataFrame] = {}

        for income_type, tps in self.income_type_tp_pairs:
            type_df = self._type_data.get(income_type, pd.DataFrame())
            if len(type_df) == 0 or not tps or "third_party" not in type_df.columns:
                continue
            gp = (
                type_df.groupby(["third_party", "date"], as_index=False)["amount"]
                .sum()
                .sort_values(["third_party", "date"], ascending=[True, True])
            )
            for tp in tps:
                d = gp[gp["third_party"] == tp][["date", "amount"]].reset_index(drop=True)
                self._type_tp_daily[(income_type, tp)] = d

        for income_type, cats in self.income_type_category_pairs:
            type_df = self._type_data.get(income_type, pd.DataFrame())
            if len(type_df) == 0 or not cats or "category" not in type_df.columns:
                continue
            gp = (
                type_df.groupby(["category", "date"], as_index=False)["amount"]
                .sum()
                .sort_values(["category", "date"], ascending=[True, True])
            )
            for cat in cats:
                d = gp[gp["category"] == cat][["date", "amount"]].reset_index(drop=True)
                self._type_category_daily[(income_type, cat)] = d

        # -------------------------
        # 消费率相关缓存：EXPENSE 日聚合 + 前缀和
        # -------------------------
        expense_df = self.raw_df[self.raw_df.get("tag_level1", pd.Series(index=self.raw_df.index)) == "EXPENSE"].copy()
        if "date" in expense_df.columns and len(expense_df) > 0:
            expense_df["date"] = pd.to_datetime(expense_df["date"], errors="coerce").dt.floor("D")
            expense_df["amount"] = pd.to_numeric(expense_df["amount"], errors="coerce").abs()
            self.expense_daily = (
                expense_df.groupby("date", as_index=False)["amount"]
                .sum()
                .sort_values("date", ascending=True)
                .reset_index(drop=True)
            )
        else:
            self.expense_daily = pd.DataFrame(columns=["date", "amount"])

        if len(self.expense_daily) > 0:
            self._expense_dates = self.expense_daily["date"].values.astype("datetime64[D]")
            self._expense_days = self._expense_dates.astype("int64")
            self._expense_amounts = self.expense_daily["amount"].to_numpy(dtype=float)
            self._expense_prefix = np.concatenate([[0.0], np.cumsum(self._expense_amounts)])
        else:
            self._expense_dates = np.array([], dtype="datetime64[D]")
            self._expense_days = np.array([], dtype=np.int64)
            self._expense_amounts = np.array([], dtype=float)
            self._expense_prefix = np.array([0.0], dtype=float)

    # =========================================================
    # 消费率辅助函数
    # =========================================================
    def _expense_sum_between(self, start_days: np.ndarray, end_days: np.ndarray) -> np.ndarray:
        """
        用前缀和快速求多个 [start, end] 区间的 expense sum
        """
        if len(self._expense_days) == 0 or len(start_days) == 0:
            return np.zeros(len(start_days), dtype=float)

        left = np.searchsorted(self._expense_days, start_days, side="left")
        right = np.searchsorted(self._expense_days, end_days, side="right")
        return self._expense_prefix[right] - self._expense_prefix[left]

    def _expense_max_between_single(self, start_day: int, end_day: int) -> float:
        if len(self._expense_days) == 0:
            return 0.0
        l = np.searchsorted(self._expense_days, start_day, side="left")
        r = np.searchsorted(self._expense_days, end_day, side="right")
        if l >= r:
            return 0.0
        return float(np.max(self._expense_amounts[l:r]))

    def _calc_consumption_metrics_from_daily(self, daily_data: pd.DataFrame, windows: List[int]) -> Dict[str, float]:
        """
        输入某个收入序列的按日聚合数据，返回：
        - 最新7天窗口最大单日消费
        - 各窗口 max/min/mean consumption rate
        """
        out = {}

        if daily_data is None or len(daily_data) == 0:
            return out

        daily_desc = daily_data.sort_values("date", ascending=False).reset_index(drop=True)
        latest_date = pd.to_datetime(daily_desc.iloc[0]["date"]).to_datetime64().astype("datetime64[D]").astype("int64")
        max_daily_consumption_7d = self._expense_max_between_single(latest_date, latest_date + 7)
        out["max_daily_consumption_7d"] = max_daily_consumption_7d

        income_dates = pd.to_datetime(daily_desc["date"]).values.astype("datetime64[D]").astype("int64")
        income_amounts = daily_desc["amount"].to_numpy(dtype=float)

        valid_mask = income_amounts > 0
        income_dates = income_dates[valid_mask]
        income_amounts = income_amounts[valid_mask]

        if len(income_dates) == 0:
            for w in windows:
                out[f"max_consumption_rate_{w}d"] = np.nan
                out[f"min_consumption_rate_{w}d"] = np.nan
                out[f"mean_consumption_rate_{w}d"] = np.nan
            return out

        for w in windows:
            expense_sum = self._expense_sum_between(income_dates, income_dates + w)
            rates = expense_sum / income_amounts
            if len(rates) > 0:
                out[f"max_consumption_rate_{w}d"] = float(np.max(rates))
                out[f"min_consumption_rate_{w}d"] = float(np.min(rates))
                out[f"mean_consumption_rate_{w}d"] = float(np.mean(rates))
            else:
                out[f"max_consumption_rate_{w}d"] = np.nan
                out[f"min_consumption_rate_{w}d"] = np.nan
                out[f"mean_consumption_rate_{w}d"] = np.nan

        return out

    # =========================================================
    # 通用实体特征生成函数
    # =========================================================
    def _generate_entity_amount_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        features = {}
        entity_key = "3rdparty" if entity_type == "tp" else "category"

        for window in self.time_windows:
            type_map = self._window_type_data.get(window, {})

            for income_type, entities in entity_pairs:
                if not entities:
                    continue

                type_data = type_map.get(income_type, pd.DataFrame())
                if len(type_data) == 0:
                    for entity in entities:
                        cleaned_entity = self._clean_entity_name(entity)
                        for stat in ["sum", "count"]:
                            features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_{stat}_{window}d"] = 0
                        for stat in ["mean", "median", "max", "min", "std", "cv"]:
                            features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_{stat}_{window}d"] = np.nan
                    continue

                if entity_col not in type_data.columns:
                    for entity in entities:
                        cleaned_entity = self._clean_entity_name(entity)
                        for stat in ["sum", "count"]:
                            features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_{stat}_{window}d"] = 0
                        for stat in ["mean", "median", "max", "min", "std", "cv"]:
                            features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_{stat}_{window}d"] = np.nan
                    continue

                grouped = {k: g["amount"].to_numpy(dtype=float) for k, g in type_data.groupby(entity_col, sort=False)}

                for entity in entities:
                    cleaned_entity = self._clean_entity_name(entity)
                    stats = self._safe_stats(grouped.get(entity))
                    for stat_name, stat_value in stats.items():
                        features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_{stat_name}_{window}d"] = stat_value

        return features
    
    def _generate_entity_ratio_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        features = {}
        entity_key = "3rdparty" if entity_type == "tp" else "category"

        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            total_amount = float(window_data["amount"].sum()) if len(window_data) > 0 else 0
            type_map = self._window_type_data.get(window, {})

            # 判断分母有效性
            denominator_total_valid = len(window_data) > 0 and total_amount > 0

            for income_type, entities in entity_pairs:
                if not entities:
                    continue

                type_data = type_map.get(income_type, pd.DataFrame())
                
                # 判断 type 分母有效性
                denominator_type_valid = len(type_data) > 0 and "amount" in type_data.columns
                
                if denominator_type_valid:
                    type_sum = float(type_data["amount"].sum())
                    denominator_type_valid = type_sum > 0  # 重新判断分母是否>0
                else:
                    type_sum = 0

                # 构建 entity_sum_map
                entity_sum_map = {}
                if denominator_type_valid and entity_col in type_data.columns:
                    entity_sum_map = type_data.groupby(entity_col)["amount"].sum().to_dict()

                for entity in entities:
                    cleaned_entity = self._clean_entity_name(entity)
                    
                    # 分子：如果没有数据，设为0
                    entity_sum = entity_sum_map.get(entity, 0)
                    
                    # 计算 type 占比 (entity_sum / type_sum)
                    if not denominator_type_valid:
                        # type 分母无效
                        features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d"] = np.nan
                    else:
                        if entity_sum == 0:
                            # 分子为0
                            features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d"] = 0
                        else:
                            # 正常计算
                            features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d"] = entity_sum / type_sum
                    
                    # 计算 total 占比 (entity_sum / total_amount)
                    if not denominator_total_valid:
                        # total 分母无效
                        features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d"] = np.nan
                    else:
                        if entity_sum == 0:
                            # 分子为0
                            features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d"] = 0
                        else:
                            # 正常计算
                            features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d"] = entity_sum / total_amount

        return features

    # def _generate_entity_ratio_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
    #     features = {}
    #     entity_key = "3rdparty" if entity_type == "tp" else "category"

    #     for window in self.time_windows:
    #         window_data = self._window_data.get(window, pd.DataFrame())

    #         if len(window_data) == 0:
    #             for income_type, entities in entity_pairs:
    #                 if not entities:
    #                     continue
    #                 for entity in entities:
    #                     cleaned_entity = self._clean_entity_name(entity)
    #                     features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d"] = np.nan
    #                     features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d"] = np.nan
    #             continue

    #         total_amount = float(window_data["amount"].sum())
    #         type_map = self._window_type_data.get(window, {})

    #         for income_type, entities in entity_pairs:
    #             if not entities:
    #                 continue

    #             type_data = type_map.get(income_type, pd.DataFrame())
    #             if len(type_data) == 0 or entity_col not in type_data.columns or total_amount <= 0:
    #                 for entity in entities:
    #                     cleaned_entity = self._clean_entity_name(entity)
    #                     features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d"] = np.nan
    #                     features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d"] = np.nan
    #                 continue

    #             type_sum = float(type_data["amount"].sum())
    #             entity_sum_map = type_data.groupby(entity_col)["amount"].sum().to_dict()

    #             for entity in entities:
    #                 cleaned_entity = self._clean_entity_name(entity)
    #                 entity_sum = entity_sum_map.get(entity, np.nan)
    #                 if pd.notna(entity_sum) and type_sum > 0:
    #                     features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d"] = entity_sum / type_sum
    #                     features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d"] = entity_sum / total_amount
    #                 else:
    #                     features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d"] = np.nan
    #                     features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d"] = np.nan

    #     return features

    def _generate_entity_latest_amount_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        features = {}
        entity_key = "3rdparty" if entity_type == "tp" else "category"

        for income_type, entities in entity_pairs:
            if not entities:
                continue

            if entity_type == "tp":
                entity_map = self._type_tp_data.get(income_type, {})
            else:
                entity_map = self._type_category_data.get(income_type, {})

            for entity in entities:
                cleaned_entity = self._clean_entity_name(entity)
                entity_data = entity_map.get(entity, pd.DataFrame())
                if entity_data is not None and len(entity_data) > 0:
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_amount"] = entity_data["amount"].iloc[0]
                else:
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_amount"] = np.nan

        return features
    
    def _generate_entity_amount_comparison_features(
        self,
        entity_type: str,
        entity_col: str,
        entity_pairs: List[List],
        windows: List[int] = [30, 90, 180],
    ) -> Dict:
        features = {}
        entity_key = "3rdparty" if entity_type == "tp" else "category"

        available_windows = [w for w in [30, 90, 180] if w in windows]
        if not available_windows:
            return features

        # 定义对比组合
        comparisons = [
            ('latest_vs_3m', 'latest', 90),
            ('1m_vs_3m', 30, 90),
            ('3m_vs_6m', 90, 180)
        ]

        for income_type, entities in entity_pairs:
            if not entities:
                continue

            if entity_type == "tp":
                entity_map = self._type_tp_data.get(income_type, {})
            else:
                entity_map = self._type_category_data.get(income_type, {})

            for entity in entities:
                cleaned_entity = self._clean_entity_name(entity)
                entity_data = entity_map.get(entity, pd.DataFrame())

                # 计算各窗口平均值和最新值
                window_avgs = {}
                latest = 0
                
                if entity_data is not None and len(entity_data) > 0:
                    for w in available_windows:
                        wd = entity_data[entity_data["trac_days"] <= w]
                        # 分子（窗口平均值）：如果窗口内无数据，设为0
                        window_avgs[w] = float(wd["amount"].mean()) if len(wd) > 0 else 0
                    
                    # 最新值（分子）：如果有数据，取第一条，否则为0
                    latest = entity_data["amount"].iloc[0]
                else:
                    # 没有数据，所有窗口平均值设为0
                    for w in available_windows:
                        window_avgs[w] = 0

                # 计算各个对比特征
                for suffix, num, denom in comparisons:
                    if suffix == 'latest_vs_3m':
                        numerator = latest
                        denominator = window_avgs.get(denom, 0)
                    else:
                        numerator = window_avgs.get(num, 0)
                        denominator = window_avgs.get(denom, 0)
                    
                    # 判断分母是否有效
                    if denominator == 0:
                        # 分母为0（无数据或实际为0），结果为nan
                        features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_{suffix}"] = np.nan
                    elif numerator == 0:
                        # 分子为0，结果为0
                        features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_{suffix}"] = 0
                    else:
                        # 正常计算
                        features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_{suffix}"] = numerator / denominator

        return features

    # def _generate_entity_amount_comparison_features(
    #     self,
    #     entity_type: str,
    #     entity_col: str,
    #     entity_pairs: List[List],
    #     windows: List[int] = [30, 90, 180],
    # ) -> Dict:
    #     features = {}
    #     entity_key = "3rdparty" if entity_type == "tp" else "category"

    #     available_windows = [w for w in [30, 90, 180] if w in windows]
    #     if not available_windows:
    #         return features

    #     for income_type, entities in entity_pairs:
    #         if not entities:
    #             continue

    #         if entity_type == "tp":
    #             entity_map = self._type_tp_data.get(income_type, {})
    #         else:
    #             entity_map = self._type_category_data.get(income_type, {})

    #         for entity in entities:
    #             cleaned_entity = self._clean_entity_name(entity)
    #             entity_data = entity_map.get(entity, pd.DataFrame())

    #             if entity_data is None or len(entity_data) == 0:
    #                 features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_vs_3m"] = np.nan
    #                 features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_1m_vs_3m"] = np.nan
    #                 features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_3m_vs_6m"] = np.nan
    #                 continue

    #             window_avgs = {}
    #             for w in available_windows:
    #                 wd = entity_data[entity_data["trac_days"] <= w]
    #                 window_avgs[w] = float(wd["amount"].mean()) if len(wd) > 0 else np.nan

    #             latest = entity_data["amount"].iloc[0]

    #             if 90 in window_avgs and pd.notna(window_avgs[90]) and window_avgs[90] > 0:
    #                 features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_vs_3m"] = latest / window_avgs[90]
    #             else:
    #                 features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_vs_3m"] = np.nan

    #             if 30 in window_avgs and 90 in window_avgs and pd.notna(window_avgs[30]) and pd.notna(window_avgs[90]) and window_avgs[90] > 0:
    #                 features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_1m_vs_3m"] = window_avgs[30] / window_avgs[90]
    #             else:
    #                 features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_1m_vs_3m"] = np.nan

    #             if 90 in window_avgs and 180 in window_avgs and pd.notna(window_avgs[90]) and pd.notna(window_avgs[180]) and window_avgs[180] > 0:
    #                 features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_3m_vs_6m"] = window_avgs[90] / window_avgs[180]
    #             else:
    #                 features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_3m_vs_6m"] = np.nan

    #     return features

    def _generate_entity_growth_rates_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        features = {}
        entity_key = "3rdparty" if entity_type == "tp" else "category"

        for income_type, entities in entity_pairs:
            if not entities:
                continue

            for entity in entities:
                cleaned_entity = self._clean_entity_name(entity)

                if entity_type == "tp":
                    daily_data = self._type_tp_daily.get((income_type, entity), pd.DataFrame())
                else:
                    daily_data = self._type_category_daily.get((income_type, entity), pd.DataFrame())

                if daily_data is None or len(daily_data) <= 1:
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_max"] = np.nan
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_min"] = np.nan
                    continue

                amounts = daily_data["amount"].to_numpy(dtype=float)
                prev_amounts = amounts[:-1]
                curr_amounts = amounts[1:]
                growth_rates = np.where(prev_amounts != 0, (curr_amounts - prev_amounts) / prev_amounts, 0)

                if len(growth_rates) > 0:
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_max"] = float(np.max(growth_rates))
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_min"] = float(np.min(growth_rates))
                else:
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_max"] = np.nan
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_min"] = np.nan

        return features

    def _generate_entity_max_consecutive_months_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        features = {}
        entity_key = "3rdparty" if entity_type == "tp" else "category"

        for income_type, entities in entity_pairs:
            if not entities:
                continue

            for entity in entities:
                cleaned_entity = self._clean_entity_name(entity)

                if entity_type == "tp":
                    entity_data = self._type_tp_data.get(income_type, {}).get(entity, pd.DataFrame())
                else:
                    entity_data = self._type_category_data.get(income_type, {}).get(entity, pd.DataFrame())

                if entity_data is None or len(entity_data) == 0:
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_max_consecutive_months"] = np.nan
                    continue

                monthly = (
                    entity_data.groupby(entity_data["date"].dt.to_period("M"))["amount"]
                    .sum()
                    .reset_index()
                )
                monthly = monthly[monthly["amount"] > 0]

                if len(monthly) == 0:
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_max_consecutive_months"] = np.nan
                    continue

                monthly["year_month"] = monthly["date"].dt.to_timestamp()
                monthly = monthly.sort_values("year_month").reset_index(drop=True)
                month_nums = (monthly["year_month"].dt.year * 12 + monthly["year_month"].dt.month).to_numpy()

                features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_max_consecutive_months"] = self._max_consecutive(month_nums, diff_target=1)

        return features

    def _generate_entity_interval_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        features = {}
        entity_key = "3rdparty" if entity_type == "tp" else "category"

        for income_type, entities in entity_pairs:
            if not entities:
                continue

            for entity in entities:
                cleaned_entity = self._clean_entity_name(entity)

                if entity_type == "tp":
                    daily_data = self._type_tp_daily.get((income_type, entity), pd.DataFrame())
                else:
                    daily_data = self._type_category_daily.get((income_type, entity), pd.DataFrame())

                if daily_data is None or len(daily_data) < 2:
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_std"] = np.nan
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_max"] = np.nan
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_gap_30_count"] = np.nan
                    continue

                dates = pd.to_datetime(daily_data["date"]).values.astype("datetime64[D]")
                intervals = np.diff(dates) / np.timedelta64(1, "D")

                if len(intervals) > 0:
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_std"] = float(np.std(intervals)) if len(intervals) > 1 else 0.0
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_max"] = float(np.max(intervals))
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_gap_30_count"] = int(np.sum(intervals > 30))
                else:
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_std"] = np.nan
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_max"] = np.nan
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_gap_30_count"] = np.nan

        return features

    def _generate_entity_trend_slope_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        features = {}
        entity_key = "3rdparty" if entity_type == "tp" else "category"

        for window in self.time_windows:
            type_map = self._window_type_data.get(window, {})

            for income_type, entities in entity_pairs:
                if not entities:
                    continue

                type_data = type_map.get(income_type, pd.DataFrame())

                if len(type_data) == 0 or entity_col not in type_data.columns:
                    for entity in entities:
                        cleaned_entity = self._clean_entity_name(entity)
                        features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_trend_slope_{window}d"] = np.nan
                    continue

                for entity in entities:
                    cleaned_entity = self._clean_entity_name(entity)
                    entity_data = type_data[type_data[entity_col] == entity]

                    if len(entity_data) < 2:
                        features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_trend_slope_{window}d"] = np.nan
                        continue

                    daily = (
                        entity_data.groupby("date", as_index=False)["amount"]
                        .sum()
                        .sort_values("date", ascending=True)
                        .reset_index(drop=True)
                    )

                    if len(daily) < 2:
                        features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_trend_slope_{window}d"] = np.nan
                        continue

                    x = (daily["date"] - daily["date"].min()).dt.days.to_numpy(dtype=float)
                    y = daily["amount"].to_numpy(dtype=float)
                    features[f"bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_trend_slope_{window}d"] = self._calc_slope(x, y)

        return features

    # =========================================================
    # 交易对手特征
    # =========================================================
    def third_party_consecutive(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("third_party_consecutive")
            self.features.update(features)
            return features

        features = {}

        for income_type in self.income_types:
            type_data = self._type_data.get(income_type, pd.DataFrame())
            if len(type_data) == 0 or "third_party" not in type_data.columns:
                features[f"bank_txn_income_{income_type}_3rdparty_max_consecutive_days"] = np.nan
                continue

            max_lens = []
            for tp, tp_data in type_data.groupby("third_party", sort=False):
                tmp = (
                    tp_data.groupby("trac_days", as_index=False)["amount"]
                    .sum()
                    .sort_values("trac_days")
                )
                days = tmp["trac_days"].to_numpy()
                if len(days) > 0:
                    max_lens.append(self._max_consecutive(days, diff_target=1))

            features[f"bank_txn_income_{income_type}_3rdparty_max_consecutive_days"] = float(np.max(max_lens)) if max_lens else np.nan

        self.features.update(features)
        return features

    def third_party_consumption_rate(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("third_party_consumption_rate")
            self.features.update(features)
            return features

        features = {}
        consumption_windows = self.time_windows

        for income_type, tps in self.income_type_tp_pairs:
            if not tps:
                continue

            for tp in tps:
                cleaned_tp = self._clean_entity_name(tp)
                daily_data = self._type_tp_daily.get((income_type, tp), pd.DataFrame())

                if daily_data is None or len(daily_data) == 0:
                    features[f"bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_max_daily_consumption_7d"] = np.nan
                    for window in consumption_windows:
                        features[f"bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_max_consumption_rate_{window}d"] = np.nan
                        features[f"bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_min_consumption_rate_{window}d"] = np.nan
                        features[f"bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_mean_consumption_rate_{window}d"] = np.nan
                    continue

                metrics = self._calc_consumption_metrics_from_daily(daily_data, consumption_windows)
                features[f"bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_max_daily_consumption_7d"] = metrics.get("max_daily_consumption_7d", np.nan)

                for window in consumption_windows:
                    features[f"bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_max_consumption_rate_{window}d"] = metrics.get(f"max_consumption_rate_{window}d", np.nan)
                    features[f"bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_min_consumption_rate_{window}d"] = metrics.get(f"min_consumption_rate_{window}d", np.nan)
                    features[f"bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_mean_consumption_rate_{window}d"] = metrics.get(f"mean_consumption_rate_{window}d", np.nan)

        self.features.update(features)
        return features

    def third_party_count(self) -> Dict:
        features = {}

        if self.third_party_col not in self.df.columns:
            return features

        for income_type, _ in self.income_type_tp_pairs:
            type_data = self._type_data.get(income_type, pd.DataFrame())
            if len(type_data) == 0:
                features[f"bank_txn_income_{income_type}_3rdparty_count"] = 0
            else:
                features[f"bank_txn_income_{income_type}_3rdparty_count"] = int(type_data["third_party"].nunique())

        self.features.update(features)
        return features

    def third_party_amount(self) -> Dict:
        features = self._generate_entity_amount_features("tp", "third_party", self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_ratio(self) -> Dict:
        features = self._generate_entity_ratio_features("tp", "third_party", self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_latest_amount(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("third_party_latest_amount")
            self.features.update(features)
            return features

        features = self._generate_entity_latest_amount_features("tp", "third_party", self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_amount_comparison(self, windows: List[int] = [30, 90, 180]) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("third_party_amount_comparison", windows=windows)
            self.features.update(features)
            return features

        features = self._generate_entity_amount_comparison_features("tp", "third_party", self.income_type_tp_pairs, windows)
        self.features.update(features)
        return features

    def third_party_growth_rates(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("third_party_growth_rates")
            self.features.update(features)
            return features

        features = self._generate_entity_growth_rates_features("tp", "third_party", self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_max_consecutive_months(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("third_party_max_consecutive_months")
            self.features.update(features)
            return features

        features = self._generate_entity_max_consecutive_months_features("tp", "third_party", self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_interval(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("third_party_interval")
            self.features.update(features)
            return features

        features = self._generate_entity_interval_features("tp", "third_party", self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_trend_slope(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("third_party_trend_slope")
            self.features.update(features)
            return features

        features = self._generate_entity_trend_slope_features("tp", "third_party", self.income_type_tp_pairs)
        self.features.update(features)
        return features

    # =========================================================
    # 类别特征
    # =========================================================
    def category_amount(self) -> Dict:
        features = self._generate_entity_amount_features("category", "category", self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_ratio(self) -> Dict:
        features = self._generate_entity_ratio_features("category", "category", self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_latest_amount(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("category_latest_amount")
            self.features.update(features)
            return features

        features = self._generate_entity_latest_amount_features("category", "category", self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_amount_comparison(self, windows: List[int] = [30, 90, 180]) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("category_amount_comparison", windows=windows)
            self.features.update(features)
            return features

        features = self._generate_entity_amount_comparison_features("category", "category", self.income_type_category_pairs, windows)
        self.features.update(features)
        return features

    def category_growth_rates(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("category_growth_rates")
            self.features.update(features)
            return features

        features = self._generate_entity_growth_rates_features("category", "category", self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_max_consecutive_months(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("category_max_consecutive_months")
            self.features.update(features)
            return features

        features = self._generate_entity_max_consecutive_months_features("category", "category", self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_interval(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("category_interval")
            self.features.update(features)
            return features

        features = self._generate_entity_interval_features("category", "category", self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_trend_slope(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("category_trend_slope")
            self.features.update(features)
            return features

        features = self._generate_entity_trend_slope_features("category", "category", self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_consumption_rate(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("category_consumption_rate")
            self.features.update(features)
            return features

        features = {}
        consumption_windows = self.time_windows

        for income_type, categories in self.income_type_category_pairs:
            if not categories:
                continue

            for category in categories:
                cleaned_category = self._clean_entity_name(category)
                daily_data = self._type_category_daily.get((income_type, category), pd.DataFrame())

                if daily_data is None or len(daily_data) == 0:
                    features[f"bank_txn_income_{income_type}_category_{cleaned_category}_max_daily_consumption_7d"] = np.nan
                    for window in consumption_windows:
                        features[f"bank_txn_income_{income_type}_category_{cleaned_category}_max_consumption_rate_{window}d"] = np.nan
                        features[f"bank_txn_income_{income_type}_category_{cleaned_category}_min_consumption_rate_{window}d"] = np.nan
                        features[f"bank_txn_income_{income_type}_category_{cleaned_category}_mean_consumption_rate_{window}d"] = np.nan
                    continue

                metrics = self._calc_consumption_metrics_from_daily(daily_data, consumption_windows)
                features[f"bank_txn_income_{income_type}_category_{cleaned_category}_max_daily_consumption_7d"] = metrics.get("max_daily_consumption_7d", np.nan)

                for window in consumption_windows:
                    features[f"bank_txn_income_{income_type}_category_{cleaned_category}_max_consumption_rate_{window}d"] = metrics.get(f"max_consumption_rate_{window}d", np.nan)
                    features[f"bank_txn_income_{income_type}_category_{cleaned_category}_min_consumption_rate_{window}d"] = metrics.get(f"min_consumption_rate_{window}d", np.nan)
                    features[f"bank_txn_income_{income_type}_category_{cleaned_category}_mean_consumption_rate_{window}d"] = metrics.get(f"mean_consumption_rate_{window}d", np.nan)

        self.features.update(features)
        return features

    # =========================================================
    # 原有全局 / type 特征
    # =========================================================
    def amount_global(self) -> Dict:
        features = {}

        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            stats = self._safe_stats(window_data["amount"].to_numpy(dtype=float) if len(window_data) > 0 else None)

            features[f"bank_txn_income_global_sum_{window}d"] = stats["sum"]
            features[f"bank_txn_income_global_count_{window}d"] = stats["count"]
            features[f"bank_txn_income_global_mean_{window}d"] = stats["mean"]
            features[f"bank_txn_income_global_median_{window}d"] = stats["median"]
            features[f"bank_txn_income_global_max_{window}d"] = stats["max"]
            features[f"bank_txn_income_global_min_{window}d"] = stats["min"]
            features[f"bank_txn_income_global_std_{window}d"] = stats["std"]
            features[f"bank_txn_income_global_cv_{window}d"] = stats["cv"]

        self.features.update(features)
        return features

    def amount_by_type(self) -> Dict:
        features = {}

        for window in self.time_windows:
            type_map = self._window_type_data.get(window, {})

            for income_type in self.income_types:
                type_data = type_map.get(income_type, pd.DataFrame())
                stats = self._safe_stats(type_data["amount"].to_numpy(dtype=float) if len(type_data) > 0 else None)

                features[f"bank_txn_income_{income_type}_sum_{window}d"] = stats["sum"]
                features[f"bank_txn_income_{income_type}_count_{window}d"] = stats["count"]
                features[f"bank_txn_income_{income_type}_mean_{window}d"] = stats["mean"]
                features[f"bank_txn_income_{income_type}_median_{window}d"] = stats["median"]
                features[f"bank_txn_income_{income_type}_max_{window}d"] = stats["max"]
                features[f"bank_txn_income_{income_type}_min_{window}d"] = stats["min"]
                features[f"bank_txn_income_{income_type}_std_{window}d"] = stats["std"]
                features[f"bank_txn_income_{income_type}_cv_{window}d"] = stats["cv"]

        self.features.update(features)
        return features

    def latest_amount(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("latest_amount")
            self.features.update(features)
            return features

        features = {}

        for income_type in self.income_types:
            type_data = self._type_data.get(income_type, pd.DataFrame())
            if len(type_data) > 0:
                features[f"bank_txn_income_{income_type}_latest_amount"] = type_data["amount"].iloc[0]
            else:
                features[f"bank_txn_income_{income_type}_latest_amount"] = np.nan

        self.features.update(features)
        return features
    
    def amount_comparison(self, windows: List[int] = [30, 90, 180]) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("amount_comparison", windows=windows)
            self.features.update(features)
            return features

        features = {}
        available_windows = [w for w in [30, 90, 180] if w in windows]
        if not available_windows:
            return features

        # 定义对比组合
        comparisons = [
            ('latest_vs_3m', 'latest', 90),
            ('1m_vs_3m', 30, 90),
            ('3m_vs_6m', 90, 180)
        ]

        for income_type in self.income_types:
            type_data = self._type_data.get(income_type, pd.DataFrame())

            # 计算各窗口平均值和最新值
            window_avgs = {}
            latest = 0
            
            if len(type_data) > 0:
                for w in available_windows:
                    wd = type_data[type_data["trac_days"] <= w]
                    # 分子（窗口平均值）：如果窗口内无数据，设为0
                    window_avgs[w] = float(wd["amount"].mean()) if len(wd) > 0 else 0
                
                # 最新值（分子）：如果有数据，取第一条，否则为0
                latest = type_data["amount"].iloc[0]
            else:
                # 没有数据，所有窗口平均值设为0
                for w in available_windows:
                    window_avgs[w] = 0

            # 计算各个对比特征
            for suffix, num, denom in comparisons:
                if suffix == 'latest_vs_3m':
                    numerator = latest
                    denominator = window_avgs.get(denom, 0)
                else:
                    numerator = window_avgs.get(num, 0)
                    denominator = window_avgs.get(denom, 0)
                
                # 判断分母是否有效
                if denominator == 0:
                    # 分母为0（无数据或实际为0），结果为nan
                    features[f"bank_txn_income_{income_type}_{suffix}"] = np.nan
                elif numerator == 0:
                    # 分子为0，结果为0
                    features[f"bank_txn_income_{income_type}_{suffix}"] = 0
                else:
                    # 正常计算
                    features[f"bank_txn_income_{income_type}_{suffix}"] = numerator / denominator

        self.features.update(features)
        return features

    # def amount_comparison(self, windows: List[int] = [30, 90, 180]) -> Dict:
    #     features = {}
    #     available_windows = [w for w in [30, 90, 180] if w in windows]
    #     if not available_windows:
    #         return features

    #     for income_type in self.income_types:
    #         type_data = self._type_data.get(income_type, pd.DataFrame())

    #         if len(type_data) == 0:
    #             features[f"bank_txn_income_{income_type}_latest_vs_3m"] = np.nan
    #             features[f"bank_txn_income_{income_type}_1m_vs_3m"] = np.nan
    #             features[f"bank_txn_income_{income_type}_3m_vs_6m"] = np.nan
    #             continue

    #         window_avgs = {}
    #         for w in available_windows:
    #             wd = type_data[type_data["trac_days"] <= w]
    #             window_avgs[w] = float(wd["amount"].mean()) if len(wd) > 0 else np.nan

    #         latest = type_data["amount"].iloc[0]

    #         if 90 in window_avgs and pd.notna(window_avgs[90]) and window_avgs[90] > 0:
    #             features[f"bank_txn_income_{income_type}_latest_vs_3m"] = latest / window_avgs[90]
    #         else:
    #             features[f"bank_txn_income_{income_type}_latest_vs_3m"] = np.nan

    #         if 30 in window_avgs and 90 in window_avgs and pd.notna(window_avgs[30]) and pd.notna(window_avgs[90]) and window_avgs[90] > 0:
    #             features[f"bank_txn_income_{income_type}_1m_vs_3m"] = window_avgs[30] / window_avgs[90]
    #         else:
    #             features[f"bank_txn_income_{income_type}_1m_vs_3m"] = np.nan

    #         if 90 in window_avgs and 180 in window_avgs and pd.notna(window_avgs[90]) and pd.notna(window_avgs[180]) and window_avgs[180] > 0:
    #             features[f"bank_txn_income_{income_type}_3m_vs_6m"] = window_avgs[90] / window_avgs[180]
    #         else:
    #             features[f"bank_txn_income_{income_type}_3m_vs_6m"] = np.nan

    #     self.features.update(features)
    #     return features

    def growth_rates(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("growth_rates")
            self.features.update(features)
            return features

        features = {}

        for income_type in self.income_types:
            daily_data = self._type_daily.get(income_type, pd.DataFrame())

            if len(daily_data) <= 1:
                features[f"bank_txn_income_{income_type}_growth_rate_max"] = np.nan
                features[f"bank_txn_income_{income_type}_growth_rate_min"] = np.nan
                continue

            amounts = daily_data["amount"].to_numpy(dtype=float)
            prev_amounts = amounts[:-1]
            curr_amounts = amounts[1:]
            growth_rates = np.where(prev_amounts != 0, (curr_amounts - prev_amounts) / prev_amounts, 0)

            if len(growth_rates) > 0:
                features[f"bank_txn_income_{income_type}_growth_rate_max"] = float(np.max(growth_rates))
                features[f"bank_txn_income_{income_type}_growth_rate_min"] = float(np.min(growth_rates))
            else:
                features[f"bank_txn_income_{income_type}_growth_rate_max"] = np.nan
                features[f"bank_txn_income_{income_type}_growth_rate_min"] = np.nan

        self.features.update(features)
        return features
    
    def ratio(self) -> Dict:
        
        features = {}

        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            # 计算总额（分母）
            total_amount = float(window_data["amount"].sum()) if len(window_data) > 0 else 0.0
            type_map = self._window_type_data.get(window, {})

            # 判断分母是否有效
            denominator_valid = len(window_data) > 0 and total_amount > 0

            for income_type in self.income_types:
                if not denominator_valid:
                    # 分母无效，结果为nan
                    features[f"bank_txn_income_{income_type}_ratio_{window}d"] = np.nan
                    continue

                type_data = type_map.get(income_type, pd.DataFrame())
                
                # 分子：如果没有数据，设为0
                type_sum = float(type_data["amount"].sum()) if len(type_data) > 0 else 0.0
                
                if type_sum == 0:
                    # 分子为0，结果为0
                    features[f"bank_txn_income_{income_type}_ratio_{window}d"] = 0
                else:
                    # 正常计算
                    features[f"bank_txn_income_{income_type}_ratio_{window}d"] = type_sum / total_amount

        self.features.update(features)
        return features

    # def ratio(self) -> Dict:
    #     features = {}

    #     for window in self.time_windows:
    #         window_data = self._window_data.get(window, pd.DataFrame())
    #         total_amount = float(window_data["amount"].sum()) if len(window_data) > 0 else 0.0
    #         type_map = self._window_type_data.get(window, {})

    #         for income_type in self.income_types:
    #             if total_amount <= 0:
    #                 features[f"bank_txn_income_{income_type}_ratio_{window}d"] = np.nan
    #                 continue

    #             type_data = type_map.get(income_type, pd.DataFrame())
    #             type_sum = float(type_data["amount"].sum()) if len(type_data) > 0 else np.nan
    #             features[f"bank_txn_income_{income_type}_ratio_{window}d"] = type_sum / total_amount if pd.notna(type_sum) else np.nan

    #     self.features.update(features)
    #     return features

    def ratio_by_type(self) -> Dict:
        features = {}

        for window in self.time_windows:
            type_map = self._window_type_data.get(window, {})

            for income_type in self.income_types:
                type_data = type_map.get(income_type, pd.DataFrame())

                if len(type_data) == 0:
                    features[f"bank_txn_income_{income_type}_max_ratio_{window}d"] = np.nan
                    features[f"bank_txn_income_{income_type}_min_ratio_{window}d"] = np.nan
                    continue

                type_sum = float(type_data["amount"].sum())
                if type_sum > 0:
                    min_value = float(type_data["amount"].min())
                    max_value = float(type_data["amount"].max())
                    features[f"bank_txn_income_{income_type}_max_ratio_{window}d"] = max_value / type_sum
                    features[f"bank_txn_income_{income_type}_min_ratio_{window}d"] = min_value / type_sum
                else:
                    features[f"bank_txn_income_{income_type}_max_ratio_{window}d"] = np.nan
                    features[f"bank_txn_income_{income_type}_min_ratio_{window}d"] = np.nan

        self.features.update(features)
        return features

    def max_consecutive_months(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("max_consecutive_months")
            self.features.update(features)
            return features

        features = {}

        for income_type in self.income_types:
            type_data = self._type_data.get(income_type, pd.DataFrame())

            if len(type_data) == 0:
                features[f"bank_txn_income_{income_type}_max_consecutive_months"] = np.nan
                continue

            monthly = (
                type_data.groupby(type_data["date"].dt.to_period("M"))["amount"]
                .sum()
                .reset_index()
            )
            monthly = monthly[monthly["amount"] > 0]

            if len(monthly) == 0:
                features[f"bank_txn_income_{income_type}_max_consecutive_months"] = np.nan
                continue

            monthly["year_month"] = monthly["date"].dt.to_timestamp()
            monthly = monthly.sort_values("year_month").reset_index(drop=True)
            month_nums = (monthly["year_month"].dt.year * 12 + monthly["year_month"].dt.month).to_numpy()

            features[f"bank_txn_income_{income_type}_max_consecutive_months"] = self._max_consecutive(month_nums, diff_target=1)

        self.features.update(features)
        return features

    def interval(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("interval")
            self.features.update(features)
            return features

        features = {}

        for income_type in self.income_types:
            daily_data = self._type_daily.get(income_type, pd.DataFrame())

            if len(daily_data) < 2:
                features[f"bank_txn_income_{income_type}_interval_std"] = np.nan
                features[f"bank_txn_income_{income_type}_interval_max"] = np.nan
                features[f"bank_txn_income_{income_type}_interval_gap_30_count"] = np.nan
                continue

            dates = pd.to_datetime(daily_data["date"]).values.astype("datetime64[D]")
            intervals = np.diff(dates) / np.timedelta64(1, "D")

            if len(intervals) > 0:
                features[f"bank_txn_income_{income_type}_interval_std"] = float(np.std(intervals)) if len(intervals) > 1 else 0.0
                features[f"bank_txn_income_{income_type}_interval_max"] = float(np.max(intervals))
                features[f"bank_txn_income_{income_type}_interval_gap_30_count"] = int(np.sum(intervals > 30))
            else:
                features[f"bank_txn_income_{income_type}_interval_std"] = np.nan
                features[f"bank_txn_income_{income_type}_interval_max"] = np.nan
                features[f"bank_txn_income_{income_type}_interval_gap_30_count"] = np.nan

        self.features.update(features)
        return features

    def fluctuation(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("fluctuation")
            self.features.update(features)
            return features

        features = {}

        for income_type in self.income_types:
            daily_data = self._type_daily.get(income_type, pd.DataFrame())

            if len(daily_data) < 2:
                features[f"bank_txn_income_{income_type}_decrease_count"] = np.nan
                continue

            amounts = daily_data["amount"].to_numpy(dtype=float)
            decreases = int(np.sum(amounts[1:] < amounts[:-1]))
            features[f"bank_txn_income_{income_type}_decrease_count"] = decreases

        self.features.update(features)
        return features

    def trend_slope(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("trend_slope")
            self.features.update(features)
            return features

        features = {}

        for window in self.time_windows:
            type_map = self._window_type_data.get(window, {})

            for income_type in self.income_types:
                type_data = type_map.get(income_type, pd.DataFrame())

                if len(type_data) < 2:
                    features[f"bank_txn_income_{income_type}_trend_slope_{window}d"] = np.nan
                    continue

                daily = (
                    type_data.groupby("date", as_index=False)["amount"]
                    .sum()
                    .sort_values("date", ascending=True)
                    .reset_index(drop=True)
                )

                if len(daily) < 2:
                    features[f"bank_txn_income_{income_type}_trend_slope_{window}d"] = np.nan
                    continue

                x = (daily["date"] - daily["date"].min()).dt.days.to_numpy(dtype=float)
                y = daily["amount"].to_numpy(dtype=float)
                features[f"bank_txn_income_{income_type}_trend_slope_{window}d"] = self._calc_slope(x, y)

        self.features.update(features)
        return features

    def consumption_rate(self) -> Dict:
        # ===== [多卡调整] =====
        if self.has_multi_accounts:
            features = self._aggregate_account_method("consumption_rate")
            self.features.update(features)
            return features

        features = {}
        consumption_windows = self.time_windows

        for income_type in self.income_types:
            daily_data = self._type_daily_desc.get(income_type, pd.DataFrame())

            if daily_data is None or len(daily_data) == 0:
                features[f"bank_txn_income_{income_type}_max_daily_consumption_7d"] = np.nan
                for window in consumption_windows:
                    features[f"bank_txn_income_{income_type}_max_consumption_rate_{window}d"] = np.nan
                    features[f"bank_txn_income_{income_type}_min_consumption_rate_{window}d"] = np.nan
                    features[f"bank_txn_income_{income_type}_mean_consumption_rate_{window}d"] = np.nan
                continue

            metrics = self._calc_consumption_metrics_from_daily(daily_data, consumption_windows)
            features[f"bank_txn_income_{income_type}_max_daily_consumption_7d"] = metrics.get("max_daily_consumption_7d", np.nan)

            for window in consumption_windows:
                features[f"bank_txn_income_{income_type}_max_consumption_rate_{window}d"] = metrics.get(f"max_consumption_rate_{window}d", np.nan)
                features[f"bank_txn_income_{income_type}_min_consumption_rate_{window}d"] = metrics.get(f"min_consumption_rate_{window}d", np.nan)
                features[f"bank_txn_income_{income_type}_mean_consumption_rate_{window}d"] = metrics.get(f"mean_consumption_rate_{window}d", np.nan)

        self.features.update(features)
        return features

    # =========================================================
    # 输出
    # =========================================================
    def get_metadata(self) -> Dict:
        return {
            "user_id": self.user_id,
            "sample_datetime": self.sample_datetime,
        }

    def generate_all_features(
        self,
        df: pd.DataFrame = None,
        feature_groups: List[str] = None,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        生成所有特征
        """
        if df is not None:
            self.original_df = df.copy()
            self.df = self._map(df)
            self.raw_df = self.df.copy()
            self._prepare_data()

        if feature_groups is None:
            feature_groups = [
                "amount_global",
                "amount_by_type",
                "latest_amount",
                "amount_comparison",
                "ratio",
                "ratio_by_type",
                "interval",
                "growth_rates",
                "fluctuation",
                "max_consecutive_months",
                "consumption_rate",
                "trend_slope",
                "third_party_count",
                "third_party_consecutive",
                "third_party_amount",
                "third_party_ratio",
                "third_party_latest_amount",
                "third_party_amount_comparison",
                "third_party_interval",
                "third_party_growth_rates",
                "third_party_max_consecutive_months",
                "third_party_consumption_rate",
                "third_party_trend_slope",
                "category_amount",
                "category_ratio",
                "category_latest_amount",
                "category_amount_comparison",
                "category_interval",
                "category_growth_rates",
                "category_max_consecutive_months",
                "category_trend_slope",
                "category_consumption_rate",
            ]

        self.features = {}

        for feature in feature_groups:
            method = getattr(self, feature, None)
            if method is None:
                print(f"警告: feature group 不存在: {feature}")
                continue
            method()

        self.features["user_id"] = self.user_id
        self.features["sample_datetime"] = self.sample_datetime

        if include_metadata:
            metadata = self.get_metadata()
            for key, value in metadata.items():
                if key not in ["user_id", "sample_datetime"]:
                    self.features[f"metadata_{key}"] = value

        out_df = pd.DataFrame([self.features])

        base_columns = ["user_id", "sample_datetime"]
        existing_base = [col for col in base_columns if col in out_df.columns]
        feature_cols = [col for col in out_df.columns if col not in existing_base]
        out_df = out_df[existing_base + feature_cols]

        out_df.columns = out_df.columns.str.strip().str.replace(" ", "", regex=False).str.lower()
        return out_df


# =========================================================
# 配置
# =========================================================
income_type_tp_pairs = [
    ["Wages", []],
    [
        "Centrelink",
        [
            "Centrelink Pension",
            "Family Benefits",
            "JobSeeker",
            "Carers Benefits",
            "Child Support",
            "Youth Allowance",
            # "Parenting Payment",
            # "National Disability Insurance",
            # "Vet Affairs",
            # "Parental Leave Pay",
            # "Child Care Subsidy",
            # "Education Entry Payment",
            # "National Disability Insurance Scheme",
            # "Child Disability Assistance Payment",
            # "Child Disability Assistance Pa",
            # "Other Centrelink/Government Payments",
            # "Other Centrelink/Government Pa",
            # "COLC Concessions",
            # "Mobility Allowance",
            # "Stillborn Pay",
            # "Schoolkids Bonus",
            # "Disability Pension",
            # "Emergency Payment",
        ],
    ],
    ["Other Income", []],
]

income_type_category_pairs = [
    ["Wages", []],
    ["Centrelink", []],
    [
        "Other Income",
        [   
             "All Other Credits",
             "Automotive",
            # "Department Stores",
            # "Dining Out",
            # "Donations",
             "Education",
            # "Entertainment",
             "External Transfers",
            # "Fees",
             "Gambling",
            # "Groceries",
             "Gyms and other memberships",
             "Health",
            # "Home Improvement",
            # "Information",
             "Insurance",
             "Internal Transfer",
            # "Overdrawn",
            # "Personal Care",
            # "Pet Care",
             "Rent",
            # "Retail",
            # "Subscription TV",
             "Telecommunications",
            # "Transport",
             "Travel",
             "Utilities",
        ],
    ],
]


def generate_income_feature(df: pd.DataFrame, feature_groups: List[str] = None):
    income_engineer = SingleApplicationIncomeFeatureEngineer(
        df=df,
        time_windows=[7, 14, 28, 56, 84, 168, 182],
        income_type_tp_pairs=income_type_tp_pairs,
        income_type_category_pairs=income_type_category_pairs,
    )
    return income_engineer.generate_all_features(df=df, feature_groups=feature_groups)