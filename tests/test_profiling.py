"""画像统计回归测试；所有输入均为合成数据，无客户样本依赖。"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import generate_report as report
import profile_analysis as pa


def fixture(rows=12):
    columns = {s["column"] for s in pa.FEATURES}
    for spec in pa.DERIVED:
        columns.update(spec["sources"])
    columns.update(pa.INCOME_ANALYSIS_COLUMNS)
    columns.update({"bank_txn_lender_disburse_competitor_example_count",
                    "bank_txn_lender_disburse_competitor_example_amount_sum"})
    data = {c: np.arange(1, rows + 1, dtype=float) for c in sorted(columns)}
    data.update({"txn_raw_input_cnt": np.ones(rows), "user_id": [f"u{i:03}" for i in range(rows)],
                 "application_id": [f"a{i:03}" for i in range(rows)],
                 "duedate_1m_5": np.arange(rows) % 2})
    return pd.DataFrame(data)


class ProfilingRegressionTests(unittest.TestCase):
    def run_fixture(self, frame, directory, **overrides):
        root = Path(directory)
        path = root / "input.csv"
        frame.to_csv(path, index=False)
        cfg = pa.load_config(overrides={
            "input_path": str(path), "output_dir": str(root / "out"),
            "cache_dir": str(root / "cache"), "export_user_scores": True,
            "allow_missing_features": True, "min_feature_valid": 2,
            "chunk_size": 5, **overrides,
        })
        with contextlib.redirect_stdout(io.StringIO()):
            pa.run_analysis(cfg)
        return pa.load_intermediates(cfg["output_dir"])

    def test_missing_registry_sources_are_skipped_independently(self):
        header = [c for c in fixture().columns if "competitor" not in c and c != pa.UNRESOLVED_COLUMNS[0]]
        _, derived, missing = pa.resolve_registry(header, {"allow_missing_features": True})
        keys = {s["key"] for s in derived}
        self.assertIn("db_lender_concentration", keys)
        self.assertNotIn("db_active_lender_cnt", keys)
        self.assertNotIn("db_unresolved_cnt", keys)
        self.assertTrue(missing)

    def test_sentinels_nonfinite_and_real_negative_amounts(self):
        got = pa.clean_numeric(pd.Series([-999999.99999, -2000000, np.inf, -np.inf]), "std")
        np.testing.assert_allclose(got, [np.nan, -2000000, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(pa.clean_source_numeric(pd.Series([-1, np.nan, 3]),
                                   "bank_txn_lender_disburse_amount_sum_l28d"), [0, np.nan, 3], equal_nan=True)
        self.assertTrue(np.isnan(pa.clean_source_numeric(pd.Series([-1]),
                                  "bank_txn_lender_disburse_single_amount_max")[0]))

    def test_partial_aggregates_remain_unknown(self):
        gate = np.array([True, True, True, False])
        source = {"x": np.array([np.nan, 1, 0, 4.]), "y": np.array([np.nan, np.nan, 0, 2.])}
        for kind in ("sum", "max", "count_positive"):
            result = pa.compute_derived({"kind": kind, "sources": ["x", "y"]}, source, gate)
            np.testing.assert_allclose(result, [np.nan, np.nan, 0, np.nan], equal_nan=True)

    def test_trend_uses_nonoverlapping_daily_windows(self):
        spec = next(s for s in pa.DERIVED if s["key"] == "tr_income_daily_ratio")
        source = {spec["numerator"]: np.array([560., 0., 100., 100.]),
                  spec["denominator"]: np.array([2100., 1540., 100., 50.])}
        actual = pa.compute_derived(spec, source, np.ones(4, dtype=bool))
        np.testing.assert_allclose(actual, [2., 0., np.nan, np.nan], equal_nan=True)

    def test_target_invalid_values_do_not_dilute_rate(self):
        stats = pa.TargetStats()
        stats.add(np.array([0, 1, -1, 2, np.nan, np.inf]))
        self.assertEqual(stats.as_dict()["valid_n"], 2)
        self.assertEqual(stats.as_dict()["rate"], 0.5)
        self.assertEqual(stats.as_dict()["other"], 2)
        custom = pa.TargetStats(2)
        custom.add(np.array([0, 2, 1, -1]))
        self.assertEqual(custom.as_dict()["rate"], 0.5)
        with self.assertRaises(ValueError):
            pa.load_config(overrides={"target_positive_value": 0})

    def test_histogram_excludes_tails_as_report_describes(self):
        for scale in ("linear", "log"):
            rows, meta = pa.make_histogram(np.arange(100.), {"key": "x", "hist_scale": scale}, 10)
            self.assertEqual(sum(row["count"] for row in rows), 98)
            self.assertEqual(meta["n_out_of_range"], 2)

    def test_unique_missing_ids_and_cap(self):
        tracker = pa.UniqueTracker(2)
        tracker.add([pd.NA, None, "", "u1", "u2", "u1"])
        self.assertEqual(tracker.n, 2)
        self.assertFalse(tracker.capped)
        tracker.add(["u3"])
        self.assertTrue(tracker.capped)

    def test_scoring_raw_smd_and_target_counts_end_to_end(self):
        frame = fixture()
        frame.loc[10, "txn_raw_input_cnt"] = 0  # 特征存在仍须服从门槛
        frame.loc[11, "duedate_1m_5"] = -1
        with tempfile.TemporaryDirectory() as directory:
            data = self.run_fixture(frame, directory, score_low=25, score_high=80,
                                    target_column="duedate_1m_5")
            users = pd.read_csv(Path(directory) / "out" / "user_scores.csv")
            self.assertTrue(pd.isna(users.loc[10, "health_score"]))
            self.assertEqual(users.loc[10, "risk_flag_count"], 0)
            self.assertEqual(users.loc[10, "segment"], "数据不足")
            self.assertTrue(np.isfinite(users.loc[0, "score_debt"]))
            debt = data["dimension_summary"].set_index("dimension").loc["debt"]
            header = list(frame.columns)
            features, derived, _ = pa.resolve_registry(header, pa.load_config())
            all_specs = features + derived
            arrays, _ = pa.build_arrays(frame, features, derived, frame.txn_raw_input_cnt.to_numpy() >= 1)
            bp = json.loads((Path(directory) / "out" / "feature_breakpoints.json").read_text())["features"]
            expected = pa.compute_scores(arrays, all_specs, bp, pa.load_config())[1]["debt"]
            np.testing.assert_allclose(users.score_debt, np.round(expected, 2), equal_nan=True)
            self.assertEqual(int(debt.features_scored), sum(s["in_score"] and s["key"] in bp for s in all_specs if s["dimension"] == "debt"))
            smd = data["target_feature_smd"].set_index("key").loc["inc_salary_interval_std"]
            self.assertAlmostEqual(smd.mean_pos, frame.loc[(frame.duedate_1m_5 == 1) & (frame.txn_raw_input_cnt >= 1),
                                                         "bank_txn_worthiness_salary_interval_std_days_182d"].mean())
            self.assertEqual(data["target"]["valid_n"], 11)
            self.assertEqual(data["segment_summary"].target_matured.sum(), 11)
            obs = data["observation_bands"]
            self.assertTrue((obs[obs.segment == "全部"].groupby("key").users.sum() == 11).all())
            self.assertIn("低分 (&lt;25)", report.build_html(data))

    def test_zero_gate_missing_target_and_sparse_columns_render(self):
        frames = [fixture().assign(txn_raw_input_cnt=0).drop(columns="duedate_1m_5"),
                  pd.DataFrame({"txn_raw_input_cnt": [1, 0], "bank_txn_income_global_sum_182d": [10, 100]})]
        for frame in frames:
            with self.subTest(columns=len(frame.columns)), tempfile.TemporaryDirectory() as directory:
                data = self.run_fixture(frame, directory)
                self.assertEqual(data["segment_summary"].set_index("segment").loc["数据不足", "users"], len(frame))
                self.assertIn("<!DOCTYPE html>", report.build_html(data))
                self.assertEqual(data["target"].get("valid_n", 0), 0)
                flags = data["risk_flags"].set_index("flag")
                self.assertEqual(flags.loc["flag_no_wage", "users"], 0)
                self.assertEqual(flags.loc["flag_no_income", "users"], 0)
                for path in (Path(directory) / "out").glob("*.json"):
                    json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda s: self.fail(f"非法JSON常量: {s}"))

    def test_gzip_input_cache_and_chunk_sizes_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = fixture()
            first = self.run_fixture(frame, directory, subset_cache="on")
            with contextlib.redirect_stdout(io.StringIO()):
                pa.run_analysis(pa.load_config(overrides=first["run_meta"]["config"]))
            reused = pa.load_intermediates(str(root / "out"))
            pd.testing.assert_frame_equal(first["feature_stats"], reused["feature_stats"])
            compressed = root / "input.csv.gz"
            frame.to_csv(compressed, index=False, compression="gzip")
            second = self.run_fixture(frame, directory, input_path=str(compressed), subset_cache="off", chunk_size=3)
            pd.testing.assert_frame_equal(first["segment_summary"], second["segment_summary"])

    def test_missing_gate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "缺少流水门槛列"):
                self.run_fixture(fixture().drop(columns="txn_raw_input_cnt"), directory)


if __name__ == "__main__":
    unittest.main()
