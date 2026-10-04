#!/usr/bin/env python3
"""
Comprehensive test suite for RQ5 predictive analysis.

Includes:
- Unit tests
- Property-based tests
- Metamorphic tests
- Negative/edge case tests
- Integration tests
"""

import json
import logging

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

import datapreprocessing_w5 as dp
import regression_energy as re
import predict_energy_w5 as pe
import predict_energy_report_w5 as per

_LOG = logging.getLogger("rq5_test")

# unit test preprocessing workflow 5


class TestUnitDatapreprocessing:
    """Unit tests for datapreprocessing_w5.py"""

    @pytest.fixture
    def sample_df(self):
        """Fixture: minimal DataFrame with date column."""
        return pd.DataFrame({"date": pd.to_datetime(
            ["2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01"])})

    def test_add_season_column_adds_season_values(self, sample_df):
        result = dp.add_season_column(sample_df.copy())
        assert "season" in result.columns
        assert list(
            result["season"]) == [
            "Winter",
            "Spring",
            "Summer",
            "Autumn"]
        assert result["season"].dtype.name == "category"

    def test_add_season_column_all_months(self):
        """Test season mapping for all 12 months."""
        df = pd.DataFrame({
            "date": pd.to_datetime([f"2021-{m:02d}-01" for m in range(1, 13)])
        })
        result = dp.add_season_column(df.copy())
        expected = [
            "Winter",
            "Winter",
            "Spring",
            "Spring",
            "Spring",
            "Summer",
            "Summer",
            "Summer",
            "Autumn",
            "Autumn",
            "Autumn",
            "Winter"]
        assert list(result["season"]) == expected

    def test_process_csvfile_rqenergy_requires_processed_filename(
            self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text(
            "date,temperature_mean_c,precipitation_mm,sunshine_hours,energy\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="preprocessing"):
            dp.process_csvfile_rqenergy(str(path), _LOG)

    def test_process_csvfile_rqenergy_reads_and_transforms(self, tmp_path):
        path = tmp_path / "processed_data.csv"
        df = pd.DataFrame({
            "date": ["2021-01-01", "2021-04-01", "2021-07-01"],
            "temperature_mean_c": [0.0, 10.0, 20.0],
            "precipitation_mm": [0.0, 1.0, 2.0],
            "sunshine_hours": [2.0, 3.0, 4.0],
            "energy": [0.5, 0.6, 0.7],
            "year": [2021, 2021, 2021],
            "month": [1, 4, 7],
            "day": [1, 1, 1],
            "weekday": [4, 3, 3],
            "is_weekend": [0, 0, 0],
            "season": ["Winter", "Spring", "Summer"],
        })
        df.to_csv(path, index=False)
        result = dp.process_csvfile_rqenergy(str(path), _LOG)
        assert "season" in result.columns
        assert result.loc[0, "season"] == "Winter"
        assert len(result) == 3


# unit test for regression(linearly as temperature and sunshine increase)
class TestUnitRegression:
    """Unit tests for regression_energy.py"""

    @pytest.fixture
    def regression_df(self):
        """Fixture: synthetic dataset with known relationship."""
        rng = np.random.default_rng(7)
        n = 80
        df = pd.DataFrame({
            "temperature_mean_c": rng.uniform(0, 25, n),
            "sunshine_hours": rng.uniform(0, 12, n),
        })
        df["energy"] = 0.35 + 0.01 * df["temperature_mean_c"] + \
            0.02 * df["sunshine_hours"]
        df["energy"] += rng.normal(0, 0.03, n)
        df["energy"] = df["energy"].clip(0.0, 1.0)
        return df

    def test_evaluate_predictive_model_returns_metrics(self, regression_df):
        metrics = re.evaluate_predictive_model(
            regression_df,
            predictors=["temperature_mean_c", "sunshine_hours"],
            target="energy",
            test_size=0.2,
            random_state=42,
        )
        assert list(metrics.columns)[:4] == ["model", "r2", "rmse", "mae"]
        assert metrics.loc[0, "n_train"] > 0
        assert metrics.loc[0, "n_test"] > 0
        assert metrics.loc[0, "r2"] > 0.0
        assert metrics.loc[0, "rmse"] >= 0.0
        assert metrics.loc[0, "mae"] >= 0.0


# unit test for predict model
class TestUnitPredict:
    """Unit tests for predict_energy_w5.py"""

    @pytest.fixture
    def daily_df(self):
        """Fixture: daily data without general columns (for lag testing)."""
        return pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=4),
            "energy": [0.1, 0.2, 0.3, 0.4],
            "temperature_mean_c": [1, 2, 3, 4],
            "sunshine_hours": [1, 2, 3, 4],
        })

    def test_add_persistence_feature_lags_correctly_and_drops_first_row(
            self, daily_df):
        result = pe.add_persistence_feature(daily_df)
        assert len(result) == 3
        assert list(result["energy_lag1"]) == pytest.approx([0.1, 0.2, 0.3])
        assert list(result["energy"]) == pytest.approx([0.2, 0.3, 0.4])

    def test_validate_file_rejects_missing_general_columns(self, tmp_path):
        bad = tmp_path / "bad_processed.csv"
        pd.DataFrame(
            {"date": ["2021-01-01"], "energy": [0.5]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="preprocessing"):
            pe.validate_file(str(bad), _LOG)

        # good file passes
        good = tmp_path / "good_processed.csv"
        pd.DataFrame({"date": ["2021-01-01"],
                      "energy": [0.5],
                      "temperature_mean_c": [5.0],
                      "precipitation_mm": [0.0],
                      "sunshine_hours": [2.0],
                      "year": [2021],
                      "month": [1],
                      "day": [1],
                      "weekday": [4],
                      "is_weekend": [0],
                      "season": ["Winter"],
                      }).to_csv(good,
                                index=False)
        pe.validate_file(str(good), _LOG)  # should not raise

    def test_load_and_aggregate_daily_missing_columns(self, tmp_path):
        path = tmp_path / "bad_processed.csv"
        pd.DataFrame({"date": ["2021-01-01"], "energy": [0.5]}
                     ).to_csv(path, index=False)
        with pytest.raises(KeyError, match="sunshine_hours"):
            pe.load_and_aggregate_daily(str(path), _LOG)

    def test_load_and_aggregate_daily_collapses_tracks_to_one_row_per_date(
            self, tmp_path):
        path = tmp_path / "tracklevel_processed.csv"
        df = pd.DataFrame({
            "date": ["2021-01-01", "2021-01-01", "2021-01-02"],
            "energy": [0.4, 0.6, 0.5],
            "temperature_mean_c": [5.0, 5.0, 6.0],
            "sunshine_hours": [2.0, 2.0, 3.0],
            "year": [2021, 2021, 2021],
            "month": [1, 1, 1],
            "day": [1, 1, 2],
            "weekday": [4, 4, 5],
            "is_weekend": [0, 0, 0],
            "season": ["Winter", "Winter", "Winter"],
        })
        df.to_csv(path, index=False)
        daily = pe.load_and_aggregate_daily(str(path), _LOG)
        assert len(daily) == 2
        assert set(["date", "energy", "temperature_mean_c",
                   "sunshine_hours"]).issubset(daily.columns)
        first_day_energy = daily.loc[daily["date"] ==
                                     pd.Timestamp("2021-01-01"), "energy"].iloc[0]
        assert first_day_energy == pytest.approx(0.5)
        assert daily["date"].is_monotonic_increasing

    def test_run_cv_raises_on_insufficient_data(self):
        daily = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=10),
            "energy": np.linspace(0.4, 0.6, 10),
            "temperature_mean_c": np.linspace(5, 15, 10),
            "sunshine_hours": np.linspace(1, 5, 10),
        })
        daily = pe.add_persistence_feature(daily)
        with pytest.raises(ValueError, match="Insufficient data"):
            pe.run_cv(daily, n_splits=5, gbr_params={}, logger=_LOG)

    def test_summarize_cv_picks_best_model_and_detects_baseline_beat(self):
        fold_results = pd.DataFrame([
            {"fold": 0, "model": "baseline_mean", "n_train": 50, "n_test": 10,
             "r2": -0.5, "rmse": 0.2, "mae": 0.15},
            {"fold": 0, "model": "baseline_persistence", "n_train": 50, "n_test": 10,
             "r2": 0.1, "rmse": 0.15, "mae": 0.1},
            {"fold": 0, "model": "linear_regression", "n_train": 50, "n_test": 10,
             "r2": 0.4, "rmse": 0.1, "mae": 0.08},
            {"fold": 0, "model": "gradient_boosting", "n_train": 50, "n_test": 10,
             "r2": 0.6, "rmse": 0.08, "mae": 0.06},
            {"fold": 1, "model": "baseline_mean", "n_train": 60, "n_test": 10,
             "r2": -0.6, "rmse": 0.21, "mae": 0.16},
            {"fold": 1, "model": "baseline_persistence", "n_train": 60, "n_test": 10,
             "r2": 0.2, "rmse": 0.14, "mae": 0.09},
            {"fold": 1, "model": "linear_regression", "n_train": 60, "n_test": 10,
             "r2": 0.5, "rmse": 0.09, "mae": 0.07},
            {"fold": 1, "model": "gradient_boosting", "n_train": 60, "n_test": 10,
             "r2": 0.65, "rmse": 0.07, "mae": 0.05},
        ])
        summary = pe.summarize_cv(fold_results)
        assert summary["best_model"] == "gradient_boosting"
        assert summary["best_model_r2_mean"] == pytest.approx(0.625, abs=1e-3)
        assert summary["beats_all_baselines"] is True

    def test_summarize_cv_detects_when_baseline_wins(self):
        fold_results = pd.DataFrame([
            {"fold": 0, "model": "baseline_mean", "n_train": 50, "n_test": 10,
             "r2": -1.0, "rmse": 0.3, "mae": 0.2},
            {"fold": 0, "model": "baseline_persistence", "n_train": 50, "n_test": 10,
             "r2": 0.7, "rmse": 0.05, "mae": 0.04},
            {"fold": 0, "model": "linear_regression", "n_train": 50, "n_test": 10,
             "r2": -0.2, "rmse": 0.15, "mae": 0.1},
            {"fold": 0, "model": "gradient_boosting", "n_train": 50, "n_test": 10,
             "r2": 0.1, "rmse": 0.12, "mae": 0.09},
        ])
        summary = pe.summarize_cv(fold_results)
        assert summary["best_model"] == "gradient_boosting"
        assert summary["beats_all_baselines"] is False


# Unit test for predict_energy_report_w5.py

class TestUnitReport:
    """Unit tests for predict_energy_report_w5.py"""

    VALID_SUMMARY = {
        "models": {
            "baseline_mean": {
                "r2_mean": -1.5,
                "r2_std": 1.0,
                "rmse_mean": 0.09,
                "rmse_std": 0.01,
                "mae_mean": 0.07,
                "mae_std": 0.01,
                "n_folds": 4},
            "baseline_persistence": {
                "r2_mean": 0.55,
                "r2_std": 0.1,
                "rmse_mean": 0.04,
                "rmse_std": 0.005,
                "mae_mean": 0.03,
                "mae_std": 0.005,
                "n_folds": 4},
            "linear_regression": {
                "r2_mean": 0.7,
                "r2_std": 0.05,
                "rmse_mean": 0.03,
                "rmse_std": 0.004,
                "mae_mean": 0.025,
                "mae_std": 0.004,
                "n_folds": 4},
            "gradient_boosting": {
                "r2_mean": 0.68,
                "r2_std": 0.08,
                "rmse_mean": 0.032,
                "rmse_std": 0.006,
                "mae_mean": 0.027,
                "mae_std": 0.005,
                "n_folds": 4},
        },
        "best_model": "linear_regression",
        "best_model_r2_mean": 0.7,
        "beats_all_baselines": True,
        "predictors": [
            "temperature_mean_c",
            "sunshine_hours"],
        "target": "energy",
        "n_days_used": 396,
    }

    def test_load_summary_missing_keys_raises(self, tmp_path):
        path = tmp_path / "bad_summary.json"
        path.write_text(json.dumps({"models": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing expected keys"):
            per.load_summary(str(path), _LOG)

    def test_load_summary_valid_returns_dict(self, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(self.VALID_SUMMARY), encoding="utf-8")
        result = per.load_summary(str(path), _LOG)
        assert result["best_model"] == "linear_regression"
        assert result["beats_all_baselines"] is True

    def test_plot_cv_comparison_creates_nonempty_file(self, tmp_path):
        output_path = tmp_path / "cv_plot.png"
        per.plot_cv_comparison(self.VALID_SUMMARY, str(output_path), _LOG)
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_write_summary_reflects_beats_baselines_true(self, tmp_path):
        output_path = tmp_path / "summary.md"
        per.write_summary(self.VALID_SUMMARY, str(output_path), 0.05, _LOG)
        content = output_path.read_text(encoding="utf-8")
        assert "linear_regression" in content
        assert "clearly outperforms" in content

    def test_write_summary_reflects_beats_baselines_false(self, tmp_path):
        losing_summary = dict(self.VALID_SUMMARY)
        losing_summary["beats_all_baselines"] = False
        output_path = tmp_path / "summary_losing.md"
        per.write_summary(losing_summary, str(output_path), 0.05, _LOG)
        content = output_path.read_text(encoding="utf-8")
        assert "does not consistently outperform" in content


class TestPropertyPredictiveModel:
    """Property-based tests for predictive model invariants."""

    @given(
        st.lists(
            st.floats(
                min_value=0,
                max_value=100,
                allow_subnormal=False),
            min_size=20,
            max_size=100),
        st.lists(
            st.floats(
                min_value=0,
                max_value=12,
                allow_subnormal=False),
            min_size=20,
            max_size=100),
    )
    def test_rmse_non_negative(self, temps, sunshine):
        min_len = min(len(temps), len(sunshine))
        df = pd.DataFrame({
            'temperature_mean_c': temps[:min_len],
            'sunshine_hours': sunshine[:min_len],
            'energy': np.random.default_rng(0).uniform(0, 1, min_len)
        })
        if len(df) < 10:
            pytest.skip("Not enough samples")
        metrics = re.evaluate_predictive_model(
            df, predictors=['temperature_mean_c', 'sunshine_hours'],
            target='energy', test_size=0.2, random_state=42
        )
        assert metrics.loc[0, 'rmse'] >= 0
        assert metrics.loc[0, 'mae'] >= 0

    @given(
        st.lists(
            st.floats(
                min_value=0,
                max_value=100,
                allow_subnormal=False),
            min_size=20,
            max_size=100),
        st.lists(
            st.floats(
                min_value=0,
                max_value=12,
                allow_subnormal=False),
            min_size=20,
            max_size=100),
    )
    def test_r2_between_negative_and_one(self, temps, sunshine):
        min_len = min(len(temps), len(sunshine))
        df = pd.DataFrame({
            'temperature_mean_c': temps[:min_len],
            'sunshine_hours': sunshine[:min_len],
            'energy': np.random.default_rng(0).uniform(0, 1, min_len)
        })
        if len(df) < 10:
            pytest.skip("Not enough samples")
        metrics = re.evaluate_predictive_model(
            df, predictors=['temperature_mean_c', 'sunshine_hours'],
            target='energy', test_size=0.2, random_state=42
        )
        assert metrics.loc[0, 'r2'] <= 1

    @given(
        st.lists(
            st.floats(
                min_value=0,
                max_value=100,
                allow_subnormal=False),
            min_size=20,
            max_size=100),
        st.lists(
            st.floats(
                min_value=0,
                max_value=12,
                allow_subnormal=False),
            min_size=20,
            max_size=100),
        st.integers(min_value=10, max_value=90),
    )
    def test_random_splits_have_valid_sample_counts(
            self, temps, sunshine, test_size_pct):
        min_len = min(len(temps), len(sunshine))
        df = pd.DataFrame({
            'temperature_mean_c': temps[:min_len],
            'sunshine_hours': sunshine[:min_len],
            'energy': np.random.default_rng(0).uniform(0, 1, min_len)
        })
        if len(df) < 20:
            pytest.skip("Not enough samples")
        test_size = test_size_pct / 100
        metrics1 = re.evaluate_predictive_model(
            df, predictors=['temperature_mean_c', 'sunshine_hours'],
            target='energy', test_size=test_size, random_state=42
        )
        metrics2 = re.evaluate_predictive_model(
            df, predictors=['temperature_mean_c', 'sunshine_hours'],
            target='energy', test_size=test_size, random_state=123
        )
        assert metrics1.loc[0, 'n_train'] > 0
        assert metrics2.loc[0, 'n_train'] > 0
        assert metrics1.loc[0, 'n_test'] > 0
        assert metrics2.loc[0, 'n_test'] > 0
        assert metrics1.loc[0, 'n_train'] + \
            metrics1.loc[0, 'n_test'] == len(df)
        assert metrics2.loc[0, 'n_train'] + \
            metrics2.loc[0, 'n_test'] == len(df)


class TestMetamorphicPredictive:
    """Metamorphic tests for predictive model invariants."""

    def test_regression_invariant_scaling(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'temperature_mean_c': np.random.uniform(0, 25, 100),
            'sunshine_hours': np.random.uniform(0, 12, 100),
        })
        df['energy'] = 0.35 + 0.01 * df['temperature_mean_c'] + \
            0.02 * df['sunshine_hours']
        df['energy'] += np.random.normal(0, 0.03, 100)
        df['energy'] = df['energy'].clip(0, 1)

        r1 = re.evaluate_predictive_model(
            df, predictors=['temperature_mean_c', 'sunshine_hours'],
            target='energy', test_size=0.2, random_state=42
        )
        df_scaled = df.copy()
        df_scaled['temperature_mean_c'] *= 2
        df_scaled['sunshine_hours'] *= 2
        r2 = re.evaluate_predictive_model(
            df_scaled, predictors=['temperature_mean_c', 'sunshine_hours'],
            target='energy', test_size=0.2, random_state=42
        )
        assert r1.loc[0, 'r2'] == pytest.approx(r2.loc[0, 'r2'], abs=0.01)

    def test_regression_invariant_adding_constant(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'temperature_mean_c': np.random.uniform(0, 25, 100),
            'sunshine_hours': np.random.uniform(0, 12, 100),
        })
        df['energy'] = 0.35 + 0.01 * df['temperature_mean_c'] + \
            0.02 * df['sunshine_hours']
        df['energy'] += np.random.normal(0, 0.03, 100)
        df['energy'] = df['energy'].clip(0, 1)

        r1 = re.evaluate_predictive_model(
            df, predictors=['temperature_mean_c', 'sunshine_hours'],
            target='energy', test_size=0.2, random_state=42
        )
        df_shifted = df.copy()
        df_shifted['temperature_mean_c'] += 100
        df_shifted['sunshine_hours'] += 100
        r2 = re.evaluate_predictive_model(
            df_shifted, predictors=['temperature_mean_c', 'sunshine_hours'],
            target='energy', test_size=0.2, random_state=42
        )
        assert r1.loc[0, 'r2'] == pytest.approx(r2.loc[0, 'r2'], abs=0.01)

    def test_regression_invariant_predictor_reordering(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'temperature_mean_c': np.random.uniform(0, 25, 100),
            'sunshine_hours': np.random.uniform(0, 12, 100),
        })
        df['energy'] = 0.35 + 0.01 * df['temperature_mean_c'] + \
            0.02 * df['sunshine_hours']
        df['energy'] += np.random.normal(0, 0.03, 100)
        df['energy'] = df['energy'].clip(0, 1)

        r1 = re.evaluate_predictive_model(
            df, predictors=['temperature_mean_c', 'sunshine_hours'],
            target='energy', test_size=0.2, random_state=42
        )
        r2 = re.evaluate_predictive_model(
            df, predictors=['sunshine_hours', 'temperature_mean_c'],
            target='energy', test_size=0.2, random_state=42
        )
        assert r1.loc[0, 'r2'] == pytest.approx(r2.loc[0, 'r2'], abs=1e-10)

# Negative test cases for edge cases and invalid inputs


class TestNegativeCases:
    """Negative tests for edge cases and invalid inputs."""

    def test_process_rejects_unprocessed_input(self, tmp_path):
        csv_path = tmp_path / "rq5_raw.csv"
        pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=10).strftime("%Y-%m-%d"),
            "energy": np.linspace(0.4, 0.6, 10),
            "temperature_mean_c": np.linspace(5, 15, 10),
            "sunshine_hours": np.linspace(1, 5, 10),
        }).to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="preprocessing"):
            pe.process(str(csv_path), n_splits=4, gbr_params={}, logger=_LOG)

    def test_validate_file_fails_on_missing_columns(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"date": ["2021-01-01"]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="preprocessing"):
            pe.validate_file(str(bad), _LOG)

# Integration tests


def _make_track_level_csv(path, n_days=60, tracks_per_day=5, seed=1,
                          weather_effect=True, autocorrelated=False):
    """Helper to generate synthetic track-level CSV."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days)
    day_idx = np.arange(n_days)
    temp = 10 + 5 * np.sin(2 * np.pi * day_idx / 365) + \
        rng.normal(0, 2, n_days)
    sunshine = np.clip(4 + 3 * np.sin(2 * np.pi * day_idx / 365)
                       + rng.normal(0, 1, n_days), 0, 16)
    if autocorrelated:
        noise = np.zeros(n_days)
        for i in range(1, n_days):
            noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.03)
    else:
        noise = rng.normal(0, 0.03, n_days)
    if weather_effect:
        daily_energy = np.clip(
            0.4 +
            0.01 *
            temp +
            0.015 *
            sunshine +
            noise,
            0,
            1)
    else:
        daily_energy = np.clip(0.5 + noise, 0, 1)
    rows = []
    for i, d in enumerate(dates):
        for t in range(tracks_per_day):
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "track_id": f"track_{t}",
                "energy": np.clip(daily_energy[i] + rng.normal(0, 0.02), 0, 1),
                "temperature_mean_c": temp[i],
                "sunshine_hours": sunshine[i],
                "year": d.year,
                "month": d.month,
                "day": d.day,
                "weekday": d.weekday(),
                "is_weekend": int(d.weekday() >= 5),
                "season": "Winter",
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


class TestIntegration:
    """Integration tests: run full pipeline and check outputs."""

    def test_process_end_to_end_smoke(self, tmp_path):
        csv_path = tmp_path / "rq5_smoke_processed.csv"
        _make_track_level_csv(
            csv_path,
            n_days=80,
            weather_effect=True,
            autocorrelated=False)
        fold_results, summary = pe.process(
            str(csv_path), n_splits=4, gbr_params={}, logger=_LOG)
        assert isinstance(fold_results, pd.DataFrame)
        assert set(
            fold_results["model"].unique()) == {
            "baseline_mean",
            "baseline_persistence",
            "linear_regression",
            "gradient_boosting"}
        assert summary["n_days_used"] > 0
        assert summary["best_model"] in {
            "linear_regression", "gradient_boosting"}
