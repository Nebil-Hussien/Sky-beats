#!/usr/bin/env python3
"""
Comprehensive test suite for RQ4 extreme temperature analysis.
Includes:
- Unit tests
- Property-based
- Metamorphic tests for temperature deviation
- Integration tests for the full preprocessing pipeline
- Negative tests for edge cases and invalid inputs
"""

import logging
from datetime import datetime
from unittest.mock import Mock

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st

# Import functions directly (matching test_rq2 style)
from datapreprocessing_w4 import (
    add_temperature_deviation,
    flag_extreme_deviation,
    add_hotcold_binary,
    add_period_columns,
    process_csvfile_w4,
    DEVIATION_LABELS,
)

_LOG = logging.getLogger("rq4_test")

# Unit tests


class TestUnitTemperature:
    """Unit tests for temperature deviation functions (imported directly)."""

    @pytest.fixture
    def sample_df(self):
        """Fixture: synthetic DataFrame with required columns."""
        return pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=10),
            'temperature_mean_c': np.random.uniform(-5, 30, 10),
            'acousticness': np.random.uniform(0, 1, 10),
            'energy': np.random.uniform(0, 1, 10),
            'tempo': np.random.uniform(60, 180, 10),
            'valence': np.random.uniform(0, 1, 10),
            'danceability': np.random.uniform(0, 1, 10),
        })

    def test_add_temperature_deviation_output_shape(self, sample_df):
        """Test that deviation column is added and row count preserved."""
        df = add_temperature_deviation(sample_df.copy())
        assert 'temperature_deviation' in df.columns
        assert len(df) == len(sample_df)

    def test_add_temperature_deviation_calculation(self):
        """Test temperature deviation calculation with known values."""
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=3),
            "temperature_mean_c": [0.0, 1.0, 2.0]
        })
        result = add_temperature_deviation(df.copy())
        expected_dev = [-1.0, 0.0, 1.0]
        np.testing.assert_array_almost_equal(
            result["temperature_deviation"].values,
            expected_dev
        )

    def test_flag_extreme_deviation_output_shape(self, sample_df):
        """Test that extreme column is added and row count preserved."""
        df = add_temperature_deviation(sample_df.copy())
        result = flag_extreme_deviation(df.copy(), low_pct=25, high_pct=75)
        assert 'temperature_deviation_extreme' in result.columns
        assert len(result) == len(df)

    def test_flag_extreme_deviation_labels(self):
        """Test that extreme labels are from the expected set."""
        df = pd.DataFrame(
            {"temperature_deviation": [-2.0, -1.0, 0.0, 1.0, 2.0]})
        result = flag_extreme_deviation(df.copy(), low_pct=20, high_pct=80)
        cats = result["temperature_deviation_extreme"].cat.categories
        assert set(cats) == set(DEVIATION_LABELS)

    def test_flag_extreme_deviation_constant_data(self):
        """Test that constant deviation assigns all rows to 'Normal'."""
        df = pd.DataFrame({"temperature_deviation": [0.0, 0.0, 0.0, 0.0, 0.0]})
        result = flag_extreme_deviation(df.copy(), low_pct=25, high_pct=75)
        assert (result["temperature_deviation_extreme"] == "Normal").all()

    def test_add_hotcold_binary(self):
        """Test binary mapping of extremes."""
        df = pd.DataFrame({"temperature_deviation_extreme": [
                          "Cold extreme", "Normal", "Hot extreme"]})
        result = add_hotcold_binary(df.copy())
        col = "temperature_deviation_extreme_hotcold"
        assert col in result.columns
        assert result.loc[0, col] == 0
        assert pd.isna(result.loc[1, col])
        assert result.loc[2, col] == 1

    def test_add_period_columns(self, sample_df):
        """Test that week_label and month_label are added correctly."""
        df = add_period_columns(sample_df.copy())
        assert 'week_label' in df.columns
        assert 'month_label' in df.columns
        assert df['week_label'].str.match(r'\d{4}-W\d{2}').all()
        assert df['month_label'].str.match(r'\d{4}-\d{2}').all()

    def test_process_csvfile_w4_validation(self, tmp_path, sample_df):
        """Test that process rejects files without general preprocessing."""
        bad_path = tmp_path / "bad.csv"
        sample_df[['date', 'temperature_mean_c']].to_csv(bad_path, index=False)
        mock_logger = Mock()

        with pytest.raises(ValueError, match="missing general preprocessing"):
            process_csvfile_w4(
                str(bad_path),
                low_pct=25.0,
                high_pct=75.0,
                logger=mock_logger)

    def test_process_csvfile_w4_reads_and_transforms(self, tmp_path):
        """Test full preprocessing pipeline with valid data."""
        path = tmp_path / "processed_data.csv"
        df = pd.DataFrame({
            "date": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "temperature_mean_c": [0.0, 1.0, 2.0],
            "acousticness": [0.1, 0.2, 0.3],
            "energy": [0.5, 0.6, 0.7],
            "tempo": [100.0, 110.0, 120.0],
            "valence": [0.4, 0.5, 0.6],
            "danceability": [0.7, 0.8, 0.9],
            "year": [2021, 2021, 2021],
            "month": [1, 1, 1],
            "day": [1, 2, 3],
            "weekday": [4, 5, 6],
            "is_weekend": [0, 1, 1],
            "season": ["Winter", "Winter", "Winter"],
        })
        df.to_csv(path, index=False)

        result = process_csvfile_w4(
            str(path),
            low_pct=25.0,
            high_pct=75.0,
            logger=_LOG)

        expected_cols = [
            "temperature_deviation",
            "temperature_deviation_extreme",
            "temperature_deviation_extreme_hotcold",
            "week_label",
            "month_label",
        ]
        for col in expected_cols:
            assert col in result.columns
        assert len(result) == 3


# propertty based tests
def nonconstant_list(min_size=10, max_size=100):
    return st.lists(
        st.floats(
            min_value=-30,
            max_value=50,
            allow_nan=False,
            allow_infinity=False),
        min_size=min_size,
        max_size=max_size
    ).filter(lambda lst: len(set(lst)) >= 2)


class TestPropertyTemperatureDeviation:
    """Property-based tests for temperature deviation."""

    @given(
        nonconstant_list(),
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2025, 12, 31)
        ),
    )
    def test_deviation_mean_zero_per_month(self, temps, start_date):
        """Property: Mean of deviations for each month should be approximately 0."""
        df = pd.DataFrame({
            'date': pd.date_range(start_date, periods=len(temps)),
            'temperature_mean_c': temps
        })
        df = add_temperature_deviation(df.copy())
        monthly_means = df.groupby(df['date'].dt.month)[
            'temperature_deviation'].mean()
        # For months with data, mean deviation should be close to 0
        assert (monthly_means.abs() < 1e-6).all()

    @given(
        nonconstant_list(),
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2025, 12, 31)
        ),
    )
    def test_deviation_preserves_row_count(self, temps, start_date):
        """Property: add_temperature_deviation preserves row count."""
        df = pd.DataFrame({
            'date': pd.date_range(start_date, periods=len(temps)),
            'temperature_mean_c': temps
        })
        result = add_temperature_deviation(df.copy())
        assert len(result) == len(df)
        assert 'temperature_deviation' in result.columns

    @given(
        nonconstant_list(),
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2025, 12, 31)
        ),
    )
    def test_deviation_equals_temperature_minus_monthly_mean(
            self, temps, start_date):
        """Property: deviation = temperature - monthly mean."""
        df = pd.DataFrame({
            'date': pd.date_range(start_date, periods=len(temps)),
            'temperature_mean_c': temps
        })
        df = add_temperature_deviation(df.copy())
        # For each row, deviation should equal temp - monthly_mean
        monthly_means = df.groupby(df['date'].dt.month)[
            'temperature_mean_c'].transform('mean')
        np.testing.assert_array_almost_equal(
            df['temperature_deviation'].values,
            (df['temperature_mean_c'] - monthly_means).values,
            decimal=6
        )


class TestMetamorphicTemperature:
    """Metamorphic tests for temperature deviation."""

    def test_deviation_invariant_scaling(self):
        """Adding a constant to all temperatures should not change deviations."""
        np.random.seed(42)
        df = pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=10),
            'temperature_mean_c': np.random.uniform(-5, 30, 10)
        })
        df_orig = add_temperature_deviation(df.copy())

        # Add constant to all temperatures
        df_shifted = df.copy()
        df_shifted['temperature_mean_c'] = df_shifted['temperature_mean_c'] + 10
        df_shifted = add_temperature_deviation(df_shifted)

        # Deviations should be identical
        np.testing.assert_array_almost_equal(
            df_orig['temperature_deviation'].values,
            df_shifted['temperature_deviation'].values
        )

    def test_deviation_invariant_reordering(self):
        """MR: Reordering rows should not change deviations."""
        np.random.seed(42)
        df = pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=10),
            'temperature_mean_c': np.random.uniform(-5, 30, 10)
        })
        df_orig = add_temperature_deviation(df.copy())

        # Shuffle rows
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df_shuffled = add_temperature_deviation(df_shuffled)

        # Deviations should be identical (sorted by date)
        df_orig_sorted = df_orig.sort_values('date').reset_index(drop=True)
        df_shuffled_sorted = df_shuffled.sort_values(
            'date').reset_index(drop=True)

        np.testing.assert_array_almost_equal(
            df_orig_sorted['temperature_deviation'].values,
            df_shuffled_sorted['temperature_deviation'].values
        )


class TestNegativeCases:
    """Negative tests for edge cases and invalid inputs."""

    def test_process_constant_temperature(self, tmp_path):
        """Test that processing constant temperature values works."""
        path = tmp_path / "constant_data.csv"
        df = pd.DataFrame({
            "date": pd.date_range("2021-01-01", periods=10),
            "temperature_mean_c": [20.0] * 10,
            "acousticness": [0.5] * 10,
            "energy": [0.5] * 10,
            "tempo": [120.0] * 10,
            "valence": [0.5] * 10,
            "danceability": [0.5] * 10,
            "year": [2021] * 10,
            "month": list(range(1, 11)),
            "day": list(range(1, 11)),
            "weekday": [0] * 10,
            "is_weekend": [0] * 10,
            "season": ["Winter"] * 10,
        })
        df.to_csv(path, index=False)

        result = process_csvfile_w4(
            str(path),
            low_pct=25.0,
            high_pct=75.0,
            logger=_LOG)
        assert 'temperature_deviation_extreme' in result.columns
        # All deviations should be 0
        assert (result['temperature_deviation'].abs() < 1e-6).all()
        # All rows should be classified as "Normal"
        assert (result['temperature_deviation_extreme'] == "Normal").all()

    def test_process_missing_general_columns(self, tmp_path):
        """Test that process rejects files missing general preprocessing columns."""
        path = tmp_path / "missing_columns.csv"
        df = pd.DataFrame({
            "date": ["2021-01-01"],
            "temperature_mean_c": [20.0],
            "acousticness": [0.5],
            "energy": [0.5],
            "tempo": [120.0],
            "valence": [0.5],
            "danceability": [0.5],
        })
        df.to_csv(path, index=False)
        mock_logger = Mock()
        with pytest.raises(ValueError, match="missing general preprocessing"):
            process_csvfile_w4(
                str(path),
                low_pct=25.0,
                high_pct=75.0,
                logger=mock_logger)

    def test_process_empty_file(self, tmp_path):
        """Test that process rejects empty CSV files."""
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        mock_logger = Mock()
        with pytest.raises(ValueError, match="Could not read"):
            process_csvfile_w4(
                str(path),
                low_pct=25.0,
                high_pct=75.0,
                logger=mock_logger)

    def test_flag_extreme_deviation_all_equal(self):
        """Test that flag_extreme_deviation handles all-equal data gracefully."""
        df = pd.DataFrame({"temperature_deviation": [0.0, 0.0, 0.0, 0.0, 0.0]})
        result = flag_extreme_deviation(df.copy(), low_pct=25, high_pct=75)
        assert (result["temperature_deviation_extreme"] == "Normal").all()

    def test_flag_extreme_deviation_insufficient_variation(self):
        """Test that flag_extreme_deviation handles near-constant data."""
        df = pd.DataFrame(
            {"temperature_deviation": [0.0, 0.0, 0.0, 0.0, 0.001]})
        result = flag_extreme_deviation(df.copy(), low_pct=25, high_pct=75)
        assert "temperature_deviation_extreme" in result.columns
