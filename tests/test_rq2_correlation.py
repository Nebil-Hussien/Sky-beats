#!/usr/bin/env python3
"""
A comprehensive test suite for RQ2 correlation analysis and preprocessing.
It Includes:
- Unit tests for core functions
- Property-based tests in a form of hypothesis for statistical invariants
- Metamorphic tests for Spearman and Kruskal-Wallis
- Integration tests for the full reporting pipeline
- Negative tests for edge cases and invalid inputs
The categories of test taken was mostly based on the Advanced Testing Class only specific model tests (Spearman and Kruskal-Wallis).
"""

import sys
import subprocess
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings, strategies as st

from datapreprocessing_w2 import (
    create_precipitation_bins_percentile,
    add_period_columns,
    process_csvfile_rq2,
    PRECIPITATION_AMOUNT_LABELS,
)
from spearman_correlation import compute_spearman
from kruskal_wallis import run_kruskal_wallis, group_summary

# Paths
SRC_DIR = Path(__file__).parent.parent / "src" / "rq2"

# unit tests


class TestUnitPreprocessing:
    """Unit tests for RQ2 preprocessing functions."""
    @pytest.fixture
    def sample_df(self):
        """Fixture: synthetic DataFrame with required columns."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20),
            'precipitation_mm': [0, 1, 2, 3, 5, 10] * 3 + [0, 0],
            'acousticness': np.random.uniform(0, 1, 20),
            'energy': np.random.uniform(0, 1, 20),
            'tempo': np.random.uniform(60, 180, 20),
            'valence': np.random.uniform(0, 1, 20),
            'danceability': np.random.uniform(0, 1, 20),
        })

    def test_create_precipitation_bins_percentile_output_shape(
            self, sample_df):
        """Test that the output DataFrame has the correct shape."""
        df = create_precipitation_bins_percentile(
            sample_df, 'precipitation_mm', p1=25, p2=50, p3=75)
        assert 'precipitation_mm_cat' in df.columns
        assert len(df) == len(sample_df)

    def test_create_precipitation_bins_percentile_labels(self, sample_df):
        """Test that the precipitation categories are within the expected labels."""
        df = create_precipitation_bins_percentile(
            sample_df, 'precipitation_mm', p1=25, p2=50, p3=75)
        valid_labels = set(PRECIPITATION_AMOUNT_LABELS)
        actual_labels = set(df['precipitation_mm_cat'].unique())
        assert actual_labels.issubset(valid_labels)

    def test_create_precipitation_bins_percentile_bucket_order(
            self, sample_df):
        """Test that the buckets are ordered correctly according to the percentiles."""
        df = create_precipitation_bins_percentile(
            sample_df, 'precipitation_mm', p1=25, p2=50, p3=75)
        categories = df['precipitation_mm_cat'].cat.categories.tolist()
        assert categories == ['No rain', 'Low', 'Moderate', 'High']

    def test_add_period_columns(self, sample_df):
        """Test that week and month labels are added correctly."""
        df = add_period_columns(sample_df)
        assert 'week_label' in df.columns
        assert 'month_label' in df.columns
        assert df['week_label'].str.match(r'\d{4}-W\d{2}').all()
        assert df['month_label'].str.match(r'\d{4}-\d{2}').all()

    def test_process_csvfile_rq2_validation(self, tmp_path, sample_df):
        """Test that process_csvfile_rq2 raises ValueError for missing required columns."""
        bad_path = tmp_path / "bad.csv"
        sample_df[['date', 'precipitation_mm']].to_csv(bad_path, index=False)
        mock_logger = Mock()
        with pytest.raises(ValueError, match="missing columns"):
            process_csvfile_rq2(
                str(bad_path),
                p1=25,
                p2=50,
                p3=75,
                logger=mock_logger)


class TestUnitSpearman:
    """Unit tests for Spearman correlation functions."""

    def test_compute_spearman_basic(self):
        """Test that compute_spearman returns expected rho and direction for a simple positive correlation."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = compute_spearman(
            x, y, alpha=0.05, confidence=0.95, min_sample_size=2)
        assert result['rho'] == pytest.approx(1.0, abs=0.001)
        assert result['direction'] == 'positive'

    def test_compute_spearman_negative(self):
        """Test that compute_spearman returns expected rho and direction for a simple negative correlation."""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        result = compute_spearman(
            x, y, alpha=0.05, confidence=0.95, min_sample_size=2)
        assert result['rho'] == pytest.approx(-1.0, abs=0.001)
        assert result['direction'] == 'negative'

    def test_compute_spearman_insufficient_data(self):
        """Should raise ValueError when n < min_sample_size."""
        x = [1]
        y = [2]
        with pytest.raises(ValueError, match="Insufficient data"):
            compute_spearman(
                x,
                y,
                alpha=0.05,
                confidence=0.95,
                min_sample_size=2)

    def test_compute_spearman_constant_variable(self):
        """Should raise ValueError when one variable is constant."""
        x = [1, 2, 3, 4, 5]
        y = [1, 1, 1, 1, 1]
        with pytest.raises(ValueError, match="constant"):
            compute_spearman(
                x,
                y,
                alpha=0.05,
                confidence=0.95,
                min_sample_size=2)


class TestUnitKruskal:
    """Unit tests for Kruskal-Wallis functions."""

    def test_group_summary_basic(self):
        df = pd.DataFrame({
            'precipitation_mm_cat': ['No rain', 'No rain', 'Low', 'Low', 'Moderate', 'Moderate'],
            'energy': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })
        summary = group_summary(df, 'precipitation_mm_cat', 'energy')
        assert summary['No rain']['n'] == 2
        assert summary['No rain']['median'] == 0.55

    def test_kruskal_insufficient_groups(self):
        df = pd.DataFrame({
            'precipitation_mm_cat': ['No rain'] * 10,
            'energy': np.random.uniform(0, 1, 10)
        })
        mock_logger = Mock()
        with pytest.raises(ValueError, match="Need at least 2 buckets"):
            run_kruskal_wallis(df, 'precipitation_mm_cat', 'energy',
                               min_group_size=5, logger=mock_logger)
# property based test


def nonconstant_list(min_size=10, max_size=100):
    """ Generate a list of floats with at least two distinct values for property-based testing."""
    return st.lists(
        st.floats(
            min_value=0,
            max_value=100,
            allow_nan=False,
            allow_infinity=False),
        min_size=min_size,
        max_size=max_size
    ).filter(lambda lst: len(set(lst)) >= 2)


class TestPropertySpearman:
    """Property-based tests for Spearman correlation."""

    @given(nonconstant_list(), nonconstant_list())
    def test_spearman_rho_between_minus_one_and_one(self, x, y):
        """Test that Spearman's rho is always between -1 and 1."""
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        result = compute_spearman(
            x, y, alpha=0.05, confidence=0.95, min_sample_size=2)
        assert -1 <= result['rho'] <= 1

    @given(nonconstant_list())
    @settings(deadline=None)
    def test_spearman_identity_rho_one(self, x):
        """Test that Spearman's rho is 1 for identical lists."""
        result = compute_spearman(
            x, x, alpha=0.05, confidence=0.95, min_sample_size=2)
        assert result['rho'] == pytest.approx(1.0, abs=0.001)

    @given(nonconstant_list())
    @settings(deadline=None)
    def test_spearman_negated_rho_minus_one(self, x):
        neg_x = [-v for v in x]
        result = compute_spearman(
            x,
            neg_x,
            alpha=0.05,
            confidence=0.95,
            min_sample_size=2)
        assert result['rho'] == pytest.approx(-1.0, abs=0.001)

    @given(nonconstant_list(), nonconstant_list())
    @settings(deadline=None)
    def test_spearman_rho_symmetric(self, x, y):
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        r1 = compute_spearman(
            x,
            y,
            alpha=0.05,
            confidence=0.95,
            min_sample_size=2)
        r2 = compute_spearman(
            y,
            x,
            alpha=0.05,
            confidence=0.95,
            min_sample_size=2)
        assert r1['rho'] == r2['rho']

    @given(nonconstant_list(), nonconstant_list())
    def test_spearman_rho_p_value_between_zero_and_one(self, x, y):
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        result = compute_spearman(
            x, y, alpha=0.05, confidence=0.95, min_sample_size=2)
        assert 0 <= result['p_value'] <= 1


class TestPropertyPrecipBucketing:
    """Property-based tests for precipitation bucketing."""

    @given(nonconstant_list(min_size=10, max_size=100))
    def test_bucketing_preserves_row_count(self, precip_values):
        df = pd.DataFrame({
            'precipitation_mm': precip_values,
            'date': pd.date_range('2024-01-01', periods=len(precip_values))
        })
        df = create_precipitation_bins_percentile(
            df, 'precipitation_mm', p1=25, p2=50, p3=75)
        assert len(df) == len(precip_values)

    @given(nonconstant_list(min_size=10, max_size=100))
    def test_bucketing_categories_are_in_expected_set(self, precip_values):
        df = pd.DataFrame({
            'precipitation_mm': precip_values,
            'date': pd.date_range('2024-01-01', periods=len(precip_values))
        })
        df = create_precipitation_bins_percentile(
            df, 'precipitation_mm', p1=25, p2=50, p3=75)
        valid_labels = set(PRECIPITATION_AMOUNT_LABELS)
        actual_labels = set(df['precipitation_mm_cat'].unique())
        assert actual_labels.issubset(valid_labels)

# Metamorphic teest


class TestMetamorphicSpearman:
    """Metamorphic tests for Spearman correlation."""

    def test_spearman_invariant_scaling(self):
        """Test that scaling both variables by a positive constant does not change Spearmans rho."""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 50)
        y = np.random.uniform(0, 100, 50)
        r_orig = compute_spearman(
            x, y, alpha=0.05, confidence=0.95, min_sample_size=2)
        r_scaled = compute_spearman(
            x * 10,
            y * 10,
            alpha=0.05,
            confidence=0.95,
            min_sample_size=2)
        assert r_orig['rho'] == pytest.approx(r_scaled['rho'], abs=0.01)

    def test_spearman_invariant_monotonic_transform(self):
        np.random.seed(42)
        x = np.random.uniform(0.001, 100, 50)
        y = np.random.uniform(0.001, 100, 50)
        r_orig = compute_spearman(
            x, y, alpha=0.05, confidence=0.95, min_sample_size=2)
        r_log = compute_spearman(
            np.log(x),
            np.log(y),
            alpha=0.05,
            confidence=0.95,
            min_sample_size=2)
        assert r_orig['rho'] == pytest.approx(r_log['rho'], abs=0.01)

    def test_spearman_invariant_reordering(self):
        np.random.seed(42)
        x = np.random.uniform(0, 100, 50)
        y = np.random.uniform(0, 100, 50)
        r_orig = compute_spearman(
            x, y, alpha=0.05, confidence=0.95, min_sample_size=2)
        idx = np.random.permutation(len(x))
        r_shuffled = compute_spearman(
            x[idx],
            y[idx],
            alpha=0.05,
            confidence=0.95,
            min_sample_size=2)
        assert r_orig['rho'] == pytest.approx(r_shuffled['rho'], abs=0.01)

    def test_spearman_invariant_adding_constant(self):
        np.random.seed(42)
        x = np.random.uniform(0, 100, 50)
        y = np.random.uniform(0, 100, 50)
        r_orig = compute_spearman(
            x, y, alpha=0.05, confidence=0.95, min_sample_size=2)
        r_added = compute_spearman(
            x + 1000,
            y + 1000,
            alpha=0.05,
            confidence=0.95,
            min_sample_size=2)
        assert r_orig['rho'] == pytest.approx(r_added['rho'], abs=0.01)

# Edge Case Tests


class TestNegativeCases:
    """Negative tests for edge cases and invalid inputs."""

    def test_process_csvfile_rq2_empty_file(self, tmp_path):
        """This test checks that process_csvfile_rq2 raises a ValueError when the input CSV is empty."""
        empty_path = tmp_path / "empty.csv"
        empty_path.write_text("")
        mock_logger = Mock()
        with pytest.raises(ValueError, match="Could not read file"):
            process_csvfile_rq2(
                str(empty_path),
                p1=25,
                p2=50,
                p3=75,
                logger=mock_logger)

    def test_process_csvfile_rq2_missing_columns(self, tmp_path):
        """This test checks that process_csvfile_rq2 raises a ValueError when the input CSV is missing required columns."""
        bad_path = tmp_path / "bad.csv"
        pd.DataFrame({'date': [1, 2, 3]}).to_csv(bad_path, index=False)
        mock_logger = Mock()
        with pytest.raises(ValueError, match="missing columns"):
            process_csvfile_rq2(
                str(bad_path),
                p1=25,
                p2=50,
                p3=75,
                logger=mock_logger)

    def test_compute_spearman_empty_data(self):
        with pytest.raises(ValueError, match="Insufficient data"):
            compute_spearman(
                [],
                [],
                alpha=0.05,
                confidence=0.95,
                min_sample_size=2)

    def test_compute_spearman_mismatched_lengths(self):
        x = [1, 2, 3]
        y = [1, 2]
        with pytest.raises(ValueError, match="Mismatched lengths"):
            compute_spearman(
                x, y, alpha=0.05, confidence=0.95, min_sample_size=2
            )

# Integration Test


def create_test_csv(tmp_path, filename="processed_test_data.csv"):
    """Create a minimal CSV with the required columns for integration tests."""
    data = {
        "date": pd.date_range("2021-01-01", periods=20),
        "precipitation_mm": [0.0, 0.0, 0.0, 0.0, 0.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             5.0, 5.0, 5.0, 5.0, 5.0,
                             10.0, 10.0, 10.0, 10.0, 10.0],
        "acousticness": [0.1, 0.2, 0.15, 0.25, 0.3,
                         0.4, 0.45, 0.5, 0.55, 0.6,
                         0.7, 0.75, 0.8, 0.85, 0.9,
                         0.2, 0.3, 0.4, 0.5, 0.6],
        "energy": [0.8, 0.85, 0.9, 0.95, 0.7,
                   0.6, 0.65, 0.7, 0.75, 0.8,
                   0.5, 0.55, 0.6, 0.65, 0.7,
                   0.9, 0.85, 0.8, 0.75, 0.7],
        "tempo": [120, 121, 119, 122, 120,
                  110, 112, 111, 113, 110,
                  130, 131, 129, 132, 130,
                  105, 106, 104, 107, 105],
        "valence": [0.5, 0.6, 0.55, 0.65, 0.7,
                    0.4, 0.45, 0.5, 0.55, 0.6,
                    0.3, 0.35, 0.4, 0.45, 0.5,
                    0.8, 0.85, 0.9, 0.95, 0.7],
        "danceability": [0.7, 0.75, 0.8, 0.85, 0.9,
                         0.6, 0.65, 0.7, 0.75, 0.8,
                         0.5, 0.55, 0.6, 0.65, 0.7,
                         0.9, 0.85, 0.8, 0.75, 0.7],
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    df['season'] = (df['month'] % 12 + 3) // 3
    csv_path = tmp_path / filename
    df.to_csv(csv_path, index=False)
    return csv_path


def run_script(script_name, args, cwd=None):
    script_path = SRC_DIR / script_name
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return result.stdout, result.stderr, result.returncode


@pytest.fixture
def test_env(tmp_path):
    csv_path = create_test_csv(tmp_path)
    out_csv = tmp_path / "processed_w2.csv"
    prep_args = ["-i", str(csv_path), "-o", str(out_csv)]
    stdout, stderr, rc = run_script("datapreprocessing_w2.py", prep_args)
    assert rc == 0, f"Preprocessing failed: {stderr}"
    return tmp_path, out_csv


class TestIntegration:
    """Integration tests for the full RQ2 pipeline."""

    def test_kruskal_report(self, test_env):
        """ Test the full Kruskal-Wallis reporting pipeline end-to-end."""
        tmp_path, csv_path = test_env
        feature = "acousticness"
        group_col = "precipitation_mm_cat"

        json_out = tmp_path / "kruskal_result.json"
        kw_args = [
            str(csv_path),
            "-y",
            feature,
            "-c",
            group_col,
            "-o",
            str(json_out)]
        stdout, stderr, rc = run_script("kruskal_wallis.py", kw_args)
        assert rc == 0, f"Kruskal-Wallis failed: {stderr}"

        with open(json_out) as f:
            result = json.load(f)
        assert 'h_statistic' in result
        assert result['y_variable'] == feature

        png_out = tmp_path / "kruskal_plot.png"
        md_out = tmp_path / "kruskal_summary.md"
        report_args = [
            "-j",
            str(json_out),
            "-o",
            str(png_out),
            "-m",
            str(md_out)]
        stdout, stderr, rc = run_script("kruskal_report.py", report_args)
        assert rc == 0
        assert png_out.exists() and png_out.stat().st_size > 0
        assert md_out.exists() and md_out.stat().st_size > 0
        assert "Kruskal-Wallis" in md_out.read_text()

    def test_spearman_report(self, test_env):
        """ Test the full Spearman correlation reporting pipeline end-to-end."""
        tmp_path, csv_path = test_env
        x_var = "precipitation_mm"
        y_var = "energy"

        json_out = tmp_path / "spearman_result.json"
        sp_args = [
            str(csv_path),
            "-x",
            x_var,
            "-y",
            y_var,
            "-o",
            str(json_out)]
        stdout, stderr, rc = run_script("spearman_correlation.py", sp_args)
        assert rc == 0

        with open(json_out) as f:
            result = json.load(f)
        assert 'rho' in result

        png_out = tmp_path / "spearman_plot.png"
        md_out = tmp_path / "spearman_summary.md"
        report_args = [
            "-j",
            str(json_out),
            "-c",
            str(csv_path),
            "-o",
            str(png_out),
            "-m",
            str(md_out)]
        stdout, stderr, rc = run_script("spearman_report.py", report_args)
        assert rc == 0
        assert png_out.exists() and png_out.stat().st_size > 0
        assert md_out.exists() and md_out.stat().st_size > 0
        assert "Spearman" in md_out.read_text()

    def test_bar_plot(self, test_env):
        """Test the bar plot integrated into the Spearman report."""
        tmp_path, csv_path = test_env
        y_var = "tempo"
        group_col = "precipitation_mm_cat"

        json_out = tmp_path / "spearman_result.json"
        sp_args = [
            str(csv_path),
            "-x",
            "precipitation_mm",
            "-y",
            y_var,
            "-o",
            str(json_out)]
        stdout, stderr, rc = run_script("spearman_correlation.py", sp_args)
        assert rc == 0

        png_out = tmp_path / "bar_plot.png"
        scatter_out = tmp_path / "spearman_plot.png"
        md_out = tmp_path / "spearman_summary.md"
        bar_args = [
            "-j",
            str(json_out),
            "-c",
            str(csv_path),
            "-o",
            str(scatter_out),
            "-m",
            str(md_out),
            "--group",
            group_col,
            "--bar-output",
            str(png_out)]
        stdout, stderr, rc = run_script("spearman_report.py", bar_args)
        assert rc == 0
        assert png_out.exists() and png_out.stat().st_size > 0
        assert scatter_out.exists() and scatter_out.stat().st_size > 0
        assert md_out.exists() and md_out.stat().st_size > 0
        with open(png_out, "rb") as f:
            assert f.read(8).startswith(b'\x89PNG\r\n\x1a\n')
