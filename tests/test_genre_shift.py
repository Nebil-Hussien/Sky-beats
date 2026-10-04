"""
Tests for src/rq3/genre_shift.py (RQ3).

Run with:  pytest tests/test_genre_shift.py
"""

import pandas as pd
import pytest

import genre_shift  # from src/ (see tests/conftest.py)


def _toy_df():
    """
    Small synthetic dataset with a clear, hand-checkable answer.

    Two driest months (low precip) are dominated by 'dance';
    two rainiest months (high precip) are dominated by 'pop'.
    Every row has 100 streams, so stream shares == count shares here.
    """
    rows = [
        # date,        precip, genre,   streams
        ("2020-01-01", 1.0, "dance", 100),
        ("2020-01-01", 1.0, "dance", 100),
        ("2020-01-01", 1.0, "pop", 100),
        ("2020-02-01", 2.0, "dance", 100),
        ("2020-02-01", 2.0, "dance", 100),
        ("2020-03-01", 10.0, "pop", 100),
        ("2020-03-01", 10.0, "pop", 100),
        ("2020-03-01", 10.0, "dance", 100),
        ("2020-04-01", 11.0, "pop", 100),
        ("2020-04-01", 11.0, "pop", 100),
    ]
    df = pd.DataFrame(rows, columns=["date", "precipitation_mm", "track_genre", "streams"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_genre_shares_sum_to_one():
    """Each share column should sum to 1 (it's a share of the group)."""
    shares = genre_shift.genre_shares(_toy_df())
    assert shares["stream_share"].sum() == pytest.approx(1.0)
    assert shares["count_share"].sum() == pytest.approx(1.0)


def test_shift_is_rainy_minus_dry():
    """stream_shift must equal rainy_stream_share - dry_stream_share for every genre."""
    shift = genre_shift.compute_genre_shift(_toy_df(), n_months=2)
    expected = shift["rainy_stream_share"] - shift["dry_stream_share"]
    pd.testing.assert_series_equal(shift["stream_shift"], expected, check_names=False)


def test_expected_direction_and_magnitude():
    """pop should gain in rainy months, dance should lose, by a symmetric amount."""
    shift = genre_shift.compute_genre_shift(_toy_df(), n_months=2)
    # dry group: 4 dance + 1 pop -> dance 0.8, pop 0.2
    # rainy group: 4 pop + 1 dance -> pop 0.8, dance 0.2
    assert shift.loc["pop", "stream_shift"] == pytest.approx(0.6)
    assert shift.loc["dance", "stream_shift"] == pytest.approx(-0.6)


def test_too_many_months_raises():
    """Asking for more months than the data supports should fail loudly."""
    with pytest.raises(SystemExit):
        genre_shift.compute_genre_shift(_toy_df(), n_months=3)  # needs 6 months, only 4 exist
