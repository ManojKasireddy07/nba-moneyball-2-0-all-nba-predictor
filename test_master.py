"""
Basic unit tests for data processing utilities.

These tests provide sanity checks for functions in ``build_master.py``.
They can be run with ``pytest``.  More comprehensive tests should be
added as the project evolves.
"""
from src.data.build_master import compute_years_in_league
import pandas as pd


def test_compute_years_in_league():
    data = [
        {"name": "Alice", "season_end_year": 2000},
        {"name": "Alice", "season_end_year": 2001},
        {"name": "Bob", "season_end_year": 2003},
    ]
    df = pd.DataFrame(data)
    result = compute_years_in_league(df)
    years = result.set_index(["name", "season_end_year"])["Years_in_League"].to_dict()
    assert years[("Alice", 2000)] == 0
    assert years[("Alice", 2001)] == 1
    assert years[("Bob", 2003)] == 0