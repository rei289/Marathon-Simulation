"""Unit tests for the data processing module."""

import logging

import pandas as pd
import pytest

from src.process_runs.process_data import DataProcessor


@pytest.fixture
def base_json_data() -> dict:
    """Use to provide a base json data dictionary for testing."""

    return {
        "start_date_local": "2025-01-01T00:00:00",
        "weather": {
            "winddir": 0,
            "windspeed": 36,
        },
    }


def _make_raw_parquet_data() -> dict:
    return {
        "time": [0, 2],
        "heartrate": [120, 126],
        "cadence": [180, 186],
        "distance": [0.0, 6.0],
        "altitude": [10.0, 10.2],
        "velocity": [3.0, 3.0],
        "grade": [0.0, 0.0],
        "moving": [True, True],
        "latitude": [40.0, 40.0],
        "longitude": [-74.0, -74.0],
    }


def test_rename_columns_maps_expected_names(base_json_data: dict) -> None:
	"""Use to test that the rename_columns method correctly maps the raw column names to the expected column names."""
	processor = DataProcessor(logging.getLogger("test"), _make_raw_parquet_data(), base_json_data)
	assert "time_datetime" in processor.parquet_data.columns
	assert "heartrate_bpm" in processor.parquet_data.columns
	assert "cadence_rpm" in processor.parquet_data.columns
	assert "velocity_mps" in processor.parquet_data.columns
	assert "latitude_degree" in processor.parquet_data.columns
	assert "longitude_degree" in processor.parquet_data.columns


def test_minute_to_second_creates_per_second_columns(base_json_data: dict) -> None:
	processor = DataProcessor(logging.getLogger("test"), _make_raw_parquet_data(), base_json_data)

	processor._minute_to_second()

	assert "heartrate_bps" in processor.parquet_data.columns
	assert "cadence_rps" in processor.parquet_data.columns
	assert processor.parquet_data.loc[0, "heartrate_bps"] == pytest.approx(2.0)
	assert processor.parquet_data.loc[0, "cadence_rps"] == pytest.approx(3.0)


def test_interpolate_missing_data_fills_one_second_gap(base_json_data: dict) -> None:
	processor = DataProcessor(logging.getLogger("test"), _make_raw_parquet_data(), base_json_data)

	processor.interpolate_missing_data()

	assert len(processor.parquet_data) == 3
	assert processor.parquet_data.loc[1, "heartrate_bpm"] == pytest.approx(123.0)
	assert processor.parquet_data.loc[1, "time_datetime"] == pd.Timestamp("2025-01-01 00:00:01")


def test_feature_engineering_sets_stride_length_and_wind_features(base_json_data: dict) -> None:
	parquet_data = {
		"time_datetime": [
			pd.Timestamp("2025-01-01 00:00:00"),
			pd.Timestamp("2025-01-01 00:00:01"),
		],
		"latitude_degree": [40.0, 40.0],
		"longitude_degree": [-74.0, -73.999],
		"smooth_altitude_m": [10.0, 10.1],
		"smooth_heartrate_bps": [2.0, 2.1],
		"smooth_velocity_mps": [4.0, 4.0],
		"smooth_cadence_rps": [0.0, 2.0],
	}
	processor = DataProcessor(logging.getLogger("test"), parquet_data, base_json_data)

	processor.feature_engineering()

	assert "headwind_mps" in processor.parquet_data.columns
	assert "crosswind_mps" in processor.parquet_data.columns
	assert processor.parquet_data.loc[0, "stride_length_m"] == 0
	assert "winddir_degree" not in processor.parquet_data.columns
	assert "windspeed_mps" not in processor.parquet_data.columns
