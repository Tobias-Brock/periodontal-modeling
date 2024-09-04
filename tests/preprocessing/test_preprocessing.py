import pytest
import pandas as pd
from unittest.mock import patch
from pamod.preprocessing import StaticProcessEngine


# Sample data for testing
@pytest.fixture
def sample_data():
    data = {
        "id_patient": [1, 2],
        "tooth": [11, 12],
        "toothtype": [1, 2],
        "rootnumber": [1, 2],
        "mobility": [1, 1],
        "restoration": [0, 1],
        "percussion-sensitivity": [0, 1],
        "sensitivity": [1, 1],
        "furcationbaseline": [1, 0],
        "side": [1, 2],
        "pdbaseline": [4, 5],
        "recbaseline": [3, 2],
        "plaque": [1, 2],
        "bop": [1, 0],
        "age": [45, 50],
        "gender": [1, 2],
        "bodymassindex": [25.0, 30.0],
        "periofamilyhistory": [2, 1],
        "diabetes": [1, 2],
        "smokingtype": [1, 0],
        "cigarettenumber": [10, 5],
        "antibiotictreatment": [1, 0],
        "stresslvl": [3, 2],
        "pdrevaluation": [4, 5],
        "boprevaluation": [2, 1],
    }
    return pd.DataFrame(data)


# Test for loading data
@patch("os.path.join")
@patch("pandas.read_excel")
def test_load_data(mock_read_excel, mock_path_join, sample_data):
    mock_read_excel.return_value = sample_data
    mock_path_join.return_value = "dummy_path"

    engine = StaticProcessEngine(behavior=False, scale=False, encoding="one_hot")
    df = engine._load_data()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Check if data is loaded correctly
    assert "id_patient" in df.columns


# Test for scaling and encoding
def test_scale_and_encode(sample_data):
    engine = StaticProcessEngine(behavior=False, scale=True, encoding="one_hot")
    engine.df = sample_data

    df_scaled = engine._scale_and_encode()

    assert isinstance(df_scaled, pd.DataFrame)
    # Use lowercase column names in assertions
    assert "pdbaseline" in df_scaled.columns
    assert any(col.startswith("side_") for col in df_scaled.columns)  # Example of one-hot encoding


# Test for data processing
def test_process_data(sample_data):
    engine = StaticProcessEngine(behavior=False, scale=False, encoding="one_hot")
    engine.df = sample_data

    df_processed = engine.process_data()

    assert isinstance(df_processed, pd.DataFrame)
    assert "side_infected" in df_processed.columns
    assert "tooth_infected" in df_processed.columns
    assert "recbaseline" in df_processed.columns
    assert df_processed["side_infected"].sum() > 0  # Ensure that infection is detected
    assert "plaque" in df_processed.columns  # Make sure plaque exists after processing


# Test for behavior columns
def test_behavior_columns(sample_data):
    engine = StaticProcessEngine(behavior=True, scale=False, encoding="one_hot")
    engine.df = sample_data

    df = engine.process_data()

    # Ensure that behavior columns are included when the behavior flag is set
    for col in engine.behavior_columns.get("binary", []) + engine.behavior_columns.get("categorical", []):
        assert col.lower() in df.columns


# Test for invalid encoding type
def test_invalid_encoding(sample_data):
    engine = StaticProcessEngine(behavior=False, scale=False, encoding="invalid")
    engine.df = sample_data

    with pytest.raises(ValueError):
        engine._scale_and_encode()


# Test for missing required columns
@patch("os.path.join")
@patch("pandas.read_excel")
def test_missing_required_columns(mock_read_excel, mock_path_join):
    mock_read_excel.return_value = pd.DataFrame({"missing_column": [1, 2, 3]})
    mock_path_join.return_value = "dummy_path"

    with pytest.raises(ValueError):
        engine = StaticProcessEngine(behavior=False, scale=False, encoding="one_hot")
        engine._load_data()
