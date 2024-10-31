import numpy as np
import pandas as pd
import pytest

from periomod.data._helpers import ProcessDataHelper


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing.

    Returns:
        pd.DataFrame: A DataFrame containing sample data for testing purposes.
    """
    data = {
        "id_patient": [1, 1, 1, 2, 2],
        "tooth": [11, 12, 13, 21, 22],
        "side": [1, 2, 1, 1, 2],
        "pdbaseline": [4, 3, 5, 6, 2],
        "recbaseline": [2, 2, 2, 3, 1],
        "plaque": [None, 1, 2, None, 1],
        "furcationbaseline": [None, 1, None, None, None],
        "side_infected": [None, None, None, None, None],
        "boprevaluation": [0, 0, 0, 0, 0],  # Required for the methods
    }
    return pd.DataFrame(data)


def test_check_infection(sample_data):
    """Test the `check_infection` method of `FunctionPreprocessor`.

    Args:
        sample_data (pd.DataFrame): The sample data fixture.
    """
    preprocessor = ProcessDataHelper()

    # Test various pocket depths and boprevaluation values
    assert preprocessor.check_infection(2, 1) == 0  # Depth = 2, bop = 1 (Healthy)
    assert preprocessor.check_infection(4, 2) == 1  # Depth = 4, bop = 2 (Infected)
    assert preprocessor.check_infection(5, 1) == 1  # Depth > 4 (Infected)


def test_tooth_neighbor(sample_data):
    """Test the `tooth_neighbor` method of `ProcessDataHelper`.

    Args:
        sample_data (pd.DataFrame): The sample data fixture.
    """
    preprocessor = ProcessDataHelper()

    # Test if neighbors are correct
    assert np.array_equal(preprocessor.tooth_neighbor(11), [12, 21])
    assert np.array_equal(preprocessor.tooth_neighbor(12), [11, 13])
    assert preprocessor.tooth_neighbor(50) == "No tooth"  # Invalid tooth


def test_get_adjacent_infected_teeth_count(sample_data):
    """Test the `get_adjacent_infected_teeth_count` method.

    Args:
        sample_data (pd.DataFrame): The sample data fixture.
    """
    preprocessor = ProcessDataHelper()

    # Modify sample data to mark some teeth as infected
    sample_data.loc[0, "side_infected"] = 1  # Tooth 11 infected
    sample_data.loc[2, "side_infected"] = 1  # Tooth 13 infected
    sample_data = preprocessor.get_adjacent_infected_teeth_count(
        sample_data, "id_patient", "tooth", "side_infected"
    )

    # Tooth 12 should have two infected neighbors
    assert (
        sample_data.loc[sample_data["tooth"] == 12, "infected_neighbors"].values[0] == 2
    )


def test_plaque_values(sample_data):
    """Test the `plaque_values` method of `ProcessDataHelper`.

    Args:
        sample_data (pd.DataFrame): The sample data fixture.
    """
    preprocessor = ProcessDataHelper()

    modes_dict = {(11, 1, 0): 1, (12, 2, 0): 2}  # Simulate mode calculation
    row = {
        "plaque_all_na": 1,
        "tooth": 11,
        "side": 1,
        "pdbaseline_grouped": 0,
        "plaque": np.nan,
    }

    # Should return mode value 1
    assert preprocessor.plaque_values(row, modes_dict) == 1


def test_plaque_imputation(sample_data):
    """Test the `plaque_imputation` method of `ProcessDataHelper`.

    Args:
        sample_data (pd.DataFrame): The sample data fixture.
    """
    preprocessor = ProcessDataHelper()

    # Run plaque imputation
    data_imputed = preprocessor.plaque_imputation(sample_data)

    # Check if the imputation has worked
    assert data_imputed["plaque"].isna().sum() == 0  # No missing values left
    assert data_imputed["plaque"].iloc[0] == 1  # Check imputed values


def test_fur_values(sample_data):
    """Test the `fur_values` method of `ProcessDataHelper`.

    Args:
        sample_data (pd.DataFrame): The sample data fixture.
    """
    preprocessor = ProcessDataHelper()

    # Test case where pdbaseline + recbaseline = 7
    row = {
        "tooth": 16,
        "side": 2,
        "pdbaseline": 5,
        "recbaseline": 2,
        "furcationbaseline_all_na": 1,
        "furcationbaseline": np.nan,
    }
    assert preprocessor.fur_values(row) == 2  # Should return furcation score 2

    # Test case where pdbaseline + recbaseline = 3
    row = {
        "tooth": 16,
        "side": 2,
        "pdbaseline": 2,
        "recbaseline": 1,
        "furcationbaseline_all_na": 1,
        "furcationbaseline": np.nan,
    }
    assert preprocessor.fur_values(row) == 0  # Should return furcation score 0


def test_fur_imputation(sample_data):
    """Test the `fur_imputation` method of `ProcessDataHelper`.

    Args:
        sample_data (pd.DataFrame): The sample data fixture.
    """
    preprocessor = ProcessDataHelper()
    data_imputed = preprocessor.fur_imputation(sample_data)

    assert data_imputed["furcationbaseline"].isna().sum() == 0
