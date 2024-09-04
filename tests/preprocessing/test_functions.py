import pytest
import pandas as pd
import numpy as np
from pamod.preprocessing._functions import FunctionPreprocessor


# Create some sample data
@pytest.fixture
def sample_data():
    data = {
        "id_patient": [1, 1, 1, 2, 2],
        "tooth": [11, 12, 13, 21, 22],
        "side": [1, 2, 1, 1, 2],
        "pdbaseline": [4, 3, 5, 6, 2],
        "recbaseline": [2, 2, 2, 3, 1],
        "plaque": [None, 1, 2, None, 1],
        "furcationbaseline": [None, 1, None, None, None],
        "side_infected": [None, None, None, None, None],
    }
    return pd.DataFrame(data)


# Test check_infection method
def test_check_infection(sample_data):
    preprocessor = FunctionPreprocessor(sample_data)

    # Test various pocket depths and boprevaluation values
    assert preprocessor.check_infection(2, 1) == 0  # Depth = 2, boprevaluation = 1 (Healthy)
    assert preprocessor.check_infection(4, 2) == 1  # Depth = 4, boprevaluation = 2 (Infected)
    assert preprocessor.check_infection(5, 1) == 1  # Depth > 4 (Infected)


# Test tooth_neighbor method
def test_tooth_neighbor(sample_data):
    preprocessor = FunctionPreprocessor(sample_data)

    # Test if neighbors are correct
    assert np.array_equal(preprocessor.tooth_neighbor(11), [12, 21])
    assert np.array_equal(preprocessor.tooth_neighbor(12), [11, 13])
    assert preprocessor.tooth_neighbor(50) == "No tooth."  # Invalid tooth


# Test get_adjacent_infected_teeth_count method
def test_get_adjacent_infected_teeth_count(sample_data):
    preprocessor = FunctionPreprocessor(sample_data)

    # Modify sample data to mark some teeth as infected
    sample_data.loc[0, "side_infected"] = 1  # Tooth 11 infected
    sample_data.loc[2, "side_infected"] = 1  # Tooth 13 infected
    sample_data = preprocessor.get_adjacent_infected_teeth_count(sample_data, "id_patient", "tooth", "side_infected")

    assert (
        sample_data.loc[sample_data["tooth"] == 12, "infected_neighbors"].values[0] == 2
    )  # Tooth 12 has two infected neighbors


# Test plaque_values method
def test_plaque_values(sample_data):
    preprocessor = FunctionPreprocessor(sample_data)

    modes_dict = {(11, 1, 0): 1, (12, 2, 0): 2}  # Simulate mode calculation
    row = {"plaque_all_na": 1, "tooth": 11, "side": 1, "pdbaseline_grouped": 0, "plaque": np.nan}

    assert preprocessor.plaque_values(row, modes_dict) == 1  # Should return mode value 1


# Test plaque_imputation method
def test_plaque_imputation(sample_data):
    preprocessor = FunctionPreprocessor(sample_data)
    preprocessor.data = sample_data

    # Run plaque imputation
    data_imputed = preprocessor.plaque_imputation()

    # Check if the imputation has worked (non-NaN values in the 'plaque' column)
    assert data_imputed["plaque"].isna().sum() == 0  # No missing values left
    assert data_imputed["plaque"].iloc[0] == 1  # Check imputed values


# Test fur_values method
def test_fur_values(sample_data):
    preprocessor = FunctionPreprocessor(sample_data)

    row = {
        "tooth": 16,
        "side": 2,
        "pdbaseline": 5,
        "recbaseline": 2,
        "furcationbaseline_all_na": 1,
        "furcationbaseline": np.nan,
    }
    # pdbaseline + recbaseline = 5 + 2 = 7, which according to fur_values logic should return 2
    assert preprocessor.fur_values(row) == 2  # Should return correct furcation score

    row = {
        "tooth": 16,
        "side": 2,
        "pdbaseline": 2,
        "recbaseline": 1,
        "furcationbaseline_all_na": 1,
        "furcationbaseline": np.nan,
    }
    # pdbaseline + recbaseline = 2 + 1 = 3, which should return 0
    assert preprocessor.fur_values(row) == 0  # Should return correct furcation score


# Test fur_imputation method
def test_fur_imputation(sample_data):
    preprocessor = FunctionPreprocessor(sample_data)
    preprocessor.data = sample_data

    # Run furcation imputation
    data_imputed = preprocessor.fur_imputation()

    # Ensure that no missing values remain
    assert data_imputed["furcationbaseline"].isna().sum() == 0
