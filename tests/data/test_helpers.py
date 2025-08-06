"""Tests for helpers."""

import numpy as np
import pandas as pd

from periomod.data._helpers import ProcessDataHelper


def test_check_infection():
    """Test the check_infection static method."""
    helper = ProcessDataHelper()
    assert helper.check_infection(depth=5, boprevaluation=1) == 1
    assert helper.check_infection(depth=4, boprevaluation=2) == 1
    assert helper.check_infection(depth=4, boprevaluation=1) == 0
    assert helper.check_infection(depth=3, boprevaluation=2) == 0


def test_get_adjacent_infected_teeth_count():
    """Test counting adjacent infected teeth."""
    helper = ProcessDataHelper()
    data = pd.DataFrame({
        "id_patient": [1, 1, 1, 1],
        "tooth": [11, 12, 13, 14],
        "tooth_infected": [1, 0, 1, 0],
    })
    df_result = helper.get_adjacent_infected_teeth_count(
        data=data,
        patient_col="id_patient",
        tooth_col="tooth",
        infection_col="tooth_infected",
    )
    assert "infected_neighbors" in df_result.columns
    # Expected counts:
    # Tooth 11: neighbor 12 (not infected), so count = 0
    # Tooth 12: neighbors 11 (infected), 13 (infected), so count = 2
    # Tooth 13: neighbors 12 (not infected), 14 (not infected), so count = 0
    # Tooth 14: neighbor 13 (infected), so count = 1
    expected_counts = [0, 2, 0, 1]
    assert df_result["infected_neighbors"].tolist() == expected_counts


def test_plaque_imputation():
    """Test the plaque_imputation method."""
    helper = ProcessDataHelper()
    data = pd.DataFrame({
        "tooth": [11, 12],
        "side": [1, 2],
        "pdbaseline": [2, 5],
        "plaque": [np.nan, np.nan],
        "id_patient": [1, 1],
    })
    helper.group_col = "id_patient"
    df_imputed = helper.plaque_imputation(data)
    assert all(df_imputed["plaque"] == 1)


def test_fur_imputation():
    """Test the fur_imputation method."""
    helper = ProcessDataHelper()
    data = pd.DataFrame({
        "tooth": [16, 11],
        "side": [2, 1],
        "pdbaseline": [5, 3],
        "recbaseline": [2, 1],
        "furcationbaseline": [np.nan, np.nan],
        "id_patient": [1, 1],
    })
    helper.group_col = "id_patient"
    df_imputed = helper.fur_imputation(data)
    # For tooth 16 side 2, furcation should be imputed based on rules
    # For tooth 11, which doesn't have furcation, furcationbaseline should be 0
    assert df_imputed.loc[0, "furcationbaseline"] in [0, 1, 2]
    assert df_imputed.loc[1, "furcationbaseline"] == 0


def test_tooth_neighbor():
    """Test the _tooth_neighbor method."""
    helper = ProcessDataHelper()
    neighbors = helper._tooth_neighbor(11)
    assert np.array_equal(neighbors, np.array([12, 21]))
    neighbors = helper._tooth_neighbor(48)
    assert np.array_equal(neighbors, np.array([47]))
    neighbors = helper._tooth_neighbor(99)
    assert neighbors == "No tooth"


def test_fur_side():
    """Test the _fur_side method."""
    helper = ProcessDataHelper()
    sides = helper._fur_side(16)
    assert np.array_equal(sides, np.array([2, 4, 6]))
    sides = helper._fur_side(36)
    assert np.array_equal(sides, np.array([2, 5]))
    sides = helper._fur_side(11)
    assert sides == "Tooth without Furkation"
