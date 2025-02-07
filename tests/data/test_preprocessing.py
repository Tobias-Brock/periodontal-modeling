"""Tests for preprocessing engine."""

import numpy as np
import pandas as pd

from periomod.data._preprocessing import StaticProcessEngine


def test_static_process_engine_initialization():
    """Test initialization of StaticProcessEngine."""
    engine = StaticProcessEngine(behavior=True, verbose=False)
    assert engine.behavior is True
    assert engine.verbose is False


def test_impute_missing_values():
    """Test impute_missing_values method."""
    data = pd.DataFrame({
        "boprevaluation": [np.nan, "NA", "-", " "],
        "recbaseline": [np.nan, 1, 2, 3],
        "bop": [np.nan, 1, 2, np.nan],
        "bodymassindex": ["25", "invalid", np.nan, "30"],
        "periofamilyhistory": [np.nan, 1, 2, np.nan],
        "smokingtype": [np.nan, 1, 2, np.nan],
        "cigarettenumber": [np.nan, 5, 10, np.nan],
        "diabetes": [np.nan, 1, 2, np.nan],
        "stresslvl": [np.nan, 3, 5, 8],
        "toothtype": [np.nan, 1, 2, np.nan],
        "rootnumber": [np.nan, 1, 2, np.nan],
        "tooth": [11, 12, 43, 33],
    })
    engine = StaticProcessEngine(behavior=False, verbose=False)
    df_imputed = engine.impute_missing_values(data)
    assert df_imputed["boprevaluation"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert df_imputed["recbaseline"].tolist() == [1.0, 1.0, 2.0, 3.0]
    assert df_imputed["bop"].tolist() == [1.0, 1.0, 2.0, 1.0]
    assert df_imputed["bodymassindex"].tolist() == [
        25.0,
        27.5,
        27.5,
        30.0,
    ]
    assert df_imputed["periofamilyhistory"].tolist() == [2, 1, 2, 2]
    assert df_imputed["smokingtype"].tolist() == [1, 1, 2, 1]
    assert df_imputed["cigarettenumber"].tolist() == [0.0, 5.0, 10.0, 0.0]
    assert df_imputed["diabetes"].tolist() == [1, 1, 2, 1]
    assert df_imputed["stresslvl"].tolist() == [1, 0, 1, 2]


def test_create_tooth_features():
    """Test create_tooth_features method."""
    data = pd.DataFrame({
        "id_patient": [1, 1],
        "pdbaseline": [5, 3],
        "bop": [2, 1],
        "tooth": [11, 12],
        "side": [1, 2],
    })
    engine = StaticProcessEngine(behavior=False, verbose=False)
    df_features = engine.create_tooth_features(data)
    assert "side_infected" in df_features.columns
    assert "tooth_infected" in df_features.columns
    assert df_features["side_infected"].tolist() == [1, 0]
    assert df_features["tooth_infected"].tolist() == [1, 0]


def test_create_outcome_variables():
    """Test create_outcome_variables method."""
    data = pd.DataFrame({
        "pdrevaluation": [4, 5, 3],
        "pdbaseline": [5, 4, 3],
        "boprevaluation": [2, 1, 1],
    })
    data = StaticProcessEngine.create_outcome_variables(data)

    assert "pocketclosure" in data.columns
    assert "pdgroupbase" in data.columns
    assert "pdgrouprevaluation" in data.columns
    assert "improvement" in data.columns
    assert data["pocketclosure"].tolist() == [0, 0, 1]
    assert data["pdgroupbase"].tolist() == [1, 1, 0]
    assert data["pdgrouprevaluation"].tolist() == [1, 1, 0]
    assert data["improvement"].tolist() == [1, 0, 0]


def test_process_data():
    """Test the full data processing pipeline."""
    data = pd.DataFrame({
        "id_patient": [1, 1],
        "plaque": [1, 2],
        "recbaseline": [1, 2],
        "age": [25, 19],
        "pregnant": [1, 1],
        "pdbaseline": [5, 3],
        "pdrevaluation": [4, 2],
        "bop": [2, 1],
        "boprevaluation": [2, 1],
        "tooth": [11, 12],
        "furcationbaseline": [1, 2],
        "side": [1, 2],
        "toothtype": (1, 2),
        "rootnumber": (1, 1),
    })
    engine = StaticProcessEngine(behavior=False, verbose=False)
    df_processed = engine.process_data(data)
    assert df_processed["age"].min() >= 18
    expected_columns = [
        "id_patient",
        "age",
        "pdbaseline",
        "pdrevaluation",
        "bop",
        "boprevaluation",
        "tooth",
        "side",
        "side_infected",
        "tooth_infected",
        "pocketclosure",
        "pdgroupbase",
        "pdgrouprevaluation",
        "improvement",
        "plaque",
        "furcationbaseline",
        "toothtype",
    ]
    assert all(col in df_processed.columns for col in expected_columns)
    assert not df_processed.isna().to_numpy().any()
