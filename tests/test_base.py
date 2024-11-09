"""Tests for base module."""

from unittest.mock import patch

import pandas as pd
import pytest

from periomod.base import (
    BaseValidator,
    Patient,
    Side,
    Tooth,
    patient_to_df,
)


def test_side_dataclass():
    """Test creation of Side dataclass instance."""
    side = Side(
        furcationbaseline=1,
        side=1,
        pdbaseline=2,
        recbaseline=2,
        plaque=1,
        bop=1,
    )
    assert side.furcationbaseline == 1
    assert side.side == 1
    assert side.pdbaseline == 2
    assert side.recbaseline == 2
    assert side.plaque == 1
    assert side.bop == 1


def test_tooth_dataclass():
    """Test creation of Tooth dataclass instance."""
    side_1 = Side(
        furcationbaseline=1,
        side=1,
        pdbaseline=2,
        recbaseline=2,
        plaque=1,
        bop=1,
    )
    side_2 = Side(
        furcationbaseline=2,
        side=2,
        pdbaseline=3,
        recbaseline=3,
        plaque=1,
        bop=0,
    )
    tooth = Tooth(
        tooth=11,
        toothtype=2,
        rootnumber=1,
        mobility=1,
        restoration=0,
        percussion=0,
        sensitivity=1,
        sides=[side_1, side_2],
    )
    assert tooth.tooth == 11
    assert tooth.toothtype == 2
    assert len(tooth.sides) == 2


def test_patient_dataclass():
    """Test creation of Patient dataclass instance."""
    side_1 = Side(
        furcationbaseline=1,
        side=1,
        pdbaseline=2,
        recbaseline=2,
        plaque=1,
        bop=1,
    )
    tooth = Tooth(
        tooth=11,
        toothtype=2,
        rootnumber=1,
        mobility=1,
        restoration=0,
        percussion=0,
        sensitivity=1,
        sides=[side_1],
    )
    patient = Patient(
        age=45,
        gender=1,
        bodymassindex=23.5,
        periofamilyhistory=1,
        diabetes=0,
        smokingtype=2,
        cigarettenumber=10,
        antibiotictreatment=0,
        stresslvl=2,
        teeth=[tooth],
    )
    assert patient.age == 45
    assert len(patient.teeth) == 1


def test_patient_to_df():
    """Test conversion of Patient instance to DataFrame."""
    side_1 = Side(
        furcationbaseline=1,
        side=1,
        pdbaseline=2,
        recbaseline=2,
        plaque=1,
        bop=1,
    )
    side_2 = Side(
        furcationbaseline=2,
        side=2,
        pdbaseline=3,
        recbaseline=3,
        plaque=1,
        bop=0,
    )
    tooth = Tooth(
        tooth=11,
        toothtype=2,
        rootnumber=1,
        mobility=1,
        restoration=0,
        percussion=0,
        sensitivity=1,
        sides=[side_1, side_2],
    )
    patient = Patient(
        age=45,
        gender=1,
        bodymassindex=23.5,
        periofamilyhistory=1,
        diabetes=0,
        smokingtype=2,
        cigarettenumber=10,
        antibiotictreatment=0,
        stresslvl=2,
        teeth=[tooth],
    )
    df = patient_to_df(patient)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2  # Two sides
    assert "age" in df.columns
    assert "tooth" in df.columns
    assert "side" in df.columns
    assert df.iloc[0]["age"] == 45
    assert df.iloc[0]["tooth"] == 11
    assert df.iloc[0]["side"] == 1
    assert df.iloc[1]["side"] == 2


def test_base_validator_valid():
    """Test BaseValidator with valid inputs."""
    with patch("periomod.base.BaseConfig.__init__", return_value=None):
        validator = BaseValidator(classification="binary", criterion="f1")
        assert validator.classification == "binary"
        assert validator.criterion == "f1"
        assert validator.hpo is None
        assert validator.tuning is None


def test_base_validator_invalid_criterion():
    """Test BaseValidator with invalid criterion."""
    with patch("periomod.base.BaseConfig.__init__", return_value=None):
        with pytest.raises(ValueError, match="Unsupported criterion"):
            BaseValidator(classification="binary", criterion="invalid_criterion")


def test_base_validator_invalid_tuning():
    """Test BaseValidator with invalid tuning method."""
    with patch("periomod.base.BaseConfig.__init__", return_value=None):
        with pytest.raises(ValueError, match="Unsupported tuning method"):
            BaseValidator(
                classification="binary", criterion="f1", tuning="invalid_tuning"
            )


def test_base_validator_invalid_classification():
    """Test BaseValidator with invalid classification type."""
    with patch("periomod.base.BaseConfig.__init__", return_value=None):
        with pytest.raises(ValueError, match="invalid classification type"):
            BaseValidator(classification="invalid_classification", criterion="f1")


def test_base_validator_invalid_hpo():
    """Test BaseValidator with invalid HPO type."""
    with patch("periomod.base.BaseConfig.__init__", return_value=None):
        with pytest.raises(ValueError, match="unsupported HPO type"):
            BaseValidator(classification="binary", criterion="f1", hpo="invalid_hpo")


def test_validate_classification():
    """Test the _validate_classification method with valid and invalid inputs."""
    with patch("periomod.base.BaseConfig.__init__", return_value=None):
        validator = BaseValidator(classification="binary", criterion="f1")
        validator.classification = "binary"
        validator._validate_classification()  # Should pass without exception
        validator.classification = "multiclass"
        validator._validate_classification()  # Should pass without exception
        validator.classification = "invalid_classification"
        with pytest.raises(ValueError, match="invalid classification type"):
            validator._validate_classification()


def test_validate_hpo():
    """Test the _validate_hpo method with valid and invalid inputs."""
    with patch("periomod.base.BaseConfig.__init__", return_value=None):
        validator = BaseValidator(classification="binary", criterion="f1")
        validator.hpo = None
        validator._validate_hpo()  # Should pass without exception
        validator.hpo = "rs"
        validator._validate_hpo()  # Should pass without exception
        validator.hpo = "hebo"
        validator._validate_hpo()  # Should pass without exception
        validator.hpo = "invalid_hpo"
        with pytest.raises(ValueError, match="unsupported HPO type"):
            validator._validate_hpo()


def test_validate_criterion():
    """Test the _validate_criterion method with valid and invalid inputs."""
    with patch("periomod.base.BaseConfig.__init__", return_value=None):
        validator = BaseValidator(classification="binary", criterion="f1")

        validator.criterion = "f1"
        validator._validate_criterion()  # Should pass without exception
        validator.criterion = "macro_f1"
        validator._validate_criterion()  # Should pass without exception
        validator.criterion = "brier_score"
        validator._validate_criterion()  # Should pass without exceptio
        validator.criterion = "invalid_criterion"
        with pytest.raises(ValueError, match="Unsupported criterion"):
            validator._validate_criterion()


def test_validate_tuning():
    """Test the _validate_tuning method with valid and invalid inputs."""
    with patch("periomod.base.BaseConfig.__init__", return_value=None):
        validator = BaseValidator(classification="binary", criterion="f1")
        validator.tuning = None
        validator._validate_tuning()  # Should pass without exception
        validator.tuning = "holdout"
        validator._validate_tuning()  # Should pass without exception
        validator.tuning = "cv"
        validator._validate_tuning()  # Should pass without exception

        validator.tuning = "invalid_tuning"
        with pytest.raises(ValueError, match="Unsupported tuning method"):
            validator._validate_tuning()
