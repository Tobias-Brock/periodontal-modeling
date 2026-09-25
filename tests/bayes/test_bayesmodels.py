"""Tests for the hierarchical logistic model."""

import numpy as np
import pytest

from periomod.bayes import (
    HierarchicalDataTransformer,
    HierarchicalLogisticModel,
    SpatialConfig,
    thin_draws,
)


@pytest.fixture
def train(data):
    """Provides the training design of the synthetic dataset.

    Args:
        data (pd.DataFrame): Synthetic processed dataset.

    Returns:
        HierarchicalData: Design matrix and nesting structure.
    """
    transformer = HierarchicalDataTransformer(task="pocketclosure", verbose=False)
    return transformer.transform_splits(data=data)["train"]


def test_nested_model_variables(train):
    """Test that the nested model uses non-centered random intercepts."""
    model = HierarchicalLogisticModel(spatial=SpatialConfig(mode="none"))
    pymc_model = model.build(data=train)
    names = {variable.name for variable in pymc_model.free_RVs}

    assert {"alpha", "beta", "sigma_patient", "z_patient"} <= names
    assert {"sigma_tooth", "z_tooth", "sigma_toothnum", "z_side"} <= names
    assert not any(name.startswith("phi_") for name in names)
    assert pymc_model.observed_RVs[0].name == "y_obs"


def test_tooth_level_spatial_model(train):
    """Test that the tooth CAR prior adds a field and a mixing weight."""
    model = HierarchicalLogisticModel(spatial=SpatialConfig(mode="tooth"))
    pymc_model = model.build(data=train)
    names = {variable.name for variable in pymc_model.free_RVs}
    potentials = {potential.name for potential in pymc_model.potentials}

    assert "phi_tooth" in names
    assert "rho_tooth" in names
    assert {"car_tooth", "zerosum_tooth"} <= potentials


def test_site_level_spatial_model(train):
    """Test that the site CAR prior adds a structured site field."""
    model = HierarchicalLogisticModel(spatial=SpatialConfig(mode="site"))
    pymc_model = model.build(data=train)
    names = {variable.name for variable in pymc_model.free_RVs}

    assert {"phi_site", "sigma_car_site"} <= names
    assert "rho_tooth" not in names


def test_direct_effects_can_be_disabled(train):
    """Test that the tooth number and side effects are optional."""
    model = HierarchicalLogisticModel(tooth_number_effect=False, side_effect=False)
    names = {variable.name for variable in model.build(data=train).free_RVs}

    assert "z_toothnum" not in names
    assert "z_side" not in names


def test_thin_draws():
    """Test that posterior draws are thinned to the requested maximum."""
    assert thin_draws(n_draws=100, max_draws=None) == slice(None)
    assert thin_draws(n_draws=100, max_draws=500) == slice(None)
    assert len(np.arange(100)[thin_draws(n_draws=100, max_draws=25)]) <= 25
