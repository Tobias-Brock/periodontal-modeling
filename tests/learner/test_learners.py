"""Tests for leaner module."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from periomod.learner._learners import Model


def test_model_initialization():
    """Test initialization of Model class."""
    model = Model(learner="rf", classification="binary")
    assert model.learner == "rf"
    assert model.classification == "binary"
    assert model.hpo is None


def test_get_model_instance_rf():
    """Test getting RandomForest model instance."""
    model = Model(learner="rf", classification="binary")
    rf_model = model._get_model_instance()
    assert isinstance(rf_model, RandomForestClassifier)


def test_get_model_instance_mlp():
    """Test getting MLP model instance."""
    model = Model(learner="mlp", classification="binary")
    mlp_model = model._get_model_instance()
    assert isinstance(mlp_model, MLPClassifier)


def test_get_model_instance_xgb_binary():
    """Test getting XGBoost model instance for binary classification."""
    model = Model(learner="xgb", classification="binary")
    xgb_model = model._get_model_instance()
    assert isinstance(xgb_model, xgb.XGBClassifier)
    assert xgb_model.get_params()["objective"] == model.xgb_obj_binary


def test_get_model_instance_lr_multiclass():
    """Test getting LogisticRegression model instance for multiclass classification."""
    model = Model(learner="lr", classification="multiclass")
    lr_model = model._get_model_instance()
    assert isinstance(lr_model, LogisticRegression)
    assert lr_model.multi_class == model.lr_multi_loss


def test_get_with_hpo_rs():
    """Test get method with random search HPO."""
    model_instance, param_grid = Model.get(
        learner="rf", classification="binary", hpo="rs"
    )
    assert isinstance(model_instance, RandomForestClassifier)
    assert isinstance(param_grid, dict)


def test_get_with_hpo_hebo():
    """Test get method with HEBO HPO."""
    model_instance, search_space, get_params_func = Model.get(
        learner="rf", classification="binary", hpo="hebo"
    )
    assert isinstance(model_instance, RandomForestClassifier)
    assert callable(get_params_func)
    assert isinstance(search_space, list)


def test_get_model_classmethod():
    """Test get_model class method."""
    model_instance = Model.get_model(learner="xgb", classification="binary")
    assert isinstance(model_instance, xgb.XGBClassifier)


def test_get_invalid_learner():
    """Test that invalid learner raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported learner type: invalid"):
        Model.get(learner="invalid", classification="binary", hpo="rs")


def test_get_invalid_hpo():
    """Test that invalid HPO method raises ValueError."""
    with pytest.raises(
        ValueError, match="Unsupported hpo type 'invalid_hpo' or learner type 'rf'"
    ):
        Model.get(learner="rf", classification="binary", hpo="invalid_hpo")
