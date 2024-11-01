"""Tests for Trainer."""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from periomod.resampling._resampler import Resampler
from periomod.training._trainer import Trainer


def create_sample_data(n_samples=100, n_features=5, n_classes=2, random_state=42):
    """Creates a sample dataset for testing."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_classes=n_classes,
        random_state=random_state,
        weights=[0.7, 0.3] if n_classes == 2 else None,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    df["y"] = y
    df["group"] = [i // 10 for i in range(n_samples)]  # Create group identifiers
    return df


def test_trainer_initialization():
    """Test initialization of Trainer class."""
    trainer = Trainer(
        classification="binary",
        criterion="f1",
        tuning="cv",
        hpo="hebo",
        mlp_training=True,
        threshold_tuning=True,
    )
    assert trainer.classification == "binary"
    assert trainer.criterion == "f1"
    assert trainer.tuning == "cv"
    assert trainer.hpo == "hebo"
    assert trainer.mlp_training is True
    assert trainer.threshold_tuning is True


def test_train_standard_model():
    """Test training a standard model."""
    df = create_sample_data()
    X_train, X_val = df.iloc[:80], df.iloc[80:]
    y_train, y_val = X_train["y"], X_val["y"]
    X_train = X_train.drop(columns=["y", "group"])
    X_val = X_val.drop(columns=["y", "group"])
    model = LogisticRegression()
    trainer = Trainer(
        classification="binary",
        criterion="f1",
        tuning="cv",
        hpo="hebo",
        mlp_training=False,
    )
    score, trained_model, best_threshold = trainer.train(
        model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
    )
    assert isinstance(score, float)
    assert isinstance(trained_model, LogisticRegression)
    assert best_threshold is None


def test_train_mlp_model():
    """Test training an MLP model."""
    df = create_sample_data()
    X_train, X_val = df.iloc[:80], df.iloc[80:]
    y_train, y_val = X_train["y"], X_val["y"]
    X_train = X_train.drop(columns=["y", "group"])
    X_val = X_val.drop(columns=["y", "group"])
    model = MLPClassifier(max_iter=10)
    trainer = Trainer(
        classification="binary",
        criterion="f1",
        tuning="cv",
        hpo="hebo",
        mlp_training=True,
    )
    score, trained_model, best_threshold = trainer.train(
        model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
    )
    assert isinstance(score, float)
    assert isinstance(trained_model, MLPClassifier)
    assert best_threshold is None or isinstance(best_threshold, float)


def test_evaluate_cv():
    """Test evaluate_cv method."""
    df = create_sample_data()
    X_train, X_val = df.iloc[:80], df.iloc[80:]
    y_train, y_val = X_train["y"].values, X_val["y"].values
    X_train = X_train.drop(columns=["y", "group"])
    X_val = X_val.drop(columns=["y", "group"])
    model = LogisticRegression()
    trainer = Trainer(
        classification="binary",
        criterion="f1",
        tuning="cv",
        hpo="hebo",
        mlp_training=False,
    )
    fold = ((X_train, y_train), (X_val, y_val))
    score = trainer.evaluate_cv(model=model, fold=fold)
    assert isinstance(score, float)


def test_optimize_threshold():
    """Test optimize_threshold method."""
    df = create_sample_data()
    resampler = Resampler(classification="binary", encoding="one_hot")
    resampler.y = "y"
    resampler.group_col = "group"
    outer_splits, _ = resampler.cv_folds(df, n_folds=3, seed=42)

    model = LogisticRegression()
    trainer = Trainer(
        classification="binary",
        criterion="f1",
        tuning="cv",
        hpo="hebo",
        mlp_training=False,
    )
    best_threshold = trainer.optimize_threshold(
        model=model, outer_splits=outer_splits, n_jobs=1
    )
    assert isinstance(best_threshold, float)


def test_train_final_model():
    """Test train_final_model method."""
    df = create_sample_data()
    resampler = Resampler(classification="binary", encoding="one_hot")
    resampler.y = "y"
    resampler.group_col = "group"
    resampler.all_cat_vars = []
    model_params = ("lr", {}, 0.5)

    trainer = Trainer(
        classification="binary",
        criterion="f1",
        tuning="cv",
        hpo="hebo",
        mlp_training=False,
    )
    result = trainer.train_final_model(
        df=df,
        resampler=resampler,
        model=model_params,
        sampling=None,
        factor=None,
        n_jobs=1,
        seed=42,
        test_size=0.2,
        verbose=False,
    )
    assert "model" in result
    assert "metrics" in result
    assert isinstance(result["model"], LogisticRegression)
    assert isinstance(result["metrics"], dict)
