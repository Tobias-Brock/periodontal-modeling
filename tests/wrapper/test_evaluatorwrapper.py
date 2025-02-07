"""Tests for EvaluatorWrapper."""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from periomod.base import Patient, Side, Tooth
from periomod.wrapper import EvaluatorWrapper, ModelExtractor


def test_basemodel_extractor_initialization():
    """Test initialization of BaseModelExtractor."""
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": MagicMock()
    }

    extractor = ModelExtractor(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
        random_state=42,
    )

    assert extractor.learners_dict == learners_dict
    assert extractor.criterion == "f1"
    assert extractor.aggregate is True
    assert extractor.verbose is True
    assert extractor.random_state == 42
    assert extractor.classification == "binary"


def test_basemodel_extractor_criterion_setter():
    """Test setting criterion in BaseModelExtractor."""
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": MagicMock(),
        "pocketclosure_lr_cv_hebo_brier_score_one_hot_no_sampling_factor2_rank1_": (
            MagicMock()
        ),
    }

    extractor = ModelExtractor(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
        random_state=42,
    )

    assert extractor.criterion == "f1"

    extractor.criterion = "brier_score"
    assert extractor.criterion == "brier_score"

    with pytest.raises(ValueError, match="Unsupported criterion"):
        extractor.criterion = "unsupported_criterion"


def test_basemodel_extractor_get_best():
    """Test the _get_best method of BaseModelExtractor."""
    model_mock = MagicMock()
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": model_mock
    }

    extractor = ModelExtractor(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
        random_state=42,
    )

    best_model, encoding, learner, task, factor, sampling = extractor._get_best()

    assert best_model == model_mock
    assert encoding == "one_hot"
    assert learner == "lr"
    assert task == "pocketclosure"
    assert factor == 2
    assert sampling is None


def test_basemodel_extractor_update_best_model():
    """Test the _update_best_model method to confirm correct model assignment."""
    model_mock = MagicMock()
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": model_mock
    }

    extractor = ModelExtractor(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
        random_state=42,
    )

    extractor._update_best_model()
    assert extractor.model == model_mock


def test_basemodel_extractor_missing_model_for_criterion():
    """Test behavior when no model with rank1 for specified criterion is found."""
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank2_": MagicMock()
    }

    with pytest.raises(
        ValueError, match="No model with rank1 found for criterion 'f1' in dict"
    ):
        ModelExtractor(
            learners_dict=learners_dict,
            criterion="f1",
            aggregate=True,
            verbose=True,
            random_state=42,
        )


def test_basemodel_extractor_encoding_detection():
    """Test encoding determination in _get_best based on model key."""
    model_mock = MagicMock()
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_target_no_sampling_factor2_rank1_": model_mock
    }

    extractor = ModelExtractor(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
        random_state=42,
    )

    _, encoding, _, _, _, _ = extractor._get_best()
    assert encoding == "target"


def test_basemodel_extractor_sampling_detection():
    """Test sampling determination in _get_best based on model key."""
    model_mock = MagicMock()
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_upsampling_factor2_rank1_": model_mock
    }

    extractor = ModelExtractor(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
        random_state=42,
    )

    _, _, _, _, _, sampling = extractor._get_best()
    assert sampling == "upsampling"


def test_evaluator_wrapper_initialization():
    """Test init."""
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": MagicMock()
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    assert wrapper.learners_dict == learners_dict
    assert wrapper.criterion == "f1"
    assert wrapper.aggregate is True
    assert wrapper.verbose is True
    assert (
        wrapper.model
        == learners_dict[
            "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_"
        ]
    )
    assert wrapper.encoding == "one_hot"
    assert wrapper.learner == "lr"
    assert wrapper.task == "pocketclosure"
    assert wrapper.factor == 2
    assert wrapper.sampling is None


def test_evaluator_wrapper_get_best():
    """Test obtaining best model."""
    model_mock = MagicMock()
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": model_mock
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    model, encoding, learner, task, factor, sampling = wrapper._get_best()

    assert model == model_mock
    assert encoding == "one_hot"
    assert learner == "lr"
    assert task == "pocketclosure"
    assert factor == 2
    assert sampling is None


@pytest.fixture
def mock_evaluator_wrapper():
    """Fixture for a BaseEvaluatorWrapper instance with mocked dependencies.

    Returns:
        EvaluatorWrapper: A mock instance of `EvaluatorWrapper` with dependencies
        replaced by MagicMock.
    """
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": MagicMock()
    }
    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
        random_state=42,
    )
    wrapper.evaluator = MagicMock()
    wrapper.dataloader = MagicMock()
    wrapper.resampler = MagicMock()
    wrapper.trainer = MagicMock()
    wrapper.inference_engine = MagicMock()
    return wrapper


def test_initialization(mock_evaluator_wrapper):
    """Tests that BaseEvaluatorWrapper initializes with correct attributes."""
    wrapper = mock_evaluator_wrapper
    assert wrapper.criterion == "f1"
    assert wrapper.aggregate is True
    assert wrapper.verbose is True
    assert wrapper.random_state == 42
    assert wrapper.model is not None


def test_subset_test_set(mock_evaluator_wrapper):
    """Tests the _subset_test_set method for correct subsetting."""
    wrapper = mock_evaluator_wrapper
    wrapper.df = pd.DataFrame({
        "pdgroupbase": [1, 2, 1, 1],
        "pdgrouprevaluation": [2, 2, 1, 3],
        "feature": [5, 6, 7, 8],
    })
    wrapper.X_test = pd.DataFrame({"feature": [5, 6, 7, 8]})
    wrapper.y_test = pd.Series([0, 1, 0, 1])

    X_subset, y_subset = wrapper._subset_test_set(
        base="pdgroupbase", revaluation="pdgrouprevaluation"
    )
    assert len(X_subset) == 2  # Ensure only rows with differences are included
    assert len(y_subset) == 2
    assert all(
        wrapper.df.loc[X_subset.index, "pdgroupbase"]
        != wrapper.df.loc[X_subset.index, "pdgrouprevaluation"]
    )


def test_test_filters(mock_evaluator_wrapper):
    """Test filtering of test data."""
    wrapper = mock_evaluator_wrapper
    wrapper.evaluator.X = pd.DataFrame({"feature": [5, 6, 7, 8]})
    wrapper.evaluator.y = pd.Series([1, 0, 1, 0])

    wrapper.evaluator.model_predictions = MagicMock(
        return_value=pd.Series([1, 0, 1, 1], index=wrapper.evaluator.y.index)
    )
    wrapper.evaluator.brier_scores = MagicMock(
        return_value=pd.Series([0.01, 0.2, 0.05, 0.3], index=wrapper.evaluator.y.index)
    )

    X_filtered, y_filtered, _ = wrapper._test_filters(
        X=wrapper.evaluator.X,
        y=wrapper.evaluator.y,
        base=None,
        revaluation=None,
        true_preds=True,
        brier_threshold=None,
    )
    assert len(X_filtered) == 3
    assert all(y_filtered == wrapper.evaluator.y.loc[X_filtered.index])

    X_filtered, y_filtered, _ = wrapper._test_filters(
        X=X_filtered,
        y=y_filtered,
        base=None,
        revaluation=None,
        true_preds=False,
        brier_threshold=0.1,
    )
    assert len(X_filtered) == 2

    X_filtered, y_filtered, _ = wrapper._test_filters(
        X=X_filtered,
        y=y_filtered,
        base=None,
        revaluation=None,
        true_preds=True,
        brier_threshold=0.05,
    )
    assert len(X_filtered) == 1


@patch("periomod.data.ProcessedDataLoader")
@patch("periomod.resampling.Resampler")
def test_evaluator_wrapper_prepare_data(mock_resampler_class, mock_dataloader_class):
    """Test data preperation."""
    mock_dataloader_instance = mock_dataloader_class.return_value
    mock_resampler_instance = mock_resampler_class.return_value

    mock_dataloader_instance.load_data.return_value = pd.DataFrame({
        "feature": [1, 2, 3]
    })
    mock_dataloader_instance.transform_data.return_value = pd.DataFrame({
        "feature": [1, 2, 3]
    })
    mock_resampler_instance.split_train_test_df.return_value = (
        pd.DataFrame(),
        pd.DataFrame(),
    )
    mock_resampler_instance.split_x_y.return_value = (
        pd.DataFrame(),
        pd.Series(),
        pd.DataFrame(),
        pd.Series(),
    )

    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": MagicMock()
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    wrapper._prepare_data_for_evaluation()


@patch("periomod.evaluation._baseeval.get_probs")
def test_evaluator_wrapper_wrapped_evaluation(mock_get_probs):
    """Test wrapped evaluation."""
    model_mock = MagicMock()
    model_mock.best_threshold = 0.5
    model_mock.classes_ = np.array([0, 1])
    mock_get_probs.return_value = np.array([0.6, 0.4, 0.7])

    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": model_mock
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    wrapper.evaluator.X = pd.DataFrame({"feature1": [1, 2, 3]})
    wrapper.evaluator.y = pd.Series([1, 0, 1])

    wrapper.wrapped_evaluation(cm=True, cm_base=False, brier_groups=False)


def test_evaluator_wrapper_evaluate_feature_importance():
    """Test feature importance."""
    rng = np.random.default_rng()
    model = LogisticRegression()
    model.coef_ = rng.random((1, 79))
    model.intercept_ = rng.random(1)
    model.classes_ = np.array([0, 1])

    model.predict_proba = MagicMock(return_value=rng.random((100, 2)))

    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": model
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    wrapper.evaluator.X = pd.DataFrame(
        rng.random((100, 79)), columns=[f"feature_{i}" for i in range(79)]
    )
    wrapper.evaluator.y = pd.Series(rng.integers(0, 2, size=100))

    wrapper.evaluate_feature_importance(fi_types=["shap", "permutation"])


@patch("periomod.inference.ModelInference", new_callable=MagicMock)
@patch("periomod.base.patient_to_df")
def test_evaluator_wrapper_wrapped_patient_inference(
    mock_patient_to_df, mock_inference_engine
):
    """Test wrapped patient inference."""
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
        teeth=[
            Tooth(
                tooth=11,
                toothtype=1,
                rootnumber=1,
                mobility=1,
                restoration=0,
                percussion=0,
                sensitivity=1,
                sides=[
                    Side(
                        furcationbaseline=1,
                        side=1,
                        pdbaseline=2,
                        recbaseline=2,
                        plaque=1,
                        bop=1,
                    ),
                ],
            ),
        ],
    )

    mock_model_inference_instance = MagicMock()
    mock_inference_engine.return_value = mock_model_inference_instance

    mock_model_inference_instance.prepare_inference.return_value = (
        pd.DataFrame({
            "pdbaseline": [2],
            "age": [45],
            "bodymassindex": [23.5],
            "recbaseline": [2],
            "cigarettenumber": [10],
            # Add other necessary columns
        }),
        pd.DataFrame(),
    )

    mock_model_inference_instance.patient_inference.return_value = pd.DataFrame({
        "Prediction": [1]
    })

    mock_patient_to_df.return_value = pd.DataFrame({
        "age": [45],
        "gender": [1],
        "bodymassindex": [23.5],
        "periofamilyhistory": [1],
        "diabetes": [0],
        "smokingtype": [2],
        "cigarettenumber": [10],
        "antibiotictreatment": [0],
        "stresslvl": [2],
        "id_patient": [1],
        "tooth": [11],
        "toothtype": [1],
        "rootnumber": [1],
        "mobility": [1],
        "restoration": [0],
        "percussion": [0],
        "sensitivity": [1],
        "furcationbaseline": [1],
        "side": [1],
        "pdbaseline": [2],
        "recbaseline": [2],
        "plaque": [1],
        "bop": [1],
        # Add missing columns for scaling
        "furcation12": [0],
        "furcation13": [0],
        "furcation23": [0],
        "pdbaseline_median": [2],
        "pdbaseline_max": [2],
        "recbaseline_median": [2],
        "recbaseline_max": [2],
        "furcationbaseline_median": [1],
        "furcationbaseline_max": [1],
    })

    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": MagicMock()
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    wrapper.inference_engine = mock_model_inference_instance
    result = wrapper.wrapped_patient_inference(patient=patient)
    mock_model_inference_instance.prepare_inference.assert_called_once()
    mock_model_inference_instance.patient_inference.assert_called_once()
    assert isinstance(result, pd.DataFrame)
    assert "Prediction" in result.columns


def test_evaluator_wrapper_wrapped_jackknife():
    """Test wrapped jackknife."""
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
        teeth=[
            Tooth(
                tooth=11,
                toothtype=1,
                rootnumber=1,
                mobility=1,
                restoration=0,
                percussion=0,
                sensitivity=1,
                sides=[
                    Side(
                        furcationbaseline=1,
                        side=1,
                        pdbaseline=2,
                        recbaseline=2,
                        plaque=1,
                        bop=1,
                    ),
                ],
            ),
        ],
    )

    mock_model_inference_instance = MagicMock()
    mock_model_inference_instance.prepare_inference.return_value = (
        pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]}),
        pd.DataFrame(),
    )
    mock_model_inference_instance.jackknife_inference.return_value = (
        pd.DataFrame({"Jackknife_Result": [0.1, 0.2, 0.3]}),
        MagicMock(),
    )

    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": MagicMock()
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    wrapper.inference_engine = mock_model_inference_instance

    results = pd.DataFrame()
    jackknife_results, _ = wrapper.wrapped_jackknife(
        patient=patient,
        results=results,
        sample_fraction=1.0,
        n_jobs=1,
        max_plots=192,
    )

    assert isinstance(jackknife_results, pd.DataFrame)
    assert "Jackknife_Result" in jackknife_results.columns


@patch("periomod.evaluation._eval.ModelEvaluator.plot_confusion_matrix")
@patch("periomod.evaluation._eval.ModelEvaluator.brier_score_groups")
@patch("periomod.evaluation._eval.ModelEvaluator.calibration_plot")
def test_wrapped_evaluation(mock_calibration, mock_brier_groups, mock_confusion_matrix):
    """Test wrapped_evaluation method."""
    model_mock = MagicMock()
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": model_mock
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    wrapper.evaluator.X = pd.DataFrame({"feature1": [1, 2, 3]})
    wrapper.evaluator.y = pd.Series([1, 0, 1])

    wrapper.wrapped_evaluation(
        cm=True,
        cm_base=False,
        brier_groups=True,
        calibration=True,
        tight_layout=True,
    )

    mock_confusion_matrix.assert_called_once_with(
        tight_layout=True,
        normalize="rows",
        task="pocketclosure",
        save=False,
        name="cm_predictionEval",
    )
    mock_brier_groups.assert_called_once_with(
        tight_layout=True, task="pocketclosure", save=False, name=None
    )
    mock_calibration.assert_called_once_with(
        task="pocketclosure", tight_layout=True, save=False, name=None
    )


@patch("periomod.evaluation._eval.ModelEvaluator.bss_comparison")
@patch("periomod.benchmarking._baseline.Baseline.train_baselines")
@patch("periomod.wrapper._wrapper.EvaluatorWrapper._test_filters")
def test_compare_bss(mock_test_filters, mock_train_baselines, mock_bss_comparison):
    """Test compare_bss method."""
    model_mock = MagicMock()
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": model_mock
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    mock_test_filters.return_value = (MagicMock(), MagicMock(), 150)
    mock_train_baselines.return_value = ({}, None, None)

    wrapper.compare_bss(
        base="baseline_var",
        revaluation="revaluation_var",
        true_preds=True,
        brier_threshold=0.1,
        tight_layout=True,
    )

    mock_test_filters.assert_called_once()
    mock_bss_comparison.assert_called_once_with(
        baseline_models={},
        classification=wrapper.classification,
        num_patients=150,
        tight_layout=True,
        save=False,
        name=None,
    )


@patch("periomod.evaluation._eval.ModelEvaluator.analyze_brier_within_clusters")
@patch("periomod.wrapper._wrapper.EvaluatorWrapper._test_filters")
def test_evaluate_cluster(mock_test_filters, mock_analyze_clusters):
    """Test evaluate_cluster method."""
    model_mock = MagicMock()
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": model_mock
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    mock_test_filters.return_value = (MagicMock(), MagicMock(), 120)

    wrapper.evaluate_cluster(
        n_cluster=3,
        base="baseline_var",
        revaluation="revaluation_var",
        true_preds=True,
        brier_threshold=0.05,
        tight_layout=True,
    )

    mock_test_filters.assert_called_once()
    mock_analyze_clusters.assert_called_once_with(n_clusters=3, tight_layout=True)


@patch("periomod.evaluation._eval.ModelEvaluator.evaluate_feature_importance")
@patch("periomod.wrapper._wrapper.EvaluatorWrapper._test_filters")
def test_evaluate_feature_importance(mock_test_filters, mock_evaluate_fi):
    """Test evaluate_feature_importance method."""
    model_mock = MagicMock()
    learners_dict = {
        "pocketclosure_lr_cv_hebo_f1_one_hot_no_sampling_factor2_rank1_": model_mock
    }

    wrapper = EvaluatorWrapper(
        learners_dict=learners_dict,
        criterion="f1",
        aggregate=True,
        verbose=True,
    )

    mock_test_filters.return_value = (MagicMock(), MagicMock(), 100)

    wrapper.evaluate_feature_importance(
        fi_types=["shap", "permutation"],
        base="baseline_var",
        revaluation="revaluation_var",
        true_preds=False,
        brier_threshold=None,
    )

    mock_test_filters.assert_called_once()
    mock_evaluate_fi.assert_called_once_with(
        fi_types=["shap", "permutation"], save=False, name=None
    )
