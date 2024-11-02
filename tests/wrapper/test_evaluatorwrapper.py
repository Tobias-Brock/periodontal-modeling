"""Tests for EvaluatorWrapper."""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from periomod.base import Patient, Side, Tooth
from periomod.wrapper import EvaluatorWrapper


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


@patch("periomod.data.ProcessedDataLoader")
@patch("periomod.resampling.Resampler")
def test_evaluator_wrapper_prepare_data(mock_resampler_class, mock_dataloader_class):
    """Test data preperation."""
    mock_dataloader_instance = mock_dataloader_class.return_value
    mock_resampler_instance = mock_resampler_class.return_value

    mock_dataloader_instance.load_data.return_value = pd.DataFrame(
        {"feature": [1, 2, 3]}
    )
    mock_dataloader_instance.transform_data.return_value = pd.DataFrame(
        {"feature": [1, 2, 3]}
    )
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

    df, train_df, test_df, X_train, y_train, X_test, y_test, base_target = (
        wrapper._prepare_data_for_evaluation()
    )


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

    wrapper.wrapped_evaluation(
        cm=True, cm_base=False, brier_groups=False, cluster=False
    )


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
        pd.DataFrame(
            {
                "pdbaseline": [2],
                "age": [45],
                "bodymassindex": [23.5],
                "recbaseline": [2],
                "cigarettenumber": [10],
                # Add other necessary columns
            }
        ),
        pd.DataFrame(),
    )

    mock_model_inference_instance.patient_inference.return_value = pd.DataFrame(
        {"Prediction": [1]}
    )

    mock_patient_to_df.return_value = pd.DataFrame(
        {
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
        }
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
