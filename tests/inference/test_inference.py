"""Tests for inference module."""

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from periomod.inference import ModelInference


def test_model_inference_initialization():
    """Test for ModelInference init."""
    mock_model = MagicMock()
    mock_model.predict_proba = MagicMock()
    mock_model.classes_ = np.array([0, 1])

    model_inference = ModelInference(
        classification="binary", model=mock_model, verbose=True
    )

    assert model_inference.classification == "binary"
    assert model_inference.model == mock_model
    assert model_inference.verbose is True


def test_model_inference_predict():
    """Test predict method."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
    mock_model.classes_ = np.array([0, 1])
    mock_model.predict.return_value = np.array([1, 0])
    mock_model.best_threshold = None

    model_inference = ModelInference(
        classification="binary", model=mock_model, verbose=False
    )

    input_data = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})

    result = model_inference.predict(input_data)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["0", "1", "prediction"]
    assert result["prediction"].tolist() == [1, 0]
    np.testing.assert_array_almost_equal(result["0"].values, [0.3, 0.6])
    np.testing.assert_array_almost_equal(result["1"].values, [0.7, 0.4])


@patch("periomod.inference._baseinference.Resampler")
@patch("periomod.inference._baseinference.ProcessedDataLoader")
@patch("periomod.inference._baseinference.StaticProcessEngine")
def test_model_inference_prepare_inference(
    mock_engine_class, mock_dataloader_class, mock_resampler_class
):
    """Test data preprocessing for inference."""
    mock_engine_instance = mock_engine_class.return_value
    mock_dataloader_instance = mock_dataloader_class.return_value
    mock_resampler_instance = mock_resampler_class.return_value
    mock_engine_instance.create_tooth_features.return_value = pd.DataFrame({
        "feature": [1, 2, 3]
    })
    mock_dataloader_instance.encode_categorical_columns.return_value = pd.DataFrame({
        "feature": [1, 2, 3]
    })
    mock_dataloader_instance.scale_numeric_columns.return_value = pd.DataFrame({
        "scaled_feature": [0.1, 0.2, 0.3]
    })
    mock_resampler_instance.apply_target_encoding.return_value = (
        pd.DataFrame(),
        pd.DataFrame(),
    )

    mock_dataloader_instance.scale_vars = ["feature"]
    mock_model = MagicMock()
    mock_model.feature_names_in_ = ["feature"]
    model_inference = ModelInference(
        classification="binary", model=mock_model, verbose=False
    )

    model_inference.group_col = "patient_id"
    model_inference.cat_vars = []
    model_inference.infect_vars = []
    model_inference.cat_map = {}
    model_inference.target_cols = []

    patient_data = pd.DataFrame({
        "id_patient": [1],
        "age": [45],
        "gender": [1],
        "bodymassindex": [25],
        "periofamilyhistory": [2],
        "diabetes": [2],
        "smokingtype": [1],
        "cigarettenumber": [0],
        "antibiotictreatment": [0],
        "tooth": [11],
        "feature": [1],
        "pdbaseline": [3],
        "bop": [1],
        "side": [1],
        "recbaseline": [2],
        "plaque": [1],
        "furcationbaseline": [2],
    })
    X_train = pd.DataFrame()
    y_train = pd.Series()

    predict_data, patient_data_processed = model_inference.prepare_inference(
        task="pocketclosure",
        patient_data=patient_data,
        encoding="target",
        X_train=X_train,
        y_train=y_train,
    )

    assert isinstance(predict_data, pd.DataFrame)
    assert isinstance(patient_data_processed, pd.DataFrame)
    mock_engine_instance.create_tooth_features.assert_called_once()
    mock_dataloader_instance.encode_categorical_columns.assert_called()
    mock_dataloader_instance.scale_numeric_columns.assert_called()


def test_model_inference_patient_inference():
    """Test patient inference."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
    mock_model.classes_ = np.array([0, 1])
    mock_model.predict.return_value = np.array([1])

    model_inference = ModelInference(
        classification="binary", model=mock_model, verbose=False
    )

    predict_data = pd.DataFrame({"feature": [1]})
    patient_data = pd.DataFrame({"tooth": [11], "side": ["left"]})

    predict_data_result, output_data, results = model_inference.patient_inference(
        predict_data=predict_data, patient_data=patient_data
    )

    assert isinstance(predict_data_result, pd.DataFrame)
    assert isinstance(output_data, pd.DataFrame)
    assert isinstance(results, pd.DataFrame)
    assert "prediction" in output_data.columns
    assert "probability" in output_data.columns
    assert output_data["prediction"].iloc[0] == 1
    np.testing.assert_almost_equal(output_data["probability"].iloc[0], 0.6)


def test_model_inference_jackknife_resampling():
    """Test jackknife resampling."""
    mock_model = MagicMock()
    model_inference = ModelInference(
        classification="binary", model=mock_model, verbose=False
    )

    with patch.object(model_inference, "process_patient") as mock_process_patient:
        mock_process_patient.return_value = pd.DataFrame({
            "0": [0.5],
            "1": [0.5],
            "prediction": [1],
            "iteration": [1],
            "data_index": [0],
        })

        train_df = pd.DataFrame({
            "patient_id": [1, 2],
            "feature": [1, 2],
            "target": [0, 1],
        })
        model_inference.group_col = "patient_id"
        patient_data = pd.DataFrame({"feature": [1]})
        model_params = {}

        jackknife_results = model_inference.jackknife_resampling(
            train_df=train_df,
            patient_data=patient_data,
            encoding="one_hot",
            model_params=model_params,
            sample_fraction=1.0,
            n_jobs=1,
        )

        assert isinstance(jackknife_results, pd.DataFrame)
        assert "prediction" in jackknife_results.columns
        assert "iteration" in jackknife_results.columns
        assert len(jackknife_results) == 2
        mock_process_patient.assert_called()


def test_model_inference_jackknife_confidence_intervals():
    """Test jackknife confidence intervals."""
    model_inference = ModelInference(
        classification="binary", model=MagicMock(), verbose=False
    )
    jackknife_results = pd.DataFrame({
        "0": [0.4, 0.5, 0.6],
        "1": [0.6, 0.5, 0.4],
        "prediction": [1, 1, 0],
        "iteration": [1, 2, 3],
        "data_index": [0, 0, 0],
    })

    ci_dict = model_inference.jackknife_confidence_intervals(
        jackknife_results=jackknife_results, alpha=0.05
    )

    assert isinstance(ci_dict, dict)
    assert 0 in ci_dict  # data_index
    assert "0" in ci_dict[0]
    assert "1" in ci_dict[0]
    for class_name in ["0", "1"]:
        assert "mean" in ci_dict[0][class_name]
        assert "lower" in ci_dict[0][class_name]
        assert "upper" in ci_dict[0][class_name]


def test_model_inference_plot_jackknife_intervals():
    """Test jackknife plots."""
    model_inference = ModelInference(
        classification="binary", model=MagicMock(), verbose=False
    )

    ci_dict = {
        0: {
            "0": {"mean": 0.5, "lower": 0.4, "upper": 0.6},
            "1": {"mean": 0.5, "lower": 0.4, "upper": 0.6},
        }
    }
    data_indices = [0]
    original_preds = pd.DataFrame(
        {"0": [0.5], "1": [0.5], "prediction": [1]}, index=[0]
    )

    fig = model_inference.plot_jackknife_intervals(
        ci_dict=ci_dict, data_indices=data_indices, original_preds=original_preds
    )

    assert isinstance(fig, plt.Figure)


@patch.object(ModelInference, "jackknife_resampling")
@patch.object(ModelInference, "jackknife_confidence_intervals")
@patch.object(ModelInference, "plot_jackknife_intervals")
def test_model_inference_jackknife_inference(
    mock_plot_jackknife_intervals,
    mock_jackknife_confidence_intervals,
    mock_jackknife_resampling,
):
    """Test jackknife inference."""
    mock_jackknife_resampling.return_value = pd.DataFrame({
        "0": [0.5],
        "1": [0.5],
        "prediction": [1],
        "iteration": [1],
        "data_index": [0],
    })
    mock_jackknife_confidence_intervals.return_value = {
        0: {
            "0": {"mean": 0.5, "lower": 0.4, "upper": 0.6},
            "1": {"mean": 0.5, "lower": 0.4, "upper": 0.6},
        }
    }
    mock_plot_jackknife_intervals.return_value = plt.figure()

    model_inference = ModelInference(
        classification="binary", model=MagicMock(), verbose=False
    )

    model = MagicMock()
    train_df = pd.DataFrame()
    patient_data = pd.DataFrame()
    inference_results = pd.DataFrame({"prediction": [1]})

    jackknife_results, ci_plot = model_inference.jackknife_inference(
        model=model,
        train_df=train_df,
        patient_data=patient_data,
        encoding="one_hot",
        inference_results=inference_results,
        alpha=0.05,
        sample_fraction=1.0,
        n_jobs=1,
        max_plots=12,
    )

    assert isinstance(jackknife_results, pd.DataFrame)
    assert isinstance(ci_plot, plt.Figure)
    mock_jackknife_resampling.assert_called_once()
    mock_jackknife_confidence_intervals.assert_called_once()
    mock_plot_jackknife_intervals.assert_called_once()
