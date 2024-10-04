"""Base classes."""

from dataclasses import dataclass
from typing import Optional

import hydra
import pandas as pd


def validate_classification(classification: str) -> None:
    """Validates the classification type.

    Args:
        classification (str): The type of classification ('binary' or 'multiclass').
    """
    if classification.lower().strip() not in ["binary", "multiclass"]:
        raise ValueError(
            "Invalid classification type. Choose 'binary' or 'multiclass'."
        )


def validate_hpo(hpo: Optional[str]) -> None:
    """Validates the hpo.

    Args:
        hpo (Optional[str]): The type of hpo ('rs' or 'hebo').
    """
    if hpo not in [None, "rs", "hebo"]:
        raise ValueError("Unsupported HPO. Choose either 'rs' or 'hebo'.")


class BaseValidator:
    """BaseValidator method."""

    def __init__(self, classification: str, hpo: Optional[str] = None) -> None:
        """Base class to provide validation and error handling for other classes.

        This class handles DataFrame validation, column checking, and numerical
        input checking.

        Args:
            classification (str): The type of classification ('binary' or
                'multiclass').
            hpo (Optional[str], optional): The hyperparameter optimization type
                ('rs' or 'hebo'). Defaults to None.
        """
        with hydra.initialize(config_path="../config", version_base="1.2"):
            cfg = hydra.compose(config_name="config")
        validate_classification(classification)
        validate_hpo(hpo)
        self.classification = classification
        self.hpo = hpo
        self.random_state_sampling = cfg.resample.random_state_sampling
        self.random_state_split = cfg.resample.random_state_split
        self.random_state_cv = cfg.resample.random_state_cv
        self.random_state_val = cfg.tuning.random_state_val
        self.test_set_size = cfg.resample.test_set_size
        self.group_col = cfg.resample.group_col
        self.n_folds = cfg.resample.n_folds
        self.random_state_model = cfg.learner.random_state_model
        self.xgb_obj_binary = cfg.learner.xgb_obj_binary
        self.xgb_loss_binary = cfg.learner.xgb_loss_binary
        self.xgb_obj_multi = cfg.learner.xgb_obj_multi
        self.xgb_loss_multi = cfg.learner.xgb_loss_multi
        self.lr_solver_binary = cfg.learner.lr_solver_binary
        self.lr_solver_multi = cfg.learner.lr_solver_multi
        self.lr_multi_loss = cfg.learner.lr_multi_loss

    def validate_dataframe(self, df: pd.DataFrame, required_columns: list) -> None:
        """Validates that the input is a pandas DataFrame and contains required columns.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            required_columns (list): A list of column names that are required in the
                DataFrame.

        Raises:
            TypeError: If the input is not a pandas DataFrame.
            ValueError: If required columns are missing from the DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected input to be a pandas DataFrame, but got {type(df)}."
            )

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"The following required columns are missing: "
                f"{', '.join(missing_columns)}."
            )

    def validate_n_folds(self, n_folds: int) -> None:
        """Validates the number of folds used in cross-validation.

        Args:
            n_folds (int): The number of folds for cross-validation.

        Raises:
            ValueError: If the number of folds is not a positive integer.
        """
        if not isinstance(n_folds, int) or n_folds <= 0:
            raise ValueError("'n_folds' must be a positive integer.")

    def validate_sampling_strategy(self, sampling: str) -> None:
        """Validates the sampling strategy.

        Args:
            sampling (str): The sampling strategy to validate.

        Raises:
            ValueError: If the sampling strategy is invalid.
        """
        valid_strategies = ["smote", "upsampling", "downsampling", None]
        if sampling not in valid_strategies:
            raise ValueError(
                f"Invalid sampling strategy: {sampling}. Valid options are "
                f"{valid_strategies}."
            )


class BaseEvaluator:
    """_BaseEvaluator method."""

    def __init__(
        self,
        classification: str,
        criterion: str,
        tuning: Optional[str] = None,
        hpo: Optional[str] = None,
    ) -> None:
        """Base class to initialize classification, criterion, tuning, and hpo.

        Args:
            classification (str): The type of classification ('binary' or
                'multiclass').
            criterion (str): The evaluation criterion.
            tuning (Optional[str], optional): The tuning method ('holdout' or 'cv').
                Defaults to None.
            hpo (Optional[str], optional): The hyperparameter optimization type ('rs'
                or 'hebo'). Defaults to None.
        """
        with hydra.initialize(config_path="../config", version_base="1.2"):
            self.cfg = hydra.compose(config_name="config")
        validate_classification(classification)
        self._validate_criterion(criterion)
        validate_hpo(hpo)
        self._validate_tuning(tuning)
        self.classification = classification
        self.criterion = criterion
        self.hpo = hpo
        self.tuning = tuning
        self.random_state = self.cfg.resample.random_state_cv
        self.random_state_val = self.cfg.tuning.random_state_val
        self.tol = self.cfg.mlp.mlp_tol
        self.n_iter_no_change = self.cfg.mlp.mlp_no_improve

    def _validate_criterion(self, criterion: str) -> None:
        """Validates the evaluation criterion.

        Args:
            criterion (str): The evaluation criterion ('f1', 'macro_f1', or
                'brier_score').

        Raises:
            ValueError: If the criterion is unsupported.
        """
        if criterion not in ["f1", "macro_f1", "brier_score"]:
            raise ValueError(
                "Unsupported criterion. Choose either 'f1', 'macro_f1', or "
                "'brier_score'."
            )

    def _validate_tuning(self, tuning: Optional[str]) -> None:
        """Validates the tuning method.

        Args:
            tuning (Optional[str]): The type of tuning ('holdout' or 'cv').

        Raises:
            ValueError: If the tuning method is unsupported.
        """
        if tuning not in [None, "holdout", "cv"]:
            raise ValueError(
                "Unsupported tuning method. Choose either 'holdout' or 'cv'."
            )


class BaseData:
    """Base class for common data attributes used in processing periodontal datasets."""

    def __init__(self) -> None:
        """Initializes the BaseData class with shared attributes."""
        self.required_columns = [
            "ID_patient",
            "Tooth",
            "Toothtype",
            "RootNumber",
            "Mobility",
            "Restoration",
            "Percussion-sensitivity",
            "Sensitivity",
            "FurcationBaseline",
            "Side",
            "PdBaseline",
            "RecBaseline",
            "Plaque",
            "BOP",
            "Age",
            "Gender",
            "BodyMassIndex",
            "PerioFamilyHistory",
            "Diabetes",
            "SmokingType",
            "CigaretteNumber",
            "AntibioticTreatment",
            "Stresslvl",
            "PdRevaluation",
            "BOPRevaluation",
            "Pregnant",
        ]
        self.cat_vars = [
            "side",
            "restoration",
            "periofamilyhistory",
            "diabetes",
            "toothtype",
            "tooth",
            "furcationbaseline",
            "smokingtype",
            "stresslvl",
            "toothside",
        ]
        self.bin_var = [
            "antibiotictreatment",
            "boprevaluation",
            "plaque",
            "bop",
            "mobility",
            "percussion-sensitivity",
            "sensitivity",
            "rootnumber",
            "gender",
        ]
        self.scale_vars = [
            "pdbaseline",
            "age",
            "bodymassindex",
            "recbaseline",
            "cigarettenumber",
        ]
        self.behavior_columns = {
            "binary": ["Flossing", "IDB", "SweetFood", "SweetDrinks", "ErosiveDrinks"],
            "categorical": [
                "OrthoddonticHistory",
                "DentalVisits",
                "Toothbrushing",
                "DryMouth",
            ],
        }
        self.all_cat_vars = self.cat_vars + self.behavior_columns["categorical"]


@dataclass
class InferenceInput:
    """Dataclass required for predictions.

    Attributes:
        tooth (int): The tooth number provided for inference.
        toothtype (int): The type or class of the tooth for prediction.
        rootnumber (int): Number of roots associated with the tooth.
        mobility (int): The mobility level of the tooth.
        restoration (int): Restoration status of the tooth.
        percussion (int): Percussion sensitivity level of the tooth.
        sensitivity (int): General sensitivity of the tooth.
        furcation (int): The furcation baseline value.
        side (int): Side of the tooth for prediction.
        pdbaseline (int): Periodontal depth baseline value.
        recbaseline (int): Recession baseline value.
        plaque (int): Plaque level associated with the tooth.
        bop (int): Bleeding on probing (BOP) status.
        age (int): Age of the patient.
        gender (int): Gender of the patient.
        bmi (float): Body mass index of the patient.
        perio_history (int): Family history of periodontal disease.
        diabetes (int): Diabetes status of the patient.
        smokingtype (int): Type of smoking habits of the patient.
        cigarettenumber (int): Number of cigarettes smoked per day.
        antibiotics (int): Antibiotic treatment status.
        stresslvl (str): Stress level of the patient.

    Methods:
        to_dict() -> dict: Converts fields of dataclass into a dictionary.
    """

    tooth: int
    toothtype: int
    rootnumber: int
    mobility: int
    restoration: int
    percussion: int
    sensitivity: int
    furcation: int
    side: int
    pdbaseline: int
    recbaseline: int
    plaque: int
    bop: int
    age: int
    gender: int
    bmi: float
    perio_history: int
    diabetes: int
    smokingtype: int
    cigarettenumber: int
    antibiotics: int
    stresslvl: str

    def to_dict(self) -> dict:
        """Convert the dataclass fields to a dictionary."""
        return {
            "tooth": self.tooth,
            "toothtype": self.toothtype,
            "side": self.side,
            "rootnumber": self.rootnumber,
            "furcationbaseline": self.furcation,
            "mobility": self.mobility,
            "percussion-sensitivity": self.percussion,
            "sensitivity": self.sensitivity,
            "pdbaseline": self.pdbaseline,
            "recbaseline": self.recbaseline,
            "plaque": self.plaque,
            "bop": self.bop,
            "age": self.age,
            "gender": self.gender,
            "bodymassindex": self.bmi,
            "cigarettenumber": self.cigarettenumber,
            "antibiotictreatment": self.antibiotics,
            "restoration": self.restoration,
            "periofamilyhistory": self.perio_history,
            "diabetes": self.diabetes,
            "smokingtype": self.smokingtype,
            "stresslvl": self.stresslvl,
        }


def create_predict_data(
    raw_data: pd.DataFrame, input_data: InferenceInput, encoding: str, model
) -> pd.DataFrame:
    """Creates prediction data for model inference.

    Args:
        raw_data (pd.DataFrame): The raw, unencoded data for the input instance.
        input_data (InferenceInput): Input data provided by the user for inference.
        encoding (str): Type of encoding used ('one_hot' or 'target').
        model: The trained model to retrieve feature names from.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared data for model prediction.

    """
    base_data = raw_data.copy()

    if encoding == "one_hot":
        encoded_data = {f"side_{i}": 0 for i in range(1, 7)}
        encoded_data.update({"infected_neighbors": 0})

        for tooth_num in range(11, 49):
            if tooth_num % 10 == 0 or tooth_num % 10 == 9:
                continue
            encoded_data[f"tooth_{tooth_num}"] = 0

        for feature, max_val in [
            ("restoration", 3),
            ("periofamilyhistory", 3),
            ("diabetes", 4),
            ("furcationbaseline", 3),
            ("smokingtype", 5),
            ("toothtype", 3),
        ]:
            for i in range(1, max_val + 1):
                encoded_data[f"{feature}_{i}"] = 0

        for stresslvl in ["high", "low", "medium"]:
            encoded_data[f"stresslvl_{stresslvl}"] = 0

        encoded_data[f"toothtype_{input_data.toothtype}"] = 1
        encoded_data[f"side_{input_data.side}"] = 1
        encoded_data[f"furcationbaseline_{input_data.furcation}"] = 1
        encoded_data[f"smokingtype_{input_data.smokingtype}"] = 1
        encoded_data[f"restoration_{input_data.restoration}"] = 1
        encoded_data[f"periofamilyhistory_{input_data.perio_history}"] = 1
        encoded_data[f"diabetes_{input_data.diabetes}"] = 1
        encoded_data[f"tooth_{input_data.tooth}"] = 1
        encoded_data[f"stresslvl_{input_data.stresslvl}"] = 1

        drop_columns = [
            "tooth",
            "side",
            "restoration",
            "periofamilyhistory",
            "diabetes",
            "toothtype",
            "furcationbaseline",
            "smokingtype",
            "stresslvl",
        ]
        base_data = base_data.drop(columns=drop_columns, errors="ignore")
        complete_data = {**base_data.iloc[0].to_dict(), **encoded_data}

    elif encoding == "target":
        complete_data = base_data.iloc[0].to_dict()
        complete_data.update({"infected_neighbors": 0})
        complete_data.update(
            {
                "rootnumber": input_data.rootnumber,
                "mobility": input_data.mobility,
                "percussion-sensitivity": input_data.percussion,
                "sensitivity": input_data.sensitivity,
                "pdbaseline": input_data.pdbaseline,
                "recbaseline": input_data.recbaseline,
                "plaque": input_data.plaque,
                "bop": input_data.bop,
                "age": input_data.age,
                "gender": input_data.gender,
                "bodymassindex": input_data.bmi,
                "cigarettenumber": input_data.cigarettenumber,
                "antibiotictreatment": input_data.antibiotics,
            }
        )

    model_features = model.get_booster().feature_names
    predict_data = pd.DataFrame([complete_data])
    predict_data = predict_data[model_features]

    return predict_data
