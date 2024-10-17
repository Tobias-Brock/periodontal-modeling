"""Base Methods."""

from dataclasses import dataclass, field
from typing import List, Optional

import hydra
import pandas as pd


def validate_classification(classification: str) -> None:
    """Validates the classification type.

    Args:
        classification (str): The type of classification ('binary' or 'multiclass').
    """
    if classification.lower().strip() not in ["binary", "multiclass"]:
        raise ValueError(
            f"{classification} is an invalid classification type."
            f"Choose 'binary' or 'multiclass'."
        )


def validate_hpo(hpo: Optional[str]) -> None:
    """Validates the hpo.

    Args:
        hpo (Optional[str]): The type of hpo ('rs' or 'hebo').
    """
    if hpo not in [None, "rs", "hebo"]:
        raise ValueError(
            f" {hpo} is an unsupported HPO type." f"Choose either 'rs' or 'hebo'."
        )


class BaseHydra:
    """Base class to initialize Hydra configuration."""

    def __init__(
        self, config_path: str = "../config", config_name: str = "config"
    ) -> None:
        """Initializes the Hydra configuration for use in other classes.

        Args:
            config_path (str): Path to the Hydra config directory.
            config_name (str): The name of the config file.
        """
        with hydra.initialize(config_path=config_path, version_base="1.2"):
            self.cfg = hydra.compose(config_name=config_name)

        self.random_state_sampling = self.cfg.resample.random_state_sampling
        self.random_state_split = self.cfg.resample.random_state_split
        self.random_state_cv = self.cfg.resample.random_state_cv
        self.random_state_val = self.cfg.tuning.random_state_val
        self.test_set_size = self.cfg.resample.test_set_size
        self.val_set_size = self.cfg.resample.val_set_size
        self.group_col = self.cfg.resample.group_col
        self.n_folds = self.cfg.resample.n_folds
        self.target = self.cfg.resample.target
        self.random_state_model = self.cfg.learner.random_state_model
        self.xgb_obj_binary = self.cfg.learner.xgb_obj_binary
        self.xgb_loss_binary = self.cfg.learner.xgb_loss_binary
        self.xgb_obj_multi = self.cfg.learner.xgb_obj_multi
        self.xgb_loss_multi = self.cfg.learner.xgb_loss_multi
        self.lr_solver_binary = self.cfg.learner.lr_solver_binary
        self.lr_solver_multi = self.cfg.learner.lr_solver_multi
        self.lr_multi_loss = self.cfg.learner.lr_multi_loss
        self.random_state = self.cfg.resample.random_state_cv
        self.random_state_val = self.cfg.tuning.random_state_val
        self.tol = self.cfg.mlp.mlp_tol
        self.n_iter_no_change = self.cfg.mlp.mlp_no_improve
        self.mlp_training = self.cfg.mlp.mlp_training


class BaseValidator(BaseHydra):
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
        super().__init__()
        validate_classification(classification)
        validate_hpo(hpo)
        self.classification = classification
        self.hpo = hpo

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: list) -> None:
        """Validate input is a pandas DataFrame and contains required columns.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            required_columns (list): A list of column names that are required in
                the DataFrame.

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

    @staticmethod
    def validate_n_folds(n_folds: int) -> None:
        """Validates the number of folds used in cross-validation.

        Args:
            n_folds (int): The number of folds for cross-validation.

        Raises:
            ValueError: If the number of folds is not a positive integer.
        """
        if not isinstance(n_folds, int) or n_folds <= 0:
            raise ValueError("'n_folds' must be a positive integer.")

    @staticmethod
    def validate_sampling_strategy(sampling: str) -> None:
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


class BaseEvaluator(BaseHydra):
    """BaseEvaluator method."""

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
            tuning (Optional[str], optional): The tuning method ('holdout' or
                'cv'). Defaults to None.
            hpo (Optional[str], optional): The hyperparameter optimization type
                ('rs' or 'hebo'). Defaults to None.
        """
        super().__init__()  # Initialize BaseHydra
        validate_classification(classification)
        self._validate_criterion(criterion)
        validate_hpo(hpo)
        self._validate_tuning(tuning)
        self.classification = classification
        self.criterion = criterion
        self.hpo = hpo
        self.tuning = tuning

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
        self.patient_columns = [
            "ID_patient",
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
        self.tooth_columns = [
            "Tooth",
            "Toothtype",
            "RootNumber",
            "Mobility",
            "Restoration",
            "Percussion-sensitivity",
            "Sensitivity",
        ]
        self.side_columns = [
            "FurcationBaseline",
            "Side",
            "PdBaseline",
            "RecBaseline",
            "Plaque",
            "BOP",
        ]
        self.required_columns = (
            self.patient_columns + self.tooth_columns + self.side_columns
        )
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
class Side:
    """Dataclass to represent a single side of a tooth."""

    furcationbaseline: Optional[int]
    side: int
    pdbaseline: Optional[int]
    recbaseline: Optional[int]
    plaque: Optional[int]
    bop: Optional[int]


@dataclass
class Tooth:
    """Dataclass to represent a tooth, which contains up to 6 sides."""

    tooth: int
    toothtype: int
    rootnumber: int
    mobility: Optional[int]
    restoration: Optional[int]
    percussion: Optional[int]
    sensitivity: Optional[int]
    sides: List[Side] = field(default_factory=list)


@dataclass
class Patient:
    """Dataclass to represent teeth and patient-level information."""

    age: int
    gender: int
    bmi: float
    perio_history: int
    diabetes: int
    smokingtype: int
    cigarettenumber: int
    antibiotics: int
    stresslvl: int
    teeth: List[Tooth] = field(default_factory=list)


def create_predict_data(
    raw_data: pd.DataFrame, patient_data: pd.DataFrame, encoding: str, model
) -> pd.DataFrame:
    """Creates prediction data for model inference.

    Args:
        raw_data (pd.DataFrame): The raw, preprocessed data.
        patient_data (pd.DataFrame): Original patient data before preprocessing.
        encoding (str): Type of encoding used ('one_hot' or 'target').
        model: The trained model to retrieve feature names from.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared data for model prediction.
    """
    base_data = raw_data.copy()

    if encoding == "one_hot":
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
            "infected_neighbors",
            "side_infected",
            "tooth_infected",
        ]
        base_data = base_data.drop(columns=drop_columns, errors="ignore")
        encoded_data = pd.DataFrame(index=base_data.index)

        for i in range(1, 7):
            encoded_data[f"side_{i}"] = 0

        for tooth_num in range(11, 49):
            if tooth_num % 10 == 0 or tooth_num % 10 == 9:
                continue
            encoded_data[f"tooth_{tooth_num}"] = 0

        categorical_features = [
            ("restoration", 3),
            ("periofamilyhistory", 3),
            ("diabetes", 4),
            ("furcationbaseline", 3),
            ("smokingtype", 5),
            ("stresslvl", 3),
            ("toothtype", 3),
        ]

        for feature, max_val in categorical_features:
            for i in range(0, max_val + 1):
                encoded_data[f"{feature}_{i}"] = 0

        for idx, row in patient_data.iterrows():
            encoded_data.at[idx, f"tooth_{row['tooth']}"] = 1
            encoded_data.at[idx, f"side_{row['side']}"] = 1

            encoded_data.at[idx, f"toothtype_{row['toothtype']}"] = 1
            encoded_data.at[idx, f"furcationbaseline_{row['furcationbaseline']}"] = 1
            encoded_data.at[idx, f"smokingtype_{row['smokingtype']}"] = 1
            encoded_data.at[idx, f"restoration_{row['restoration']}"] = 1
            encoded_data.at[idx, f"periofamilyhistory_{row['periofamilyhistory']}"] = 1
            encoded_data.at[idx, f"diabetes_{row['diabetes']}"] = 1
            encoded_data.at[idx, f"stresslvl_{row['stresslvl']}"] = 1

        complete_data = pd.concat(
            [base_data.reset_index(drop=True), encoded_data.reset_index(drop=True)],
            axis=1,
        )

        complete_data = complete_data.loc[:, ~complete_data.columns.duplicated()]
        duplicates = complete_data.columns[complete_data.columns.duplicated()].unique()
        if len(duplicates) > 0:
            print("Duplicate columns found:", duplicates)

    elif encoding == "target":
        complete_data = base_data.copy()
        numerical_columns = {
            "mobility": "mobility",
            "percussion-sensitivity": "percussion-sensitivity",
            "sensitivity": "sensitivity",
            "pdbaseline": "pdbaseline",
            "recbaseline": "recbaseline",
            "plaque": "plaque",
            "bop": "bop",
            "age": "age",
            "gender": "gender",
            "bodymassindex": "bodymassindex",
            "cigarettenumber": "cigarettenumber",
            "antibiotictreatment": "antibiotictreatment",
            "rootnumber": "rootnumber",
        }
        for key, value in numerical_columns.items():
            if value in patient_data.columns:
                complete_data[key] = patient_data[value].values

    else:
        raise ValueError(f"Unsupported encoding type: {encoding}")

    if hasattr(model, "get_booster"):
        model_features = model.get_booster().feature_names
    elif hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_
    else:
        raise ValueError("Model type not supported for feature extraction")

    for feature in model_features:
        if feature not in complete_data.columns:
            complete_data[feature] = 0

    predict_data = complete_data[model_features]

    return predict_data
