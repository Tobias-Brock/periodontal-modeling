"""Base Methods."""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import hydra
import pandas as pd


def _validate_classification(classification: str) -> None:
    """Validates the classification type.

    Args:
        classification (str): The type of classification ('binary' or 'multiclass').
    """
    if classification.lower().strip() not in ["binary", "multiclass"]:
        raise ValueError(
            f"{classification} is an invalid classification type."
            f"Choose 'binary' or 'multiclass'."
        )


def _validate_hpo(hpo: Optional[str]) -> None:
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
            cfg = hydra.compose(config_name=config_name)

        self.random_state_sampling = cfg.resample.random_state_sampling
        self.random_state_split = cfg.resample.random_state_split
        self.random_state_cv = cfg.resample.random_state_cv
        self.random_state_val = cfg.tuning.random_state_val
        self.test_set_size = cfg.resample.test_set_size
        self.val_set_size = cfg.resample.val_set_size
        self.group_col = cfg.resample.group_col
        self.n_folds = cfg.resample.n_folds
        self.y = cfg.resample.y
        self.random_state_model = cfg.learner.random_state_model
        self.xgb_obj_binary = cfg.learner.xgb_obj_binary
        self.xgb_loss_binary = cfg.learner.xgb_loss_binary
        self.xgb_obj_multi = cfg.learner.xgb_obj_multi
        self.xgb_loss_multi = cfg.learner.xgb_loss_multi
        self.lr_solver_binary = cfg.learner.lr_solver_binary
        self.lr_solver_multi = cfg.learner.lr_solver_multi
        self.lr_multi_loss = cfg.learner.lr_multi_loss
        self.random_state = cfg.resample.random_state_cv
        self.tol = cfg.mlp.mlp_tol
        self.n_iter_no_change = cfg.mlp.mlp_no_improve
        self.mlp_training = cfg.mlp.mlp_training
        self.patient_columns = cfg.data.patient_columns
        self.tooth_columns = cfg.data.tooth_columns
        self.side_columns = cfg.data.side_columns
        self.cat_vars = cfg.data.cat_vars
        self.bin_vars = cfg.data.bin_vars
        self.scale_vars = cfg.data.scale_vars
        self.behavior_columns = cfg.data.behavior_columns
        self.task_cols = cfg.data.task_cols
        self.no_train_cols = cfg.data.no_train_cols
        self.infect_vars = cfg.data.infect_cols
        self.cat_map = cfg.data.cat_map
        self.target_cols = cfg.data.target_cols
        self.all_cat_vars = self.cat_vars + cfg.data.behavior_columns["categorical"]
        self.required_columns = (
            self.patient_columns + self.tooth_columns + self.side_columns
        )


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
        _validate_classification(classification)
        _validate_hpo(hpo)
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
        super().__init__()
        _validate_classification(classification)
        _validate_hpo(hpo)
        self.classification = classification
        self.criterion = criterion
        self.hpo = hpo
        self.tuning = tuning
        self._validate_criterion()
        self._validate_tuning()

    def _validate_criterion(self) -> None:
        """Validates the evaluation criterion.

        Raises:
            ValueError: If the criterion is unsupported.
        """
        if self.criterion not in ["f1", "macro_f1", "brier_score"]:
            raise ValueError(
                "Unsupported criterion. Choose either 'f1', 'macro_f1', or "
                "'brier_score'."
            )

    def _validate_tuning(self) -> None:
        """Validates the tuning method.

        Raises:
            ValueError: If the tuning method is unsupported.
        """
        if self.tuning not in [None, "holdout", "cv"]:
            raise ValueError(
                "Unsupported tuning method. Choose either 'holdout' or 'cv'."
            )


@dataclass
class Side:
    """Dataclass to represent a single side of a tooth with required field names."""

    furcationbaseline: Optional[int]
    side: int
    pdbaseline: Optional[int]
    recbaseline: Optional[int]
    plaque: Optional[int]
    bop: Optional[int]


@dataclass
class Tooth:
    """Tooth dataclass, which contains up to 6 sides with required field names."""

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
    """Patient-level information with required field names."""

    age: int
    gender: int
    bodymassindex: float
    periofamilyhistory: int
    diabetes: int
    smokingtype: int
    cigarettenumber: int
    antibiotictreatment: int
    stresslvl: int
    teeth: List[Tooth] = field(default_factory=list)


def patient_to_dataframe(patient: Patient) -> pd.DataFrame:
    """Converts a Patient instance into a DataFrame suitable for prediction.

    Args:
        patient (Patient): The Patient dataclass instance.

    Returns:
        pd.DataFrame: DataFrame where each row represents a tooth side.
    """
    rows = []
    patient_dict = asdict(patient)

    for tooth in patient_dict["teeth"]:
        for side in tooth["sides"]:
            data = {
                **{k: v for k, v in patient_dict.items() if k != "teeth"},
                **{k: v for k, v in tooth.items() if k != "sides"},
                **side,
            }
            rows.append(data)

    return pd.DataFrame(rows)
