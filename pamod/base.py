"""Base classes."""

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
