"""Base Methods."""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import hydra
import pandas as pd


class BaseConfig:
    """Base class to initialize Hydra configuration.

    This class loads and sets various configuration parameters used across the package,
    providing easy access to these parameters by initializing them from a Hydra
    configuration file.

    Args:
        config_path (str): Path to the Hydra config directory.
        config_name (str): Name of the configuration file (without extension).

    Attributes:
        group_col (str): Column name used for group-based splitting.
        y (str): Target column name in the dataset.
        target_state (int): Random state of target encoding.
        learner_state (int): Random state of learners.
        xgb_obj_binary (str): Objective function for binary classification in XGBoost.
        xgb_loss_binary (str): Loss function for binary classification in XGBoost.
        xgb_obj_multi (str): Objective function for multiclass classification in
            XGBoost.
        xgb_loss_multi (str): Loss function for multiclass classification in XGBoost.
        lr_solver_binary (str): Solver type for binary classification in logistic
            regression.
        lr_solver_multi (str): Solver type for multiclass classification in logistic
            regression.
        lr_multi_loss (str): Loss function for multiclass logistic regression.
        patient_columns (List[str]): List of column names related to patient data.
        tooth_columns (List[str]): List of column names related to tooth data.
        side_columns (List[str]): List of column names related to side data.
        feature_mapping (dict[str, str]): Mapping of feature names for plotting.
        cat_vars (List[str]): List of categorical variables in the dataset.
        bin_vars (List[str]): List of binary variables in the dataset.
        scale_vars (List[str]): List of numeric variables to scale in preprocessing.
        behavior_columns (Dict[str, List[str]]): Dictionary categorizing
            behavior-related columns by type.
        task_cols (List[str]): List of task-specific columns in the dataset.
        no_train_cols (List[str]): Columns excluded from training.
        infect_vars (List[str]): Columns indicating infection status.
        cat_map (Dict[str, int]): Mapping of categorical features and their maximum
            values for encoding.
        target_cols (List[str]): Columns related to the prediction target.
        all_cat_vars (List[str]): Combined list of categorical variables for encoding.
        required_columns (List[str]): Combined list of columns required in the dataset
            for analysis.
        rs_state (int): State for random search parameter selection.

    Example:
        ```
        config = BaseConfig()
        print(config.tooth_columns)
        ```

    Note:
        This class assumes Hydra configuration files are correctly set up and stored at
        `config_path`. Make sure the file structure and values are properly defined
        within the configuration.
    """

    def __init__(
        self, config_path: str = "../config", config_name: str = "config"
    ) -> None:
        """Initializes the Hydra configuration for use in other classes."""
        with hydra.initialize(config_path=config_path, version_base="1.2"):
            cfg = hydra.compose(config_name=config_name)

        self.group_col = cfg.resample.group_col
        self.y = cfg.resample.y
        self.target_state = cfg.resample.target_state
        self.learner_state = cfg.learner.learner_state
        self.xgb_obj_binary = cfg.learner.xgb_obj_binary
        self.xgb_loss_binary = cfg.learner.xgb_loss_binary
        self.xgb_obj_multi = cfg.learner.xgb_obj_multi
        self.xgb_loss_multi = cfg.learner.xgb_loss_multi
        self.lr_solver_binary = cfg.learner.lr_solver_binary
        self.lr_solver_multi = cfg.learner.lr_solver_multi
        self.lr_multi_loss = cfg.learner.lr_multi_loss
        self.patient_columns = cfg.data.patient_columns
        self.tooth_columns = cfg.data.tooth_columns
        self.side_columns = cfg.data.side_columns
        self.feature_mapping = cfg.data.feature_mapping
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
        self.rs_state = cfg.tuning.rs_state


class BaseValidator(BaseConfig):
    """Base class for initializing classification, criterion, tuning, and HPO.

    This class extends `BaseConfig` and validates classification types, evaluation
    criteria, and tuning methods.

    Inherits:
        - `BaseLoader`: Provides loading and saving capabilities for processed data.

    Args:
        classification (str): Type of classification, either 'binary' or 'multiclass'.
        criterion (str): Evaluation criterion (e.g., 'f1', 'macro_f1').
        tuning (Optional[str], optional): Tuning method, either 'holdout' or 'cv'.
            Defaults to None.
        hpo (Optional[str], optional): Hyperparameter optimization type, either 'rs' or
            'hebo'. Defaults to None.

    Attributes:
        classification (str): Type of classification, either 'binary' or 'multiclass'.
        criterion (str): Evaluation criterion for model performance.
        tuning (Optional[str]): Tuning method ('holdout' or 'cv').
        hpo (Optional[str]): Type of hyperparameter optimization ('rs' or 'hebo').

    Raises:
        ValueError: If the classification, criterion, or tuning method is invalid.

    Example:
        ```
        validator = BaseValidator(classification="binary", criterion="f1")
        print(validator.criterion)
        ```
    """

    def __init__(
        self,
        classification: str,
        criterion: str,
        tuning: Optional[str] = None,
        hpo: Optional[str] = None,
    ) -> None:
        """Initializes BaseValidator method for with validation functions for inputs."""
        super().__init__()
        self.classification = classification
        self.criterion = criterion
        self.hpo = hpo
        self.tuning = tuning
        self._validate_classification()
        self._validate_hpo()
        self._validate_criterion()
        self._validate_tuning()

    def _validate_classification(self) -> None:
        """Validates the classification type for the model.

        Raises:
            ValueError: If `self.classification` is not 'binary' or 'multiclass'.

        Expected classification types:
            - 'binary'
            - 'multiclass'
        """
        if self.classification.lower().strip() not in ["binary", "multiclass"]:
            raise ValueError(
                f"{self.classification} is an invalid classification type. "
                f"Choose 'binary' or 'multiclass'."
            )

    def _validate_hpo(self) -> None:
        """Validates the hyperparameter optimization (HPO) type.

        Raises:
            ValueError: If `self.hpo` is not None, 'rs', or 'hebo'.

        Supported HPO types:
            - None
            - 'rs' (Random Search)
            - 'hebo' (Heteroscedastic Bayesian Optimization)
        """
        if self.hpo not in [None, "rs", "hebo"]:
            raise ValueError(
                f"{self.hpo} is an unsupported HPO type. Choose 'rs' or 'hebo'."
            )

    def _validate_criterion(self) -> None:
        """Validates the evaluation criterion for model performance.

        Raises:
            ValueError: If `self.criterion` is not a supported evaluation metrics.

        Supported evaluation criteria:
            - 'f1'
            - 'macro_f1'
            - 'brier_score'
        """
        if self.criterion not in ["f1", "macro_f1", "brier_score"]:
            raise ValueError(
                "Unsupported criterion. Choose either 'f1', 'macro_f1', or "
                "'brier_score'."
            )

    def _validate_tuning(self) -> None:
        """Validates the tuning method for hyperparameter optimization.

        Raises:
            ValueError: If `self.tuning` is not None, 'holdout', or 'cv'.

        Supported tuning methods:
            - None
            - 'holdout'
            - 'cv' (Cross-Validation)
        """
        if self.tuning not in [None, "holdout", "cv"]:
            raise ValueError(
                "Unsupported tuning method. Choose either 'holdout' or 'cv'."
            )


@dataclass
class Side:
    """Represents a single side of a tooth with relevant features.

    Attributes:
        furcationbaseline (Optional[int]): Baseline furcation measurement, if available.
        side (int): Identifier for the side of the tooth.
        pdbaseline (Optional[int]): Baseline probing depth measurement.
        recbaseline (Optional[int]): Baseline recession measurement.
        plaque (Optional[int]): Plaque presence status.
        bop (Optional[int]): Bleeding on probing status.

    Example:
        ```
        side_1 = Side(
            furcationbaseline=1,
            side=1,
            pdbaseline=2,
            recbaseline=2,
            plaque=1,
            bop=1
        )
        ```
    """

    furcationbaseline: Optional[int]
    side: int
    pdbaseline: Optional[int]
    recbaseline: Optional[int]
    plaque: Optional[int]
    bop: Optional[int]


@dataclass
class Tooth:
    """Represents a tooth with specific features and associated sides.

    Attributes:
        tooth (int): Identifier number for the tooth.
        toothtype (int): Type classification of the tooth (e.g., molar, incisor).
        rootnumber (int): Count of roots associated with the tooth.
        mobility (Optional[int]): Mobility status of the tooth.
        restoration (Optional[int]): Restoration status of the tooth.
        percussion (Optional[int]): Percussion sensitivity, if applicable.
        sensitivity (Optional[int]): Sensitivity status of the tooth.
        sides (List[Side]): Collection of `Side` instances for each side of the tooth.

    Example:
        ```
        side_1 = Side(
            furcationbaseline=1, side=1, pdbaseline=2, recbaseline=2, plaque=1, bop=1
            )
        side_2 = Side(
            furcationbaseline=2, side=2, pdbaseline=3, recbaseline=3, plaque=1, bop=0
            )

        tooth = Tooth(
            tooth=11,
            toothtype=2,
            rootnumber=1,
            mobility=1,
            restoration=0,
            percussion=0,
            sensitivity=1,
            sides=[side_1, side_2]
        )
        ```
    """

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
    """Contains patient-level data along with dental health information for each tooth.

    This dataclass encapsulates patient demographic and health information, along with
    detailed data about each tooth and its associated sides. It serves as a structured
    container for organizing patient records in dental health applications.

    Attributes:
        age (int): The age of the patient in years.
        gender (int): Gender code for the patient (e.g., 0 for female, 1 for male).
        bodymassindex (float): Body Mass Index (BMI) of the patient.
        periofamilyhistory (int): Indicator of family history with periodontal disease.
        diabetes (int): Diabetes status, where 0 indicates no diabetes and 1 indicates
            diabetes.
        smokingtype (int): Type of smoking habit (e.g., 0 for non-smoker, 1 for
            occasional, 2 for frequent).
        cigarettenumber (int): Number of cigarettes smoked per day.
        antibiotictreatment (int): Indicator of antibiotic treatment history, where 0
            means no treatment and 1 indicates treatment.
        stresslvl (int): Stress level rating on a scale (e.g., 0 to 3).
        teeth (List[Tooth]): A list of `Tooth` instances containing specific tooth
            data, where each tooth may have up to 6 sides with separate health metrics.

    Example:
        ```
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
                            bop=1
                            ),
                        Side(
                            furcationbaseline=2,
                            side=2,
                            pdbaseline=3,
                            recbaseline=3,
                            plaque=1,
                            bop=1
                            ),
                        # Additional sides can be added similarly
                    ]
                ),
                Tooth(
                    tooth=18,
                    toothtype=3,
                    rootnumber=2,
                    mobility=0,
                    restoration=1,
                    percussion=1,
                    sensitivity=0,
                    sides=[
                        Side(
                            furcationbaseline=3,
                            side=1,
                            pdbaseline=4,
                            recbaseline=5,
                            plaque=2,
                            bop=0
                            ),
                        # Additional sides can be added similarly
                    ]
                )
            ]
        )
        ```
    """

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


def patient_to_df(patient: Patient) -> pd.DataFrame:
    """Converts a Patient instance into a DataFrame suitable for prediction.

    This function takes a `Patient` dataclass instance and flattens its attributes
    along with nested `Tooth` and `Side` instances to generate a DataFrame. Each row
    in the DataFrame corresponds to a side of a tooth, with all relevant patient,
    tooth, and side attributes in a single row.

    Args:
        patient (Patient): The Patient dataclass instance.

    Returns:
        pd.DataFrame: DataFrame where each row represents a tooth side.

    Example:
        ```
        patient = Patient(..)
        patient_data = patient_to_df(patient=patient)
        ```
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
