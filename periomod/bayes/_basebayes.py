from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd

from ..base import BaseConfig, BaseValidator

NODE_LEVELS = ("patient", "tooth", "site")
SPATIAL_MODES = ("none", "tooth", "site", "both")
RANDOM_EFFECT_MODES = ("sample", "zero")
BAYES_TASKS = ("pocketclosure", "pocketclosureinf", "improvement")


def load_bayes_config(
    config_path: str = "../../config", config_name: str = "bayesconfig"
):
    """Composes the Hydra configuration of the Bayesian submodule.

    Args:
        config_path (str): Path to the Hydra config directory, relative to this
            file. Defaults to "../../config".
        config_name (str): Name of the configuration file without extension.
            Defaults to "bayesconfig".

    Returns:
        DictConfig: Composed configuration containing the 'bayes' group.
    """
    with hydra.initialize(config_path=config_path, version_base="1.2"):
        return hydra.compose(config_name=config_name)


@dataclass
class PatientSplit:
    """Patient identifiers of the fixed train, validation and test split.

    Attributes:
        train (List[int]): Patient identifiers of the training set.
        val (List[int]): Patient identifiers of the validation set.
        test (List[int]): Patient identifiers of the test set.

    Example:
        ```
        split = PatientSplit(train=[1, 2], val=[3], test=[4])
        print(split.sizes)
        ```
    """

    train: List[int] = field(default_factory=list)
    val: List[int] = field(default_factory=list)
    test: List[int] = field(default_factory=list)

    @property
    def sizes(self) -> Dict[str, int]:
        """Number of patients per split.

        Returns:
            Dict[str, int]: Mapping of split name to number of patients.
        """
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def items(self) -> List[Tuple[str, List[int]]]:
        """Splits as name and patient identifier pairs.

        Returns:
            List[Tuple[str, List[int]]]: Split name and patient identifiers.
        """
        return [("train", self.train), ("val", self.val), ("test", self.test)]


@dataclass
class HierarchicalData:
    """Design matrix and nesting structure of one split.

    Observations are the target sites of the task. Sites are nested within
    teeth and teeth within patients, which is encoded by the index arrays
    `tooth_idx`, `patient_idx` and `tooth_patient`.

    Attributes:
        X (np.ndarray): Design matrix of the fixed effects, shape (n_obs, p).
        y (np.ndarray): Site level outcome, shape (n_obs,).
        feature_names (List[str]): Column names of the design matrix.
        keys (pd.DataFrame): Patient, tooth and side of every observation.
        patient_idx (np.ndarray): Patient index of every observation.
        tooth_idx (np.ndarray): Tooth index of every observation.
        side_idx (np.ndarray): Side position of every observation, 0 to 5.
        tooth_patient (np.ndarray): Patient index of every tooth.
        tooth_number (np.ndarray): FDI number of every tooth.
        toothnum_idx (np.ndarray): Index of the FDI number of every tooth.
        patients (np.ndarray): Patient identifiers in index order.

    Example:
        ```
        data = transformer.transform_data(data=df)
        print(data.n_obs, data.n_teeth, data.n_patients)
        ```
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    keys: pd.DataFrame
    patient_idx: np.ndarray
    tooth_idx: np.ndarray
    side_idx: np.ndarray
    tooth_patient: np.ndarray
    tooth_number: np.ndarray
    toothnum_idx: np.ndarray
    patients: np.ndarray

    @property
    def n_obs(self) -> int:
        """Number of target sites in the split.

        Returns:
            int: Number of observations.
        """
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of fixed effects in the design matrix.

        Returns:
            int: Number of columns of the design matrix.
        """
        return int(self.X.shape[1])

    @property
    def n_patients(self) -> int:
        """Number of patients in the split.

        Returns:
            int: Number of patient level groups.
        """
        return int(len(self.patients))

    @property
    def n_teeth(self) -> int:
        """Number of teeth in the split.

        Returns:
            int: Number of tooth level groups.
        """
        return int(len(self.tooth_patient))


@dataclass
class SpatialConfig:
    """Configuration of the anatomical spatial component.

    The spatial stage adds intrinsic conditional autoregressive (CAR) effects
    on top of the nested random intercepts. Tooth effects borrow strength from
    adjacent teeth within an arch, site effects from the anatomically
    neighboring sites around a tooth and from the interproximal contacts of
    adjacent teeth. The adjacency is the same anatomy the graph submodule uses.

    Attributes:
        mode (str): Spatial component. Choose 'none' for the nested model,
            'tooth' for a CAR prior on the tooth effects, 'site' for a CAR
            prior on the site effects or 'both'.
        tooth_neighbor (bool): Connects adjacent teeth within an arch.
        site_neighbor (bool): Connects neighboring sites around a tooth.
        interproximal (bool): Connects interproximal sites of adjacent teeth.
        name (str): Label of the configuration, used in benchmark results.

    Example:
        ```
        from periomod.bayes import SpatialConfig

        # nested model without spatial component
        spatial = SpatialConfig(mode="none", name="nested")

        # CAR prior on tooth effects
        spatial = SpatialConfig(mode="tooth", name="car_tooth")
        ```
    """

    mode: str = "none"
    tooth_neighbor: bool = True
    site_neighbor: bool = True
    interproximal: bool = True
    name: str = "nested"

    def __post_init__(self) -> None:
        """Validates the spatial mode.

        Raises:
            ValueError: If `mode` is not a supported spatial component.
        """
        if self.mode not in SPATIAL_MODES:
            raise ValueError(
                f"{self.mode} is an invalid spatial mode. "
                f"Choose one of {list(SPATIAL_MODES)}."
            )

    @classmethod
    def from_config(
        cls, mode: str, relations: Dict[str, bool], name: str = "nested"
    ) -> "SpatialConfig":
        """Creates a SpatialConfig from the Hydra configuration.

        Args:
            mode (str): Spatial component of the model.
            relations (Dict[str, bool]): Relation switches from the config.
            name (str): Label of the configuration. Defaults to "nested".

        Returns:
            SpatialConfig: Instantiated spatial configuration.
        """
        return cls(mode=mode, **dict(relations), name=name)

    @classmethod
    def stages(cls) -> Dict[str, "SpatialConfig"]:
        """Returns the predefined spatial stages of the modeling workflow.

        Returns:
            Dict[str, SpatialConfig]: Mapping of stage name to configuration.
        """
        return {
            "nested": cls(mode="none", name="nested"),
            "car_tooth": cls(mode="tooth", name="car_tooth"),
            "car_site": cls(mode="site", name="car_site"),
            "car_both": cls(mode="both", name="car_both"),
        }

    @property
    def on_teeth(self) -> bool:
        """Indicates whether tooth effects receive a CAR prior.

        Returns:
            bool: True if the tooth level is modeled spatially.
        """
        return self.mode in ("tooth", "both")

    @property
    def on_sites(self) -> bool:
        """Indicates whether site effects receive a CAR prior.

        Returns:
            bool: True if the site level is modeled spatially.
        """
        return self.mode in ("site", "both")


class BayesConfigMixin:
    """Mixin that loads the configuration of the Bayesian submodule.

    Provides `_init_bayes_config`, which attaches all parameters of the 'bayes'
    config group to the instance. It is applied on top of `BaseConfig` or
    `BaseValidator`, which load the main package configuration.

    Attributes:
        exclude_columns (List[str]): Columns dropped from the design matrix.
        infect_levels (Dict[str, List[str]]): Engineered infection features per
            level.
        include_infect (bool): Includes the engineered infection features.
        direct_effects (Dict[str, bool]): Categorical predictors modeled as
            hierarchical effects instead of dummy variables.
        side_ring (List[Tuple[int, int]]): Anatomical ring of the six sides.
        interproximal_aspects (List[Tuple[int, int]]): Interproximal contacts.
        spatial (str): Default spatial component of the model.
        spatial_relations (Dict[str, bool]): Default adjacency switches.
        intercept_sd (float): Prior scale of the intercept.
        beta_sd (float): Prior scale of the fixed effects.
        sigma_sd (float): Prior scale of the random effect standard deviations.
        spatial_sigma_sd (float): Prior scale of the CAR standard deviations.
        zero_sum_sd (float): Scale of the soft sum-to-zero constraint.
        draws (int): Number of posterior draws per chain.
        tune (int): Number of tuning steps per chain.
        chains (int): Number of chains.
        cores (int): Number of cores used for sampling.
        target_accept (float): Target acceptance rate of NUTS.
        sampler_seed (int): Random state of the sampler.
        progressbar (bool): Displays the sampler progress bar.
        mp_ctx (str): Multiprocessing context used for parallel chains.
        hdi_prob (float): Mass of the reported credible intervals.
        random_effects (str): Treatment of random effects for unseen patients.
        val_size (float): Default proportion of patients in the validation set.
        test_size (float): Default proportion of patients in the test set.
        split_seed (int): Default random state of the patient level split.
        target_sites (str): Default site selection of the likelihood.
    """

    def _init_bayes_config(self) -> None:
        """Loads the Bayesian configuration into instance attributes."""
        cfg = load_bayes_config()
        self.exclude_columns = [col.lower() for col in cfg.bayes.exclude_columns]
        self.infect_levels = {
            level: [col.lower() for col in cfg.bayes.infect_levels[level]]
            for level in NODE_LEVELS
        }
        self.include_infect = cfg.bayes.include_infect
        self.direct_effects = dict(cfg.bayes.direct_effects)
        self.side_ring = [tuple(pair) for pair in cfg.bayes.side_ring]
        self.interproximal_aspects = [
            tuple(pair) for pair in cfg.bayes.interproximal_aspects
        ]
        self.spatial = cfg.bayes.spatial
        self.spatial_relations = dict(cfg.bayes.spatial_relations)
        self.intercept_sd = cfg.bayes.intercept_sd
        self.beta_sd = cfg.bayes.beta_sd
        self.sigma_sd = cfg.bayes.sigma_sd
        self.spatial_sigma_sd = cfg.bayes.spatial_sigma_sd
        self.zero_sum_sd = cfg.bayes.zero_sum_sd
        self.draws = cfg.bayes.draws
        self.tune = cfg.bayes.tune
        self.chains = cfg.bayes.chains
        self.cores = cfg.bayes.cores
        self.target_accept = cfg.bayes.target_accept
        self.sampler_seed = cfg.bayes.sampler_seed
        self.progressbar = cfg.bayes.progressbar
        self.mp_ctx = cfg.bayes.mp_ctx
        self.hdi_prob = cfg.bayes.hdi_prob
        self.random_effects = cfg.bayes.random_effects
        self.val_size = cfg.bayes.val_size
        self.test_size = cfg.bayes.test_size
        self.split_seed = cfg.bayes.split_seed
        self.target_sites = cfg.bayes.target_sites

    def default_spatial(self) -> SpatialConfig:
        """Builds the default spatial configuration from the Hydra config.

        Returns:
            SpatialConfig: Spatial configuration with the configured defaults.
        """
        return SpatialConfig.from_config(
            mode=self.spatial, relations=self.spatial_relations
        )


class BaseBayesConfig(BaseConfig, BayesConfigMixin):
    """Base class providing the package and Bayesian submodule configuration.

    Extends `BaseConfig` with the parameters of the 'bayes' config group, which
    are composed from `config/bayesconfig.yaml`.

    Inherits:
        - `BaseConfig`: Provides configuration settings of the package.
        - `BayesConfigMixin`: Loads the Bayesian configuration.

    Example:
        ```
        config = BaseBayesConfig()
        print(config.draws)
        ```
    """

    def __init__(self) -> None:
        """Initializes the package and Bayesian submodule configuration."""
        super().__init__()
        self._init_bayes_config()


class BaseBayesValidator(BaseValidator, BayesConfigMixin):
    """Base class validating classification and criterion for Bayesian models.

    Inherits:
        - `BaseValidator`: Validates instance-level variables and parameters.
        - `BayesConfigMixin`: Loads the Bayesian configuration.

    Args:
        classification (str): Type of classification. Only 'binary' is
            supported by the hierarchical logistic model.
        criterion (str): Evaluation criterion (e.g., 'f1', 'brier_score').

    Example:
        ```
        validator = BaseBayesValidator(classification="binary", criterion="f1")
        print(validator.target_accept)
        ```
    """

    def __init__(self, classification: str, criterion: str) -> None:
        """Initializes the validator with the Bayesian configuration."""
        super().__init__(classification=classification, criterion=criterion)
        self._init_bayes_config()


class BaseHierarchicalTransformer(BaseBayesConfig, ABC):
    """Abstract base class for split-aware preprocessing of nested data.

    Provides the resolution of the dataset columns to the patient, tooth and
    site level, as well as the patient level split that precedes any
    preprocessing. Fitting of imputation, dummy coding and standardization is
    left to the subclass and must be restricted to training patients.

    Inherits:
        - `BaseBayesConfig`: Provides package and Bayesian configuration.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        task (str): Task name. Can be 'pocketclosure', 'pocketclosureinf' or
            'improvement'.
        include_infect (Optional[bool]): Includes the engineered infection
            features. Defaults to the configured value.

    Attributes:
        task (str): Task name used to derive the site level outcome.
        include_infect (bool): Indicates whether engineered infection features
            enter the design matrix.
        levels (Dict[str, Dict[str, List[str]]]): Raw columns per level and
            column kind ('numeric', 'binary' or 'categorical').

    Abstract Methods:
        - `fit`: Fits imputation, dummy coding and scaling on training patients.
        - `transform_data`: Applies the fitted transformations to a dataset.
    """

    def __init__(self, task: str, include_infect: Optional[bool] = None) -> None:
        """Initializes the transformer with task and feature configuration."""
        super().__init__()
        self.task = task
        self.include_infect = (
            self.include_infect if include_infect is None else include_infect
        )
        self.levels: Dict[str, Dict[str, List[str]]] = {}

    def _level_columns(self, level: str) -> List[str]:
        """Collects the candidate columns of a level.

        Args:
            level (str): Level, either 'patient', 'tooth' or 'site'.

        Returns:
            List[str]: Lowercased column names assigned to the level.

        Raises:
            ValueError: If `level` is not a valid level.
        """
        if level == "patient":
            columns = list(self.patient_columns)
        elif level == "tooth":
            columns = list(self.tooth_columns)
        elif level == "site":
            columns = list(self.side_columns)
        else:
            raise ValueError(
                f"{level} is an invalid level. Choose 'patient', 'tooth' or 'site'."
            )
        columns = [col.lower() for col in columns]
        if self.include_infect:
            columns += self.infect_levels[level]

        excluded = set(self.exclude_columns + list(self.task_cols) + self.no_train_cols)
        excluded |= {column for column, direct in self.direct_effects.items() if direct}
        return [col for col in columns if col not in excluded]

    def _column_kind(self, column: str) -> str:
        """Determines the kind of a column from the package configuration.

        Args:
            column (str): Lowercased column name.

        Returns:
            str: Column kind, either 'categorical', 'binary' or 'numeric'.
        """
        if column in self.all_cat_vars:
            return "categorical"
        if column in self.bin_vars:
            return "binary"
        return "numeric"

    def resolve_levels(self, data: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """Assigns the available columns to levels and column kinds.

        Args:
            data (pd.DataFrame): Processed dataset with lowercased columns.

        Returns:
            Dict[str, Dict[str, List[str]]]: Columns per level and kind.
        """
        levels: Dict[str, Dict[str, List[str]]] = {}
        for level in NODE_LEVELS:
            columns = [col for col in self._level_columns(level) if col in data.columns]
            levels[level] = {
                kind: [col for col in columns if self._column_kind(col) == kind]
                for kind in ("numeric", "binary", "categorical")
            }
        self.levels = levels
        return levels

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fits imputation, dummy coding and scaling on training patients.

        Args:
            data (pd.DataFrame): Dataset restricted to training patients.
        """

    @abstractmethod
    def transform_data(self, data: pd.DataFrame):
        """Applies the fitted transformations to a dataset.

        Args:
            data (pd.DataFrame): Dataset to transform.
        """


class BaseHierarchicalModel(BaseBayesConfig, ABC):
    """Abstract base class for hierarchical Bayesian models of site outcomes.

    Inherits:
        - `BaseBayesConfig`: Provides package and Bayesian configuration.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        spatial (Optional[SpatialConfig]): Spatial component of the model.
            Defaults to the configured component.

    Attributes:
        spatial (SpatialConfig): Spatial component of the model.

    Abstract Methods:
        - `build`: Builds the model for a design matrix and nesting structure.
        - `linear_predictor`: Computes the posterior linear predictor.
    """

    def __init__(self, spatial: Optional[SpatialConfig] = None) -> None:
        """Initializes the model with its spatial configuration."""
        super().__init__()
        self.spatial = self.default_spatial() if spatial is None else spatial

    @abstractmethod
    def build(self, data: HierarchicalData):
        """Builds the Bayesian model of a split.

        Args:
            data (HierarchicalData): Design matrix and nesting structure.
        """

    @abstractmethod
    def linear_predictor(self, idata, data: HierarchicalData, in_sample: bool):
        """Computes the posterior draws of the linear predictor.

        Args:
            idata (Any): Posterior draws of the fitted model.
            data (HierarchicalData): Design matrix and nesting structure.
            in_sample (bool): Uses the fitted group effects if True.
        """


class BaseBayesTrainer(BaseBayesValidator, ABC):
    """Abstract base class for sampling and evaluating hierarchical models.

    Inherits:
        - `BaseBayesValidator`: Validates classification and criterion.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        classification (str): Type of classification. Only 'binary' is
            supported.
        criterion (str): Evaluation criterion (e.g., 'f1', 'brier_score').

    Abstract Methods:
        - `fit`: Draws from the posterior of a hierarchical model.
        - `predict`: Computes posterior predictive probabilities.
    """

    def __init__(self, classification: str, criterion: str) -> None:
        """Initializes the trainer with classification type and criterion."""
        super().__init__(classification=classification, criterion=criterion)

    @abstractmethod
    def fit(self, model, data: HierarchicalData):
        """Draws from the posterior of a hierarchical model.

        Args:
            model (Any): Hierarchical model to fit.
            data (HierarchicalData): Training design matrix and structure.
        """

    @abstractmethod
    def predict(self, model, idata, data: HierarchicalData, in_sample: bool):
        """Computes posterior predictive probabilities of the target sites.

        Args:
            model (Any): Fitted hierarchical model.
            idata (Any): Posterior draws of the fitted model.
            data (HierarchicalData): Design matrix and nesting structure.
            in_sample (bool): Uses the fitted group effects if True.
        """
