from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from pamod.base import BaseConfig, BaseValidator
from pamod.resampling import Resampler
from pamod.training import Trainer
from pamod.tuning import HEBOTuner, RandomSearchTuner


class BaseExperiment(BaseValidator, ABC):
    """Base class for experiment workflows with model benchmarking.

    This class provides a shared framework for setting up and running
    experiments with model training, resampling, tuning, and evaluation. It
    supports configurations for task-specific classification, tuning methods,
    hyperparameter optimization, and sampling strategies, providing core methods
    to set up tuning, training, and evaluation for different machine learning
    tasks.

    Inherits:
        - BaseValidator: Validates instance-level variables and parameters.
        - ABC: Specifies abstract methods for subclasses to implement.

    Args:
        df (pd.DataFrame): The preloaded dataset used for training and evaluation.
        task (str): Task name, used to determine classification type.
        learner (str): Specifies the model or algorithm for evaluation.
        criterion (str): Criterion for performance evaluation ('macro_f1' or
            'brier_score').
        encoding (str): Encoding type for categorical features ('one_hot' or 'binary').
        tuning (Optional[str]): Method of tuning to apply ('holdout' or 'cv').
        hpo (Optional[str]): Hyperparameter optimization strategy ('rs', 'hebo').
        sampling (Optional[str]): Resampling strategy to handle class imbalance.
        factor (Optional[float]): Factor applied during resampling.
        n_configs (int): Number of configurations for hyperparameter tuning.
        racing_folds (Optional[int]): Number of racing folds for random search.
        n_jobs (Optional[int]): Number of parallel jobs for processing.
        cv_folds (Optional[int]): Number of folds for cross-validation; defaults to
            the value set in `self.n_folds` if None.
        test_seed (Optional[int]): Seed for random train-test split; defaults to
            `self.random_state_split` if None.
        test_size (Optional[float]): Proportion of data used for testing; defaults to
            `self.test_set_size` if None.
        val_size (Optional[float]): Proportion of data used for validation in holdout;
            defaults to `self.val_set_size` if None.
        cv_seed (Optional[int]): Seed for cross-validation; defaults to
            `self.random_state_cv` if None.
        mlp_flag (Optional[bool]): Whether to enable MLP training with early stopping;
            defaults to `self.mlp_training`.
        threshold_tuning (bool): If True, tunes the decision threshold for binary
            classification when optimizing `f1`.
        verbose (bool): If True, enables detailed logging of model and tuning
            processes.

    Attributes:
        task (str): Name of the task used for model evaluation.
        classification (str): Classification type ('binary' or 'multiclass') based on
            task.
        df (pd.DataFrame): Loaded dataset for training, validation, and testing.
        learner (str): Model or algorithm used for evaluation.
        encoding (str): Encoding type for categorical features.
        sampling (str): Resampling strategy for handling class imbalance.
        factor (float): Factor for applying the specified resampling strategy.
        n_configs (int): Number of configurations for hyperparameter tuning.
        racing_folds (int): Number of racing folds for random search.
        n_jobs (int): Number of parallel jobs for processing.
        cv_folds (int): Number of cross-validation folds for training.
        test_seed (int): Seed for random train-test split for reproducibility.
        test_size (float): Proportion of data for test split.
        val_size (float): Proportion of data for validation split in holdout.
        cv_seed (int): Seed for cross-validation for reproducibility.
        mlp_flag (bool): Enables MLP training with early stopping.
        threshold_tuning (bool): Enables threshold tuning for binary classification.
        verbose (bool): Enables verbose logging during model tuning and evaluation.
        resampler (Resampler): Resampler instance for handling data resampling.
        trainer (Trainer): Trainer instance for managing model training.
        tuner (Tuner): Tuner instance for hyperparameter optimization.

    Abstract Method:
        - `perform_evaluation`: Abstract method to handle the model evaluation process.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        learner: str,
        criterion: str,
        encoding: str,
        tuning: Optional[str],
        hpo: Optional[str],
        sampling: Optional[str],
        factor: Optional[float],
        n_configs: int,
        racing_folds: Optional[int],
        n_jobs: Optional[int],
        cv_folds: Optional[int],
        test_seed: Optional[int],
        test_size: Optional[float],
        val_size: Optional[float],
        cv_seed: Optional[int],
        mlp_flag: Optional[bool],
        threshold_tuning: bool,
        verbose: bool,
    ) -> None:
        """Initialize the Experiment class with tuning parameters."""
        self.task = task
        classification = self._determine_classification()
        super().__init__(
            classification=classification, criterion=criterion, tuning=tuning, hpo=hpo
        )
        self.df = df
        self.learner = learner
        self.encoding = encoding
        self.sampling = sampling
        self.factor = factor
        self.n_configs = n_configs
        self.racing_folds = racing_folds
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds if cv_folds is not None else self.n_folds
        self.test_seed = test_seed if test_seed is not None else self.random_state_split
        self.test_size = test_size if test_size is not None else self.test_set_size
        self.val_size = val_size if val_size is not None else self.val_set_size
        self.cv_seed = cv_seed if cv_seed is not None else self.random_state_cv
        self.mlp_flag = mlp_flag if mlp_flag is not None else self.mlp_training
        self.threshold_tuning = threshold_tuning
        self.verbose = verbose
        self.resampler = Resampler(self.classification, self.encoding)
        self.trainer = Trainer(
            self.classification,
            self.criterion,
            tuning=self.tuning,
            hpo=self.hpo,
            mlp_training=self.mlp_flag,
            threshold_tuning=self.threshold_tuning,
        )
        self.tuner = self._initialize_tuner()

    def _determine_classification(self) -> str:
        """Determine classification type based on the task name.

        Returns:
            str: The classification type ('binary' or 'multiclass').
        """
        if self.task in ["pocketclosure", "improve"]:
            return "binary"
        elif self.task == "pdgrouprevaluation":
            return "multiclass"
        else:
            raise ValueError(
                f"Unknown task: {self.task}. Unable to determine classification."
            )

    def _initialize_tuner(self):
        """Initialize the appropriate tuner based on the hpo method."""
        if self.hpo == "rs":
            return RandomSearchTuner(
                classification=self.classification,
                criterion=self.criterion,
                tuning=self.tuning,
                hpo=self.hpo,
                n_configs=self.n_configs,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                trainer=self.trainer,
                mlp_training=self.mlp_flag,
                threshold_tuning=self.threshold_tuning,
            )
        elif self.hpo == "hebo":
            return HEBOTuner(
                classification=self.classification,
                criterion=self.criterion,
                tuning=self.tuning,
                hpo=self.hpo,
                n_configs=self.n_configs,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                trainer=self.trainer,
                mlp_training=self.mlp_flag,
                threshold_tuning=self.threshold_tuning,
            )
        else:
            raise ValueError(f"Unsupported HPO method: {self.hpo}")

    def _train_final_model(
        self, final_model_tuple: Tuple[str, Dict, Optional[float]]
    ) -> dict:
        """Helper method to train the final model with best parameters.

        Args:
            final_model_tuple (Tuple[str, Dict, Optional[float]]): A tuple containing
                the learner name, best hyperparameters, and an optional best threshold.

        Returns:
            dict: A dictionary containing the trained model and its evaluation metrics.
        """
        return self.trainer.train_final_model(
            df=self.df,
            resampler=self.resampler,
            model=final_model_tuple,
            sampling=self.sampling,
            factor=self.factor,
            n_jobs=self.n_jobs,
            seed=self.test_seed,
            test_size=self.test_size,
            verbose=self.verbose,
        )

    @abstractmethod
    def perform_evaluation(self) -> dict:
        """Perform model evaluation and return final metrics."""

    @abstractmethod
    def _evaluate_holdout(self, train_df: pd.DataFrame) -> dict:
        """Perform holdout validation and return the final model metrics.

        Args:
            train_df (pd.DataFrame): train df for holdout tuning.
        """

    @abstractmethod
    def _evaluate_cv(self) -> dict:
        """Perform cross-validation and return the final model metrics."""


class BaseBenchmark(BaseConfig):
    """Base class for benchmarking models on specified tasks with various settings.

    This class initializes common parameters for benchmarking, including task
    specifications, encoding and sampling methods, tuning strategies, and model
    evaluation criteria.

    Inherits:
        BaseConfig: Base configuration class providing configuration loading.

    Args:
        task (str): Task for evaluation, defining classification type.
        learners (List[str]): List of learners (models) for benchmarking.
        tuning_methods (List[str]): List of tuning methods for learners.
        hpo_methods (List[str]): Hyperparameter optimization methods.
        criteria (List[str]): Evaluation criteria (e.g., 'f1', 'brier_score').
        encodings (List[str]): Encoding types for categorical features.
        sampling (Optional[List[Union[str, None]]]): Sampling strategy to use.
        factor (Optional[float]): Factor for resampling.
        n_configs (int): Number of configurations for hyperparameter tuning.
        n_jobs (Optional[int]): Number of parallel jobs.
        cv_folds (Optional[int]): Number of folds for cross-validation.
        racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
        test_seed (Optional[int]): Random seed for test set splitting.
        test_size (Optional[float]): Size of the test set as a fraction.
        val_size (Optional[float]): Size of validation set as a fraction for holdout.
        cv_seed (Optional[int]): Seed for cross-validation splitting.
        mlp_flag (Optional[bool]): Flag for MLP training with early stopping.
        threshold_tuning (bool): Enables threshold tuning if criterion is 'f1'.
        verbose (bool): Enables verbose logging if set to True.
        path (Path): Directory path for storing processed data.
        name (str): Filename for the processed data file.

    Attributes:
        task (str): Task for classification or regression.
        learners (List[str]): Selected models to benchmark.
        tuning_methods (List[str]): Tuning methods for optimization.
        hpo_methods (List[str]): Hyperparameter optimization strategies.
        criteria (List[str]): Criteria for evaluating model performance.
        encodings (List[str]): Encoding schemes for data transformation.
        sampling (Optional[List[Union[str, None]]]): Sampling strategies.
        factor (Optional[float]): Factor used in sampling strategy.
        n_configs (int): Number of HPO configurations to test.
        n_jobs (Optional[int]): Parallel jobs for model training and evaluation.
        cv_folds (Optional[int]): Number of cross-validation folds.
        racing_folds (Optional[int]): Racing folds for optimization.
        test_seed (Optional[int]): Seed for test-train splits.
        test_size (Optional[float]): Test set fraction.
        val_size (Optional[float]): Validation set fraction for holdout tuning.
        cv_seed (Optional[int]): Seed for cross-validation splitting.
        mlp_flag (Optional[bool]): Flag indicating MLP usage with early stopping.
        threshold_tuning (bool): Enables threshold tuning for 'f1' optimization.
        verbose (bool): Enables verbose logging.
        path (Path): Directory path for processed data storage.
        name (str): Name of processed data file.
    """

    def __init__(
        self,
        task: str,
        learners: List[str],
        tuning_methods: List[str],
        hpo_methods: List[str],
        criteria: List[str],
        encodings: List[str],
        sampling: Optional[List[Union[str, None]]],
        factor: Optional[float],
        n_configs: int,
        n_jobs: Optional[int],
        cv_folds: Optional[int],
        racing_folds: Optional[int],
        test_seed: Optional[int],
        test_size: Optional[float],
        val_size: Optional[float],
        cv_seed: Optional[int],
        mlp_flag: Optional[bool],
        threshold_tuning: bool,
        verbose: bool,
        path: Path,
        name: str,
    ) -> None:
        """Initialize the base benchmark class with common parameters."""
        super().__init__()
        self.task = task
        self.learners = learners
        self.tuning_methods = tuning_methods
        self.hpo_methods = hpo_methods
        self.criteria = criteria
        self.encodings = encodings
        self.sampling = sampling
        self.factor = factor
        self.n_configs = n_configs
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cv_folds = cv_folds
        self.racing_folds = racing_folds
        self.test_seed = test_seed
        self.test_size = test_size
        self.val_size = val_size
        self.cv_seed = cv_seed
        self.mlp_flag = mlp_flag
        self.threshold_tuning = threshold_tuning
        self.path = path
        self.name = name
