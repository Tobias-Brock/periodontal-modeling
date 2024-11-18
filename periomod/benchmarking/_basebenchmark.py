from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from periomod.base import BaseConfig, BaseValidator
from periomod.resampling import Resampler
from periomod.training import Trainer
from periomod.tuning import HEBOTuner, RandomSearchTuner


class BaseExperiment(BaseValidator, ABC):
    """Base class for experiment workflows with model benchmarking.

    This class provides a shared framework for setting up and running
    experiments with model training, resampling, tuning, and evaluation. It
    supports configurations for task-specific classification, tuning methods,
    hyperparameter optimization, and sampling strategies, providing core methods
    to set up tuning, training, and evaluation for different machine learning
    tasks.

    Inherits:
        - `BaseValidator`: Validates instance-level variables and parameters.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        df (pd.DataFrame): The preloaded dataset used for training and evaluation.
        task (str): Task name, used to determine the classification type based on the
            Can be 'pocketclosure', 'pocketclosureinf', 'improvement', or
            'pdgrouprevaluation'.
        learner (str): Specifies the machine learning model or algorithm to use for
            evaluation, including 'xgb', 'rf', 'lr' or 'mlp'.
        criterion (str): Evaluation criterion for model performance. Options are
            'f1' and 'macro_f1' for F1 score and 'brier_score' for Brier Score.
        encoding (str): Encoding type for categorical features. Choose between
            'one_hot' or 'target' encoding based on model requirements.
        tuning (Optional[str]): The tuning method to apply during model training,
            either 'holdout' or 'cv' for cross-validation.
        hpo (Optional[str]): Hyperparameter optimization strategy. Options include
            'rs' (Random Search) and 'hebo'.
        sampling (Optional[str]): Sampling strategy to address class imbalance in
            the dataset. Includes None, 'upsampling', 'downsampling', and 'smote'.
        factor (Optional[float]): Factor used during resampling, specifying the
            amount of class balancing to apply.
        n_configs (int): Number of configurations to evaluate during hyperparameter
            tuning, used to limit the search space.
        racing_folds (Optional[int]): Number of racing folds used during random
            search for efficient hyperparameter optimization.
        n_jobs (int): Number of parallel jobs to use for processing.
            Set to -1 to use all available cores.
        cv_folds (int): Number of folds for cross-validation.
        test_seed (int): Seed for random train-test split for reproducibility.
        test_size (float): Proportion of data to use for testing.
        val_size (float): Proportion of data to use for validation in a
            holdout strategy.
        cv_seed (int): Seed for cross-validation splits for reproducibility.
        mlp_flag (Optional[bool]): If True, enables training with a Multi-Layer
            Perceptron (MLP) with early stopping. Defaults to `self.mlp_training`.
        threshold_tuning (bool): If True, tunes the decision threshold in binary
            classification to optimize for `f1` score.
        verbose (bool): If True, enables detailed logging of the model training,
            tuning, and evaluation processes for better traceability.

    Attributes:
        task (str): The task name, used to set the evaluation objective.
        classification (str): Classification type derived from the task ('binary'
            or 'multiclass') for configuring the evaluation.
        df (pd.DataFrame): DataFrame containing the dataset for training, validation,
            and testing purposes.
        learner (str): The chosen machine learning model or algorithm for evaluation.
        encoding (str): Encoding type applied to categorical features, either
            'one_hot' or 'target'.
        sampling (str): Resampling strategy used to address class imbalance in
            the dataset.
        factor (float): Resampling factor applied to balance classes as per
            the chosen sampling strategy.
        n_configs (int): Number of configurations evaluated during hyperparameter
            tuning.
        racing_folds (int): Number of racing folds applied during random search for
            efficient tuning.
        n_jobs (int): Number of parallel jobs used for model training and evaluation.
        cv_folds (int): Number of folds used for cross-validation.
        test_seed (int): Seed for splitting data into training and test sets,
            ensuring reproducibility.
        test_size (float): Proportion of the dataset assigned to the test split.
        val_size (float): Proportion of the dataset assigned to validation split in
            holdout validation.
        cv_seed (int): Seed for cross-validation splits to ensure consistency across
            runs.
        mlp_flag (bool): Enables training with a Multi-Layer Perceptron (MLP) and
            early stopping.
        threshold_tuning (bool): Enables tuning of the classification threshold
            in binary classification for optimizing the F1 score.
        verbose (bool): Controls the verbosity level of the output for detailed
            logs during training and evaluation.
        resampler (Resampler): Instance of the `Resampler` class for handling
            dataset resampling based on the specified strategy.
        trainer (Trainer): Instance of the `Trainer` class for managing the model
            training process.
        tuner (Tuner): Instance of the `Tuner` class used for performing
            hyperparameter optimization.


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
        n_jobs: int,
        cv_folds: Optional[int],
        test_seed: int,
        test_size: float,
        val_size: Optional[float],
        cv_seed: Optional[int],
        mlp_flag: Optional[bool],
        threshold_tuning: Optional[bool],
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
        self.cv_folds = cv_folds
        self.test_seed = test_seed
        self.test_size = test_size
        self.val_size = val_size
        self.cv_seed = cv_seed
        self.mlp_flag = mlp_flag
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
        if self.task in ["pocketclosure", "pocketclosureinf", "improvement"]:
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
        - `BaseConfig`: Base configuration class providing configuration loading.

    Args:
        task (str): Task for evaluation (pocketclosure', 'pocketclosureinf',
            'improvement', or 'pdgrouprevaluation'.).
        learners (List[str]): List of models or algorithms to benchmark,
            including 'xgb', 'rf', 'lr' or 'mlp'.
        tuning_methods (List[str]): List of tuning methods for model training,
            such as 'holdout' or 'cv'.
        hpo_methods (List[str]): Hyperparameter optimization strategies to apply,
            includes 'rs' and 'hebo'.
        criteria (List[str]): List of evaluation criteria ('f1', 'macro_f1',
            'brier_score').
        encodings (List[str]): Encoding types to transform categorical features,
            can either be 'one_hot' or 'target' encoding.
        sampling (Optional[List[Union[str, None]]]): Sampling strategies to handle
            class imbalance, options include None, 'upsampling', 'downsampling', or
            'smote'.
        factor (Optional[float]): Factor specifying the amount of sampling to apply
            during resampling, if applicable.
        n_configs (int): Number of configurations to evaluate in hyperparameter tuning.
        n_jobs (int): Number of parallel jobs to use for processing; set
            to -1 to utilize all available cores.
        cv_folds (int): Number of cross-validation folds for model
            training. Defaults to None.
        racing_folds (Optional[int]): Number of racing folds to use in Random Search
            (rs) for optimized tuning.
        test_seed (int): Random seed for reproducible train-test splits.
        test_size (float): Fraction of the dataset to allocate to test set.
        val_size (float): Fraction of the dataset to allocate to validation
            in a holdout setup.
        cv_seed (int): Seed for cross-validation splitting.
        mlp_flag (bool): If True, enables Multi-Layer Perceptron (MLP)
            training with early stopping.
        threshold_tuning (bool): Enables decision threshold tuning for binary
            classification when optimizing for 'f1'.
        verbose (bool): Enables detailed logging of processes if set to True.
        path (Path): Directory path where processed data will be stored.

    Attributes:
        task (str): Task used for model classification or regression evaluation.
        learners (List[str]): Selected models or algorithms for benchmarking.
        tuning_methods (List[str]): List of model tuning approaches.
        hpo_methods (List[str]): Hyperparameter optimization techniques to apply.
        criteria (List[str]): Criteria used to evaluate model performance.
        encodings (List[str]): Encoding methods applied to categorical features.
        sampling (Optional[List[Union[str, None]]]): Sampling strategies employed
            to address class imbalance.
        factor (Optional[float]): Specifies the degree of sampling applied
            within the chosen strategy.
        n_configs (int): Number of configurations assessed during hyperparameter
            optimization.
        n_jobs (int): Number of parallel processes for model training
            and evaluation.
        cv_folds (int): Number of cross-validation folds for model training.
        racing_folds (Optional[int]): Racing folds used in tuning with cross-validation
            and random search..
        test_seed (int): Seed for consistent test-train splitting.
        test_size (float): Proportion of the data set aside for testing.
        val_size (float): Proportion of data allocated to validation
            in holdout tuning.
        cv_seed (int): Seed for cross-validation splitting.
        mlp_flag (bool): Flag for MLP training with early stopping.
        threshold_tuning (bool): Enables threshold adjustment for optimizing F1
            in binary classification tasks.
        verbose (bool): Flag to enable detailed logging during training and evaluation.
        path (Path): Path where processed data is saved.

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
        n_jobs: int,
        cv_folds: Optional[int],
        racing_folds: Optional[int],
        test_seed: int,
        test_size: float,
        val_size: Optional[float],
        cv_seed: Optional[int],
        mlp_flag: Optional[bool],
        threshold_tuning: Optional[bool],
        verbose: bool,
        path: Path,
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
        self._validate_task()

    def _validate_task(self) -> None:
        """Validates the task type for the model.

        Raises:
            ValueError: If `self.task` is not one of the recognized task types.

        Supported task types:
            - "pocketclosure"
            - "pocketclosureinf"
            - "improvement"
            - "pdgrouprevaluation"
        """
        if self.task not in {
            "pocketclosure",
            "pocketclosureinf",
            "improvement",
            "pdgrouprevaluation",
        }:
            raise ValueError(
                f"Unknown task: {self.task}. Unable to determine classification."
            )
