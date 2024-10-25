from pathlib import Path
from typing import List, Optional, Union

from pamod.base import BaseEvaluator, BaseHydra
from pamod.resampling import Resampler
from pamod.training import Trainer
from pamod.tuning import HEBOTuner, RandomSearchTuner


class BaseExperiment(BaseEvaluator):
    """Base class to handle common attributes for benchmarking-related classes."""

    def __init__(
        self,
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
        verbosity: bool,
    ) -> None:
        """Initialize the Experiment class with tuning parameters.

        Args:
            df (pd.DataFrame): The preloaded data.
            task (str): The task name used to determine classification type.
            learner (str): The machine learning model to evaluate.
            criterion (str): Criterion for optimization ('macro_f1' or 'brier_score').
            encoding (str): Encoding type ('one_hot' or 'binary')
            tuning (Optional[str]): The tuning method ('holdout' or 'cv'). Can be None.
            hpo (Optional[str]): The hyperparameter optimization method. Can be None.
            sampling (Optional[str]): Sampling strategy to use.
            factor (Optional[float]): Factor for resampling.
            n_configs (int): Number of configurations for hyperparameter tuning.
            racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
            n_jobs (Optional[int]): Number of parallel jobs to run for evaluation.
            cv_folds (Optional[int], optional): Number of folds for cross-validation.
            test_seed (Optional[int], optional): Random seed for splitting.
            test_size (Optional[float]): Size of grouped train test split.
            val_size (Optional[float]): Size of grouped train test split for holdout.
            cv_seed (int): Seed for splitting CV folds.
            mlp_flag (bool): Flag for MLP training with early stopping.
            threshold_tuning (bool): Perform threshold tuning for binary classification
                if the criterion is "f1".
            verbosity (bool): Enables verbose output if set to True.
        """
        self.task = task
        classification = self._determine_classification()
        super().__init__(classification, criterion, tuning, hpo)
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
        self.verbosity = verbosity
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
                self.classification,
                self.criterion,
                self.tuning,
                self.hpo,
                self.n_configs,
                self.n_jobs,
                self.verbosity,
                self.trainer,
                self.mlp_flag,
                self.threshold_tuning,
            )
        elif self.hpo == "hebo":
            return HEBOTuner(
                self.classification,
                self.criterion,
                self.tuning,
                self.hpo,
                self.n_configs,
                self.n_jobs,
                self.verbosity,
                self.trainer,
                self.mlp_flag,
                self.threshold_tuning,
            )
        else:
            raise ValueError(f"Unsupported HPO method: {self.hpo}")


class BaseBenchmark(BaseHydra):
    """Base class to handle common attributes for benchmarking-related classes."""

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
        verbosity: bool,
        path: Path,
        name: str,
    ) -> None:
        """Initialize the base benchmark class with common parameters.

        Args:
            task (str): Task for evaluation.
            learners (List[str]): List of learners for benchmarking.
            tuning_methods (List[str]): Tuning methods for learners.
            hpo_methods (List[str]): Hyperparameter optimization methods.
            criteria (List[str]): Evaluation criteria for benchmarking.
            encodings (List[str]): Type of encoding ('one_hot' or 'target').
            sampling (Optional[List[str]]): Sampling strategy to use.
            factor (Optional[float]): Factor for resampling.
            n_configs (int): Number of configurations for hyperparameter tuning.
            n_jobs (Optional[int]): Number of parallel jobs.
            cv_folds (Optional[int]): Number of folds for cross-validation.
            racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
            test_seed (Optional[int]): Random seed for splitting.
            test_size (Optional[float]): Size of grouped train test split.
            val_size (Optional[float]): Size of grouped train test split for holdout.
            cv_seed (Optional[int]): Seed for splitting CV folds.
            mlp_flag (Optional[bool]): Flag for MLP training with early stopping.
            threshold_tuning (bool): Perform threshold tuning for binary classification
                if the criterion is "f1".
            verbosity (bool): Enables verbose output if True.
            path (str): Directory path for the processed data.
            name (str): File name for the processed data.
        """
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
        self.verbosity = verbosity
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
