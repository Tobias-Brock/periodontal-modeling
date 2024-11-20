from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BaseConfig, Patient
from ..benchmarking import Baseline
from ..data import ProcessedDataLoader
from ..evaluation import ModelEvaluator
from ..inference import ModelInference
from ..resampling import Resampler
from ..training import Trainer


class ModelExtractor(BaseConfig):
    """Extracts best model from learner dictionary.

    Inherits:
        `BaseConfig`: Loads configuration parameters.

    Args:
        learners_dict (Dict): Dictionary containing models and their metadata.
        criterion (str): Criterion for selecting models (e.g., 'f1', 'brier_score').
        aggregate (bool): Whether to aggregate metrics.
        verbose (bool): Controls verbose in the evaluation process.
        random_state (int): Random state for resampling.

    Attributes:
        learners_dict (Dict): Holds learners and metadata.
        criterion (str): Evaluation criterion to select the optimal model.
        aggregate (bool): Indicates if metrics should be aggregated.
        verbose (bool): Flag for controlling logging verbose.
        random_state (int): Random state for resampling.
        classification (str): Classification type ('binary' or 'multiclass').

    Properties:
        - `criterion (str)`: Retrieves or sets current evaluation criterion for model
            selection. Supports 'f1', 'brier_score', and 'macro_f1'.
        - `model (object)`: Retrieves best-ranked model dynamically based on the current
            criterion. Recalculates when criterion is updated.
    """

    def __init__(
        self,
        learners_dict: Dict,
        criterion: str,
        aggregate: bool,
        verbose: bool,
        random_state: int,
    ):
        """Initializes ModelExtractor."""
        super().__init__()
        self.learners_dict = learners_dict
        self.criterion = criterion
        self.aggregate = aggregate
        self.verbose = verbose
        self.random_state = random_state
        self._update_best_model()
        self.classification = (
            "multiclass" if self.task == "pdgrouprevaluation" else "binary"
        )

    @property
    def criterion(self) -> str:
        """The current evaluation criterion used to select the best model.

        Returns:
            str: The current criterion for model selection (e.g., 'f1', 'brier_score').

        Raises:
            ValueError: If an unsupported criterion is assigned.
        """
        return self._criterion

    @criterion.setter
    def criterion(self, value: str) -> None:
        """Sets the evaluation criterion and updates related attributes accordingly.

        Args:
            value (str): The criterion for selecting the model ('f1', 'brier_score',
                or 'macro_f1').

        Raises:
            ValueError: If the provided criterion is unsupported.
        """
        if value not in ["f1", "brier_score", "macro_f1"]:
            raise ValueError(
                "Unsupported criterion. Choose 'f1', 'macro_f1', or 'brier_score'."
            )
        self._criterion = value
        self._update_best_model()

    @property
    def model(self) -> Any:
        """Retrieves the best model based on the current criterion.

        Returns:
            Any: The model object selected according to the current criterion.

        Raises:
            ValueError: If no model matching the criterion and rank is found.
        """
        return self._model

    def _update_best_model(self) -> None:
        """Retrieves and updates the best model based on the current criterion."""
        (
            best_model,
            self.encoding,
            self.learner,
            self.task,
            self.factor,
            self.sampling,
        ) = self._get_best()
        self._model = best_model

    def _get_best(self) -> Tuple[Any, str, str, str, Optional[float], Optional[str]]:
        """Retrieves best model entities.

        Returns:
            Tuple: A tuple containing the best model, encoding ('one_hot' or 'target'),
                learner, task, factor, and sampling type (if applicable).

        Raises:
            ValueError: If model with rank1 is not found, or any component cannot be
                determined.
        """
        best_model_key = next(
            (
                key
                for key in self.learners_dict
                if f"_{self.criterion}_" in key and "rank1" in key
            ),
            None,
        )

        if not best_model_key:
            raise ValueError(
                f"No model with rank1 found for criterion '{self.criterion}' in dict."
            )

        best_model = self.learners_dict[best_model_key]

        if "one_hot" in best_model_key:
            encoding = "one_hot"
        elif "target" in best_model_key:
            encoding = "target"
        else:
            raise ValueError("Unable to determine encoding from the model key.")

        if "upsampling" in best_model_key:
            sampling = "upsampling"
        elif "downsampling" in best_model_key:
            sampling = "downsampling"
        elif "smote" in best_model_key:
            sampling = "smote"
        else:
            sampling = None

        key_parts = best_model_key.split("_")
        task = key_parts[0]
        learner = key_parts[1]

        for part in key_parts:
            if part.startswith("factor"):
                factor_value = part.replace("factor", "")
                if factor_value.isdigit():
                    factor = float(factor_value)
                else:
                    factor = None

        return best_model, encoding, learner, task, factor, sampling


class BaseEvaluatorWrapper(ModelExtractor, ABC):
    """Base class for wrappers handling model evaluation processes.

    This class serves as a foundational structure for evaluator wrappers, offering
    methods to initialize, prepare, and evaluate models according to specified
    parameters. It provides core functionality to streamline evaluation, feature
    importance analysis, patient inference, and jackknife resampling.

    Inherits:
        - `BaseModelExtractor`: Loads configuration parameters and model extraction.
        - `ABC`: Specifies abstract methods that must be implemented by subclasses.

    Args:
        learners_dict (Dict): Dictionary containing models and their metadata.
        criterion (str): Criterion for selecting models (e.g., 'f1', 'brier_score').
        aggregate (bool): Whether to aggregate metrics.
        verbose (bool): Controls verbose in the evaluation process.
        random_state (int): Random state for resampling.
        path (Path): Path to the directory containing processed data files.

    Attributes:
        learners_dict (Dict): Holds learners and metadata.
        criterion (str): Evaluation criterion to select the optimal model.
        aggregate (bool): Indicates if metrics should be aggregated.
        verbose (bool): Flag for controlling logging verbose.
        random_state (int): Random state for resampling.
        model (object): Best-ranked model for the given criterion.
        encoding (str): Encoding type, either 'one_hot' or 'target'.
        learner (str): The learner associated with the best model.
        task (str): Task associated with the model ('pocketclosure', 'improve', etc.).
        factor (Optional[float]): Resampling factor if applicable.
        sampling (Optional[str]): Resampling strategy used (e.g., 'smote').
        classification (str): Classification type ('binary' or 'multiclass').
        dataloader (ProcessedDataLoader): Data loader and transformer.
        resampler (Resampler): Resampling strategy for training and testing.
        df (pd.DataFrame): Loaded dataset.
        df_processed (pd.DataFrame): Processed dataset.
        train_df (pd.DataFrame): Training data after splitting.
        test_df (pd.DataFrame): Test data after splitting.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        base_target (Optional[np.ndarray]): Baseline target for evaluations.
        baseline (Baseline): Basline class for model analysis.
        evaluator (ModelEvaluator): Evaluator for model metrics and feature importance.
        inference_engine (ModelInference): Model inference manager.
        trainer (Trainer): Trainer for model evaluation and optimization.

    Inherited Properties:
        - `criterion (str)`: Retrieves or sets current evaluation criterion for model
            selection. Supports 'f1', 'brier_score', and 'macro_f1'.
        - `model (object)`: Retrieves best-ranked model dynamically based on the current
            criterion. Recalculates when criterion is updated.

    Abstract Methods:
        - `wrapped_evaluation`: Performs model evaluation and generates specified plots.
        - `evaluate_cluster`: Performs clustering and calculates Brier scores.
        - `evaluate_feature_importance`: Computes feature importance using specified
          methods.
        - `average_over_splits`: Aggregates metrics over multiple splits for model
          robustness.
        - `wrapped_patient_inference`: Runs inference on individual patient data.
        - `wrapped_jackknife`: Executes jackknife resampling on patient data for
          confidence interval estimation.
    """

    def __init__(
        self,
        learners_dict: Dict,
        criterion: str,
        aggregate: bool,
        verbose: bool,
        random_state: int,
        path: Path,
    ):
        """Base class for EvaluatorWrapper, initializing common parameters."""
        super().__init__(
            learners_dict=learners_dict,
            criterion=criterion,
            aggregate=aggregate,
            verbose=verbose,
            random_state=random_state,
        )
        self.path = path
        self.dataloader = ProcessedDataLoader(task=self.task, encoding=self.encoding)
        self.resampler = Resampler(
            classification=self.classification, encoding=self.encoding
        )
        (
            self.df,
            self.df_processed,
            self.train_df,
            self.test_df,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.base_target,
        ) = self._prepare_data_for_evaluation()
        self.baseline = Baseline(
            task=self.task,
            encoding=self.encoding,
            random_state=self.random_state,
            path=self.path,
        )
        self.evaluator = ModelEvaluator(
            model=self.model,
            X=self.X_test,
            y=self.y_test,
            encoding=self.encoding,
            aggregate=self.aggregate,
        )
        self.inference_engine = ModelInference(
            classification=self.classification,
            model=self.model,
            verbose=self.verbose,
        )
        self.trainer = Trainer(
            classification=self.classification,
            criterion=self.criterion,
            tuning=None,
            hpo=None,
        )

    def _prepare_data_for_evaluation(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        Optional[np.ndarray],
    ]:
        """Prepares data for evaluation.

        Returns:
            Tuple: df, df_processed, train_df, test_df, X_train, y_train, X_test,
                y_test, and optionally base_target.
        """
        df = self.dataloader.load_data(path=self.path)

        task = "pocketclosure" if self.task == "pocketclosureinf" else self.task

        if task in ["pocketclosure", "pdgrouprevaluation"]:
            base_target = self._generate_base_target(df=df)
        else:
            base_target = None

        df_processed = self.dataloader.transform_data(df=df)
        train_df, test_df = self.resampler.split_train_test_df(
            df=df_processed, seed=self.random_state
        )
        if task in ["pocketclosure", "pdgrouprevaluation"] and base_target is not None:
            test_patient_ids = test_df[self.group_col]
            base_target = (
                base_target.reindex(df_processed.index)
                .loc[df_processed[self.group_col].isin(test_patient_ids)]
                .values
            )

        X_train, y_train, X_test, y_test = self.resampler.split_x_y(
            train_df=train_df, test_df=test_df
        )

        return (
            df,
            df_processed,
            train_df,
            test_df,
            X_train,
            y_train,
            X_test,
            y_test,
            base_target,
        )

    def _generate_base_target(self, df: pd.DataFrame) -> pd.Series:
        """Generates the target column before treatment based on the task.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.Series: The target before column for evaluation.
        """
        if self.task in ["pocketclosure", "pocketclosureinf"]:
            return df.apply(
                lambda row: (
                    0
                    if row["pdbaseline"] == 4
                    and row["bop"] == 2
                    or row["pdbaseline"] > 4
                    else 1
                ),
                axis=1,
            )
        elif self.task == "pdgrouprevaluation":
            return df["pdgroupbase"]
        else:
            raise ValueError(f"Task '{self.task}' is not recognized.")

    def _train_and_get_metrics(
        self, seed: int, learner: str, test_set_size: float = 0.2, n_jobs: int = -1
    ) -> dict:
        """Helper function to run `train_final_model` with a specific seed.

        Args:
            seed (int): Seed value for train-test split.
            learner (str): Type of learner, used for MLP-specific training logic.
            test_set_size (float): Size of test set. Defaults to 0.2.
            n_jobs (int): Number of parallel jobs. Defaults to -1 (use all processors).

        Returns:
            dict: Metrics from `train_final_model`.
        """
        best_params = (
            self.model.get_params() if hasattr(self.model, "get_params") else {}
        )
        best_threshold = getattr(self.model, "best_threshold", None)
        model_tuple = (learner, best_params, best_threshold)

        result = self.trainer.train_final_model(
            df=self.df_processed,
            resampler=self.resampler,
            model=model_tuple,
            sampling=self.sampling,
            factor=self.factor,
            n_jobs=n_jobs,
            seed=seed,
            test_size=test_set_size,
            verbose=self.verbose,
        )
        return result["metrics"]

    def _subset_test_set(
        self, base: str, revaluation: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Creates a subset of the test set based on differences in raw data variables.

        Args:
            base (str): Baseline variable to compare against in `df_raw`.
            revaluation (str): Revaluation variable to check for changes in `df_raw`.

        Returns:
            Tuple: Subsets of X_test and y_test where
                `revaluation` differs from `base`.
        """
        changed_indices = self.df.index[self.df[revaluation] != self.df[base]]
        X_test_subset = self.X_test.reindex(changed_indices)
        y_test_subset = self.y_test.reindex(changed_indices)
        return X_test_subset, y_test_subset

    def _test_filters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base: Optional[str],
        revaluation: Optional[str],
        true_preds: bool,
        brier_threshold: Optional[float],
    ) -> Tuple[pd.DataFrame, pd.Series, int]:
        """Applies subsetting filters to the evaluator's test set.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Label set.
            base (Optional[str]): Baseline variable for comparison. If provided with
                `revaluation`, subsets to cases where `revaluation` differs from `base`.
            revaluation (Optional[str]): Revaluation variable for comparison. Used only
                if `base` is also provided.
            true_preds (bool): If True, further subsets to cases where the model's
                predictions match the true labels.
            brier_threshold (Optional[float]): Threshold for filtering Brier scores. If
                provided, further subsets to cases with Brier scores below threshold.

        Returns:
            Tuple: Filtered feature set, labels and number of unique patients.
        """
        if base and revaluation:
            X, y = self._subset_test_set(base=base, revaluation=revaluation)
            X, y = X.dropna(), y.dropna()

        if true_preds:
            pred = self.evaluator.model_predictions().reindex(y.index)
            correct_indices = y.index[pred == y]
            X, y = X.loc[correct_indices].dropna(), y.loc[correct_indices].dropna()

        if brier_threshold is not None:
            brier_scores = self.evaluator.brier_scores().reindex(y.index)
            threshold_indices = brier_scores[brier_scores < brier_threshold].index
            X, y = X.loc[threshold_indices].dropna(), y.loc[threshold_indices].dropna()

        subset_patient_ids = self.test_df.loc[y.index, self.group_col]
        num_patients = subset_patient_ids.nunique()

        return X, y, num_patients

    @abstractmethod
    def wrapped_evaluation(
        self,
        cm: bool,
        cm_base: bool,
        brier_groups: bool,
        calibration: bool,
        tight_layout: bool,
    ):
        """Runs evaluation on the best-ranked model based on specified criteria.

        Args:
            cm (bool): If True, plots the confusion matrix.
            cm_base (bool): If True, plots the confusion matrix against the
                value before treatment. Only applicable for specific tasks.
            brier_groups (bool): If True, calculates Brier score groups.
            calibration (bool): If True, plots model calibration.
            tight_layout (bool): If True, applies tight layout to the plot.
        """

    @abstractmethod
    def evaluate_cluster(
        self,
        n_cluster: int,
        base: Optional[str],
        revaluation: Optional[str],
        true_preds: bool,
        brier_threshold: Optional[float],
        tight_layout: bool,
    ):
        """Performs cluster analysis with Brier scores, with optional subsetting.

        Args:
            n_cluster (int): Number of clusters for Brier score clustering analysis.
            base (Optional[str]): Baseline variable for comparison.
            revaluation (Optional[str]): Revaluation variable for comparison.
            true_preds (bool): If True, further subsets to cases where model predictions
                match the true labels.
            brier_threshold (Optional[float]): Threshold for Brier score filtering.
            tight_layout (bool): If True, applies tight layout to the plot.
        """

    @abstractmethod
    def evaluate_feature_importance(
        self,
        fi_types: List[str],
        base: Optional[str],
        revaluation: Optional[str],
        true_preds: bool,
        brier_threshold: Optional[float],
    ):
        """Evaluates feature importance using specified types, with optional subsetting.

        Args:
            fi_types (List[str]): List of feature importance types to evaluate.
            base (Optional[str]): Baseline variable for comparison.
            revaluation (Optional[str]): Revaluation variable for comparison.
            true_preds (bool): If True, further subsets to cases where model predictions
                match the true labels.
            brier_threshold (Optional[float]): Threshold for Brier score filtering.
        """

    @abstractmethod
    def average_over_splits(self, num_splits: int, n_jobs: int):
        """Trains the final model over multiple splits with different seeds.

        Args:
            num_splits (int): Number of random seeds/splits to train the model on.
            n_jobs (int): Number of parallel jobs.
        """

    @abstractmethod
    def wrapped_patient_inference(
        self,
        patient: Patient,
    ):
        """Runs inference on the patient's data using the best-ranked model.

        Args:
            patient (Patient): A `Patient` dataclass instance containing patient-level,
                tooth-level, and side-level information.
        """

    @abstractmethod
    def wrapped_jackknife(
        self,
        patient: Patient,
        results: pd.DataFrame,
        sample_fraction: float,
        n_jobs: int,
        max_plots: int,
    ) -> pd.DataFrame:
        """Runs jackknife resampling for inference on a given patient's data.

        Args:
            patient (Patient): `Patient` dataclass instance containing patient-level
                information, tooth-level, and side-level details.
            results (pd.DataFrame): DataFrame to store results from jackknife inference.
            sample_fraction (float, optional): The fraction of patient data to use for
                jackknife resampling.
            n_jobs (int, optional): The number of parallel jobs to run.
            max_plots (int): Maximum number of plots for jackknife intervals.
        """
