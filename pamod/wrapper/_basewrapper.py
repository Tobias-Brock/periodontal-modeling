from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..base import BaseHydra, Patient
from ..data import ProcessedDataLoader
from ..evaluation import ModelEvaluator
from ..inference import ModelInference
from ..resampling import Resampler
from ..training import Trainer


class BaseEvaluatorWrapper(BaseHydra, ABC):
    def __init__(
        self,
        learners_dict: Dict,
        criterion: str,
        aggregate: bool,
        verbosity: bool,
    ):
        """Base class for EvaluatorWrapper, initializing common parameters.

        Args:
            learners_dict (dict): Dictionary containing models and their metadata.
            criterion (str): Criterion for selecting model (e.g., 'f1', 'brier_score').
            aggregate (bool): Method for aggregating metrics.
            verbosity (bool): Verbosity flag.
        """
        super().__init__()
        self.learners_dict = learners_dict
        self.criterion = criterion
        self.aggregate = aggregate
        self.verbosity = verbosity
        (
            self.model,
            self.encoding,
            self.learner,
            self.task,
            self.factor,
            self.sampling,
        ) = self._get_best()
        self.classification = (
            "multiclass" if self.task == "pdgrouprevaluation" else "binary"
        )
        self.dataloader = ProcessedDataLoader(task=self.task, encoding=self.encoding)
        self.resampler = Resampler(
            classification=self.classification, encoding=self.encoding
        )
        (
            self.df,
            self.train_df,
            self._test_df,
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            self.base_target,
        ) = self._prepare_data_for_evaluation()
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
            verbosity=self.verbosity,
        )
        self.trainer = Trainer(
            classification=self.classification,
            criterion=self.criterion,
            tuning=None,
            hpo=None,
        )

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

    def _prepare_data_for_evaluation(
        self,
        seed: Optional[int] = None,
    ) -> Tuple[
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

        Args:
            seed (Optional[int]): Seed for train test split. Defaults to None.

        Returns:
            Tuple: df, train_df, test_df, X_train, y_train, X_test, y_test,
                and optionally base_target.
        """
        df = self.dataloader.load_data()

        if self.task in ["pocketclosure", "pdgrouprevaluation"]:
            base_target = self._generate_base_target(df=df)

        df = self.dataloader.transform_data(df=df)
        seed = seed if seed is not None else self.random_state_split
        train_df, test_df = self.resampler.split_train_test_df(df=df, seed=seed)

        if self.task in ["pocketclosure", "pdgrouprevaluation"]:
            test_patient_ids = test_df[self.group_col]
            base_target = base_target[df[self.group_col].isin(test_patient_ids)].values

        X_train, y_train, X_test, y_test = self.resampler.split_x_y(
            train_df=train_df, test_df=test_df
        )

        return df, train_df, test_df, X_train, y_train, X_test, y_test, base_target

    def _generate_base_target(self, df: pd.DataFrame) -> pd.Series:
        """Generates the target column before treatment based on the task.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.Series: The target before column for evaluation.
        """
        if self.task == "pocketclosure":
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

    def _train_and_get_metrics(self, seed: int, learner: str, n_jobs: int = -1) -> dict:
        """Helper function to run `train_final_model` with a specific seed.

        Args:
            seed (int): Seed value for train-test split.
            learner (str): Type of learner, used for MLP-specific training logic.
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
            df=self.df,
            resampler=self.resampler,
            model=model_tuple,
            sampling=self.sampling,
            factor=self.factor,
            n_jobs=n_jobs,
            seed=seed,
            test_size=self.test_set_size,
            verbosity=self.verbosity,
        )
        return result["metrics"]

    @abstractmethod
    def wrapped_evaluation(
        self,
        cm: bool,
        cm_base: bool,
        brier_groups: bool,
        cluster: bool,
        n_cluster: int,
    ):
        """Abstract method for evaluating the best-ranked model."""
        pass

    @abstractmethod
    def evaluate_feature_importance(self, importance_types: List[str]):
        """Evaluates feature importance using the provided evaluator.

        Args:
            importance_types (List[str]): List of feature importance types.
        """

    @abstractmethod
    def average_over_splits(self, num_splits: int, n_jobs: int) -> pd.DataFrame:
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
