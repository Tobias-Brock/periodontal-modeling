from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import clone

from pamod.learner import Model
from pamod.tuning._basetuner import BaseTuner, MetaTuner


class HEBOTuner(BaseTuner, MetaTuner):
    """Hyperparameter tuning class using HEBO (Bayesian Optimization)."""

    def __init__(
        self, classification: str, criterion: str, tuning: str, hpo: str = "HEBO"
    ) -> None:
        """Initialize HEBOTuner with classification, criterion, and tuning method.

        Args:
            classification (str): The type of classification ('binary' or 'multiclass').
            criterion (str): The evaluation criterion (e.g., 'f1', 'brier_score').
            tuning (str): The type of tuning ('holdout' or 'cv').
            hpo (str): The hyperparameter optimization method (default is 'HEBO').
        """
        super().__init__(classification, criterion, tuning, hpo)

    def holdout(
        self,
        learner: str,
        X_train_h: pd.DataFrame,
        y_train_h: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_configs: int,
        n_jobs: int,
        verbosity: bool,
    ) -> Tuple[Dict[str, Union[float, int]], Optional[float]]:
        """Perform Bayesian Optimization using HEBO for holdout validation.

        Args:
            learner (str): The machine learning model to evaluate.
            X_train_h (pd.DataFrame): The training features for the holdout set.
            y_train_h (pd.Series): The training labels for the holdout set.
            X_val (pd.DataFrame): The validation features for the holdout set.
            y_val (pd.Series): The validation labels for the holdout set.
            n_configs (int): The number of configurations to evaluate during HPO.
            n_jobs (int): The number of parallel jobs for model training.
            verbosity (bool): Whether to print detailed logs during HEBO optimization.

        Returns:
            Tuple[Dict[str, Union[float, int]], Optional[float]]:
                The best hyperparameters and the best threshold.
        """
        return self._run_optimization(
            learner,
            X_train_h,
            y_train_h,
            X_val,
            y_val,
            None,
            n_configs,
            n_jobs,
            verbosity,
        )

    def cv(
        self,
        learner: str,
        outer_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
        n_configs: int,
        n_jobs: int,
        verbosity: bool,
    ) -> Tuple[Dict[str, Union[float, int]], Optional[float]]:
        """Perform Bayesian Optimization using HEBO with cross-validation.

        Args:
            learner (str): The machine learning model to evaluate.
            outer_splits (List[Tuple[pd.DataFrame, pd.DataFrame]]):
                List of cross-validation folds.
            n_configs (int): The number of configurations to evaluate during HPO.
            n_jobs (int): The number of parallel jobs for model training.
            verbosity (bool): Whether to print detailed logs during HEBO optimization.

        Returns:
            Tuple[Dict[str, Union[float, int]], Optional[float]]:
                The best hyperparameters and the best threshold.
        """
        return self._run_optimization(
            learner, None, None, None, None, outer_splits, n_configs, n_jobs, verbosity
        )

    def _run_optimization(
        self,
        learner: str,
        X_train_h: Optional[pd.DataFrame],
        y_train_h: Optional[pd.Series],
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        outer_splits: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]],
        n_configs: int,
        n_jobs: int,
        verbosity: bool,
    ) -> Tuple[Dict[str, Union[float, int]], Optional[float]]:
        """Perform Bayesian Optimization using HEBO for holdout and cross-validation.

        Args:
            learner (str): The machine learning model to evaluate.
            X_train_h (Optional[pd.DataFrame]): Training features for the holdout set
                (None if using CV).
            y_train_h (Optional[pd.Series]): Training labels for the holdout set
                (None if using CV).
            X_val (Optional[pd.DataFrame]): Validation features for the holdout set
                (None if using CV).
            y_val (Optional[pd.Series]): Validation labels for the holdout set
                (None if using CV).
            outer_splits (Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]]):
                Cross-validation folds (None if using holdout).
            n_configs (int): The number of configurations to evaluate during HPO.
            n_jobs (int): The number of parallel jobs for model training.
            verbosity (bool): Whether to print detailed logs during HEBO optimization.

        Returns:
            Tuple[Dict[str, Union[float, int]], Optional[float]]:
                The best hyperparameters and the best threshold.
        """
        model, search_space, params_func = Model.get(
            learner, self.classification, self.hpo
        )
        space = DesignSpace().parse(search_space)
        optimizer = HEBO(space)

        for i in range(n_configs):
            params_suggestion = optimizer.suggest(n_suggestions=1).iloc[0]
            params_dict = params_func(params_suggestion)

            score = self._objective(
                model,
                params_dict,
                X_train_h,
                y_train_h,
                X_val,
                y_val,
                outer_splits,
                n_jobs,
            )
            optimizer.observe(pd.DataFrame([params_suggestion]), np.array([score]))

            if verbosity:
                self._print_iteration_info(i, model, params_dict, score)

        # Extract best parameters
        best_params_idx = optimizer.y.argmin()
        best_params_df = optimizer.X.iloc[best_params_idx]
        best_params = params_func(best_params_df)
        best_threshold = None
        if self.classification == "binary":
            model_clone = clone(model).set_params(**best_params)
            if self.criterion == "f1":
                if self.tuning == "holdout":
                    model_clone.fit(X_train_h, y_train_h)
                    probs = model_clone.predict_proba(X_val)[:, 1]
                    _, best_threshold = self.metric_evaluator.evaluate(y_val, probs)

                elif self.tuning == "cv":
                    best_threshold = self.trainer.optimize_threshold(
                        model, outer_splits
                    )

        return best_params, best_threshold

    def _objective(
        self,
        model,
        params_dict: Dict[str, Union[float, int]],
        X_train_h: Optional[pd.DataFrame],
        y_train_h: Optional[pd.Series],
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        outer_splits: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]],
        n_jobs: int,
    ) -> float:
        """Evaluate the model performance for both holdout and cross-validation.

        Args:
            model: The machine learning model to evaluate.
            params_dict (Dict[str, Union[float, int]]): The suggested hyperparameters
                as a dictionary.
            X_train_h (Optional[pd.DataFrame]): Training features for the holdout set
                (None for CV).
            y_train_h (Optional[pd.Series]): Training labels for the holdout set
                (None for CV).
            X_val (Optional[pd.DataFrame]): Validation features for the holdout set
                (None for CV).
            y_val (Optional[pd.Series]): Validation labels for the holdout set
                (None for CV).
            outer_splits (Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]]):
                Cross-validation folds (None for holdout).
            n_jobs (int): The number of parallel jobs for model training.

        Returns:
            float: The evaluation score to be minimized by HEBO.
        """
        model_clone = clone(model)
        model_clone.set_params(**params_dict)

        if "n_jobs" in model_clone.get_params():
            model_clone.set_params(n_jobs=n_jobs if n_jobs is not None else 1)

        score = self.evaluate_objective(
            model_clone,
            X_train_h,
            y_train_h,
            X_val,
            y_val,
            outer_splits,
            n_jobs if n_jobs is not None else 1,
        )

        return -score if self.criterion in ["f1", "macro_f1"] else score

    def evaluate_objective(
        self,
        model_clone,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        outer_splits: Optional[List[Tuple[pd.DataFrame, pd.Series]]],
        n_jobs: int,
    ) -> float:
        """Evaluates the model's performance based on the tuning strategy.

        The tuning strategy can be either 'holdout' or 'cv' (cross-validation).

        Args:
            model_clone: The cloned machine learning model to be
                evaluated.
            X_train (pd.DataFrame): Training features for the holdout set.
            y_train (pd.Series): Training labels for the holdout set.
            X_val (pd.DataFrame): Validation features for the holdout set (used for
                'holdout' tuning).
            y_val (pd.Series): Validation labels for the holdout set (used for
                'holdout' tuning).
            outer_splits (List[tuple]): List of cross-validation folds, each a tuple
                containing (X_train_fold, y_train_fold).
            n_jobs (int): Number of parallel jobs to use for cross-validation.

        Returns:
            float: The model's performance metric based on tuning strategy.
        """
        if self.tuning == "holdout":
            score, _, _ = self.trainer.train(
                model_clone, X_train, y_train, X_val, y_val
            )
            return score

        elif self.tuning == "cv":
            if outer_splits is None:
                raise ValueError(
                    "outer_splits cannot be None when using cross-validation."
                )
            scores = Parallel(n_jobs=n_jobs)(
                delayed(self.trainer.evaluate_cv)(deepcopy(model_clone), fold)
                for fold in outer_splits
            )
            print(f"CV scores: {scores}.")
            return np.mean(scores)

        raise ValueError(f"Unsupported criterion: {self.tuning}")
