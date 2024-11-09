from copy import deepcopy
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import clone

from ..learner import Model
from ..training import Trainer
from ._basetuner import BaseTuner


class RandomSearchTuner(BaseTuner):
    """Random Search hyperparameter tuning class.

    This class performs hyperparameter tuning using random search, supporting
    both holdout and cross-validation (CV) tuning methods.

    Inherits:
        - `BaseTuner`: Provides a framework for implementing HPO strategies,
          including shared evaluation and logging functions.

    Args:
        classification (str): The type of classification ('binary' or 'multiclass').
        criterion (str): The evaluation criterion (e.g., 'f1', 'brier_score').
        tuning (str): The type of tuning ('holdout' or 'cv').
        hpo (str): The hyperparameter optimization method, default is 'rs'.
        n_configs (int): Number of configurations to evaluate. Defaults to 10.
        n_jobs (int): Number of parallel jobs for model training.
            Defaults to 1.
        verbose (bool): Whether to print detailed logs during optimization.
            Defaults to True.
        trainer (Optional[Trainer]): Trainer instance for model training.
        mlp_training (bool): Enables MLP-specific training with early stopping.
        threshold_tuning (bool): Enables threshold tuning for binary classification
            when the criterion is "f1".

    Attributes:
        classification (str): Type of classification ('binary' or 'multiclass').
        criterion (str): Performance criterion for optimization
            ('f1', 'brier_score' or 'macro_f1').
        tuning (str): Tuning approach ('holdout' or 'cv').
        hpo (str): Hyperparameter optimization method (default is 'rs').
        n_configs (int): Number of configurations to evaluate.
        n_jobs (int): Number of parallel jobs for training.
        verbose (bool): Flag to enable detailed logs during optimization.
        mlp_training (bool): Enables MLP training with early stopping.
        threshold_tuning (bool): Enables threshold tuning if criterion is 'f1'.
        trainer (Trainer): Trainer instance for model evaluation.

    Methods:
        holdout: Optimizes hyperparameters using random search for
            holdout validation.
        cv: Optimizes hyperparameters using random search with
            cross-validation.

    Example:
        ```
        trainer = Trainer(
            classification="binary",
            criterion="f1",
            tuning="cv",
            hpo="rs",
            mlp_training=True,
            threshold_tuning=True,
        )

        tuner = RandomSearchTuner(
            classification="binary",
            criterion="f1",
            tuning="cv",
            hpo="rs",
            n_configs=15,
            n_jobs=4,
            verbose=True,
            trainer=trainer,
            mlp_training=True,
            threshold_tuning=True,
        )

        # Running holdout-based tuning
        best_params, best_threshold = tuner.holdout(
            learner="rf",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        # Running cross-validation tuning
        best_params, best_threshold = tuner.cv(
            learner="rf",
            outer_splits=cross_val_splits
        )
        ```
    """

    def __init__(
        self,
        classification: str,
        criterion: str,
        tuning: str,
        hpo: str = "rs",
        n_configs: int = 10,
        n_jobs: int = 1,
        verbose: bool = True,
        trainer: Optional[Trainer] = None,
        mlp_training: bool = True,
        threshold_tuning: bool = True,
    ) -> None:
        """Initialize RandomSearchTuner."""
        super().__init__(
            classification=classification,
            criterion=criterion,
            tuning=tuning,
            hpo=hpo,
            n_configs=n_configs,
            n_jobs=n_jobs,
            verbose=verbose,
            trainer=trainer,
            mlp_training=mlp_training,
            threshold_tuning=threshold_tuning,
        )

    def holdout(
        self,
        learner: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Union[float, int]], Union[float, None]]:
        """Perform random search on the holdout set for binary and multiclass .

        Args:
            learner (str): The machine learning model used for evaluation.
            X_train (pd.DataFrame): Training features for the holdout set.
            y_train (pd.Series): Training labels for the holdout set.
            X_val (pd.DataFrame): Validation features for the holdout set.
            y_val (pd.Series): Validation labels for the holdout set.

        Returns:
            tuple:
                - Best score (float)
                - Best hyperparameters (dict)
                - Best threshold (float or None, applicable for binary classification).
        """
        (
            best_score,
            best_threshold,
            best_params,
            param_grid,
            model,
        ) = self._initialize_search(learner=learner, random_state=self.rs_state)

        for i in range(self.n_configs):
            params = self._sample_params(
                param_grid=param_grid, iteration=i, random_state=self.rs_state
            )
            model_clone = clone(model).set_params(**params)
            if "n_jobs" in model_clone.get_params():
                model_clone.set_params(n_jobs=self.n_jobs)

            score, model_clone, threshold = self.trainer.train(
                model_clone, X_train, y_train, X_val, y_val
            )

            best_score, best_params, best_threshold = self._update_best(
                current_score=score,
                params=params,
                threshold=threshold,
                best_score=best_score,
                best_params=best_params,
                best_threshold=best_threshold,
            )

            if self.verbose:
                self._print_iteration_info(
                    iteration=i,
                    model=model_clone,
                    params_dict=params,
                    score=score,
                    threshold=best_threshold,
                )

        return best_params, best_threshold

    def cv(
        self,
        learner: str,
        outer_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
        racing_folds: Union[int, None],
    ) -> Tuple[dict, Union[float, None]]:
        """Perform cross-validation with optional racing and hyperparameter tuning.

        Args:
            learner: The machine learning model to evaluate.
            outer_splits: List of training and validation splits.
            racing_folds (int or None): Number of folds for racing; None uses all folds.

        Returns:
            tuple: Best hyperparameters, and optimal threshold (if applicable).
        """
        best_score, _, best_params, param_grid, model = self._initialize_search(
            learner=learner, random_state=self.rs_state
        )

        for i in range(self.n_configs):
            params = self._sample_params(param_grid, random_state=self.rs_state)
            model_clone = clone(model).set_params(**params)
            if "n_jobs" in model_clone.get_params():
                model_clone.set_params(n_jobs=self.n_jobs)
            scores = self._evaluate_folds(
                model=model_clone,
                best_score=best_score,
                outer_splits=outer_splits,
                racing_folds=racing_folds,
            )
            avg_score = np.mean(scores)
            best_score, best_params, _ = self._update_best(
                current_score=avg_score,
                params=params,
                threshold=None,
                best_score=best_score,
                best_params=best_params,
                best_threshold=None,
            )

            if self.verbose:
                self._print_iteration_info(
                    iteration=i, model=model_clone, params_dict=params, score=avg_score
                )

        if (
            self.classification == "binary"
            and self.criterion == "f1"
            and self.threshold_tuning
        ):
            optimal_threshold = self.trainer.optimize_threshold(
                model=model_clone, outer_splits=outer_splits, n_jobs=self.n_jobs
            )
        else:
            optimal_threshold = None

        return best_params, optimal_threshold

    def _evaluate_folds(
        self,
        model: Any,
        best_score: float,
        outer_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
        racing_folds: Union[int, None],
    ) -> list:
        """Evaluate the model across folds using cross-validation or racing strategy.

        Args:
            model (Any): The cloned model to evaluate.
            best_score (float): The best score recorded so far.
            outer_splits (list of tuples): List of training/validation folds.
            racing_folds (int or None): Number of folds to use for the racing strategy.

        Returns:
            scores: Scores from each fold evaluation.
        """
        num_folds = len(outer_splits)
        if racing_folds is None or racing_folds >= num_folds:
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self.trainer.evaluate_cv)(deepcopy(model), fold)
                for fold in outer_splits
            )
        else:
            selected_indices = random.sample(range(num_folds), racing_folds)
            selected_folds = [outer_splits[i] for i in selected_indices]
            initial_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self.trainer.evaluate_cv)(deepcopy(model), fold)
                for fold in selected_folds
            )
            avg_initial_score = np.mean(initial_scores)

            if (
                self.criterion in ["f1", "macro_f1"] and avg_initial_score > best_score
            ) or (self.criterion == "brier_score" and avg_initial_score < best_score):
                remaining_folds = [
                    outer_splits[i]
                    for i in range(num_folds)
                    if i not in selected_indices
                ]
                continued_scores = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.trainer.evaluate_cv)(deepcopy(model), fold)
                    for fold in remaining_folds
                )
                scores = initial_scores + continued_scores
            else:
                scores = initial_scores

        return scores

    def _initialize_search(
        self, learner: str, random_state: int
    ) -> Tuple[float, Union[float, None], Dict[str, Union[float, int]], dict, object]:
        """Initialize search with random seed, best score, parameters, and model.

        Args:
            learner (str): The learner type to be used for training the model.
            random_state (int): Random state.

        Returns:
            Tuple:
                - best_score: Initialized best score based on the criterion.
                - best_threshold: None or a placeholder threshold for binary
                    classification.
                - param_grid: The parameter grid for the specified model.
                - model: The model instance.
        """
        random.seed(random_state)
        best_score = (
            -float("inf") if self.criterion in ["f1", "macro_f1"] else float("inf")
        )
        best_threshold = None
        best_params: Dict[str, Union[float, int]] = {}
        model, param_grid = Model.get(
            learner=learner, classification=self.classification, hpo=self.hpo
        )

        return best_score, best_threshold, best_params, param_grid, model

    def _update_best(
        self,
        current_score: float,
        params: dict,
        threshold: Union[float, None],
        best_score: float,
        best_params: dict,
        best_threshold: Union[float, None],
    ) -> Tuple[float, dict, Union[float, None]]:
        """Update best score, parameters, and threshold if current score is better.

        Args:
            current_score (float): The current score obtained.
            params (dict): The parameters associated with the current score.
            threshold (float or None): The threshold associated with the current score.
            best_score (float): The best score recorded so far.
            best_params (dict): The best parameters recorded so far.
            best_threshold (float or None): The best threshold recorded so far.

        Returns:
            tuple: Updated best score, best parameters, and best threshold (optional).
        """
        if (self.criterion in ["f1", "macro_f1"] and current_score > best_score) or (
            self.criterion == "brier_score" and current_score < best_score
        ):
            best_score = current_score
            best_params = params
            best_threshold = threshold if self.classification == "binary" else None

        return best_score, best_params, best_threshold

    def _sample_params(
        self,
        param_grid: Dict[str, Union[list, object]],
        iteration: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Union[float, int]]:
        """Sample a set of hyperparameters from the provided grid.

        Args:
            param_grid (dict): Hyperparameter grid.
            iteration (Optional[int]): Current iteration index for random seed
                adjustment. If None, the iteration seed will not be adjusted.
            random_state (Optional[int]): Random state

        Returns:
            dict: Sampled hyperparameters.
        """
        iteration_seed = (
            random_state + iteration
            if random_state is not None and iteration is not None
            else None
        )

        params = {}
        for k, v in param_grid.items():
            if hasattr(v, "rvs"):
                params[k] = v.rvs(random_state=iteration_seed)
            elif isinstance(v, list):
                if iteration_seed is not None:
                    random.seed(iteration_seed)
                params[k] = random.choice(v)
            elif isinstance(v, np.ndarray):
                if iteration_seed is not None:
                    random.seed(iteration_seed)
                params[k] = random.choice(v.tolist())
            else:
                raise TypeError(f"Unsupported type for parameter '{k}': {type(v)}")

        return params
