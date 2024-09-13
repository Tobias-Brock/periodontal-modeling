import random
import numpy as np
import pandas as pd
from sklearn.base import clone
from joblib import Parallel, delayed
from copy import deepcopy
from typing import Tuple, Dict, Union

from pamod.base import BaseEvaluator
from pamod.training._trainer import Trainer
from pamod.learner import Model
from pamod.tuning._thresholdopt import ThresholdOptimizer


class RandomSearchTuner(BaseEvaluator):
    def __init__(self, classification: str, criterion: str) -> None:
        """
        Initializes the CrossValidationEvaluator with the provided trainer, classification type, and metric evaluator.

        Args:
            trainer (Trainer): An instance of the Trainer class responsible for training models.
            classification (str): The type of classification ('binary' or 'multiclass').
        """
        super().__init__(classification, criterion)
        self.random_state = self.random_state_val
        self.trainer = Trainer(self.classification, self.criterion)
        self.threshold_optimizer = ThresholdOptimizer(self.classification, self.criterion)

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
    ) -> Tuple[float, Dict[str, Union[float, int]], Union[float, None]]:
        """
        Performs random search on the holdout validation set for both binary and multiclass models.

        Args:
            learner (str): The machine learning model used for evaluation.
            X_train_h (pd.DataFrame): Training features for the holdout set.
            y_train_h (pd.Series): Training labels for the holdout set.
            X_val (pd.DataFrame): Validation features for the holdout set.
            y_val (pd.Series): Validation labels for the holdout set.
            n_configs (int): The number of configurations to evaluate during random search.
            n_jobs (int): The number of parallel jobs for model training.
            verbosity (bool): Whether to print detailed logs during random search.

        Returns:
            tuple:
                - Best score (float)
                - Best hyperparameters (dict)
                - Best threshold (float or None, applicable for binary classification).
        """
        best_score, best_threshold, param_grid, model = self._initialize_search(learner)

        best_params = {}

        for i in range(n_configs):
            # Sample hyperparameters from the grid
            params = self._sample_params(param_grid, i)
            model_clone = clone(model).set_params(**params)
            if "n_jobs" in model_clone.get_params():
                model_clone.set_params(n_jobs=n_jobs)

            # Train and evaluate the model using the provided Trainer class
            score, model_clone, threshold = self.trainer.train(model_clone, X_train_h, y_train_h, X_val, y_val)

            best_score, best_params, best_threshold = self._update_best(
                score, params, threshold, best_score, best_params, best_threshold
            )

            if verbosity:
                self._print_iteration_info(i, model_clone, params, score, mode="val_split", threshold=best_threshold)

        return best_score, best_params, best_threshold

    def cv(
        self, learner, outer_splits, n_configs: int, racing_folds, n_jobs: int, verbosity: bool
    ) -> Tuple[float, Dict[str, Union[float, int]], Union[float, None]]:
        """
        Performs cross-validation with an optional racing strategy and hyperparameter optimization to evaluate a model.

        Args:
            model (sklearn estimator): The machine learning model used for evaluation.
            param_grid (dict): Hyperparameter grid for tuning the model with Random Search.
            outer_splits (list of tuples): Each tuple represents a cross-validation fold with training and validation data.
            n_configs (int): The number of configurations to evaluate during hyperparameter optimization.
            racing_folds (int or None): Specifies the number of folds to use for the racing strategy. If None, evaluates all folds without racing.
            criterion (str): Metric used for evaluation (choices: 'f1', 'brier_score', 'macro_f1').
            n_jobs (int): The number of parallel jobs to run for evaluation.
            verbosity (bool): Activates verbosity during model evaluation process if set to True.

        Returns:
            tuple: Contains the best score achieved, the best hyperparameters, and (for binary classification) the optimal threshold.
        """
        best_score, _, param_grid, model = self._initialize_search(learner)

        for i in range(n_configs):
            params = self._sample_params(param_grid)
            model_clone = clone(model).set_params(**params)
            if "n_jobs" in model_clone.get_params():
                model_clone.set_params(n_jobs=n_jobs)

            scores = self._evaluate_folds(model_clone, best_score, outer_splits, racing_folds, n_jobs)
            avg_score = np.mean(scores)  # Calculate average score across all evaluated folds

            best_score, best_params, _ = self._update_best(avg_score, params, None, best_score, best_params, None)

            if verbosity:
                self._print_iteration_info(i, model_clone, params, avg_score, mode="cv")

        # If binary classification, perform threshold optimization
        if self.classification == "binary" and self.criterion == "f1":
            optimal_threshold = self.threshold_optimizer.optimize_threshold(model, best_params, outer_splits)

        return best_score, best_params, optimal_threshold

    def _evaluate_folds(self, model_clone, best_score, outer_splits, racing_folds, n_jobs):
        """
        Evaluates the model across the specified folds using either standard cross-validation or racing strategy.

        Args:
            model_clone (sklearn estimator): The cloned model to evaluate.
            outer_splits (list of tuples): List of training/validation folds.
            racing_folds (int or None): Number of folds to use for the racing strategy.
            criterion (str): The evaluation criterion.
            n_jobs (int): Number of parallel jobs.

        Returns:
            list: Scores from each fold evaluation.
        """
        num_folds = len(outer_splits)
        if racing_folds is None or racing_folds >= num_folds:
            # Standard cross-validation evaluation
            scores = Parallel(n_jobs=n_jobs)(
                delayed(self.trainer.evaluate_cv)(deepcopy(model_clone), fold) for fold in outer_splits
            )
        else:
            # Racing strategy: evaluate on a subset of folds first
            selected_indices = random.sample(range(num_folds), racing_folds)
            selected_folds = [outer_splits[i] for i in selected_indices]
            initial_scores = Parallel(n_jobs=n_jobs)(
                delayed(self.trainer.evaluate_cv)(deepcopy(model_clone), fold) for fold in selected_folds
            )
            avg_initial_score = np.mean(initial_scores)

            # Continue evaluation on remaining folds if initial score is promising
            if (self.criterion in ["f1", "macro_f1"] and avg_initial_score > best_score) or (
                self.criterion == "brier_score" and avg_initial_score < best_score
            ):
                remaining_folds = [outer_splits[i] for i in range(num_folds) if i not in selected_indices]
                continued_scores = Parallel(n_jobs=n_jobs)(
                    delayed(self.trainer.evaluate_cv)(deepcopy(model_clone), fold) for fold in remaining_folds
                )
                scores = initial_scores + continued_scores
            else:
                scores = initial_scores

        return scores

    def _initialize_search(self, learner: str) -> Tuple[float, Union[float, None], dict, object]:
        """
        Initializes the search by setting up the random seed, best score, parameters, threshold, and model.

        Args:
            learner (str): The learner type to be used for training the model.

        Returns:
            Tuple:
                - best_score: Initialized best score based on the criterion.
                - best_threshold: None or a threshold placeholder for binary classification.
                - param_grid: The parameter grid for the specified model.
                - model: The model instance.
        """
        random.seed(self.random_state)
        best_score = -float("inf") if self.criterion in ["f1", "macro_f1"] else float("inf")
        best_threshold = None  # Threshold is only applicable for binary classification
        model, param_grid = Model.get(learner, self.classification)

        return best_score, best_threshold, param_grid, model

    def _update_best(
        self,
        current_score: float,
        params: dict,
        threshold: Union[float, None],
        best_score: float,
        best_params: dict,
        best_threshold: Union[float, None],
    ) -> Tuple[float, dict, Union[float, None]]:
        """
        Updates the best score, parameters, and threshold if the current score is better.

        Args:
            current_score (float): The current score obtained.
            params (dict): The parameters associated with the current score.
            threshold (float or None): The threshold associated with the current score.
            best_score (float): The best score recorded so far.
            best_params (dict): The best parameters recorded so far.
            best_threshold (float or None): The best threshold recorded so far.

        Returns:
            tuple: Updated best score, best parameters, and best threshold (if applicable).
        """
        if (self.criterion in ["f1", "macro_f1"] and current_score > best_score) or (
            self.criterion == "brier_score" and current_score < best_score
        ):
            best_score = current_score
            best_params = params
            best_threshold = threshold if self.classification == "binary" else None

        return best_score, best_params, best_threshold

    def _sample_params(
        self, param_grid: Dict[str, Union[list, object]], iteration: int = None
    ) -> Dict[str, Union[float, int]]:
        """
        Samples a set of hyperparameters from the provided grid.

        Args:
            param_grid (dict): Hyperparameter grid.
            iteration (int, optional): Current iteration index for random seed adjustment.
                                    If None, the iteration seed will not be adjusted.

        Returns:
            dict: Sampled hyperparameters.
        """
        # Calculate iteration_seed if an iteration value is provided
        iteration_seed = (
            self.random_state + iteration if self.random_state is not None and iteration is not None else None
        )

        params = {}
        for k, v in param_grid.items():
            if hasattr(v, "rvs"):  # For distributions like scipy.stats distributions
                params[k] = v.rvs(random_state=iteration_seed)
            else:
                if iteration_seed is not None:
                    random.seed(iteration_seed)
                params[k] = random.choice(v)  # For list-based grids

        return params

    def _print_iteration_info(
        self, iteration: int, model, params: dict, score: float, mode: str = "cv", threshold: float = None
    ) -> None:
        """
        Prints detailed iteration information during either cross-validation or validation split tuning.

        Args:
            iteration (int): Current iteration index.
            model (sklearn estimator): The evaluated model.
            params (dict): The hyperparameters used in the current iteration.
            score (float): The score achieved in the current iteration.
            mode (str): Specifies whether it's for 'cv' (cross-validation) or 'val_split' (validation split tuning).
            threshold (float, optional): The best threshold if applicable (for validation split tuning).
        """
        model_name = model.__class__.__name__
        params_str = ", ".join([f"{key}={value}" for key, value in params.items()])

        if mode == "cv":
            print(f"RS CV iteration {iteration + 1} {model_name}: '{params_str}', {self.criterion}={score}")
        elif mode == "val_split":
            print(
                f"RS val_split iteration {iteration + 1} {model_name}: '{params_str}', {self.criterion}={score}, threshold={threshold}"
            )
