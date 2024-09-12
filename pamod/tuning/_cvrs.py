import random
import numpy as np
import hydra
from sklearn.base import clone
from joblib import Parallel, delayed
from copy import deepcopy

from pamod.resampling import MetricEvaluator
from pamod.training import Trainer
from pamod.tuning._thresholdopt import ThresholdOptimizer


class CrossValidationEvaluator:
    def __init__(self, classification: str, criterion: str) -> None:
        """
        Initializes the CrossValidationEvaluator with the provided trainer, classification type, and metric evaluator.

        Args:
            trainer (Trainer): An instance of the Trainer class responsible for training models.
            classification (str): The type of classification ('binary' or 'multiclass').
        """
        with hydra.initialize(config_path="../../config", version_base="1.2"):
            cfg = hydra.compose(config_name="config")
        self.classification = classification
        self.criterion = criterion
        self.random_state = cfg.resample.random_state_cv
        self.metric_evaluator = MetricEvaluator(self.classification, self.criterion)
        self.trainer = Trainer(self.classification, self.criterion)

    def evaluate_with_cv(self, model, param_grid, outer_splits, n_configs, racing_folds, n_jobs, verbosity):
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
        random.seed(self.random_state)
        best_score = -float("inf") if self.criterion in ["f1", "macro_f1"] else float("inf")
        best_params = None
        optimal_threshold = None

        # Iterate through the specified number of hyperparameter configurations
        for i in range(n_configs):
            iteration_seed = self.random_state + i if self.random_state is not None else None

            # Sample hyperparameters
            params = self._sample_params(param_grid, iteration_seed)

            model_clone = clone(model).set_params(**params)
            if "n_jobs" in model_clone.get_params():
                model_clone.set_params(n_jobs=n_jobs)

            # Evaluate across folds using standard CV or racing strategy
            scores = self._evaluate_folds(model_clone, best_score, outer_splits, racing_folds, n_jobs)

            avg_score = np.mean(scores)  # Calculate average score across all evaluated folds

            # Update best score and parameters if current configuration is better
            if (self.criterion in ["f1", "macro_f1"] and avg_score > best_score) or (
                self.criterion == "brier_score" and avg_score < best_score
            ):
                best_score = avg_score
                best_params = params

            if verbosity:
                self._print_iteration_info(i, model_clone, params, avg_score, self.criterion)

        # If binary classification, perform threshold optimization
        if self.classification == "binary" and self.criterion == "f1":
            threshold_optimizer = ThresholdOptimizer(self.criterion, self.classification)
            optimal_threshold = threshold_optimizer.optimize_threshold(model, best_params, outer_splits)

        return best_score, best_params, optimal_threshold

    def _sample_params(self, param_grid, iteration_seed):
        """
        Samples a set of hyperparameters from the provided grid.

        Args:
            param_grid (dict): Hyperparameter grid.
            iteration_seed (int): Random seed for reproducibility.

        Returns:
            dict: Sampled hyperparameters.
        """
        params = {}
        for k, v in param_grid.items():
            if hasattr(v, "rvs"):
                params[k] = v.rvs(random_state=iteration_seed)
            else:
                if iteration_seed is not None:
                    random.seed(iteration_seed)
                params[k] = random.choice(v)
        return params

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
                delayed(self.trainer.evaluate_cv)(deepcopy(model_clone), fold, self.criterion) for fold in outer_splits
            )
        else:
            # Racing strategy: evaluate on a subset of folds first
            selected_indices = random.sample(range(num_folds), racing_folds)
            selected_folds = [outer_splits[i] for i in selected_indices]
            initial_scores = Parallel(n_jobs=n_jobs)(
                delayed(self.trainer.evaluate_cv)(deepcopy(model_clone), fold, self.criterion)
                for fold in selected_folds
            )
            avg_initial_score = np.mean(initial_scores)

            # Continue evaluation on remaining folds if initial score is promising
            if (self.criterion in ["f1", "macro_f1"] and avg_initial_score > best_score) or (
                self.criterion == "brier_score" and avg_initial_score < best_score
            ):
                remaining_folds = [outer_splits[i] for i in range(num_folds) if i not in selected_indices]
                continued_scores = Parallel(n_jobs=n_jobs)(
                    delayed(self.trainer.evaluate_cv)(deepcopy(model_clone), fold, self.criterion)
                    for fold in remaining_folds
                )
                scores = initial_scores + continued_scores
            else:
                scores = initial_scores

        return scores

    def _print_iteration_info(self, iteration, model, params, score, criterion):
        """
        Prints detailed iteration information during cross-validation.

        Args:
            iteration (int): Current iteration index.
            model (sklearn estimator): The evaluated model.
            params (dict): The hyperparameters used in the current iteration.
            score (float): The score achieved in the current iteration.
            criterion (str): The evaluation criterion.
        """
        model_name = model.__class__.__name__
        params_str = ", ".join([f"{key}={value}" for key, value in params.items()])
        print(f"RS CV iteration {iteration + 1} {model_name}: '{params_str}', {criterion}={score}")
