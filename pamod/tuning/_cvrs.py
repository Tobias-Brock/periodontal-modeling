import random
import numpy as np
from sklearn.base import clone
from joblib import Parallel, delayed
from copy import deepcopy


class CrossValidationEvaluator:
    def __init__(self, trainer, classification_type, metric_evaluator):
        """
        Initializes the CrossValidationEvaluator with the provided trainer, classification type, and metric evaluator.

        Args:
            trainer (Trainer): An instance of the Trainer class responsible for training models.
            classification_type (str): The type of classification ('binary' or 'multiclass').
            metric_evaluator (MetricEvaluator): An instance of the MetricEvaluator class for calculating evaluation metrics.
        """
        self.trainer = trainer
        self.classification_type = classification_type
        self.metric_evaluator = metric_evaluator

    def evaluate_with_cv(
        self, model, param_grid, outer_splits, n_configs, racing_folds, criterion, n_jobs, verbosity, random_state
    ):
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
            random_state (int): Seed for reproducibility.

        Returns:
            tuple: Contains the best score achieved, the best hyperparameters, and (for binary classification) the optimal threshold.
        """
        random.seed(random_state)
        best_score = -float("inf") if criterion in ["f1", "macro_f1"] else float("inf")
        best_params = None
        optimal_threshold = None

        # Iterate through the specified number of hyperparameter configurations
        for i in range(n_configs):
            iteration_seed = random_state + i if random_state is not None else None

            # Sample hyperparameters
            params = self._sample_params(param_grid, iteration_seed)

            model_clone = clone(model).set_params(**params)
            if "n_jobs" in model_clone.get_params():
                model_clone.set_params(n_jobs=n_jobs)

            # Evaluate across folds using standard CV or racing strategy
            scores = self._evaluate_folds(model_clone, outer_splits, racing_folds, criterion, n_jobs)

            avg_score = np.mean(scores)  # Calculate average score across all evaluated folds

            # Update best score and parameters if current configuration is better
            if (criterion in ["f1", "macro_f1"] and avg_score > best_score) or (
                criterion == "brier_score" and avg_score < best_score
            ):
                best_score = avg_score
                best_params = params

            if verbosity:
                self._print_iteration_info(i, model_clone, params, avg_score, criterion)

        # If binary classification, perform threshold optimization
        if self.classification_type == "binary":
            optimal_threshold = self._threshold_optimization(model, best_params, outer_splits, criterion)

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

    def _evaluate_folds(self, model_clone, outer_splits, racing_folds, criterion, n_jobs):
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
                delayed(self.trainer.train)(
                    deepcopy(model_clone), fold[0][0], fold[0][1], fold[1][0], fold[1][1], criterion
                )
                for fold in outer_splits
            )
        else:
            # Racing strategy: evaluate on a subset of folds first
            selected_indices = random.sample(range(num_folds), racing_folds)
            selected_folds = [outer_splits[i] for i in selected_indices]
            remaining_folds = [outer_splits[i] for i in range(num_folds) if i not in selected_indices]

            initial_scores = Parallel(n_jobs=n_jobs)(
                delayed(self.trainer.train)(
                    deepcopy(model_clone), fold[0][0], fold[0][1], fold[1][0], fold[1][1], criterion
                )
                for fold in selected_folds
            )
            avg_initial_score = np.mean(initial_scores)

            # Continue evaluation on remaining folds if initial score is promising
            if (criterion == "macro_f1" and avg_initial_score > best_score) or (
                criterion == "brier_score" and avg_initial_score < best_score
            ):
                continued_scores = Parallel(n_jobs=n_jobs)(
                    delayed(self.trainer.train)(
                        deepcopy(model_clone), fold[0][0], fold[0][1], fold[1][0], fold[1][1], criterion
                    )
                    for fold in remaining_folds
                )
                scores = initial_scores + continued_scores
            else:
                scores = initial_scores

        return scores

    def _threshold_optimization(self, model, best_params, outer_splits, criterion):
        """
        Performs threshold optimization for the best model parameters across folds.

        Args:
            model (sklearn estimator): The machine learning model used for evaluation.
            best_params (dict): The best hyperparameters obtained during optimization.
            outer_splits (list of tuples): A list of tuples containing the train and validation data for each fold.
            criterion (str): The evaluation criterion.

        Returns:
            float: The optimal threshold (for binary classification) or None (for multiclass).
        """
        # Implement threshold optimization using a similar strategy as shown previously
        return threshold_optimization(model, best_params, outer_splits, criterion)

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
