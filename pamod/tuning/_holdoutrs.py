import random
import hydra
from sklearn.base import clone


class RandomSearchHoldout:
    def __init__(self, classification_type, n_configs, n_jobs, verbosity):
        """
        Initializes the RandomSearchHoldout for hyperparameter tuning using a holdout set.

        Args:
            classification_type (str): The type of classification ('binary' or 'multiclass').
            n_configs (int): The number of configurations to evaluate during random search.
            n_jobs (int): The number of parallel jobs for model training.
            verbosity (bool): Whether to print detailed logs during random search.
            random_state (int, optional): Random seed for reproducibility.
        """
        with hydra.initialize(config_path="../../config", version_base="1.2"):
            cfg = hydra.compose(config_name="config")
        self.classification_type = classification_type
        self.n_configs = n_configs
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.random_state = cfg.tuning.random_state_val

    def evaluate_with_holdout(self, model, param_grid, X_train_h, y_train_h, X_val, y_val, criterion, trainer):
        """
        Performs random search on the holdout validation set for both binary and multiclass models.

        Args:
            model (sklearn estimator): The machine learning model used for evaluation.
            param_grid (dict): Hyperparameter grid for random search.
            X_train_h (pd.DataFrame): Training features.
            y_train_h (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            criterion (str): Criterion for optimization ('f1', 'brier_score' for binary; 'macro_f1', 'brier_score' for multiclass).
            trainer (Trainer): An instance of the Trainer class to handle the training process.

        Returns:
            tuple: Best score, best hyperparameters, and the best threshold (if applicable for binary).
        """
        random.seed(self.random_state)
        best_score = -float("inf") if criterion in ["f1", "macro_f1"] else float("inf")
        best_params = None
        best_threshold = None  # Threshold is only applicable for binary classification

        for i in range(self.n_configs):
            # Sample hyperparameters from the grid
            params = self._sample_params(param_grid, i)
            model_clone = clone(model).set_params(**params)
            if "n_jobs" in model_clone.get_params():
                model_clone.set_params(n_jobs=self.n_jobs)

            # Train and evaluate the model using the provided Trainer class
            score, model_clone, threshold = trainer.train(model_clone, X_train_h, y_train_h, X_val, y_val, criterion)

            # Update best score and params if current is better
            if (criterion in ["f1", "macro_f1"] and score > best_score) or (
                criterion == "brier_score" and score < best_score
            ):
                best_score = score
                best_params = params
                best_threshold = threshold if self.classification_type == "binary" else None

            # Verbosity
            if self.verbosity:
                model_name = model.__class__.__name__
                params_str = ", ".join([f"{key}={value}" for key, value in params.items()])
                print(
                    f"RS val_split iteration {i + 1} {model_name}: '{params_str}', {criterion}={score}, threshold={best_threshold}"
                )

        return best_score, best_params, best_threshold

    def _sample_params(self, param_grid, iteration):
        """
        Samples a set of hyperparameters from the grid.

        Args:
            param_grid (dict): Hyperparameter grid.
            iteration (int): Current iteration index for random seed adjustment.

        Returns:
            dict: Sampled hyperparameters.
        """
        iteration_seed = self.random_state + iteration if self.random_state is not None else None
        params = {}
        for k, v in param_grid.items():
            if hasattr(v, "rvs"):
                params[k] = v.rvs(random_state=iteration_seed)
            else:
                if iteration_seed is not None:
                    random.seed(iteration_seed)
                params[k] = random.choice(v)
        return params
