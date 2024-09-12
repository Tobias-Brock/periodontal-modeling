import random
import hydra
from sklearn.base import clone
from typing import Tuple, Dict, Union

from pamod.training import Trainer


class RandomSearchHoldout:
    def __init__(self, classification: str, criterion: str, n_configs: int, n_jobs: int, verbosity: bool) -> None:
        """
        Initializes the RandomSearchHoldout for hyperparameter tuning using a holdout set.

        Args:
            classification (str): The type of classification ('binary' or 'multiclass').
            criterion (str): The evaluation criterion ('f1', 'brier_score', 'macro_f1').
            n_configs (int): The number of configurations to evaluate during random search.
            n_jobs (int): The number of parallel jobs for model training.
            verbosity (bool): Whether to print detailed logs during random search.
        """
        with hydra.initialize(config_path="../../config", version_base="1.2"):
            cfg = hydra.compose(config_name="config")
        self.classification = classification
        self.criterion = criterion
        self.n_configs = n_configs
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.random_state = cfg.tuning.random_state_val
        self.trainer = Trainer(self.classification)

    def holdout_rs(
        self, model, param_grid: Dict[str, Union[list, object]], X_train_h, y_train_h, X_val, y_val
    ) -> Tuple[float, Dict[str, Union[float, int]], Union[float, None]]:
        """
        Performs random search on the holdout validation set for both binary and multiclass models.

        Args:
            model (sklearn estimator): The machine learning model used for evaluation.
            param_grid (dict): Hyperparameter grid for random search.
            X_train_h (pd.DataFrame): Training features for the holdout set.
            y_train_h (pd.Series): Training labels for the holdout set.
            X_val (pd.DataFrame): Validation features for the holdout set.
            y_val (pd.Series): Validation labels for the holdout set.

        Returns:
            tuple:
                - Best score (float)
                - Best hyperparameters (dict)
                - Best threshold (float or None, applicable for binary classification).
        """
        random.seed(self.random_state)
        best_score = -float("inf") if self.criterion in ["f1", "macro_f1"] else float("inf")
        best_params = None
        best_threshold = None  # Threshold is only applicable for binary classification

        for i in range(self.n_configs):
            # Sample hyperparameters from the grid
            params = self._sample_params(param_grid, i)
            model_clone = clone(model).set_params(**params)
            if "n_jobs" in model_clone.get_params():
                model_clone.set_params(n_jobs=self.n_jobs)

            # Train and evaluate the model using the provided Trainer class
            score, model_clone, threshold = self.trainer.train(
                model_clone, X_train_h, y_train_h, X_val, y_val, self.criterion
            )

            # Update best score and params if current is better
            if (self.criterion in ["f1", "macro_f1"] and score > best_score) or (
                self.criterion == "brier_score" and score < best_score
            ):
                best_score = score
                best_params = params
                best_threshold = threshold if self.classification == "binary" else None

            # Verbosity
            if self.verbosity:
                model_name = model.__class__.__name__
                params_str = ", ".join([f"{key}={value}" for key, value in params.items()])
                print(
                    f"RS val_split iteration {i + 1} {model_name}: '{params_str}', {self.criterion}={score}, threshold={best_threshold}"
                )

        return best_score, best_params, best_threshold

    def _sample_params(
        self, param_grid: Dict[str, Union[list, object]], iteration: int
    ) -> Dict[str, Union[float, int]]:
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
