from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import clone

from ..learner import Model
from ..training import Trainer
from ._basetuner import BaseTuner


class HEBOTuner(BaseTuner):
    """HEBO (Bayesian Optimization) hyperparameter tuning class.

    This class performs hyperparameter tuning for machine learning models
    using Bayesian Optimization with the HEBO library, supporting both holdout
    and cross-validation (CV) tuning methods.

    Inherits:
        - `BaseTuner`: Provides a framework for implementing HPO strategies,
          including shared evaluation and logging functions.

    Args:
        classification (str): The type of classification ('binary' or 'multiclass').
        criterion (str): The evaluation criterion (e.g., 'f1', 'brier_score').
        tuning (str): The type of tuning ('holdout' or 'cv').
        hpo (str): The hyperparameter optimization method (default is 'HEBO').
        n_configs (int): Number of configurations to evaluate. Defaults to 10.
        n_jobs (int): Number of parallel jobs for model training.
            Defaults to 1.
        verbose (bool): Whether to print detailed logs during HEBO optimization.
            Defaults to True.
        trainer (Optional[Trainer]): Trainer instance for model training.
        mlp_training (bool): Enables MLP-specific training with early stopping.
        threshold_tuning (bool): Enables threshold tuning for binary classification
            when the criterion is "f1".

    Attributes:
        classification (str): Specifies the classification type
            ('binary' or 'multiclass').
        criterion (str): The tuning criterion to optimize
            ('f1', 'brier_score' or 'macro_f1').
        tuning (str): Indicates the tuning approach ('holdout' or 'cv').
        hpo (str): Hyperparameter optimization method, default is 'HEBO'.
        n_configs (int): Number of configurations for HPO.
        n_jobs (int): Number of parallel jobs for model evaluation.
        verbose (bool): Enables logging during tuning if set to True.
        mlp_training (bool): Flag to enable MLP training with early stopping.
        threshold_tuning (bool): Enables threshold tuning for binary classification.
        trainer (Trainer): Trainer instance for managing model training and evaluation.

    Methods:
        holdout: Optimizes hyperparameters using HEBO for holdout validation.
        cv: Optimizes hyperparameters using HEBO with cross-validation.

    Example:
        ```
        trainer = Trainer(
            classification="binary",
            criterion="f1",
            tuning="holdout",
            hpo="hebo",
            mlp_training=True,
            threshold_tuning=True,
        )

        tuner = HEBOTuner(
            classification="binary",
            criterion="f1",
            tuning="holdout",
            hpo="hebo",
            n_configs=10,
            n_jobs=-1,
            verbose=True,
            trainer=trainer,
            mlp_training=True,
            threshold_tuning=True,
        )

        best_params, best_threshold = tuner.holdout(
            learner="rf",
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        # Using cross-validation
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
        hpo: str = "hebo",
        n_configs: int = 10,
        n_jobs: int = 1,
        verbose: bool = True,
        trainer: Optional[Trainer] = None,
        mlp_training: bool = True,
        threshold_tuning: bool = True,
    ) -> None:
        """Initialize HEBOTuner."""
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
    ) -> Tuple[Dict[str, Union[float, int]], Optional[float]]:
        """Perform Bayesian Optimization using hebo for holdout validation.

        Args:
            learner (str): The machine learning model to evaluate.
            X_train (pd.DataFrame): The training features for the holdout set.
            y_train (pd.Series): The training labels for the holdout set.
            X_val (pd.DataFrame): The validation features for the holdout set.
            y_val (pd.Series): The validation labels for the holdout set.

        Returns:
            Tuple: The best hyperparameters and the best threshold.
        """
        return self._run_optimization(
            learner=learner,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            outer_splits=None,
        )

    def cv(
        self,
        learner: str,
        outer_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
        racing_folds: Optional[int] = None,
    ) -> Tuple[Dict[str, Union[float, int]], Optional[float]]:
        """Perform Bayesian Optimization using HEBO with cross-validation.

        Args:
            learner (str): The machine learning model to evaluate.
            outer_splits (List[Tuple[pd.DataFrame, pd.DataFrame]]):
                List of cross-validation folds.
            racing_folds (Optional[int]): Number of racing folds; if None, regular
                cross-validation is performed.

        Returns:
            Tuple: The best hyperparameters and the best threshold.
        """
        return self._run_optimization(
            learner=learner,
            X_train=None,
            y_train=None,
            X_val=None,
            y_val=None,
            outer_splits=outer_splits,
        )

    def _run_optimization(
        self,
        learner: str,
        X_train: Optional[pd.DataFrame],
        y_train: Optional[pd.Series],
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        outer_splits: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]],
    ) -> Tuple[Dict[str, Union[float, int]], Optional[float]]:
        """Perform Bayesian Optimization using HEBO for holdout and cross-validation.

        Args:
            learner (str): The machine learning model to evaluate.
            X_train (Optional[pd.DataFrame]): Training features for the holdout set
                (None if using CV).
            y_train (Optional[pd.Series]): Training labels for the holdout set
                (None if using CV).
            X_val (Optional[pd.DataFrame]): Validation features for the holdout set
                (None if using CV).
            y_val (Optional[pd.Series]): Validation labels for the holdout set
                (None if using CV).
            outer_splits (Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]]):
                Cross-validation folds (None if using holdout).

        Returns:
            Tuple: The best hyperparameters and the best threshold.
        """
        model, search_space, params_func = Model.get(
            learner=learner, classification=self.classification, hpo=self.hpo
        )
        space = DesignSpace().parse(search_space)
        optimizer = HEBO(space)

        for i in range(self.n_configs):
            params_suggestion = optimizer.suggest(n_suggestions=1).iloc[0]
            params_dict = params_func(params_suggestion)

            score = self._objective(
                model=model,
                params_dict=params_dict,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                outer_splits=outer_splits,
            )
            optimizer.observe(pd.DataFrame([params_suggestion]), np.array([score]))

            if self.verbose:
                self._print_iteration_info(
                    iteration=i, model=model, params_dict=params_dict, score=score
                )

        best_params_idx = optimizer.y.argmin()
        best_params_df = optimizer.X.iloc[best_params_idx]
        best_params = params_func(best_params_df)
        best_threshold = None
        if self.classification == "binary" and self.threshold_tuning:
            model_clone = clone(model).set_params(**best_params)
            if self.criterion == "f1":
                if self.tuning == "holdout":
                    model_clone.fit(X_train, y_train)
                    probs = model_clone.predict_proba(X_val)[:, 1]
                    _, best_threshold = self.trainer.evaluate(
                        y_val, probs, self.threshold_tuning
                    )

                elif self.tuning == "cv":
                    best_threshold = self.trainer.optimize_threshold(
                        model=model_clone, outer_splits=outer_splits, n_jobs=self.n_jobs
                    )

        return best_params, best_threshold

    def _objective(
        self,
        model: Any,
        params_dict: Dict[str, Union[float, int]],
        X_train: Optional[pd.DataFrame],
        y_train: Optional[pd.Series],
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        outer_splits: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]],
    ) -> float:
        """Evaluate the model performance for both holdout and cross-validation.

        Args:
            model (Any): The machine learning model to evaluate.
            params_dict (Dict[str, Union[float, int]]): The suggested hyperparameters
                as a dictionary.
            X_train (Optional[pd.DataFrame]): Training features for the holdout set
                (None for CV).
            y_train (Optional[pd.Series]): Training labels for the holdout set
                (None for CV).
            X_val (Optional[pd.DataFrame]): Validation features for the holdout set
                (None for CV).
            y_val (Optional[pd.Series]): Validation labels for the holdout set
                (None for CV).
            outer_splits (Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]]):
                Cross-validation folds (None for holdout).

        Returns:
            float: The evaluation score to be minimized by HEBO.
        """
        model_clone = clone(model)
        model_clone.set_params(**params_dict)

        if "n_jobs" in model_clone.get_params():
            model_clone.set_params(n_jobs=self.n_jobs)

        score = self._evaluate_objective(
            model=model_clone,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            outer_splits=outer_splits,
        )

        return -score if self.criterion in ["f1", "macro_f1"] else score

    def _evaluate_objective(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        outer_splits: Optional[List[Tuple[pd.DataFrame, pd.Series]]],
    ) -> float:
        """Evaluates the model's performance based on the tuning strategy.

        The tuning strategy can be either 'holdout' or 'cv' (cross-validation).

        Args:
            model (Any): The cloned machine learning model to be
                evaluated.
            X_train (pd.DataFrame): Training features for the holdout set.
            y_train (pd.Series): Training labels for the holdout set.
            X_val (pd.DataFrame): Validation features for the holdout set (used for
                'holdout' tuning).
            y_val (pd.Series): Validation labels for the holdout set (used for
                'holdout' tuning).
            outer_splits (List[tuple]): List of cross-validation folds, each a tuple
                containing (X_train_fold, y_train_fold).

        Returns:
            float: The model's performance metric based on tuning strategy.
        """
        if self.tuning == "holdout":
            score, _, _ = self.trainer.train(
                model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
            )
            return score

        elif self.tuning == "cv":
            if outer_splits is None:
                raise ValueError(
                    "outer_splits cannot be None when using cross-validation."
                )
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self.trainer.evaluate_cv)(deepcopy(model), fold)
                for fold in outer_splits
            )
            return np.mean(scores)

        raise ValueError(f"Unsupported criterion: {self.tuning}")
