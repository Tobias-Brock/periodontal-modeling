from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import warnings

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.neural_network import MLPClassifier

from ..base import BaseValidator
from ..resampling import Resampler
from ._metrics import brier_loss_multi


class BaseTrainer(BaseValidator, ABC):
    """Abstract base class for training machine learning models.

    This class provides foundational methods for training and evaluating
    machine learning models, including MLP models with early stopping,
    and optimizing decision thresholds. It supports binary and multiclass
    classification and allows for various evaluation metrics, threshold
    tuning, and cross-validation procedures.

    Inherits:
        - `BaseValidator`: Validates instance level variables.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        classification (str): Specifies the type of classification ('binary'
            or 'multiclass').
        criterion (str): Defines the performance criterion to optimize (e.g.,
            'f1' or 'brier_score').
        tuning (Optional[str]): Specifies the tuning method ('holdout' or
            'cv') or None.
        hpo (Optional[str]): Specifies the hyperparameter optimization method.
        mlp_training (Optional[bool]): Flag to indicate if a separate MLP training
            procedure with early stopping is to be used.
        threshold_tuning (Optional[bool]): Determines if threshold tuning is performed
            for binary classification when the criterion is "f1".

    Attributes:
        classification (str): Type of classification ('binary' or 'multiclass').
        criterion (str): Performance criterion to optimize
            ('f1', 'brier_score' or 'macro_f1').
        tuning (Optional[str]): Tuning method ('holdout' or 'cv') or None.
        hpo (Optional[str]): Hyperparameter optimization method if specified.
        mlp_training (Optional[bool]): Indicates if MLP training with early stopping is
            applied.
        threshold_tuning (Optional[bool]): Specifies if threshold tuning is performed
            for binary classification when the criterion is 'f1'.

    Methods:
        evaluate: Determines model performance based on the specified
            classification criterion.
        optimize_threshold: Utilizes cross-validation to optimize the
            decision threshold by aggregating probability predictions.
        evaluate_cv: Evaluates a model on a training-validation fold
            based on the specified criterion, supporting cross-validation.

    Abstract Methods:
        - `train`: Trains the model with standard or custom logic depending
          on the specified learner type.
        - `train_mlp`: Trains an MLP model with early stopping and additional
          evaluation logic if required.
        - `train_final_model`: Trains the final model on the entire dataset,
          applying resampling, parallel processing, and specified sampling
          methods.
    """

    def __init__(
        self,
        classification: str,
        criterion: str,
        tuning: Optional[str],
        hpo: Optional[str],
        mlp_training: Optional[bool],
        threshold_tuning: Optional[bool],
    ) -> None:
        """Initializes the Trainer with classification type and criterion."""
        super().__init__(
            classification=classification, criterion=criterion, tuning=tuning, hpo=hpo
        )
        self.mlp_training = mlp_training
        self.threshold_tuning = threshold_tuning

    def evaluate(
        self,
        y: np.ndarray,
        probs: np.ndarray,
        threshold: Optional[bool] = None,
    ) -> Tuple[float, Optional[float]]:
        """Evaluates model performance based on the classification criterion.

        For binary or multiclass classification.

        Args:
            y (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for each class.
                For binary classification, the probability for the positive class.
                For multiclass, a 2D array with probabilities.
            threshold (bool): Flag for threshold tuning when tuning with F1.
                Defaults to None.

        Returns:
            Tuple: Score and optimal threshold (if for binary).
                For multiclass, only the score is returned.
        """
        if self.classification == "binary":
            return self._evaluate_binary(y=y, probs=probs, threshold=threshold)
        else:
            return self._evaluate_multiclass(y=y, probs=probs)

    def _evaluate_binary(
        self,
        y: np.ndarray,
        probs: np.ndarray,
        threshold: Optional[bool] = None,
    ) -> Tuple[float, Optional[float]]:
        """Evaluates binary classification metrics based on probabilities.

        Args:
            y (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for the positive class.
            threshold (bool): Flag for threshold tuning when tuning with F1.
                Defaults to None.

        Returns:
            Tuple: Score and optimal threshold (if applicable).
        """
        if self.criterion == "f1":
            if threshold:
                scores, thresholds = [], np.linspace(0, 1, 101)
                for threshold in thresholds:
                    preds = (probs >= threshold).astype(int)
                    scores.append(f1_score(y, preds, pos_label=0))
                best_idx = np.argmax(scores)
                return scores[best_idx], thresholds[best_idx]
            else:
                preds = (probs >= 0.5).astype(int)
                return f1_score(y_true=y, y_pred=preds, pos_label=0), 0.5
        else:
            return brier_score_loss(y_true=y, y_proba=probs), None

    def _evaluate_multiclass(
        self, y: np.ndarray, probs: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Evaluates multiclass classification metrics based on probabilities.

        Args:
            y (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for each class (2D array).

        Returns:
            Tuple: The calculated score and None.
        """
        preds = np.argmax(probs, axis=1)

        if self.criterion == "macro_f1":
            return f1_score(y_true=y, y_pred=preds, average="macro"), None
        else:
            return brier_loss_multi(y=y, probs=probs), None

    def evaluate_cv(
        self, model: Any, fold: Tuple, return_probs: bool = False
    ) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
        """Evaluates a model on a specific training-validation fold.

        Based on a chosen performance criterion.

        Args:
            model (Any): The machine learning model used for
                evaluation.
            fold (tuple): A tuple containing two tuples:
                - The first tuple contains the training data (features and labels).
                - The second tuple contains the validation data (features and labels).
                Specifically, it is structured as ((X_train, y_train), (X_val, y_val)),
                where X_train and X_val are the feature matrices, and y_train and y_val
                are the target vectors.
            return_probs (bool): Return predicted probabilities with score if True.

        Returns:
            Union: The calculated score of the model on the validation data, and
                optionally the true labels and predicted probabilities.
        """
        (X_train, y_train), (X_val, y_val) = fold
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            score, _, _ = self.train(
                model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
            )

            if return_probs:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_val)[:, 1]
                    return score, y_val, probs
                else:
                    raise AttributeError(
                        f"The model {type(model)} does not support predict_proba."
                    )

        return score

    def _find_optimal_threshold(
        self, true_labels: np.ndarray, probs: np.ndarray
    ) -> Union[float, None]:
        """Find the optimal threshold based on the criterion.

        Converts probabilities into binary decisions.

        Args:
            true_labels (np.ndarray): The true labels for validation or test data.
            probs (np.ndarray): Predicted probabilities for the positive class.

        Returns:
            Union: The optimal threshold for 'f1', or None if the criterion is
                'brier_score'.
        """
        if self.criterion == "brier_score":
            return None

        elif self.criterion == "f1":
            thresholds = np.linspace(0, 1, 101)
            scores = [
                f1_score(y_true=true_labels, y_pred=probs >= th, pos_label=0)
                for th in thresholds
            ]
            best_threshold = thresholds[np.argmax(scores)]
            print(f"Best threshold: {best_threshold}, Best F1 score: {np.max(scores)}")
            return best_threshold
        raise ValueError(f"Invalid criterion: {self.criterion}")

    def optimize_threshold(
        self,
        model: Any,
        outer_splits: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]],
        n_jobs: int,
    ) -> Union[float, None]:
        """Optimize the decision threshold using cross-validation.

        Aggregates probability predictions across cross-validation folds.

        Args:
            model (Any): The trained machine learning model.
            outer_splits (List[Tuple]): List of ((X_train, y_train), (X_val, y_val)).
            n_jobs (int): Number of parallel jobs to use for cross-validation.

        Returns:
            Union: The optimal threshold for 'f1', or None if the criterion is
                'brier_score'.
        """
        if outer_splits is None:
            return None

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.evaluate_cv)(model, fold, return_probs=True)
            for fold in outer_splits
        )

        all_true_labels = np.concatenate([y for _, y, _ in results])
        all_probs = np.concatenate([probs for _, _, probs in results])

        return self._find_optimal_threshold(
            true_labels=all_true_labels, probs=all_probs
        )

    @abstractmethod
    def train(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        """Trains either an MLP model with custom logic or a standard model.

        Args:
            model (Any): The machine learning model to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
        """

    @abstractmethod
    def train_mlp(
        self,
        mlp_model: MLPClassifier,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        final: bool = False,
    ):
        """Trains MLPClassifier with early stopping and evaluates performance.

        Applies evaluation for both binary and multiclass classification.

        Args:
            mlp_model (MLPClassifier): The MLPClassifier to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            final (bool): Flag for final model training.
        """

    @abstractmethod
    def train_final_model(
        self,
        df: pd.DataFrame,
        resampler: Resampler,
        model: Tuple,
        sampling: Optional[str],
        factor: Optional[float],
        n_jobs: int,
        seed: int,
        test_size: float,
        verbose: bool,
    ):
        """Trains the final model.

        Args:
            df (pandas.DataFrame): The dataset used for model evaluation.
            resampler: Resampling class.
            model (sklearn estimator): The machine learning model used for evaluation.
            sampling (str): The type of sampling to apply.
            factor (float): The factor by which to upsample or downsample.
            n_jobs (int): The number of parallel jobs to run for evaluation.
            seed (int): Seed for splitting.
            test_size (float): Size of train test split.
            verbose (bool): verbose during model evaluation process if set to True.
        """
