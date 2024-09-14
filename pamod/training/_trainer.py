from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier

from pamod.base import BaseEvaluator
from pamod.resampling import MetricEvaluator
from pamod.training import MLPTrainer


class Trainer(BaseEvaluator):
    def __init__(self, classification: str, criterion: str) -> None:
        """
        Initializes the Trainer with a classification type.

        Args:
            classification (str): The type of classification ('binary' or 'multiclass').
        """
        super().__init__(classification, criterion)
        self.metric_evaluator = MetricEvaluator(self.classification, self.criterion)

    def train(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[float, object, Union[float, None]]:
        """
        General method to train models. Detects if the model is MLP or a standard model and applies appropriate training logic.

        Args:
            model (sklearn estimator): The machine learning model to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.

        Returns:
            tuple: The evaluation score, trained model, and threshold (if applicable for binary classification).
        """
        # Handle MLP models with custom training logic
        if isinstance(model, MLPClassifier):
            mlp_trainer = MLPTrainer(max_iter=model.max_iter, classification=self.classification)
            score, trained_model, best_threshold = mlp_trainer.train(
                model, X_train, y_train, X_val, y_val, self.criterion
            )
        else:
            # For non-MLP models, perform standard training and evaluation
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)
            score, best_threshold = self._evaluate(probs, y_val)

            trained_model = model

        return score, trained_model, best_threshold

    def _evaluate(self, probs: np.ndarray, y_val: pd.Series) -> Tuple[float, Union[float, None]]:
        """
        Generalized evaluation for both binary and multiclass models based on probabilities and criterion.

        Args:
            probs (np.ndarray): Probability predictions from the model.
            y_val (pd.Series): Validation labels.

        Returns:
            tuple: The evaluation score and the best threshold (if applicable for binary classification).
        """
        if self.classification == "binary":
            probs = probs[:, 1]  # Extract probabilities for the positive class
            score, best_threshold = self.metric_evaluator.evaluate(y_val, probs)
        else:
            score, _ = self.metric_evaluator.evaluate(y_val, probs)
            best_threshold = None

        return score, best_threshold

    def evaluate_cv(self, model, fold: Tuple) -> float:
        """
        Evaluates a given machine learning model on a specific training-validation fold,
        based on a chosen performance criterion.

        Args:
            model (sklearn estimator): The machine learning model used for evaluation.
            fold (tuple): A tuple containing two tuples: the first for the training data (features and labels)
                        and the second for the validation data (features and labels). Specifically, it is
                        structured as ((X_train, y_train), (X_val, y_val)), where X_train and X_val are the
                        feature matrices, and y_train and y_val are the target vectors.

        Returns:
            float: The calculated score of the model on the validation data according to the specified criterion.
                Higher scores indicate better performance for 'f1', while lower scores are better
                for 'brier_score'.

        Raises:
            ValueError: If an invalid evaluation criterion is specified.
        """
        (X_train, y_train), (X_val, y_val) = fold

        if isinstance(model, MLPClassifier):
            mlptrainer = MLPTrainer(self.classification, self.criterion)
            _, model, _ = mlptrainer.train(model, X_train, y_train, X_val, y_val, self.criterion)
        else:
            model.fit(X_train, y_train)
        if self.classification == "binary":
            preds = (
                model.predict_proba(X_val)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X_val)
            )
        elif self.classification == "multiclass":
            if self.criterion == "macro_f1":
                preds = model.predict(X_val)
            elif self.criterion == "brier_score":
                preds = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

        return self.metric_evaluator.evaluate_metric(model, y_val, preds)

    def evaluate_objective(
        self,
        tuning: str,
        model_clone: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        outer_splits: List[tuple],
        n_jobs: int,
    ) -> float:
        """
        Evaluates the model's performance based on the tuning strategy (holdout or cross-validation).

        Args:
            tuning (str): The tuning method, either 'holdout' or 'cv' (cross-validation).
            model_clone (BaseEstimator): The cloned machine learning model to be evaluated.
            X_train (pd.DataFrame): Training features for the holdout set.
            y_train (pd.Series): Training labels for the holdout set.
            X_val (pd.DataFrame): Validation features for the holdout set (used for 'holdout' tuning).
            y_val (pd.Series): Validation labels for the holdout set (used for 'holdout' tuning).
            outer_splits (List[tuple]): List of cross-validation folds, each a tuple containing (X_train_fold, y_train_fold).
            n_jobs (int): Number of parallel jobs to use for cross-validation.

        Returns:
            float: The model's performance metric based on the specified tuning strategy.
        """
        if tuning == "holdout":
            model_clone.fit(X_train, y_train)
            probs = model_clone.predict_proba(X_val)

            if self.classification == "binary":
                probs = probs[:, 1]
                return self.metric_evaluator.evaluate_metric(model_clone, y_val, probs)
            elif self.classification == "multiclass":
                preds = np.argmax(probs, axis=1)
                return self.metric_evaluator.evaluate_metric(model_clone, y_val, preds)

        elif tuning == "cv":
            scores = Parallel(n_jobs=n_jobs)(
                delayed(self.evaluate_cv)(deepcopy(model_clone), fold) for fold in outer_splits
            )
            return np.mean(scores)
