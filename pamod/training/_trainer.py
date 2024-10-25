from typing import List, Optional, Tuple, Union
import warnings

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.neural_network import MLPClassifier

from pamod.base import BaseEvaluator
from pamod.learner import Model
from pamod.resampling import Resampler
from pamod.training import brier_loss_multi, final_metrics, get_probs


class Trainer(BaseEvaluator):
    def __init__(
        self,
        classification: str,
        criterion: str,
        tuning: Optional[str],
        hpo: Optional[str],
        mlp_training: bool = True,
        threshold_tuning: bool = True,
    ) -> None:
        """Initializes the Trainer with classification type and criterion.

        Args:
            classification (str): The type of classification ('binary' or
                'multiclass').
            criterion (str): The performance criterion to optimize (e.g., 'f1',
                'brier_score').
            tuning (Optional[str]): The tuning method ('holdout' or 'cv'). Can be None.
            hpo (Optional[str]): The hyperparameter optimization method. Can be None.
            mlp_training (bool): Flag for separate MLP training with early stopping.
            threshold_tuning (bool): Perform threshold tuning for binary classification
                if the criterion is "f1". Defaults to True.
        """
        super().__init__(classification, criterion, tuning, hpo)
        self.mlp_training = mlp_training
        self.threshold_tuning = threshold_tuning

    def evaluate(
        self,
        y_val: np.ndarray,
        probs: np.ndarray,
        threshold: bool = True,
    ) -> Tuple[float, Optional[float]]:
        """Evaluates model performance based on the classification criterion.

        For binary or multiclass classification.

        Args:
            y_val (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for each class.
                For binary classification, the probability for the positive class.
                For multiclass, a 2D array with probabilities.
            threshold (bool): Flag for threshold tuning when tuning with F1.

        Returns:
            Tuple[float, Optional[float]]: Score and optimal threshold (if for binary).
                For multiclass, only the score is returned.
        """
        if self.classification == "binary":
            return self._evaluate_binary(y_val, probs, threshold)
        else:
            return self._evaluate_multiclass(y_val, probs)

    def _evaluate_binary(
        self,
        y_val: np.ndarray,
        probs: np.ndarray,
        threshold: bool = True,
    ) -> Tuple[float, Optional[float]]:
        """Evaluates binary classification metrics based on probabilities.

        Args:
            y_val (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for the positive class.
            threshold (bool): Flag for threshold tuning when tuning with F1.

        Returns:
            Tuple[float, Optional[float]]: Score and optimal threshold (if applicable).
        """
        if self.criterion == "f1":
            if threshold:
                scores, thresholds = [], np.linspace(0, 1, 101)
                for threshold in thresholds:
                    preds = (probs >= threshold).astype(int)
                    scores.append(f1_score(y_val, preds, pos_label=0))
                best_idx = np.argmax(scores)
                return scores[best_idx], thresholds[best_idx]
            else:
                preds = (probs >= 0.5).astype(int)
                return f1_score(y_val, preds, pos_label=0), 0.5
        else:
            return brier_score_loss(y_val, probs), None

    def _evaluate_multiclass(
        self, y_val: np.ndarray, probs: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Evaluates multiclass classification metrics based on probabilities.

        Args:
            y_val (np.ndarray): True labels for the validation data.
            probs (np.ndarray): Probability predictions for each class (2D array).

        Returns:
            float: The calculated score.
        """
        preds = np.argmax(probs, axis=1)

        if self.criterion == "macro_f1":
            return f1_score(y_val, preds, average="macro"), None
        else:
            return brier_loss_multi(y_val, probs), None

    def train(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[float, object, Union[float, None]]:
        """Trains either an MLP model with custom logic or a standard model.

        Args:
            model (sklearn estimator): The machine learning model to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.

        Returns:
            Tuple: The evaluation score, trained model, and the best threshold.
        """
        if isinstance(model, MLPClassifier) and self.mlp_training:
            score, model, best_threshold = self.train_mlp(
                model, X_train, y_train, X_val, y_val, self.mlp_training
            )
        else:
            model.fit(X_train, y_train)
            probs = get_probs(model, self.classification, X_val)
            best_threshold = None

            if self.classification == "binary" and (
                self.tuning == "cv" or self.hpo == "hebo"
            ):
                score, _ = self.evaluate(y_val, probs, False)
            else:
                score, best_threshold = self.evaluate(
                    y_val, probs, self.threshold_tuning
                )

        return score, model, best_threshold

    def train_mlp(
        self,
        mlp_model,
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

        Returns:
            tuple: The best validation score, trained MLPClassifier, and the
                optimal threshold (None for multiclass).
        """
        best_val_score = (
            -float("inf") if self.criterion in ["f1", "macro_f1"] else float("inf")
        )
        best_threshold = None
        no_improvement_count = 0

        for _ in range(mlp_model.max_iter):
            mlp_model.partial_fit(X_train, y_train, classes=np.unique(y_train))

            probs = get_probs(mlp_model, self.classification, X_val)
            if self.classification == "binary":
                if final or (self.tuning == "cv" or self.hpo == "hebo"):
                    score, _ = self.evaluate(y_val, probs, False)
            else:
                score, best_threshold = self.evaluate(
                    y_val, probs, self.threshold_tuning
                )

            if self.criterion in ["f1", "macro_f1"]:
                improvement = score > best_val_score + self.tol
            else:
                improvement = score < best_val_score - self.tol

            if improvement:
                best_val_score = score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.n_iter_no_change:
                break

        return best_val_score, mlp_model, best_threshold

    def evaluate_cv(
        self, model, fold: Tuple, return_probs: bool = False
    ) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
        """Evaluates a model on a specific training-validation fold.

        Based on a chosen performance criterion.

        Args:
            model (sklearn estimator): The machine learning model used for
                evaluation.
            fold (tuple): A tuple containing two tuples:
                - The first tuple contains the training data (features and labels).
                - The second tuple contains the validation data (features and labels).
                Specifically, it is structured as ((X_train, y_train), (X_val, y_val)),
                where X_train and X_val are the feature matrices, and y_train and y_val
                are the target vectors.
            return_probs (bool): Return predicted probabilities with score if True.

        Returns:
            Union[float, Tuple[float, np.ndarray, np.ndarray]]: The calculated score of
                the model on the validation data, and optionally the true labels and
                predicted probabilities.
        """
        (X_train, y_train), (X_val, y_val) = fold
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

            score, _, _ = self.train(model, X_train, y_train, X_val, y_val)

            if return_probs:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_val)[:, 1]
                    return score, y_val, probs
                else:
                    raise AttributeError(
                        f"The model {type(model)} does not support predict_proba."
                    )

        return score

    def find_optimal_threshold(
        self, true_labels: np.ndarray, probs: np.ndarray
    ) -> Union[float, None]:
        """Find the optimal threshold based on the criterion.

        Converts probabilities into binary decisions.

        Args:
            true_labels (np.ndarray): The true labels for validation or test data.
            probs (np.ndarray): Predicted probabilities for the positive class.

        Returns:
            float or None: The optimal threshold for 'f1', or None if the criterion is
                'brier_score'.
        """
        if self.criterion == "brier_score":
            return None

        elif self.criterion == "f1":
            thresholds = np.linspace(0, 1, 101)
            scores = [
                f1_score(true_labels, probs >= th, pos_label=0) for th in thresholds
            ]
            best_threshold = thresholds[np.argmax(scores)]
            print(f"Best threshold: {best_threshold}, Best F1 score: {np.max(scores)}")
            return best_threshold
        raise ValueError(f"Invalid criterion: {self.criterion}")

    def optimize_threshold(
        self,
        model,
        outer_splits: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]],
        n_jobs: int,
    ) -> Union[float, None]:
        """Optimize the decision threshold using cross-validation.

        Aggregates probability predictions across cross-validation folds.

        Args:
            model (sklearn estimator): The trained machine learning model.
            best_params (dict): The best hyperparameters obtained from optimization.
            outer_splits (List[Tuple]): List of ((X_train, y_train), (X_val, y_val)).
            n_jobs (int): Number of parallel jobs to use for cross-validation.

        Returns:
            float or None: The optimal threshold for 'f1', or None if the criterion is
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

        return self.find_optimal_threshold(all_true_labels, all_probs)

    def train_final_model(
        self,
        df: pd.DataFrame,
        resampler: Resampler,
        model: Tuple,
        sampling: Optional[str],
        factor: Optional[float],
        n_jobs: Optional[int],
        seed: int,
        test_size: float,
        verbosity: bool = True,
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
            verbosity (bool): Verbosity during model evaluation process if set to True.

        Returns:
            dict: A dictionary containing the trained model and metrics.
        """
        learner, best_params, best_threshold = model
        model = Model.get_model(learner, self.classification)
        final_model = clone(model)
        final_model.set_params(**best_params)
        final_model.best_threshold = best_threshold

        if "n_jobs" in final_model.get_params():
            final_model.set_params(n_jobs=n_jobs)

        train_df, test_df = resampler.split_train_test_df(df, seed, test_size)

        X_train, y_train, X_test, y_test = resampler.split_x_y(
            train_df, test_df, sampling, factor
        )
        if learner == "mlp" and self.mlp_training:
            train_df_h, test_df_h = resampler.split_train_test_df(
                train_df, seed, test_size
            )

            X_train_h, y_train_h, X_val, y_val = resampler.split_x_y(
                train_df_h, test_df_h, sampling, factor
            )
            _, final_model, _ = self.train_mlp(
                final_model, X_train_h, y_train_h, X_val, y_val, self.mlp_training
            )
        else:
            final_model.fit(X_train, y_train)
        final_probs = get_probs(final_model, self.classification, X_test)

        if (
            self.criterion == "f1"
            and final_probs is not None
            and np.any(final_probs)
            and best_threshold is not None
        ):
            final_predictions = (final_probs >= best_threshold).astype(int)
        else:
            final_predictions = final_model.predict(X_test)

        metrics = final_metrics(
            self.classification, y_test, final_predictions, final_probs, best_threshold
        )
        if verbosity:
            unpacked_metrics = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in metrics.items()
            }
            results = {
                "Learner": learner,
                "Tuning": "final",
                "HPO": self.hpo,
                "Criterion": self.criterion,
                **unpacked_metrics,
            }

            df_results = pd.DataFrame([results])
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 1000)

            print("\nFinal Model Metrics Summary:")
            print(df_results)

        return {"model": final_model, "metrics": metrics}
