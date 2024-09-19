from typing import List, Optional, Tuple, Union
import warnings

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier

from pamod.base import BaseEvaluator
from pamod.learner import Model
from pamod.resampling import Resampler
from pamod.training import MetricEvaluator, MLPTrainer, final_metrics, get_probs


class Trainer(BaseEvaluator):
    def __init__(
        self,
        classification: str,
        criterion: str,
        tuning: Optional[str],
        hpo: Optional[str],
    ) -> None:
        """Initializes the Trainer with classification type and criterion.

        Args:
            classification (str): The type of classification ('binary' or
                'multiclass').
            criterion (str): The performance criterion to optimize (e.g., 'f1',
                'brier_score').
            tuning (Optional[str]): The tuning method ('holdout' or 'cv'). Can be None.
            hpo (Optional[str]): The hyperparameter optimization method. Can be None.
        """
        super().__init__(classification, criterion, tuning, hpo)
        self.metric_evaluator = MetricEvaluator(self.classification, self.criterion)

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
        if isinstance(model, MLPClassifier):
            mlp_trainer = MLPTrainer(
                self.classification, self.criterion, self.tuning, self.hpo
            )
            score, model, best_threshold = mlp_trainer.train(
                model, X_train, y_train, X_val, y_val
            )
        else:
            # For non-MLP models, perform standard training and evaluation
            model.fit(X_train, y_train)
            probs = get_probs(model, self.classification, X_val)
            best_threshold = None

            if self.classification == "binary" and (
                self.tuning == "cv" or self.hpo == "HEBO"
            ):
                score = self.metric_evaluator.evaluate_metric(model, y_val, probs)
            else:
                score, best_threshold = self.metric_evaluator.evaluate(y_val, probs)

        return score, model, best_threshold

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

            # Train the model and get the score
            score, _, _ = self.train(model, X_train, y_train, X_val, y_val)

            if return_probs:
                # Get predicted probabilities if requested
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
            return None  # Thresholding is not applicable for Brier score

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
        n_jobs: int,
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
            verbosity (bool): Verbosity during model evaluation process if set to True.

        Returns:
            dict: A dictionary containing the trained model and metrics.
        """
        learner, best_params, best_threshold = model
        model = Model.get_model(learner, self.classification)
        final_model = clone(model)
        final_model.set_params(**best_params)

        if "n_jobs" in final_model.get_params():
            final_model.set_params(n_jobs=n_jobs)  # Set parallel jobs if supported

        train_df, test_df = resampler.split_train_test_df(df)

        X_train, y_train, X_test, y_test = resampler.split_x_y(
            train_df, test_df, sampling, factor
        )
        if learner == "MLP":
            mlp = MLPTrainer(self.classification, self.criterion, None, None)
            train_df_h, test_df_h = resampler.split_train_test_df(train_df)

            X_train_h, y_train_h, X_val, y_val = resampler.split_x_y(
                train_df_h, test_df_h, sampling, factor
            )
            _, final_model, _ = mlp.train(
                final_model, X_train_h, y_train_h, X_val, y_val, final=True
            )
        else:
            final_model.fit(X_train, y_train)
        final_probs = get_probs(final_model, self.classification, X_test)

        if self.criterion in ["f1"] and final_probs is not None:
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
                "HPO": self.hpo,  # Final model doesn't involve HPO
                "Criterion": self.criterion,
                **unpacked_metrics,  # Unpack rounded metrics here
            }

            df_results = pd.DataFrame([results])
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 1000)

            print("\nFinal Model Metrics Summary:")
            print(df_results)

        return {"model": final_model, "metrics": metrics}
