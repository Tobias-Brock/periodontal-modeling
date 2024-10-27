from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.neural_network import MLPClassifier

from ..learner import Model
from ..resampling import Resampler
from ._basetrainer import BaseTrainer
from ._metrics import final_metrics, get_probs


class Trainer(BaseTrainer):
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
        super().__init__(
            classification=classification,
            criterion=criterion,
            tuning=tuning,
            hpo=hpo,
            mlp_training=mlp_training,
            threshold_tuning=threshold_tuning,
        )

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
                mlp_model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                final=self.mlp_training,
            )
        else:
            model.fit(X_train, y_train)
            probs = get_probs(model=model, classification=self.classification, X=X_val)
            best_threshold = None

            if self.classification == "binary" and (
                self.tuning == "cv" or self.hpo == "hebo"
            ):
                score, _ = self.evaluate(y=y_val, probs=probs, threshold=False)
            else:
                score, best_threshold = self.evaluate(
                    y=y_val, probs=probs, threshold=self.threshold_tuning
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

            probs = get_probs(
                model=mlp_model, classification=self.classification, X=X_val
            )
            if self.classification == "binary":
                if final or (self.tuning == "cv" or self.hpo == "hebo"):
                    score, _ = self.evaluate(y=y_val, probs=probs, threshold=False)
            else:
                score, best_threshold = self.evaluate(
                    y=y_val, probs=probs, threshold=self.threshold_tuning
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

        train_df, test_df = resampler.split_train_test_df(
            df=df, seed=seed, test_size=test_size
        )

        X_train, y_train, X_test, y_test = resampler.split_x_y(
            train_df=train_df, test_df=test_df, sampling=sampling, factor=factor
        )
        if learner == "mlp" and self.mlp_training:
            train_df_h, test_df_h = resampler.split_train_test_df(
                df=train_df, seed=seed, test_size=test_size
            )

            X_train_h, y_train_h, X_val, y_val = resampler.split_x_y(
                train_df=train_df_h, test_df=test_df_h, sampling=sampling, factor=factor
            )
            _, final_model, _ = self.train_mlp(
                mlp_model=final_model,
                X_train=X_train_h,
                y_train=y_train_h,
                X_val=X_val,
                y_val=y_val,
                final=self.mlp_training,
            )
        else:
            final_model.fit(X_train, y_train)
        final_probs = get_probs(
            model=final_model, classification=self.classification, X=X_test
        )

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
            classification=self.classification,
            y=y_test,
            preds=final_predictions,
            probs=final_probs,
            threshold=best_threshold,
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
            pd.set_option("display.max_columns", None, "display.width", 1000)
            print("\nFinal Model Metrics Summary:\n", df_results)

        return {"model": final_model, "metrics": metrics}
