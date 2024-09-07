import numpy as np
import hydra
from pamod.resampling import MetricEvaluator


class MLPTrainer:
    def __init__(self, max_iter, classification_type):
        """
        Initializes the MLPTrainer with metric evaluator and training parameters.

        Args:
            metric_evaluator (MetricEvaluator): An instance of MetricEvaluator for evaluating metrics.
            max_iter (int): Maximum number of iterations for training.
            tol (float): Tolerance for improvement. Stops training if improvement is less than tol.
            n_iter_no_change (int): Number of iterations with no improvement to wait before stopping.
        """
        with hydra.initialize(config_path="../../config", version_base="1.2"):
            cfg = hydra.compose(config_name="config")

        self.metric_evaluator = MetricEvaluator(classification_type)
        self.max_iter = max_iter
        self.tol = cfg.mlp.mlp_tol
        self.n_iter_no_change = cfg.mlp.mlp_no_improve

    def train(self, mlp_model, X_train, y_train, X_val, y_val, criterion):
        """
        Trains an MLPClassifier using early stopping logic, with an option for binary or multiclass evaluation.

        Args:
            mlp_model (MLPClassifier): The MLPClassifier to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            criterion (str): Criterion for optimization - 'f1', 'brier_score' (binary), or 'macro_f1', 'brier_score' (multiclass).

        Returns:
            tuple: The best validation score, trained MLPClassifier, and the optimal threshold (if applicable).
        """
        if self.metric_evaluator.classification_type == "binary":
            return self._train_model(mlp_model, X_train, y_train, X_val, y_val, criterion, binary=True)
        else:
            return self._train_model(mlp_model, X_train, y_train, X_val, y_val, criterion, binary=False)

    def _train_model(self, mlp_model, X_train, y_train, X_val, y_val, criterion, binary):
        """
        Generalized method for training MLPClassifier with early stopping and evaluation for both binary and multiclass.

        Args:
            mlp_model (MLPClassifier): The MLPClassifier to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            criterion (str): Criterion for optimization.
            binary (bool): Whether it's binary classification.

        Returns:
            tuple: The best validation score, trained MLPClassifier, and the optimal threshold (None for multiclass).
        """
        best_val_score, best_threshold = self._initialize_best_score(criterion, binary)
        no_improvement_count = 0

        for _ in range(self.max_iter):
            self._fit_model(mlp_model, X_train, y_train, binary)

            probs = self._get_probabilities(mlp_model, X_val, binary)
            score, best_threshold = self.metric_evaluator.evaluate(y_val, probs, criterion)

            if self._is_improvement(score, best_val_score, criterion):
                best_val_score = score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.n_iter_no_change:
                break  # Stop early

        return best_val_score, mlp_model, best_threshold if binary else None

    def _initialize_best_score(self, criterion, binary):
        """
        Initializes the best score and threshold for tracking improvements.

        Args:
            criterion (str): Criterion for optimization.
            binary (bool): Whether it's binary classification.

        Returns:
            tuple: Best initial score and default threshold.
        """
        best_val_score = -float("inf") if criterion in ["f1", "macro_f1"] else float("inf")
        best_threshold = 0.5 if binary else None
        return best_val_score, best_threshold

    def _fit_model(self, mlp_model, X_train, y_train, binary):
        """
        Fits the MLP model using partial_fit.

        Args:
            mlp_model (MLPClassifier): The MLPClassifier to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            binary (bool): Whether it's binary classification.
        """
        mlp_model.partial_fit(X_train, y_train, classes=np.unique(y_train))

    def _get_probabilities(self, mlp_model, X_val, binary):
        """
        Gets the predicted probabilities from the MLP model.

        Args:
            mlp_model (MLPClassifier): The trained MLPClassifier model.
            X_val (pd.DataFrame): Validation features.
            binary (bool): Whether it's binary classification.

        Returns:
            array-like: Predicted probabilities.
        """
        if binary:
            return mlp_model.predict_proba(X_val)[:, 1]
        else:
            return mlp_model.predict_proba(X_val)

    def _is_improvement(self, score, best_val_score, criterion):
        """
        Determines if there is an improvement in the validation score.

        Args:
            score (float): Current validation score.
            best_val_score (float): Best validation score so far.
            criterion (str): Criterion for optimization.

        Returns:
            bool: Whether the current score is an improvement.
        """
        if criterion in ["f1", "macro_f1"]:
            return score > best_val_score + self.tol
        else:
            return score < best_val_score - self.tol
