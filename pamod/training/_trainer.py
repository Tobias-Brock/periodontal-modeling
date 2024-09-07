from sklearn.neural_network import MLPClassifier
from pamod.training import MLPTrainer
from pamod.resampling import MetricEvaluator


class Trainer:
    def __init__(self, classification_type):
        """
        Initializes the Trainer with a classification type.

        Args:
            classification_type (str): The type of classification ('binary' or 'multiclass').
        """
        self.classification_type = classification_type

    def train(self, model, X_train, y_train, X_val, y_val, criterion):
        """
        General method to train models. Detects if the model is MLP or a standard model and applies appropriate training logic.

        Args:
            model (sklearn estimator): The machine learning model to be trained.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation labels.
            criterion (str): The evaluation criterion - 'f1', 'brier_score', 'macro_f1', etc.

        Returns:
            tuple: The evaluation score, trained model, and threshold (if applicable for binary classification).
        """
        # Handle MLP models with custom training logic
        if isinstance(model, MLPClassifier):
            mlp_trainer = MLPTrainer(max_iter=model.max_iter, classification_type=self.classification_type)
            score, trained_model, best_threshold = mlp_trainer.train(model, X_train, y_train, X_val, y_val, criterion)
        else:
            # For non-MLP models, perform standard training and evaluation
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)

            # Handle binary vs multiclass cases
            if self.classification_type == "binary":
                score, best_threshold = self._evaluate_binary(model, probs, y_val, criterion)
            else:
                score, best_threshold = self._evaluate_multiclass(model, probs, y_val, criterion)

            trained_model = model

        return score, trained_model, best_threshold

    def _evaluate_binary(self, probs, y_val, criterion):
        """
        Evaluates the binary model based on probabilities and criterion.

        Args:
            model (sklearn estimator): The trained binary classification model.
            probs (array-like): Probability predictions from the model.
            y_val (pd.Series): Validation labels.
            criterion (str): Criterion for evaluation ('f1', 'brier_score').

        Returns:
            tuple: The evaluation score and the best threshold (for F1 optimization).
        """
        probs = probs[:, 1]  # Extract probabilities for the positive class
        metric_evaluator = MetricEvaluator("binary")
        score, best_threshold = metric_evaluator.evaluate(y_val, probs, criterion)
        return score, best_threshold

    def _evaluate_multiclass(self, probs, y_val, criterion):
        """
        Evaluates the multiclass model based on probabilities and criterion.

        Args:
            model (sklearn estimator): The trained multiclass classification model.
            probs (array-like): Probability predictions from the model.
            y_val (pd.Series): Validation labels.
            criterion (str): Criterion for evaluation ('macro_f1', 'brier_score').

        Returns:
            tuple: The evaluation score and None (no threshold needed for multiclass).
        """
        metric_evaluator = MetricEvaluator("multiclass")
        score, _ = metric_evaluator.evaluate(y_val, probs, criterion)
        return score, None
