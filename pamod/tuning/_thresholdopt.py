import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier


class ThresholdOptimizer:
    def __init__(self, criterion):
        """
        Initializes the ThresholdOptimizer with the criterion for thresholding.

        Args:
            criterion (str): The criterion for optimization ('f1' or 'brier_score').
        """
        if criterion not in ["f1", "brier_score"]:
            raise ValueError("Unsupported criterion. Choose either 'f1' or 'brier_score'.")
        self.criterion = criterion

    def find_optimal_threshold(self, true_labels, probabilities):
        """
        Finds the optimal threshold based on the criterion for converting probabilities into binary decisions.

        Args:
            true_labels (np.ndarray): The true labels for validation or test data.
            probabilities (np.ndarray): Predicted probabilities for the positive class.

        Returns:
            float or None: The optimal threshold for 'f1', or None if the criterion is 'brier_score'.
        """
        if self.criterion == "brier_score":
            return None  # Thresholding is not applicable for Brier score

        thresholds = np.linspace(0, 1, 101)

        if self.criterion == "f1":
            # Calculate F1 scores for each threshold
            scores = [f1_score(true_labels, probabilities >= th) for th in thresholds]
            best_threshold = thresholds[np.argmax(scores)]
            return best_threshold

    def optimize_threshold(self, model, best_params, outer_splits):
        """
        Optimizes the decision threshold by aggregating probability predictions across cross-validation folds.

        Args:
            model (sklearn estimator): The trained machine learning model.
            best_params (dict): The best hyperparameters obtained from optimization.
            outer_splits (list of tuples): A list of tuples, where each tuple contains ((X_train, y_train), (X_val, y_val)).

        Returns:
            float or None: The optimal threshold for 'f1', or None if the criterion is 'brier_score'.
        """
        best_model = clone(model).set_params(**best_params)

        all_true_labels = []
        all_probs = []

        # Iterate through cross-validation folds
        for (X_train, y_train), (X_val, y_val) in outer_splits:
            # Custom logic for training MLPClassifier if necessary
            if isinstance(model, MLPClassifier):
                _, trained_model, _ = train_mlp_classifier_binary(
                    best_model, X_train, y_train, X_val, y_val, model.max_iter, self.criterion
                )
                probs = trained_model.predict_proba(X_val)[:, 1]
            else:
                # Standard model training and evaluation
                best_model.fit(X_train, y_train)
                probs = best_model.predict_proba(X_val)[:, 1]

            # Collect predictions and true labels
            all_probs.extend(probs)
            all_true_labels.extend(y_val)

        # Convert lists to numpy arrays
        all_true_labels = np.array(all_true_labels)
        all_probs = np.array(all_probs)

        # Find and return the optimal threshold
        return self.find_optimal_threshold(all_true_labels, all_probs)
