from pamod.resampling import Resampler, brier_loss_multi
from pamod.data import ProcessedDataLoader
from pamod.tuning import RandomSearchTuner
from pamod.learner import Model

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def train_final_model(df, classification, final_model, sampling, factor, criterion, n_jobs, verbosity):
    """
    Trains the final model on the entire training set with the best parameters found, evaluates it on the test set for binary classification,
    and performs threshold optimization based on the specified criterion (F1, Brier score, or ROC AUC).

    Args:
        df (pandas.DataFrame): The dataset used for model evaluation.
        classification (str): Determines classification type for sampling strategy.
        model (sklearn estimator): The machine learning model used for evaluation.
        best_params (dict): The best hyperparameters obtained during optimization.
        criterion (str): Criterion for optimization ('f1' or 'brier_score').
        n_jobs (int): The number of parallel jobs to run for evaluation.
        verbosity (bool): Actiavtes verbosity during model evaluation process if set to True.

    Returns:
        dict: A dictionary containing the trained final model, its evaluation metrics, and the best threshold found.
    """
    learner, best_params, best_threshold = final_model
    model = Model.get_model(learner, classification)
    resampler = Resampler(classification)
    final_model = clone(model)
    if best_params is None:
        best_params = {}
    final_model.set_params(**best_params)

    # Set n_jobs for parallel training if applicable and not an MLPClassifier
    if "n_jobs" in final_model.get_params():
        final_model.set_params(n_jobs=n_jobs)  # Set parallel jobs if supported

    train_df, test_df = resampler.split_train_test_df(df)

    X_train, y_train, X_test, y_test = resampler.split_x_y(train_df, test_df, sampling, factor)

    final_model.fit(X_train, y_train)

    if classification == "binary":
        final_probs = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, "predict_proba") else None
    elif classification == "multiclass":
        # For multiclass, don't select only one column, retain the full probability matrix
        final_probs = final_model.predict_proba(X_test) if hasattr(final_model, "predict_proba") else None

    if criterion in ["f1"] and final_probs is not None:
        final_predictions = (final_probs >= best_threshold).astype(int)
    else:
        final_predictions = final_model.predict(X_test)

    if classification == "binary":
        f1 = f1_score(y_test, final_predictions, pos_label=0)
        precision = precision_score(y_test, final_predictions, pos_label=0)
        recall = recall_score(y_test, final_predictions, pos_label=0)
        accuracy = accuracy_score(y_test, final_predictions)
        brier_score_value = brier_score_loss(y_test, final_probs) if final_probs is not None else None
        roc_auc_value = roc_auc_score(y_test, final_probs) if final_probs is not None else None
        conf_matrix = confusion_matrix(y_test, final_predictions)

        final_metrics = {
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "Brier Score": brier_score_value,
            "ROC AUC Score": roc_auc_value,
            "Confusion Matrix": conf_matrix,
            "Best Threshold": best_threshold,
        }
    elif classification == "multiclass":
        brier_score = brier_loss_multi(y_test, final_probs)

        # Compile evaluation metrics
        final_metrics = {
            "macro_f1": f1_score(y_test, final_predictions, average="macro"),
            "accuracy": accuracy_score(y_test, final_predictions),
            "class_f1_scores": f1_score(y_test, final_predictions, average=None),
            "brier_score": brier_score,
        }

    # Print final quantities if set to true
    if verbosity:
        model_name = model.__class__.__name__
        print(f"{model_name}: Final metrics: {final_metrics}")

    return {"model": final_model, "metrics": final_metrics}


# Perform model evaluation with different sampling strategies and handling of imbalanced data
def perform_model_evaluation(
    df,
    classification,
    learner,
    method,
    sampling,
    factor,
    n_configs,
    hpo,
    criterion,
    racing_folds,
    n_jobs,
    verbosity=True,
):
    """
    Perform model evaluation specifically for binary classification using Random Search (RS)
    for hyperparameter optimization.

    Args:
        df (pandas.DataFrame): The dataset used for model evaluation.
        classification (str): Determines classification type for sampling strategy.
        model (sklearn estimator): The machine learning model used for evaluation.
        param_grid (dict): Hyperparameter grid for tuning the model with Random Search.
        method (str): Method for model evaluation, one of 'cv' or 'val_split'.
        n_configs (int): The number of configurations to evaluate during HPO.
        hpo (str): Hyperparameter optimization method, one of 'RS' (Random Search) or 'HEBO'.
        racing_folds (int or None): Number of folds to use for initial evaluation in the racing strategy.
                                    If None, standard CV is performed on all folds.
        criterion (str): Criterion for optimization - 'macro_f1' or 'brier_score'.
        n_jobs (int): The number of parallel jobs to run for evaluation.
        verbosity (bool): Actiavtes verbosity during model evaluation process if set to True.

    Returns:
        dict: A dictionary containing the trained model and its evaluation metrics.

    Raises:
        ValueError: If an invalid evaluation method is specified.
    """
    resampler = Resampler(classification)
    train_df, _ = resampler.split_train_test_df(df)

    # Determine the evaluation strategy

    if hpo == "RS":
        # Initialize the tuner only for Random Search HPO
        tuner = RandomSearchTuner(classification, criterion)

        if method == "split":
            train_df_h, test_df_h = resampler.split_train_test_df(train_df)
            X_train_h, y_train_h, X_val, y_val = resampler.split_x_y(train_df_h, test_df_h, sampling, factor)

            # Perform Random Search on holdout validation set
            _, best_params, best_threshold = tuner.holdout(
                learner, X_train_h, y_train_h, X_val, y_val, n_configs, n_jobs, verbosity
            )

        elif method == "cv":
            outer_splits, _ = resampler.cv_folds(df, sampling, factor)

            _, best_params, best_threshold = tuner.cv(learner, outer_splits, n_configs, racing_folds, n_jobs, verbosity)

        else:
            raise ValueError("Invalid evaluation method specified. Choose 'cv' or 'split'.")

    final_model = (learner, best_params, best_threshold)

    # Train the final model with the best parameters found and evaluate its performance
    final_model_metrics = train_final_model(
        df,
        classification,
        final_model,
        sampling,
        factor,
        criterion,
        n_jobs,
        verbosity,
    )

    return final_model_metrics


def main():
    dataloader = ProcessedDataLoader("pdgrouprevaluation")  # Correct the target name
    df = dataloader.load_data()  # Load the dataset
    df = dataloader.transform_target(df)  # Transform the target column

    # Call model evaluation
    perform_model_evaluation(
        df=df,
        classification="multiclass",
        learner="XGB",
        method="split",
        sampling=None,
        factor=None,
        n_configs=3,
        hpo="RS",
        criterion="macro_f1",
        racing_folds=5,
        n_jobs=None,
    )


if __name__ == "__main__":
    main()
