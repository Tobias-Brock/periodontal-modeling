from typing import Optional, Tuple, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from ..base import BaseConfig
from ._parameters import (
    get_lr_params_hebo_oh,
    get_mlp_params_hebo,
    get_rf_params_hebo,
    get_xgb_params_hebo,
    lr_param_grid_oh,
    lr_search_space_hebo_oh,
    mlp_param_grid,
    mlp_search_space_hebo,
    rf_param_grid,
    rf_search_space_hebo,
    xgb_param_grid,
    xgb_search_space_hebo,
)


class Model(BaseConfig):
    """Configurable machine learning model class with HPO options.

    This class provides an interface for initializing machine learning models
    based on the specified learner type (e.g., random forest, logistic regression)
    and classification type (binary or multiclass). It supports optional
    hyperparameter optimization (HPO) configurations.

    Inherits:
        - `BaseConfig`: Provides base configuration settings.

    Args:
        learner (str): The machine learning algorithm to use. Options include:
            'rf' (random forest), 'mlp' (multi-layer perceptron), 'xgb' (XGBoost),
            or 'lr' (logistic regression).
        classification (str): Specifies the classification type. Can be either
            'binary' or 'multiclass'.
        hpo (Optional[str]): The hyperparameter optimization (HPO) method to
            use. Options are 'hebo' or 'rs'. Defaults to None, which requires
            specifying HPO in relevant methods.

    Attributes:
        learner (str): The specified machine learning algorithm for the model.
            Options include 'rf', 'mlp', 'xgb', and 'lr'.
        classification (str): Defines the type of classification task.
            Options are 'binary' or 'multiclass'.
        hpo (Optional[str]): Hyperparameter optimization method for tuning, if
            specified. Options are 'hebo' or 'rs'.

    Methods:
        get: Class method returning a model and hyperparameter search space
            or parameter grid.
        get_model: Class method that returns only the instantiated model
            without HPO options.

    Example:
        ```
        model_instance = Model.get(learner="rf", classification="binary", hpo="hebo")
        trained_model = Model.get_model(learner="mlp", classification="multiclass")
        ```
    """

    def __init__(
        self,
        learner: str,
        classification: str,
        hpo: Optional[str] = None,
    ) -> None:
        """Initializes the Model with the learner type and classification.

        Args:
            learner (str): The machine learning algorithm to use
                (e.g., 'rf', 'mlp', 'xgb', 'lr').
            classification (str): The type of classification ('binary' or 'multiclass').
            hpo (str, optional): The hyperparameter optimization method to use.
                Defaults to None.
        """
        super().__init__()
        self.classification = classification
        self.hpo = hpo
        self.learner = learner

    def _get_model_instance(self):
        """Return the machine learning model based on the learner and classification.

        Returns:
            model instance.

        Raises:
            ValueError: If an invalid learner or classification is provided.
        """
        if self.learner == "rf":
            return RandomForestClassifier(random_state=self.learner_state)
        elif self.learner == "mlp":
            return MLPClassifier(random_state=self.learner_state)
        elif self.learner == "xgb":
            if self.classification == "binary":
                return xgb.XGBClassifier(
                    objective=self.xgb_obj_binary,
                    eval_metric=self.xgb_loss_binary,
                    random_state=self.learner_state,
                )
            elif self.classification == "multiclass":
                return xgb.XGBClassifier(
                    objective=self.xgb_obj_multi,
                    eval_metric=self.xgb_loss_multi,
                    random_state=self.learner_state,
                )
        elif self.learner == "lr":
            if self.classification == "binary":
                return LogisticRegression(
                    solver=self.lr_solver_binary,
                    random_state=self.learner_state,
                )
            elif self.classification == "multiclass":
                return LogisticRegression(
                    multi_class=self.lr_multi_loss,
                    solver=self.lr_solver_multi,
                    random_state=self.learner_state,
                )
        else:
            raise ValueError(f"Unsupported learner type: {self.learner}")

    @classmethod
    def get(
        cls, learner: str, classification: str, hpo: Optional[str] = None
    ) -> Union[Tuple, Tuple]:
        """Return the machine learning model and parameter grid or HEBO search space.

        Args:
            learner (str): The machine learning algorithm to use.
            classification (str): The type of classification ('binary' or 'multiclass').
            hpo (str): The hyperparameter optimization method ('hebo' or 'rs').

        Returns:
            Union: If hpo is 'rs', returns a tuple of (model, parameter grid).
                If hpo is 'hebo', returns a tuple of (model, HEBO search space,
                transformation function).
        """
        instance = cls(learner, classification)
        model = instance._get_model_instance()

        if hpo is None:
            raise ValueError("hpo must be provided as 'hebo' or 'rs'")

        if hpo == "hebo":
            if learner == "rf":
                return model, rf_search_space_hebo, get_rf_params_hebo
            elif learner == "mlp":
                return model, mlp_search_space_hebo, get_mlp_params_hebo
            elif learner == "xgb":
                return model, xgb_search_space_hebo, get_xgb_params_hebo
            elif learner == "lr":
                return model, lr_search_space_hebo_oh, get_lr_params_hebo_oh
        elif hpo == "rs":
            if learner == "rf":
                return model, rf_param_grid
            elif learner == "mlp":
                return model, mlp_param_grid
            elif learner == "xgb":
                return model, xgb_param_grid
            elif learner == "lr":
                return model, lr_param_grid_oh

        raise ValueError(f"Unsupported hpo type '{hpo}' or learner type '{learner}'")

    @classmethod
    def get_model(cls, learner: str, classification: str) -> Union[
        RandomForestClassifier,
        LogisticRegression,
        MLPClassifier,
        xgb.XGBClassifier,
    ]:
        """Return only the machine learning model based on learner and classification.

        Args:
            learner (str): The machine learning algorithm to use.
            classification (str): Type of classification ('binary' or 'multiclass').

        Returns:
            model: model instance (Union[sklearn estiamtor]).
        """
        instance = cls(learner, classification)
        return instance._get_model_instance()
