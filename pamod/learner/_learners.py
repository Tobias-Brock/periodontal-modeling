from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from pamod.base import BaseValidator
from pamod.learner._parameters import (
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


class Model(BaseValidator):
    def __init__(
        self, learner: str, classification: str, hpo: Optional[str] = None
    ) -> None:
        """Initializes the Model with the learner type and classification.

        Args:
            learner (str): The machine learning algorithm to use
                (e.g., 'RandomForest', 'MLP', 'XGB', 'LogisticRegression').
            classification (str): The type of classification ('binary' or 'multiclass').
            hpo (str, optional): The hyperparameter optimization method to use
                (default None).
        """
        super().__init__(classification, hpo)
        self.learner = learner

    def _get_model_instance(self):
        """Return the machine learning model based on the learner and classification.

        Returns:
            model instance.

        Raises:
            ValueError: If an invalid learner or classification is provided.
        """
        if self.learner == "RandomForest":
            return RandomForestClassifier(random_state=self.random_state_model)
        elif self.learner == "MLP":
            return MLPClassifier(random_state=self.random_state_model)
        elif self.learner == "XGB":
            if self.classification == "binary":
                return xgb.XGBClassifier(
                    objective=self.xgb_obj_binary,
                    eval_metric=self.xgb_loss_binary,
                    random_state=self.random_state_model,
                )
            elif self.classification == "multiclass":
                return xgb.XGBClassifier(
                    objective=self.xgb_obj_multi,
                    eval_metric=self.xgb_loss_multi,
                    random_state=self.random_state_model,
                )
        elif self.learner == "LogisticRegression":
            if self.classification == "binary":
                return LogisticRegression(
                    solver=self.lr_solver_binary,
                    random_state=self.random_state_model,
                )
            elif self.classification == "multiclass":
                return LogisticRegression(
                    multi_class=self.lr_multi_loss,
                    solver=self.lr_solver_multi,
                    random_state=self.random_state_model,
                )
        else:
            raise ValueError(f"Unsupported learner type: {self.learner}")

    @classmethod
    def get(cls, learner: str, classification: str, hpo: Optional[str] = None):
        """Return the machine learning model and parameter grid or HEBO search space.

        Args:
            learner (str): The machine learning algorithm to use.
            classification (str): The type of classification ('binary' or 'multiclass').
            hpo (str): The hyperparameter optimization method ('HEBO' or 'RS').

        Returns:
            tuple: If hpo is 'RS', return model and parameter grid. If hpo is 'HEBO',
                return the model, HEBO search space, and transformation function.
        """
        instance = cls(learner, classification)
        model = instance._get_model_instance()

        if hpo is None:
            raise ValueError("hpo must be provided as 'HEBO' or 'RS'")

        if hpo == "HEBO":
            if learner == "RandomForest":
                return model, rf_search_space_hebo, get_rf_params_hebo
            elif learner == "MLP":
                return model, mlp_search_space_hebo, get_mlp_params_hebo
            elif learner == "XGB":
                return model, xgb_search_space_hebo, get_xgb_params_hebo
            elif learner == "LogisticRegression":
                return model, lr_search_space_hebo_oh, get_lr_params_hebo_oh
            else:
                raise ValueError(f"Unsupported learner type: {learner}")
        elif hpo == "RS":
            if learner == "RandomForest":
                return model, rf_param_grid
            elif learner == "MLP":
                return model, mlp_param_grid
            elif learner == "XGB":
                return model, xgb_param_grid
            elif learner == "LogisticRegression":
                return model, lr_param_grid_oh
            else:
                raise ValueError(f"Unsupported learner type: {learner}")

    @classmethod
    def get_model(cls, learner: str, classification: str):
        """Return only the machine learning model based on learner and classification.

        Args:
            learner (str): The machine learning algorithm to use.
            classification (str): Type of classification ('binary' or 'multiclass').

        Returns:
            model instance.
        """
        instance = cls(learner, classification)
        return instance._get_model_instance()
