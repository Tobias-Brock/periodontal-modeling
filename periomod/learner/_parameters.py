import numpy as np
from scipy.stats import loguniform, randint, uniform

###### Random Search search spaces

# XGBoost parameter grid
xgb_param_grid = {
    "learning_rate": uniform(0.01, 0.09),
    "n_estimators": [300, 500, 700],
    "max_depth": randint(4, 10),
    "min_child_weight": randint(1, 3),
    "gamma": uniform(0, 0.3),
    "subsample": uniform(0.8, 0.2),
    "colsample_bytree": uniform(0.8, 0.2),
    "lambda": uniform(0.01, 0.99),
    "alpha": uniform(0.00001, 0.99999),
    "max_delta_step": randint(0, 7),
}

# Random forest parameter grid
rf_param_grid = {
    "n_estimators": [300, 400, 500],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 4),
    "max_features": uniform(0.1, 0.9),
    "class_weight": [None, "balanced", "balanced_subsample"],
    "criterion": ["gini"],
    "max_leaf_nodes": [None, 10, 20, 30, 40],
}

# Logistic regression parameter grid for one-hot encoded datasets
lr_param_grid_oh = {
    "C": np.logspace(-4, 4, 10),
    "penalty": ["l2", None],
    "max_iter": [100, 200, 300],
    "class_weight": [None, "balanced"],
}

# Logistic regression parameter grid for target encoded datasets
lr_param_grid_t = {
    "C": np.logspace(-4, 4, 10),
    "penalty": ["elasticnet", None],
    "max_iter": [100, 200, 300],
    "class_weight": [None, "balanced"],
    "l1_ratio": np.linspace(0, 1, 10),
}

# MLP parameter grid
mlp_param_grid = {
    "hidden_layer_sizes": [
        (50,),
        (100,),
        (150,),
        (50, 50),
        (100, 100),
        (50, 100, 50),
        (100, 50, 100),
    ],
    "activation": ["tanh", "relu", "logistic"],
    "solver": ["sgd", "adam"],
    "alpha": loguniform(0.0001, 0.1),
    "batch_size": ["auto", 64, 128],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "learning_rate_init": uniform(0.001, 0.049),
    "max_iter": [200, 300, 500],
    "shuffle": [True, False],
    "tol": uniform(0.0001, 0.0099),
    "momentum": uniform(0.9, 0.09),
    "nesterovs_momentum": [True, False],
    "beta_1": uniform(0.9, 0.05),
    "beta_2": uniform(0.99, 0.009),
}

###### HEBO search spaces


# HEBO XGBoost
def get_xgb_params_hebo(params_row):
    """Extracts and transforms parameters from a HEBO optimization suggestion.

    Args:
        params_row (pd.Series): A pandas Series containing one row of parameters
        suggested by HEBO.

    Returns:
        dict: A dictionary of the model parameters where numerical parameters are
        converted to their appropriate types (e.g., integers).

    Note:
        The parameter names in `params_row` should match the expected parameter names.
    """
    params = params_row.to_dict()
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["min_child_weight"] = int(params["min_child_weight"])
    params["max_delta_step"] = int(params["max_delta_step"])

    return params


# XGBoost HEBO search space
xgb_search_space_hebo = [
    {"name": "learning_rate", "type": "num", "lb": 0.01, "ub": 0.1},
    {"name": "n_estimators", "type": "cat", "categories": [300, 500, 700]},
    {"name": "max_depth", "type": "int", "lb": 4, "ub": 10},
    {"name": "min_child_weight", "type": "int", "lb": 1, "ub": 3},
    {"name": "gamma", "type": "num", "lb": 0, "ub": 0.3},
    {"name": "subsample", "type": "num", "lb": 0.8, "ub": 1.0},
    {"name": "colsample_bytree", "type": "num", "lb": 0.8, "ub": 1.0},
    {"name": "reg_lambda", "type": "num", "lb": 0.01, "ub": 1.0},
    {"name": "reg_alpha", "type": "num", "lb": 0.01, "ub": 1.0},
    {"name": "max_delta_step", "type": "int", "lb": 0, "ub": 7},
]


# HEBO random forest
def get_rf_params_hebo(params_row):
    """Extracts and transforms parameters from a HEBO optimization suggestion.

    Args:
        params_row (pd.Series): A pandas Series containing one row of parameters
        suggested by HEBO.

    Returns:
        dict: A dictionary of the model parameters where numerical parameters are
        converted to their appropriate types (e.g., integers).

    Note:
        The parameter names in `params_row` should match the expected parameter names.
    """
    params = params_row.to_dict()
    params["min_samples_split"] = int(params["min_samples_split"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])

    return params


# Random forest HEBO search space
rf_search_space_hebo = [
    {"name": "n_estimators", "type": "cat", "categories": [300, 400, 500]},
    {"name": "max_depth", "type": "cat", "categories": [None, 10, 20, 30, 40]},
    {"name": "min_samples_split", "type": "int", "lb": 2, "ub": 10},
    {"name": "min_samples_leaf", "type": "int", "lb": 1, "ub": 4},
    {"name": "max_features", "type": "num", "lb": 0.1, "ub": 1.0},
    {
        "name": "class_weight",
        "type": "cat",
        "categories": [None, "balanced", "balanced_subsample"],
    },
    {"name": "criterion", "type": "cat", "categories": ["gini"]},
    {"name": "max_leaf_nodes", "type": "cat", "categories": [None, 10, 20, 30, 40]},
]


# HEBO for logistic regression
def get_lr_params_hebo_oh(params_row):
    """Extracts and transforms parameters from a HEBO optimization suggestion.

    Args:
        params_row (pd.Series): A pandas Series containing one row of parameters
        suggested by HEBO.

    Returns:
        dict: A dictionary of the model parameters where numerical parameters are
        converted to their appropriate types (e.g., integers).

    Note:
        The parameter names in `params_row` should match the expected parameter names.
    """
    params = params_row.to_dict()
    params["C"] = float(params["C"])

    return params


# HEBO search space for one-hot encoded logistic regression
lr_search_space_hebo_oh = [
    {"name": "C", "type": "num", "lb": 1e-4, "ub": 1e4, "scale": "log"},
    {"name": "penalty", "type": "cat", "categories": ["l2", None]},
    {"name": "max_iter", "type": "cat", "categories": [100, 200, 300]},
    {"name": "class_weight", "type": "cat", "categories": [None, "balanced"]},
]


# Logistic regression for target encoded category variables
def get_lr_params_hebo_t(params_row):
    """Extracts and transforms parameters from a HEBO optimization suggestion.

    Args:
        params_row (pd.Series): A pandas Series containing one row of parameters
        suggested by HEBO.

    Returns:
        dict: A dictionary of the model parameters where numerical parameters are
        converted to their appropriate types (e.g., integers).

    Note:
        The parameter names in `params_row` should match the expected parameter names.
    """
    params = params_row.to_dict()
    params["C"] = float(params["C"])

    if "l1_ratio" in params:
        params["l1_ratio"] = float(params["l1_ratio"])

    return params


# HEBO search space for target encoded logistic regression
lr_search_space_hebo_t = [
    {"name": "C", "type": "num", "lb": 1e-4, "ub": 1e4, "scale": "log"},
    {"name": "penalty", "type": "cat", "categories": ["elasticnet", None]},
    {"name": "max_iter", "type": "cat", "categories": [100, 200, 300]},
    {"name": "class_weight", "type": "cat", "categories": [None, "balanced"]},
    {"name": "l1_ratio", "type": "num", "lb": 0, "ub": 1},
]

hidden_layer_sizes_options = {
    0: (50,),
    1: (100,),
    2: (150,),
    3: (50, 50),
    4: (100, 100),
    5: (50, 100, 50),
    6: (100, 50, 100),
}


# HEBO MLP
def get_mlp_params_hebo(params_row):
    """Extracts and transforms parameters from a HEBO optimization suggestion.

    Args:
        params_row (pd.Series): A pandas Series containing one row of parameters
        suggested by HEBO.

    Returns:
        dict: A dictionary of the model parameters where numerical parameters are
        converted to their appropriate types (e.g., integers).

    Note:
        The parameter names in `params_row` should match the expected parameter names.
    """
    params = params_row.to_dict()
    hidden_layer_sizes_index = int(params["hidden_layer_sizes_index"])
    params["hidden_layer_sizes"] = hidden_layer_sizes_options[hidden_layer_sizes_index]
    params["alpha"] = float(params["alpha"])
    params["learning_rate_init"] = float(params["learning_rate_init"])
    params["tol"] = float(params["tol"])
    params["momentum"] = float(params["momentum"])
    params["beta_1"] = float(params["beta_1"])
    params["beta_2"] = float(params["beta_2"])

    if params["batch_size"] != "auto":
        params["batch_size"] = int(params["batch_size"])

    del params["hidden_layer_sizes_index"]

    return params


# HEBO MLP search space
mlp_search_space_hebo = [
    {"name": "hidden_layer_sizes_index", "type": "int", "lb": 0, "ub": 6},
    {"name": "activation", "type": "cat", "categories": ["tanh", "relu", "logistic"]},
    {"name": "solver", "type": "cat", "categories": ["sgd", "adam"]},
    {"name": "alpha", "type": "num", "lb": 0.0001, "ub": 0.1, "scale": "log"},
    {"name": "batch_size", "type": "cat", "categories": ["auto", "64", "128"]},
    {
        "name": "learning_rate",
        "type": "cat",
        "categories": ["constant", "invscaling", "adaptive"],
    },
    {"name": "learning_rate_init", "type": "num", "lb": 0.001, "ub": 0.05},
    {"name": "max_iter", "type": "cat", "categories": [50, 70, 100]},
    {"name": "shuffle", "type": "cat", "categories": [True, False]},
    {"name": "tol", "type": "num", "lb": 0.0001, "ub": 0.01},
    {"name": "momentum", "type": "num", "lb": 0.9, "ub": 0.99},
    {"name": "nesterovs_momentum", "type": "cat", "categories": [True, False]},
    {"name": "beta_1", "type": "num", "lb": 0.9, "ub": 0.95},
    {"name": "beta_2", "type": "num", "lb": 0.99, "ub": 0.999},
]
