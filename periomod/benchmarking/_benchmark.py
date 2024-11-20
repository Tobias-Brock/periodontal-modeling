import itertools
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from ..data import ProcessedDataLoader
from ._basebenchmark import BaseBenchmark, BaseExperiment


class Experiment(BaseExperiment):
    """Concrete implementation for performing ML experiments and evaluation.

    This class extends `BaseExperiment`, providing methods for evaluating machine
    learning models using holdout or cross-validation strategies. It performs
    hyperparameter tuning, final model training, and evaluation based on
    specified tuning and optimization methods.

    Inherits:
        `BaseExperiment`: Provides core functionality for validation, resampling,
            training, and tuning configurations.

    Args:
        df (pd.DataFrame): The preloaded data for the experiment.
        task (str): The task name used to determine classification type.
            Can be 'pocketclosure', 'pocketclosureinf', 'improvement', or
            'pdgrouprevaluation'.
        learner (str): Specifies the model or algorithm to evaluate.
            Includes 'xgb', 'rf', 'lr' or 'mlp'.
        criterion (str): Criterion for optimization ('f1', 'macro_f1' or 'brier_score').
        encoding (str): Encoding type for categorical features ('one_hot' or 'binary').
        tuning (Optional[str]): Tuning method to apply ('holdout' or 'cv'). Can be None.
        hpo (Optional[str]): Hyperparameter optimization method ('rs' or 'hebo').
            Can be None.
        sampling (Optional[str]): Resampling strategy to apply. Defaults to None.
            Includes None, 'upsampling', 'downsampling', and 'smote'.
        factor (Optional[float]): Resampling factor. Defaults to None.
        n_configs (int): Number of configurations for hyperparameter tuning.
            Defaults to 10.
        racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
            Defaults to None.
        n_jobs (int): Number of parallel jobs to run for evaluation.
            Defaults to 1.
        cv_folds (Optional[int]): Number of folds for cross-validation;
            Defaults to 10.
        test_seed (int): Random seed for test splitting. Defaults to 0.
        test_size (float): Proportion of data used for testing. Defaults to
            0.2.
        val_size (Optional[float]): Size of validation set in holdout tuning.
            Defaults to 0.2.
        cv_seed (Optional[int]): Random seed for cross-validation. Defaults to 0
        mlp_flag (Optional[bool]): Flag to enable MLP training with early stopping.
            Defaults to None.
        threshold_tuning (Optional[bool]): If True, performs threshold tuning for binary
            classification if the criterion is "f1". Defaults to None.
        verbose (bool): Enables verbose output if set to True.

    Attributes:
        df (pd.DataFrame): Dataset used for training and evaluation.
        task (str): Name of the task used to determine the classification type.
        learner (str): Model or algorithm name for the experiment.
        criterion (str): Criterion for performance evaluation.
        encoding (str): Encoding type for categorical features.
        sampling (str): Resampling method used in training.
        factor (float): Factor applied during resampling.
        n_configs (int): Number of configurations evaluated in hyperparameter tuning.
        racing_folds (int): Number of racing folds for random search.
        n_jobs (int): Number of parallel jobs used during processing.
        cv_folds (int): Number of cross-validation folds.
        test_seed (int): Seed for reproducible test splitting.
        test_size (float): Proportion of data reserved for testing.
        val_size (float): Size of the validation set in holdout tuning.
        cv_seed (int): Seed for reproducible cross-validation splits.
        mlp_flag (bool): Indicates if MLP training with early stopping is enabled.
        threshold_tuning (bool): Enables threshold tuning for binary classification.
        verbose (bool): Controls detailed output during the experiment.
        resampler (Resampler): Resampler instance for data handling.
        trainer (Trainer): Trainer instance for model training and evaluation.
        tuner (Tuner): Initialized tuner for hyperparameter optimization.

    Methods:
        perform_evaluation: Conducts evaluation based on the tuning method.

    Example:
        ```
        from periomod.benchmarking import Experiment
        from periomod.data import ProcessedDataLoader

        # Load a dataframe with the correct target and encoding selected
        dataloader = ProcessedDataLoader(task="pocketclosure", encoding="one_hot")
        df = dataloader.load_data(path="data/processed/processed_data.csv")
        df = dataloader.transform_data(df=df)

        experiment = Experiment(
            df=df,
            task="pocketclosure",
            learner="rf",
            criterion="f1",
            encoding="one_hot",
            tuning="cv",
            hpo="rs",
            sampling="upsample",
            factor=1.5,
            n_configs=20,
            racing_folds=5,
        )

        # Perform the evaluation based on cross-validation
        final_metrics = experiment.perform_evaluation()
        print(final_metrics)
        ```
    """

    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        learner: str,
        criterion: str,
        encoding: str,
        tuning: Optional[str],
        hpo: Optional[str],
        sampling: Optional[str] = None,
        factor: Optional[float] = None,
        n_configs: int = 10,
        racing_folds: Optional[int] = None,
        n_jobs: int = 1,
        cv_folds: Optional[int] = 10,
        test_seed: int = 0,
        test_size: float = 0.2,
        val_size: Optional[float] = 0.2,
        cv_seed: Optional[int] = 0,
        mlp_flag: Optional[bool] = None,
        threshold_tuning: Optional[bool] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the Experiment class with tuning parameters.

        Args:
            df (pd.DataFrame): The preloaded data for the experiment.
            task (str): The task name used to determine classification type.
                Can be 'pocketclosure', 'pocketclosureinf', 'improvement', or
                'pdgrouprevaluation'.
            learner (str): Specifies the model or algorithm to evaluate.
                Includes 'xgb', 'rf', 'lr' or 'mlp'.
            criterion (str): Criterion for optimization ('f1', 'macro_f1' or
                'brier_score').
            encoding (str): Encoding type for categorical features ('one_hot' or
                'binary').
            tuning (Optional[str]): Tuning method to apply ('holdout' or 'cv').
                Can be None.
            hpo (Optional[str]): Hyperparameter optimization method ('rs' or 'hebo').
                Can be None.
            sampling (Optional[str]): Resampling strategy to apply. Defaults to None.
                Includes None, 'upsampling', 'downsampling', and 'smote'.
            factor (Optional[float]): Resampling factor. Defaults to None.
            n_configs (int): Number of configurations for hyperparameter tuning.
                Defaults to 10.
            racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
                Defaults to None.
            n_jobs (int): Number of parallel jobs to run for evaluation.
                Defaults to 1.
            cv_folds (Optional[int]): Number of folds for cross-validation;
                Defaults to 10.
            test_seed (int): Random seed for test splitting. Defaults to 0.
            test_size (float): Proportion of data used for testing. Defaults to
                0.2.
            val_size (Optional[float]): Size of validation set in holdout tuning.
                Defaults to 0.2.
            cv_seed (Optional[int]): Random seed for cross-validation. Defaults to 0
            mlp_flag (Optional[bool]): Flag to enable MLP training with early stopping.
                Defaults to None.
            threshold_tuning (Optional[bool]): If True, performs threshold tuning for
                binary classification if the criterion is "f1". Defaults to None.
            verbose (bool): Enables verbose output if set to True.
        """
        super().__init__(
            df=df,
            task=task,
            learner=learner,
            criterion=criterion,
            encoding=encoding,
            tuning=tuning,
            hpo=hpo,
            sampling=sampling,
            factor=factor,
            n_configs=n_configs,
            racing_folds=racing_folds,
            n_jobs=n_jobs,
            cv_folds=cv_folds,
            test_seed=test_seed,
            test_size=test_size,
            val_size=val_size,
            cv_seed=cv_seed,
            mlp_flag=mlp_flag,
            threshold_tuning=threshold_tuning,
            verbose=verbose,
        )

    def perform_evaluation(self) -> dict:
        """Perform model evaluation and return final metrics.

        Returns:
            dict: A dictionary containing the trained model and its evaluation metrics.
        """
        train_df, _ = self.resampler.split_train_test_df(
            df=self.df, seed=self.test_seed, test_size=self.test_size
        )

        if self.tuning == "holdout":
            return self._evaluate_holdout(train_df=train_df)
        elif self.tuning == "cv":
            return self._evaluate_cv()
        else:
            raise ValueError(f"Unsupported tuning method: {self.tuning}")

    def _evaluate_holdout(self, train_df: pd.DataFrame) -> dict:
        """Perform holdout validation and return the final model metrics.

        Args:
            train_df (pd.DataFrame): train df for holdout tuning.

        Returns:
            dict: A dictionary of evaluation metrics for the final model.
        """
        train_df_h, test_df_h = self.resampler.split_train_test_df(
            df=train_df, seed=self.test_seed, test_size=self.val_size
        )
        X_train_h, y_train_h, X_val, y_val = self.resampler.split_x_y(
            train_df=train_df_h,
            test_df=test_df_h,
            sampling=self.sampling,
            factor=self.factor,
        )
        best_params, best_threshold = self.tuner.holdout(
            learner=self.learner,
            X_train=X_train_h,
            y_train=y_train_h,
            X_val=X_val,
            y_val=y_val,
        )
        final_model = (self.learner, best_params, best_threshold)

        return self._train_final_model(final_model)

    def _evaluate_cv(self) -> dict:
        """Perform cross-validation and return the final model metrics.

        Returns:
            dict: A dictionary of evaluation metrics for the final model.
        """
        outer_splits, _ = self.resampler.cv_folds(
            df=self.df,
            sampling=self.sampling,
            factor=self.factor,
            seed=self.cv_seed,
            n_folds=self.cv_folds,
        )
        best_params, best_threshold = self.tuner.cv(
            learner=self.learner,
            outer_splits=outer_splits,
            racing_folds=self.racing_folds,
        )
        final_model = (self.learner, best_params, best_threshold)

        return self._train_final_model(final_model_tuple=final_model)


class Benchmarker(BaseBenchmark):
    """Benchmarker for evaluating machine learning models with tuning strategies.

    This class provides functionality to benchmark various machine learning
    models, tuning methods, HPO techniques, and criteria over multiple
    encodings, sampling strategies, and evaluation criteria. It supports
    training and evaluation workflows for different tasks and handles
    configurations for holdout or cross-validation tuning with threshold
    optimization.

    Inherits:
        - `BaseBenchmark`: Provides common benchmarking attributes.

    Args:
        task (str): Task for evaluation ('pocketclosure', 'pocketclosureinf',
            'improvement', or 'pdgrouprevaluation'.).
        learners (List[str]): List of learners to benchmark ('xgb', 'rf', 'lr' or
            'mlp').
        tuning_methods (List[str]): Tuning methods for each learner ('holdout',
            'cv').
        hpo_methods (List[str]): HPO methods ('hebo' or 'rs').
        criteria (List[str]): List of evaluation criteria ('f1', 'macro_f1',
            'brier_score').
        encodings (List[str]): List of encodings ('one_hot' or 'target').
        sampling (Optional[List[str]]): Sampling strategies for class imbalance.
            Includes None, 'upsampling', 'downsampling', and 'smote'.
        factor (Optional[float]): Factor to apply during resampling.
        n_configs (int): Number of configurations for hyperparameter tuning.
            Defaults to 10.
        n_jobs (int): Number of parallel jobs for processing. Defaults to 1.
        cv_folds (Optional[int]): Number of folds for cross-validation.
            Defaults to 10.
        racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
            Defaults to None.
        test_seed (int): Random seed for test splitting. Defaults to 0.
        test_size (float): Proportion of data used for testing. Defaults to
            0.2.
        val_size (Optional[float]): Size of validation set in holdout tuning.
            Defaults to 0.2.
        cv_seed (Optional[int]): Random seed for cross-validation. Defaults to 0
        mlp_flag (Optional[bool]): Enables MLP training with early stopping.
            Defaults to None.
        threshold_tuning (Optional[bool]): Enables threshold tuning for binary
            classification. Defaults to None.
        verbose (bool): If True, enables detailed logging during benchmarking.
            Defaults to True.
        path (Path): Path to the directory containing processed data files.
            Defaults to Path("data/processed/processed_data.csv").

    Attributes:
        task (str): The specified task for evaluation.
        learners (List[str]): List of learners to evaluate.
        tuning_methods (List[str]): Tuning methods for model evaluation.
        hpo_methods (List[str]): HPO methods for hyperparameter tuning.
        criteria (List[str]): List of evaluation metrics.
        encodings (List[str]): Encoding types for categorical features.
        sampling (List[str]): Resampling strategies for class balancing.
        factor (float): Resampling factor for balancing.
        n_configs (int): Number of configurations for hyperparameter tuning.
        n_jobs (int): Number of parallel jobs for model training.
        cv_folds (int): Number of cross-validation folds.
        racing_folds (int): Number of racing folds for random search.
        test_seed (int): Seed for reproducible train-test splits.
        test_size (float): Size of the test split.
        val_size (float): Size of the validation split in holdout tuning.
        cv_seed (int): Seed for cross-validation splits.
        mlp_flag (bool): Indicates if MLP training with early stopping is used.
        threshold_tuning (bool): Enables threshold tuning for binary classification.
        verbose (bool): Enables detailed logging during benchmarking.
        path (Path): Directory path for processed data.
        name (str): File name for processed data.
        data_cache (dict): Cached data for different task and encoding combinations.

    Methods:
        run_all_benchmarks: Executes benchmarks for all combinations of learners,
            tuning methods, HPO, criteria, encodings, and sampling strategies,
            and returns a DataFrame summary and a dictionary of top models.

    Example:
        ```
        benchmarker = Benchmarker(
            task="pocketclosure",
            learners=["xgb", "rf"],
            tuning_methods=["holdout", "cv"],
            hpo_methods=["hebo", "rs"],
            criteria=["f1", "brier_score"],
            encodings=["one_hot", "target"],
            sampling=["upsampling", "downsampling"],
            factor=1.5,
            n_configs=20,
            n_jobs=4,
            cv_folds=5,
            test_seed=42,
            test_size=0.2,
            verbose=True,
            path="/data/processed/processed_data.csv",
        )

        # Running all benchmarks
        results_df, top_models = benchmarker.run_all_benchmarks()
        print(results_df)
        print(top_models)
        ```
    """

    def __init__(
        self,
        task: str,
        learners: List[str],
        tuning_methods: List[str],
        hpo_methods: List[str],
        criteria: List[str],
        encodings: List[str],
        sampling: Optional[List[Union[str, None]]] = None,
        factor: Optional[float] = None,
        n_configs: int = 10,
        n_jobs: int = 1,
        cv_folds: Optional[int] = 10,
        racing_folds: Optional[int] = None,
        test_seed: int = 0,
        test_size: float = 0.2,
        val_size: Optional[float] = 0.2,
        cv_seed: Optional[int] = 0,
        mlp_flag: Optional[bool] = None,
        threshold_tuning: Optional[bool] = None,
        verbose: bool = True,
        path: Path = Path("data/processed/processed_data.csv"),
    ) -> None:
        """Initialize the Experiment with different tasks, learners, etc.

        Args:
            task (str): Task for evaluation ('pocketclosure', 'pocketclosureinf',
                'improvement', or 'pdgrouprevaluation'.).
            learners (List[str]): List of learners to benchmark ('xgb', 'rf', 'lr' or
                'mlp').
            tuning_methods (List[str]): Tuning methods for each learner ('holdout',
                'cv').
            hpo_methods (List[str]): HPO methods ('hebo' or 'rs').
            criteria (List[str]): List of evaluation criteria ('f1', 'macro_f1',
                'brier_score').
            encodings (List[str]): List of encodings ('one_hot' or 'target').
            sampling (Optional[List[str]]): Sampling strategies for class imbalance.
                Includes None, 'upsampling', 'downsampling', and 'smote'.
            factor (Optional[float]): Factor to apply during resampling.
            n_configs (int): Number of configurations for hyperparameter tuning.
                Defaults to 10.
            n_jobs (int): Number of parallel jobs for processing. Defaults to 1.
            cv_folds (Optional[int]): Number of folds for cross-validation.
                Defaults to 10.
            racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
                Defaults to None.
            test_seed (int): Random seed for test splitting. Defaults to 0.
            test_size (float): Proportion of data used for testing. Defaults to
                0.2.
            val_size (Optional[float]): Size of validation set in holdout tuning.
                Defaults to 0.2.
            cv_seed (Optional[int]): Random seed for cross-validation. Defaults to 0
            mlp_flag (Optional[bool]): Enables MLP training with early stopping.
                Defaults to None.
            threshold_tuning (Optional[bool]): Enables threshold tuning for binary
                classification. Defaults to None.
            verbose (bool): If True, enables detailed logging during benchmarking.
                Defaults to True.
            path (Path): Path to the directory containing processed data files.
                Defaults to Path("data/processed/processed_data.csv").
        """
        super().__init__(
            task=task,
            learners=learners,
            tuning_methods=tuning_methods,
            hpo_methods=hpo_methods,
            criteria=criteria,
            encodings=encodings,
            sampling=sampling,
            factor=factor,
            n_configs=n_configs,
            n_jobs=n_jobs,
            cv_folds=cv_folds,
            racing_folds=racing_folds,
            test_seed=test_seed,
            test_size=test_size,
            val_size=val_size,
            cv_seed=cv_seed,
            mlp_flag=mlp_flag,
            threshold_tuning=threshold_tuning,
            verbose=verbose,
            path=path,
        )
        self.data_cache = self._load_data_for_tasks()

    def _load_data_for_tasks(self) -> dict:
        """Load and transform data for each task and encoding combination once.

        Returns:
            dict: A dictionary containing transformed data for each task-encoding pair.
        """
        data_cache = {}
        for encoding in self.encodings:
            cache_key = encoding

            if cache_key not in data_cache:
                dataloader = ProcessedDataLoader(task=self.task, encoding=encoding)
                df = dataloader.load_data(path=self.path)
                transformed_df = dataloader.transform_data(df)
                data_cache[cache_key] = transformed_df

        return data_cache

    def run_benchmarks(self) -> Tuple[pd.DataFrame, dict]:
        """Benchmark all combinations of inputs.

        Returns:
            tuple: DataFrame summarizing the benchmark results with metrics for each
                configuration and dictionary mapping model keys to models for top
                configurations per criterion.

        Raises:
            KeyError: If an unknown criterion is encountered in `metric_map`.
        """
        results = []
        learners_dict = {}
        top_models_per_criterion: Dict[
            str, List[Tuple[float, object, str, str, str, str]]
        ] = {criterion: [] for criterion in self.criteria}

        metric_map = {
            "f1": "F1 Score",
            "brier_score": (
                "Multiclass Brier Score"
                if self.task == "pdgrouprevaluation"
                else "Brier Score"
            ),
            "macro_f1": "Macro F1",
        }

        for learner, tuning, hpo, criterion, encoding, sampling in itertools.product(
            self.learners,
            self.tuning_methods,
            self.hpo_methods,
            self.criteria,
            self.encodings,
            self.sampling or ["no_sampling"],
        ):
            if sampling is None:
                self.factor = None

            if (criterion == "macro_f1" and self.task != "pdgrouprevaluation") or (
                criterion == "f1" and self.task == "pdgrouprevaluation"
            ):
                print(f"Criterion '{criterion}' and task '{self.task}' not valid.")
                continue
            if self.verbose:
                print(
                    f"\nRunning benchmark for Task: {self.task}, Learner: {learner}, "
                    f"Tuning: {tuning}, HPO: {hpo}, Criterion: {criterion}, "
                    f"Sampling: {sampling}, Factor: {self.factor}."
                )
            df = self.data_cache[(encoding)]

            exp = Experiment(
                df=df,
                task=self.task,
                learner=learner,
                criterion=criterion,
                encoding=encoding,
                tuning=tuning,
                hpo=hpo,
                sampling=sampling,
                factor=self.factor,
                n_configs=self.n_configs,
                racing_folds=self.racing_folds,
                n_jobs=self.n_jobs,
                cv_folds=self.cv_folds,
                test_seed=self.test_seed,
                test_size=self.test_size,
                val_size=self.val_size,
                cv_seed=self.cv_seed,
                mlp_flag=self.mlp_flag,
                threshold_tuning=self.threshold_tuning,
                verbose=self.verbose,
            )

            try:
                result = exp.perform_evaluation()
                metrics = result["metrics"]
                trained_model = result["model"]

                unpacked_metrics = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics.items()
                }
                results.append(
                    {
                        "Task": self.task,
                        "Learner": learner,
                        "Tuning": tuning,
                        "HPO": hpo,
                        "Criterion": criterion,
                        "Sampling": sampling,
                        "Factor": self.factor,
                        **unpacked_metrics,
                    }
                )

                metric_key = metric_map.get(criterion)
                if metric_key is None:
                    raise KeyError(f"Unknown criterion '{criterion}'")

                criterion_value = metrics[metric_key]

                current_model_data = (
                    criterion_value,
                    trained_model,
                    learner,
                    tuning,
                    hpo,
                    encoding,
                )

                if len(top_models_per_criterion[criterion]) < 4:
                    top_models_per_criterion[criterion].append(current_model_data)
                else:
                    worst_model_idx = min(
                        range(len(top_models_per_criterion[criterion])),
                        key=lambda idx: (
                            top_models_per_criterion[criterion][idx][0]
                            if criterion != "brier_score"
                            else -top_models_per_criterion[criterion][idx][0]
                        ),
                    )
                    worst_model_score = top_models_per_criterion[criterion][
                        worst_model_idx
                    ][0]
                    if (
                        criterion != "brier_score"
                        and criterion_value > worst_model_score
                    ) or (
                        criterion == "brier_score"
                        and criterion_value < worst_model_score
                    ):
                        top_models_per_criterion[criterion][
                            worst_model_idx
                        ] = current_model_data

            except Exception as e:
                error_message = str(e)
                if (
                    "Matrix not positive definite after repeatedly adding jitter"
                    in error_message
                    or "elements of the" in error_message
                    and "are NaN" in error_message
                    or "cholesky_cpu" in error_message
                ):
                    print(
                        f"Suppressed NotPSDError for {self.task}, {learner} due to"
                        f"convergence issue \n"
                    )
                else:
                    print(
                        f"Error running benchmark for {self.task}, {learner}: "
                        f"{error_message}\n"
                    )
                    traceback.print_exc()

        for criterion, models in top_models_per_criterion.items():
            sorted_models = sorted(
                models, key=lambda x: -x[0] if criterion != "brier_score" else x[0]
            )
            for idx, (score, model, learner, tuning, hpo, encoding) in enumerate(
                sorted_models
            ):
                learners_dict_key = (
                    f"{self.task}_{learner}_{tuning}_{hpo}_{criterion}_{encoding}_"
                    f"{sampling or 'no_sampling'}_factor{self.factor}_rank{idx+1}_"
                    f"score{round(score, 4)}"
                )
                learners_dict[learners_dict_key] = model

        df_results = pd.DataFrame(results)
        pd.set_option("display.max_columns", None, "display.width", 1000)

        if self.verbose:
            print(f"\nBenchmark Results Summary:\n{df_results}")

        return df_results, learners_dict
