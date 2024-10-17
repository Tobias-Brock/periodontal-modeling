import itertools
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from pamod.benchmarking import BaseBenchmark, BaseExperiment
from pamod.config import PROCESSED_BASE_DIR
from pamod.data import ProcessedDataLoader


class Experiment(BaseExperiment):
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
        n_jobs: Optional[int] = None,
        cv_folds: Optional[int] = None,
        test_seed: Optional[int] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        cv_seed: Optional[int] = None,
        mlp_flag: Optional[bool] = None,
        verbosity: bool = True,
    ) -> None:
        """Initialize the Experiment class with tuning parameters.

        Args:
            df (pd.DataFrame): The preloaded data.
            task (str): The task name used to determine classification type.
            learner (str): The machine learning model to evaluate.
            criterion (str): Criterion for optimization ('macro_f1' or 'brier_score').
            encoding (str): Encoding type ('one_hot' or 'binary')
            tuning (Optional[str]): The tuning method ('holdout' or 'cv'). Can be None.
            hpo (Optional[str]): The hyperparameter optimization method. Can be None.
            sampling (Optional[str]): Sampling strategy to use. Defaults to None.
            factor (Optional[float]): Factor for resampling. Defaults to None.
            n_configs (int): Number of configurations for hyperparameter tuning.
                Defaults to 10.
            racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
                Defaults to None.
            n_jobs (Optional[int]): Number of parallel jobs to run for evaluation.
                Defaults to None.
            cv_folds (Optional[int], optional): Number of folds for cross-validation.
                Defaults to None, in which case the class's `n_folds` will be used.
            test_seed (Optional[int], optional): Random seed for splitting.
                Defaults to None.
            test_size (Optional[float]): Size of grouped train test split.
            val_size (Optional[float]): Size of grouped train test split for holdout.
            cv_seed (int): Seed for splitting CV folds.
            mlp_flag (bool): Flag for MLP training with early stopping.
            verbosity (bool): Enables verbose output if set to True.
        """
        self.df = df
        super().__init__(
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
            verbosity=verbosity,
        )

    def perform_evaluation(self) -> dict:
        """Perform model evaluation and return final metrics.

        Returns:
            dict: A dictionary containing the trained model and its evaluation metrics.
        """
        train_df, _ = self.resampler.split_train_test_df(
            self.df, self.test_seed, self.test_size
        )

        if self.tuning == "holdout":
            return self._evaluate_holdout(train_df)
        elif self.tuning == "cv":
            return self._evaluate_cv()
        else:
            raise ValueError(f"Unsupported tuning method: {self.tuning}")

    def _train_final_model(
        self, final_model_tuple: Tuple[str, Dict, Optional[float]]
    ) -> dict:
        """Helper method to train the final model with best parameters.

        Args:
            final_model_tuple (Tuple[str, Dict, Optional[float]]): A tuple containing
                the learner name, best hyperparameters, and an optional best threshold.

        Returns:
            dict: A dictionary containing the trained model and its evaluation metrics.
        """
        return self.trainer.train_final_model(
            self.df,
            self.resampler,
            final_model_tuple,
            self.sampling,
            self.factor,
            self.n_jobs,
            self.test_seed,
            self.test_size,
            self.verbosity,
        )

    def _evaluate_holdout(self, train_df: pd.DataFrame) -> dict:
        """Perform holdout validation and return the final model metrics.

        Args:
            train_df (pd.DataFrame): train df for holdout tuning.

        Returns:
            dict: A dictionary of evaluation metrics for the final model.
        """
        train_df_h, test_df_h = self.resampler.split_train_test_df(
            train_df, self.test_seed, self.val_size
        )
        X_train_h, y_train_h, X_val, y_val = self.resampler.split_x_y(
            train_df_h, test_df_h, self.sampling, self.factor
        )
        best_params, best_threshold = self.tuner.holdout(
            self.learner,
            X_train_h,
            y_train_h,
            X_val,
            y_val,
        )
        final_model = (self.learner, best_params, best_threshold)

        return self._train_final_model(final_model)

    def _evaluate_cv(self) -> dict:
        """Perform cross-validation and return the final model metrics.

        Returns:
            dict: A dictionary of evaluation metrics for the final model.
        """
        outer_splits, _ = self.resampler.cv_folds(
            self.df, self.sampling, self.factor, self.cv_seed, self.cv_folds
        )
        best_params, best_threshold = self.tuner.cv(
            self.learner,
            outer_splits,
            self.n_configs,
            self.racing_folds,
        )
        final_model = (self.learner, best_params, best_threshold)

        return self._train_final_model(final_model)


class Benchmarker(BaseBenchmark):
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
        n_jobs: Optional[int] = None,
        cv_folds: Optional[int] = None,
        racing_folds: Optional[int] = None,
        test_seed: Optional[int] = None,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
        cv_seed: Optional[int] = None,
        mlp_flag: Optional[bool] = None,
        verbosity: bool = True,
        path: Path = PROCESSED_BASE_DIR,
        name: str = "processed_data.csv",
    ) -> None:
        """Initialize the Experiment with different tasks, learners, etc.

        Args:
            task (str): Task (e.g., 'pocketclosure', 'improve').
            learners (List[str]): List of learners to benchmark (e.g., 'xgb', etc.).
            tuning_methods (List[str]): List of tuning methods ('holdout', 'cv').
            hpo_methods (List[str]): List of HPO methods ('hebo', 'rs').
            criteria (List[str]): List of evaluation criteria ('f1', 'brier_score').
            encodings (List[str]): List of encodings ('one_hot' or 'target').
            sampling (OptionalList[str]]): Sampling strategy to use. Defaults to None.
            factor (Optional[float]): Factor for resampling. Defaults to None.
            n_configs (int): Number of configurations for hyperparameter tuning.
                Defaults to 10.
            n_jobs (Optional[int]): Number of parallel jobs to run for evaluation.
                Defaults to None.
            cv_folds (Optional[int], optional): Number of folds for cross-validation.
                Defaults to None, in which case the class's `n_folds` will be used.
            racing_folds (Optional[int]): Number of racing folds for Random Search (RS).
                Defaults to None.
            test_seed (Optional[int], optional): Random seed for splitting.
                Defaults to None.
            test_size (Optional[float]): Size of grouped train test split.
            val_size (Optional[float]): Size of grouped train test split for holdout.
            cv_seed (int): Seed for splitting CV folds.
            mlp_flag (Optional[bool]): Flag for MLP training with early stopping.
            verbosity (bool): Enables verbose output if True.
            path (str): Directory path for the processed data.
            name (str): File name for the processed data.
        """
        super().__init__(
            task,
            learners,
            tuning_methods,
            hpo_methods,
            criteria,
            encodings,
            sampling,
            factor,
            n_configs,
            n_jobs,
            cv_folds,
            racing_folds,
            test_seed,
            test_size,
            val_size,
            cv_seed,
            mlp_flag,
            verbosity,
            path,
            name,
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
                dataloader = ProcessedDataLoader(self.task, encoding)
                df = dataloader.load_data(self.path, self.name)
                transformed_df = dataloader.transform_data(df)
                data_cache[cache_key] = transformed_df

        return data_cache

    def run_all_benchmarks(self) -> Tuple[pd.DataFrame, dict]:
        """Benchmark all combinations of tasks, learners, tuning, HPO, and criteria."""
        results = []
        learners_dict = {}
        top_models_per_criterion: Dict[
            str, List[Tuple[float, object, str, str, str, str]]
        ] = {criterion: [] for criterion in self.criteria}
        metric_map = {
            "f1": "F1 Score",
            "brier_score": "Brier Score",
            "macro_f1": "Macro F1 Score",
        }

        for learner, tuning, hpo, criterion, encoding, sampling in itertools.product(
            self.learners,
            self.tuning_methods,
            self.hpo_methods,
            self.criteria,
            self.encodings,
            self.sampling or ["no_sampling"],
        ):
            if (criterion == "macro_f1" and self.task != "pdgrouprevaluation") or (
                criterion == "f1" and self.task == "pdgrouprevaluation"
            ):
                print(f"Criterion '{criterion}' and task '{self.task}' not valid.")
                continue
            if self.verbosity:
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
                verbosity=self.verbosity,
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
                print(f"Error running benchmark for {self.task}, {learner}: {e}\n")
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
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        if self.verbosity:
            print("\nBenchmark Results Summary:")
            print(df_results)

        return df_results, learners_dict


def main():
    task = "pdgrouprevaluation"
    # learners = ["xgb", "rf", "lr", "mlp"]
    learners = ["xgb"]
    tuning_methods = ["holdout"]
    hpo_methods = ["hebo"]
    criteria = ["macro_f1"]
    encoding = ["one_hot"]

    benchmarker = Benchmarker(
        task=task,
        learners=learners,
        tuning_methods=tuning_methods,
        hpo_methods=hpo_methods,
        criteria=criteria,
        encodings=encoding,
        sampling="upsampling",
        factor=2,
        n_configs=3,
        racing_folds=3,
        n_jobs=-1,
        verbosity=True,
    )

    benchmarker.run_all_benchmarks()


if __name__ == "__main__":
    main()
