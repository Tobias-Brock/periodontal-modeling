import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..data import ProcessedDataLoader
from ._basebayes import (
    BAYES_TASKS,
    BaseBayesConfig,
    BaseBayesValidator,
    HierarchicalData,
    SpatialConfig,
)
from ._models import HierarchicalLogisticModel
from ._trainer import BayesTrainer
from ._transform import HierarchicalDataTransformer


class BayesExperiment(BaseBayesValidator):
    """Runs a full Bayesian hierarchical experiment on site level outcomes.

    The experiment splits patients into fixed train, validation and test sets,
    fits the preprocessing on training patients, draws from the posterior of
    the multilevel logistic model and reports posterior predictive
    probabilities, credible intervals, discrimination and calibration metrics,
    convergence diagnostics and posterior predictive checks. The test split is
    only evaluated when explicitly requested.

    Validation and test patients are unseen, so their patient and tooth
    intercepts are drawn from the posterior population distribution rather than
    estimated, which is the posterior predictive distribution of a new patient.

    Inherits:
        - `BaseBayesValidator`: Validates classification and criterion.

    Args:
        data (pd.DataFrame): Processed dataset with one row per site.
        task (str): Task name. Can be 'pocketclosure', 'pocketclosureinf' or
            'improvement'.
        criterion (str): Evaluation criterion used for threshold selection.
            Defaults to "f1".
        spatial (Optional[SpatialConfig]): Spatial component of the model.
            Defaults to the configured component.
        include_infect (Optional[bool]): Includes the engineered infection
            features. Defaults to the configured value.
        target_sites (Optional[str]): Sites entering the likelihood. Defaults
            to the configured value.
        tooth_number_effect (Optional[bool]): Models the FDI tooth number as a
            hierarchical effect. Defaults to the configured value.
        side_effect (Optional[bool]): Models the site position as a
            hierarchical effect. Defaults to the configured value.
        intercept_sd (Optional[float]): Prior scale of the intercept. Defaults
            to the configured value.
        beta_sd (Optional[float]): Prior scale of the fixed effects. Defaults
            to the configured value.
        sigma_sd (Optional[float]): Prior scale of the random effect standard
            deviations. Defaults to the configured value.
        spatial_sigma_sd (Optional[float]): Prior scale of the spatial standard
            deviations. Defaults to the configured value.
        draws (Optional[int]): Number of posterior draws per chain. Defaults to
            the configured value.
        tune (Optional[int]): Number of tuning steps per chain. Defaults to the
            configured value.
        chains (Optional[int]): Number of chains. Defaults to the configured
            value.
        cores (Optional[int]): Number of cores used for sampling. Defaults to
            the configured value.
        target_accept (Optional[float]): Target acceptance rate of NUTS.
            Defaults to the configured value.
        sampler_seed (Optional[int]): Random state of the sampler. Defaults to
            the configured value.
        mp_ctx (Optional[str]): Multiprocessing context of parallel chains.
            Defaults to the configured value.
        hdi_prob (Optional[float]): Mass of the credible intervals. Defaults to
            the configured value.
        random_effects (Optional[str]): Treatment of group effects of unseen
            patients. Defaults to the configured value.
        max_draws (Optional[int]): Maximum number of posterior draws used for
            predictions. Defaults to 1000.
        val_size (Optional[float]): Proportion of patients in the validation
            set. Defaults to the configured value.
        test_size (Optional[float]): Proportion of patients in the test set.
            Defaults to the configured value.
        split_seed (Optional[int]): Random state of the patient level split.
            Defaults to the configured value.
        threshold_tuning (bool): Tunes the decision threshold on validation
            sites. Defaults to True.
        verbose (bool): Prints preprocessing and sampling progress. Defaults to
            True.

    Attributes:
        data (pd.DataFrame): Processed dataset used in the experiment.
        task (str): Task name of the experiment.
        classification (str): Classification type, always 'binary'.
        spatial (SpatialConfig): Spatial component of the model.
        transformer (HierarchicalDataTransformer): Split-aware preprocessing.
        model (HierarchicalLogisticModel): Model specification.
        trainer (BayesTrainer): Sampler and evaluator of the model.
        splits (Dict[str, HierarchicalData]): Design matrices per split.
        idata (Any): Posterior draws of the fitted model.

    Methods:
        prepare_data: Splits patients and builds the design matrices.
        perform_evaluation: Samples the posterior and evaluates target sites.

    Example:
        ```
        from periomod.bayes import BayesExperiment
        from periomod.data import ProcessedDataLoader

        df = ProcessedDataLoader.load_data(
            path="data/processed/processed_data.csv"
        )
        experiment = BayesExperiment(data=df, task="improvement", criterion="f1")
        result = experiment.perform_evaluation()
        print(result["metrics"])
        print(result["diagnostics"])
        ```
    """

    def __init__(
        self,
        data: pd.DataFrame,
        task: str,
        criterion: str = "f1",
        spatial: Optional[SpatialConfig] = None,
        include_infect: Optional[bool] = None,
        target_sites: Optional[str] = None,
        tooth_number_effect: Optional[bool] = None,
        side_effect: Optional[bool] = None,
        intercept_sd: Optional[float] = None,
        beta_sd: Optional[float] = None,
        sigma_sd: Optional[float] = None,
        spatial_sigma_sd: Optional[float] = None,
        draws: Optional[int] = None,
        tune: Optional[int] = None,
        chains: Optional[int] = None,
        cores: Optional[int] = None,
        target_accept: Optional[float] = None,
        sampler_seed: Optional[int] = None,
        mp_ctx: Optional[str] = None,
        hdi_prob: Optional[float] = None,
        random_effects: Optional[str] = None,
        max_draws: Optional[int] = 1000,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
        split_seed: Optional[int] = None,
        threshold_tuning: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initializes the Bayesian experiment with data, task and settings."""
        self.task = task
        super().__init__(classification="binary", criterion=criterion)
        self._validate_task()
        self.data = data
        self.spatial = self.default_spatial() if spatial is None else spatial
        self.verbose = verbose
        self.transformer = HierarchicalDataTransformer(
            task=task,
            include_infect=include_infect,
            target_sites=target_sites,
            val_size=val_size,
            test_size=test_size,
            split_seed=split_seed,
            verbose=verbose,
        )
        self.model = HierarchicalLogisticModel(
            spatial=self.spatial,
            tooth_number_effect=tooth_number_effect,
            side_effect=side_effect,
            intercept_sd=intercept_sd,
            beta_sd=beta_sd,
            sigma_sd=sigma_sd,
            spatial_sigma_sd=spatial_sigma_sd,
        )
        self.trainer = BayesTrainer(
            classification="binary",
            criterion=criterion,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            sampler_seed=sampler_seed,
            mp_ctx=mp_ctx,
            hdi_prob=hdi_prob,
            random_effects=random_effects,
            max_draws=max_draws,
            threshold_tuning=threshold_tuning,
            verbose=verbose,
        )
        self.splits: Dict[str, HierarchicalData] = {}
        self.idata: Any = None

    def _validate_task(self) -> None:
        """Validates the task against the tasks supported by the submodule.

        Raises:
            ValueError: If the task is not a supported binary task.
        """
        if self.task not in BAYES_TASKS:
            raise ValueError(
                f"Task '{self.task}' not supported by the Bayesian submodule. "
                f"Choose one of {list(BAYES_TASKS)}."
            )

    def prepare_data(self) -> Dict[str, HierarchicalData]:
        """Splits patients and builds the design matrices of all splits.

        Returns:
            Dict[str, HierarchicalData]: Design matrix and nesting structure of
                the 'train', 'val' and 'test' split.
        """
        self.splits = self.transformer.transform_splits(data=self.data)
        if self.verbose:
            print(
                f"Fitting the '{self.spatial.mode}' model with "
                f"{self.splits['train'].n_features} fixed effects."
            )
        return self.splits

    def perform_evaluation(self, evaluate_test: bool = False) -> Dict[str, Any]:
        """Samples the posterior and evaluates the target sites.

        Args:
            evaluate_test (bool): Evaluates the held-out test split in addition
                to the validation split. Only use for the final model.
                Defaults to False.

        Returns:
            Dict[str, Any]: Validation metrics, model, posterior draws,
                decision threshold, site level predictions with credible
                intervals, convergence diagnostics, sampler statistics,
                posterior predictive checks and calibration of the validation
                split. Test results are added when `evaluate_test` is set.
        """
        if not self.splits:
            self.prepare_data()

        self.idata = self.trainer.fit(model=self.model, data=self.splits["train"])
        probs = self.trainer.predict(
            model=self.model, idata=self.idata, data=self.splits["val"]
        )
        mean_probs = probs.mean(axis=0)
        score, threshold = self.trainer.tune_threshold(
            y=self.splits["val"].y, probs=mean_probs
        )

        result: Dict[str, Any] = {
            "metrics": self.trainer.evaluate(
                y=self.splits["val"].y, probs=mean_probs, threshold=threshold
            ),
            "model": self.model,
            "idata": self.idata,
            "threshold": threshold,
            "val_score": score,
            "predictions": self.trainer.summarize_predictions(
                probs=probs, data=self.splits["val"]
            ),
            "diagnostics": self.trainer.diagnostics(idata=self.idata),
            "sampler_stats": self.trainer.sampler_stats(idata=self.idata),
            "posterior_predictive_check": self.trainer.posterior_predictive_check(
                model=self.model, idata=self.idata, data=self.splits["train"]
            ),
            "calibration": self.trainer.calibration_table(
                y=self.splits["val"].y, probs=mean_probs
            ),
            "spatial": self.spatial,
        }

        if evaluate_test:
            test_probs = self.trainer.predict(
                model=self.model, idata=self.idata, data=self.splits["test"]
            )
            result["test_metrics"] = self.trainer.evaluate(
                y=self.splits["test"].y,
                probs=test_probs.mean(axis=0),
                threshold=threshold,
            )
            result["test_predictions"] = self.trainer.summarize_predictions(
                probs=test_probs, data=self.splits["test"]
            )

        return result


class BayesBenchmarker(BaseBayesConfig):
    """Benchmarks hierarchical models across tasks, criteria and model stages.

    The benchmarker runs one `BayesExperiment` per combination of task,
    criterion, spatial stage and seed, and collects validation metrics together
    with the convergence diagnostics of every run. It is intended for the
    two-stage workflow that compares the nested model against the models with
    an anatomical spatial component.

    Inherits:
        - `BaseBayesConfig`: Provides package and Bayesian configuration.

    Args:
        tasks (List[str]): Tasks to benchmark.
        criteria (List[str]): Evaluation criteria for threshold selection.
        stages (Optional[List[str]]): Names of the predefined spatial stages.
            Defaults to 'nested' and 'car_tooth'.
        spatial_configs (Optional[Dict[str, SpatialConfig]]): Custom spatial
            configurations, which take precedence over `stages`. Defaults to
            None.
        seeds (Optional[List[int]]): Random states of the sampler. Defaults to
            the configured value.
        path (Union[str, Path]): Path to the processed dataset. Defaults to
            Path("data/processed/processed_data.csv").
        verbose (bool): Prints the progress of the benchmark. Defaults to True.
        experiment_args (Optional[Dict[str, Any]]): Additional arguments passed
            to every `BayesExperiment`. Defaults to None.

    Attributes:
        tasks (List[str]): Tasks included in the benchmark.
        criteria (List[str]): Evaluation criteria included in the benchmark.
        spatial_configs (Dict[str, SpatialConfig]): Stages to compare.
        seeds (List[int]): Random states of the sampler.
        data (pd.DataFrame): Processed dataset used in the benchmark.
        verbose (bool): Controls verbosity of the benchmark.
        experiment_args (Dict[str, Any]): Additional experiment arguments.

    Methods:
        run_benchmarks: Runs every combination and collects the metrics.

    Example:
        ```
        from periomod.bayes import BayesBenchmarker

        benchmarker = BayesBenchmarker(
            tasks=["improvement"],
            criteria=["f1"],
            stages=["nested", "car_tooth"],
            path="data/processed/processed_data.csv",
        )
        results, posteriors = benchmarker.run_benchmarks()
        ```
    """

    def __init__(
        self,
        tasks: List[str],
        criteria: List[str],
        stages: Optional[List[str]] = None,
        spatial_configs: Optional[Dict[str, SpatialConfig]] = None,
        seeds: Optional[List[int]] = None,
        path: Union[str, Path] = Path("data/processed/processed_data.csv"),
        verbose: bool = True,
        experiment_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the benchmarker with tasks, criteria and stages."""
        super().__init__()
        self.tasks = tasks
        self.criteria = criteria
        self.seeds = [self.sampler_seed] if seeds is None else seeds
        self.verbose = verbose
        self.experiment_args = experiment_args or {}
        self.data = ProcessedDataLoader.load_data(path=path)

        if spatial_configs is None:
            presets = SpatialConfig.stages()
            names = stages or ["nested", "car_tooth"]
            spatial_configs = {name: presets[name] for name in names}
        self.spatial_configs = spatial_configs

    def run_benchmarks(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Runs every combination of task, criterion, stage and seed.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Validation metrics and
                convergence summaries of every combination, and the posterior
                draws of every run, keyed by configuration.
        """
        results, posteriors = [], {}

        for task, criterion, (name, spatial), seed in itertools.product(
            self.tasks, self.criteria, self.spatial_configs.items(), self.seeds
        ):
            if self.verbose:
                print(
                    f"\nRunning Bayesian benchmark for Task: {task}, "
                    f"Criterion: {criterion}, Stage: {name}, Seed: {seed}."
                )

            experiment = BayesExperiment(
                data=self.data,
                task=task,
                criterion=criterion,
                spatial=spatial,
                sampler_seed=seed,
                verbose=self.verbose,
                **self.experiment_args,
            )
            result = experiment.perform_evaluation()
            metrics = {
                metric: value
                for metric, value in result["metrics"].items()
                if metric != "Confusion Matrix"
            }
            diagnostics = result["diagnostics"]
            key = f"{task}_{criterion}_{name}_{seed}"
            posteriors[key] = result["idata"]
            results.append({
                "Task": task,
                "Criterion": criterion,
                "Stage": name,
                "Spatial Mode": spatial.mode,
                "Seed": seed,
                **metrics,
                "Max R-hat": float(diagnostics["r_hat"].max()),
                "Min ESS Bulk": float(diagnostics["ess_bulk"].min()),
                "Divergences": result["sampler_stats"]["divergences"],
            })

        return pd.DataFrame(results), posteriors
