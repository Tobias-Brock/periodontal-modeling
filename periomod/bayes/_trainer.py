from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss

from ..training import Trainer, final_metrics
from ._basebayes import BaseBayesTrainer, HierarchicalData
from ._models import HierarchicalLogisticModel, thin_draws

try:
    import arviz as az
    import pymc as pm
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "The Bayesian submodule requires 'pymc' and 'arviz'. Install the "
        "optional dependencies with 'pip install periomod[bayes]'."
    ) from error

POPULATION_VARS = (
    "alpha",
    "beta",
    "sigma_patient",
    "sigma_tooth",
    "sigma_toothnum",
    "sigma_side",
    "sigma_car_site",
    "rho_tooth",
)


class BayesTrainer(BaseBayesTrainer):
    """Samples, evaluates and diagnoses hierarchical models of site outcomes.

    The trainer draws from the posterior with NUTS, computes posterior
    predictive probabilities of the target sites with credible intervals and
    reports discrimination, calibration, convergence diagnostics and posterior
    predictive checks. Discrimination and calibration reuse the metrics of the
    tabular package, which makes the results comparable to the site level
    models of `periomod.benchmarking` and `periomod.graph`.

    Inherits:
        - `BaseBayesTrainer`: Validates classification and criterion.

    Args:
        classification (str): Type of classification. Only 'binary' is
            supported. Defaults to "binary".
        criterion (str): Evaluation criterion used for threshold selection
            (e.g., 'f1', 'brier_score'). Defaults to "f1".
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
        progressbar (Optional[bool]): Displays the sampler progress bar.
            Defaults to the configured value.
        mp_ctx (Optional[str]): Multiprocessing context used for parallel
            chains, e.g. 'forkserver' or 'spawn'. Defaults to the configured
            value, which avoids the crashes of forked chains on macOS.
        hdi_prob (Optional[float]): Mass of the reported credible intervals.
            Defaults to the configured value.
        random_effects (Optional[str]): Treatment of group effects for unseen
            patients, either 'sample' or 'zero'. Defaults to the configured
            value.
        max_draws (Optional[int]): Maximum number of posterior draws used for
            predictions. Defaults to 1000.
        threshold_tuning (bool): Tunes the decision threshold on the validation
            sites when the criterion is 'f1', 'recall' or 'specificity'.
            Defaults to True.
        verbose (bool): Prints sampling progress and diagnostics. Defaults to
            True.

    Attributes:
        draws (int): Number of posterior draws per chain.
        tune (int): Number of tuning steps per chain.
        chains (int): Number of chains.
        cores (int): Number of cores used for sampling.
        target_accept (float): Target acceptance rate of NUTS.
        sampler_seed (int): Random state of the sampler.
        mp_ctx (str): Multiprocessing context used for parallel chains.
        hdi_prob (float): Mass of the reported credible intervals.
        random_effects (str): Treatment of group effects for unseen patients.
        max_draws (int): Maximum number of posterior draws used.
        threshold_tuning (bool): Indicates whether thresholds are tuned.
        verbose (bool): Controls verbosity of the sampling process.

    Methods:
        fit: Draws from the posterior of a hierarchical model.
        predict: Computes posterior predictive probabilities of target sites.
        summarize_predictions: Site level probabilities with credible intervals.
        evaluate: Discrimination and calibration metrics of a split.
        calibration_table: Binned calibration of the predicted probabilities.
        tune_threshold: Selects the decision threshold on a split.
        diagnostics: Convergence diagnostics of the population parameters.
        sampler_stats: Divergences and acceptance statistics of the sampler.
        posterior_predictive_check: Posterior predictive checks of the fit.

    Example:
        ```
        from periomod.bayes import BayesTrainer, HierarchicalLogisticModel

        model = HierarchicalLogisticModel()
        trainer = BayesTrainer(criterion="f1", draws=1000, tune=1000, chains=4)
        idata = trainer.fit(model=model, data=splits["train"])
        probs = trainer.predict(model=model, idata=idata, data=splits["val"])
        metrics = trainer.evaluate(y=splits["val"].y, probs=probs.mean(axis=0))
        ```
    """

    def __init__(
        self,
        classification: str = "binary",
        criterion: str = "f1",
        draws: Optional[int] = None,
        tune: Optional[int] = None,
        chains: Optional[int] = None,
        cores: Optional[int] = None,
        target_accept: Optional[float] = None,
        sampler_seed: Optional[int] = None,
        progressbar: Optional[bool] = None,
        mp_ctx: Optional[str] = None,
        hdi_prob: Optional[float] = None,
        random_effects: Optional[str] = None,
        max_draws: Optional[int] = 1000,
        threshold_tuning: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initializes the trainer with sampler and evaluation settings."""
        super().__init__(classification=classification, criterion=criterion)
        self._validate_binary()
        self.draws = self.draws if draws is None else draws
        self.tune = self.tune if tune is None else tune
        self.chains = self.chains if chains is None else chains
        self.cores = self.cores if cores is None else cores
        self.target_accept = (
            self.target_accept if target_accept is None else target_accept
        )
        self.sampler_seed = self.sampler_seed if sampler_seed is None else sampler_seed
        self.progressbar = self.progressbar if progressbar is None else progressbar
        self.mp_ctx = self.mp_ctx if mp_ctx is None else mp_ctx
        self.hdi_prob = self.hdi_prob if hdi_prob is None else hdi_prob
        self.random_effects = (
            self.random_effects if random_effects is None else random_effects
        )
        self.max_draws = max_draws
        self.threshold_tuning = threshold_tuning
        self.verbose = verbose
        self._evaluator = Trainer(
            classification=self.classification,
            criterion=self.criterion,
            tuning=None,
            hpo=None,
        )

    def _validate_binary(self) -> None:
        """Validates that the classification type is supported.

        Raises:
            ValueError: If the classification type is not 'binary'.
        """
        if self.classification != "binary":
            raise ValueError(
                "The hierarchical logistic model only supports binary "
                "classification. Multiclass outcomes require an ordinal "
                "likelihood, which is not implemented."
            )

    def fit(self, model: HierarchicalLogisticModel, data: HierarchicalData):
        """Draws from the posterior of a hierarchical model.

        Args:
            model (HierarchicalLogisticModel): Model specification.
            data (HierarchicalData): Training design matrix and structure.

        Returns:
            arviz.InferenceData: Posterior draws and sampler statistics.
        """
        with model.build(data=data):
            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                target_accept=self.target_accept,
                random_seed=self.sampler_seed,
                progressbar=self.progressbar,
                mp_ctx=self.mp_ctx,
            )
        if self.verbose:
            stats = self.sampler_stats(idata=idata)
            print(
                f"Sampling finished with {stats['divergences']} divergences "
                f"and a mean acceptance rate of {stats['accept_rate']:.3f}."
            )
        return idata

    def predict(
        self,
        model: HierarchicalLogisticModel,
        idata,
        data: HierarchicalData,
        in_sample: bool = False,
    ) -> np.ndarray:
        """Computes posterior predictive probabilities of the target sites.

        Args:
            model (HierarchicalLogisticModel): Fitted model specification.
            idata (arviz.InferenceData): Posterior draws of the fitted model.
            data (HierarchicalData): Design matrix and nesting structure.
            in_sample (bool): Uses the fitted group effects if True. Defaults
                to False.

        Returns:
            np.ndarray: Posterior probabilities of shape (n_draws, n_obs).
        """
        eta = model.linear_predictor(
            idata=idata,
            data=data,
            in_sample=in_sample,
            random_effects=self.random_effects,
            max_draws=self.max_draws,
            seed=self.sampler_seed,
        )
        return 1.0 / (1.0 + np.exp(-eta))

    def summarize_predictions(
        self, probs: np.ndarray, data: HierarchicalData
    ) -> pd.DataFrame:
        """Summarizes site level probabilities with credible intervals.

        Args:
            probs (np.ndarray): Posterior probabilities of shape
                (n_draws, n_obs).
            data (HierarchicalData): Design matrix and nesting structure.

        Returns:
            pd.DataFrame: Target sites with outcome, posterior mean and the
                bounds of the highest density interval.
        """
        interval = az.hdi(probs[None, ...], hdi_prob=self.hdi_prob)
        predictions = data.keys.copy()
        predictions["y"] = data.y
        predictions["prob"] = probs.mean(axis=0)
        predictions["prob_sd"] = probs.std(axis=0)
        predictions["hdi_lower"] = interval[:, 0]
        predictions["hdi_upper"] = interval[:, 1]
        return predictions

    def tune_threshold(
        self, y: np.ndarray, probs: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Selects the decision threshold on a split.

        Args:
            y (np.ndarray): Observed outcomes of the target sites.
            probs (np.ndarray): Posterior mean probabilities.

        Returns:
            Tuple[float, Optional[float]]: Criterion score and threshold.
        """
        return self._evaluator.evaluate(
            y=y, probs=probs, threshold=self.threshold_tuning
        )

    def evaluate(
        self,
        y: np.ndarray,
        probs: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Computes discrimination and calibration metrics of a split.

        Args:
            y (np.ndarray): Observed outcomes of the target sites.
            probs (np.ndarray): Posterior mean probabilities.
            threshold (Optional[float]): Decision threshold. Defaults to None,
                which applies 0.5.

        Returns:
            Dict[str, Any]: Evaluation metrics of the target sites, extended by
                the log loss and the expected calibration error.
        """
        preds = (probs >= (threshold if threshold is not None else 0.5)).astype(int)
        metrics = final_metrics(
            classification=self.classification,
            y=y,
            preds=preds,
            probs=probs,
            threshold=threshold,
        )
        metrics["Log Loss"] = float(log_loss(y_true=y, y_pred=probs, labels=[0, 1]))
        metrics["ECE"] = self._expected_calibration_error(y=y, probs=probs)
        return metrics

    @staticmethod
    def _expected_calibration_error(
        y: np.ndarray, probs: np.ndarray, n_bins: int = 10
    ) -> float:
        """Computes the expected calibration error of predicted probabilities.

        Args:
            y (np.ndarray): Observed outcomes of the target sites.
            probs (np.ndarray): Posterior mean probabilities.
            n_bins (int): Number of probability bins. Defaults to 10.

        Returns:
            float: Weighted absolute deviation of observed from predicted risk.
        """
        bins = np.clip(np.digitize(probs, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, None)
        error = 0.0
        for index in np.unique(bins):
            mask = bins == index
            error += float(np.mean(mask) * abs(y[mask].mean() - probs[mask].mean()))
        return error

    def calibration_table(
        self, y: np.ndarray, probs: np.ndarray, n_bins: int = 10
    ) -> pd.DataFrame:
        """Computes the binned calibration of the predicted probabilities.

        Args:
            y (np.ndarray): Observed outcomes of the target sites.
            probs (np.ndarray): Posterior mean probabilities.
            n_bins (int): Number of probability bins. Defaults to 10.

        Returns:
            pd.DataFrame: Predicted and observed risk per probability bin.
        """
        observed, predicted = calibration_curve(
            y_true=y, y_prob=probs, n_bins=n_bins, strategy="quantile"
        )
        return pd.DataFrame({
            "Predicted": predicted,
            "Observed": observed,
            "Difference": observed - predicted,
        })

    def diagnostics(self, idata, var_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Computes convergence diagnostics of the population parameters.

        Args:
            idata (arviz.InferenceData): Posterior draws of the fitted model.
            var_names (Optional[List[str]]): Variables to summarize. Defaults
                to the population level parameters of the model.

        Returns:
            pd.DataFrame: Posterior summary with R-hat and effective sample
                sizes.
        """
        if var_names is None:
            var_names = [name for name in POPULATION_VARS if name in idata.posterior]
        summary = az.summary(
            idata, var_names=var_names, hdi_prob=self.hdi_prob, round_to=None
        )
        if self.verbose:
            worst = summary["r_hat"].max()
            smallest = summary["ess_bulk"].min()
            print(f"Convergence: max R-hat {worst:.4f}, min bulk ESS {smallest:.0f}.")
        return summary

    @staticmethod
    def sampler_stats(idata) -> Dict[str, float]:
        """Collects divergences and acceptance statistics of the sampler.

        Args:
            idata (arviz.InferenceData): Posterior draws of the fitted model.

        Returns:
            Dict[str, float]: Number of divergences, mean acceptance rate and
                maximum tree depth reached during sampling.
        """
        stats = idata.sample_stats
        return {
            "divergences": float(stats["diverging"].to_numpy().sum()),
            "accept_rate": float(stats["acceptance_rate"].to_numpy().mean()),
            "max_treedepth": float(stats["tree_depth"].to_numpy().max()),
        }

    def posterior_predictive_check(
        self,
        model: HierarchicalLogisticModel,
        idata,
        data: HierarchicalData,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Compares observed and replicated statistics of the training data.

        Replicated datasets are drawn from the in-sample posterior predictive
        distribution. The statistics target the nesting structure, so that a
        model with poorly calibrated random effects is visible as a mismatch of
        the dispersion of patient or tooth level outcome rates.

        Args:
            model (HierarchicalLogisticModel): Fitted model specification.
            idata (arviz.InferenceData): Posterior draws of the fitted model.
            data (HierarchicalData): Training design matrix and structure.
            seed (int): Random state of the replicated outcomes. Defaults to 0.

        Returns:
            pd.DataFrame: Observed statistic, mean and interval of the
                replicated statistic and the posterior predictive p-value.
        """
        probs = self.predict(model=model, idata=idata, data=data, in_sample=True)
        keep = thin_draws(n_draws=probs.shape[0], max_draws=200)
        probs = probs[keep]
        rng = np.random.default_rng(seed=seed)
        replicated = rng.binomial(n=1, p=probs).astype(float)

        statistics = {
            "Mean outcome": lambda values: values.mean(axis=-1),
            "SD of patient rates": lambda values: self._group_rates(
                values=values, index=data.patient_idx, n_groups=data.n_patients
            ).std(axis=-1),
            "SD of tooth rates": lambda values: self._group_rates(
                values=values, index=data.tooth_idx, n_groups=data.n_teeth
            ).std(axis=-1),
        }

        rows = []
        for name, statistic in statistics.items():
            observed = float(statistic(data.y[None, :])[0])
            replicates = np.asarray(statistic(replicated))
            rows.append({
                "Statistic": name,
                "Observed": observed,
                "Replicated Mean": float(replicates.mean()),
                "Replicated Lower": float(np.quantile(replicates, 0.025)),
                "Replicated Upper": float(np.quantile(replicates, 0.975)),
                "Bayesian p-value": float(np.mean(replicates >= observed)),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _group_rates(
        values: np.ndarray, index: np.ndarray, n_groups: int
    ) -> np.ndarray:
        """Computes group level outcome rates of observed or replicated data.

        Args:
            values (np.ndarray): Outcomes of shape (n_draws, n_obs).
            index (np.ndarray): Group index of every observation.
            n_groups (int): Number of groups.

        Returns:
            np.ndarray: Group rates of shape (n_draws, n_groups).
        """
        counts = np.bincount(index, minlength=n_groups).astype(float)
        sums = np.stack([
            np.bincount(index, weights=row, minlength=n_groups) for row in values
        ])
        return sums / counts
