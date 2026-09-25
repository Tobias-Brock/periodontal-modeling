from typing import Dict, Optional

import numpy as np

from ..base import all_teeth
from ._basebayes import BaseHierarchicalModel, HierarchicalData, SpatialConfig
from ._spatial import SpatialAdjacency, group_sizes

try:
    import pymc as pm
    import pytensor.tensor as pt
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "The Bayesian submodule requires 'pymc'. Install the optional "
        "dependencies with 'pip install periomod[bayes]'."
    ) from error


def thin_draws(n_draws: int, max_draws: Optional[int]) -> slice:
    """Determines a slice that thins the posterior draws.

    Posterior predictive quantities are evaluated for every draw and every
    site, which becomes memory intensive for large datasets. Thinning keeps a
    regularly spaced subset of the draws.

    Args:
        n_draws (int): Number of available posterior draws.
        max_draws (Optional[int]): Maximum number of draws to keep. Defaults to
            None, which keeps all draws.

    Returns:
        slice: Slice selecting the retained draws.
    """
    if max_draws is None or n_draws <= max_draws:
        return slice(None)
    return slice(None, None, int(np.ceil(n_draws / max_draws)))


class HierarchicalLogisticModel(BaseHierarchicalModel):
    r"""Multilevel logistic model of site level treatment outcomes.

    Sites are nested within teeth and teeth within patients. The linear
    predictor of site $s$ of tooth $t$ of patient $i$ is

    $$
    \mathrm{logit}\, P(Y_{its} = 1)
    = \alpha + X_{its}\beta + u_i + v_{it} + a_{\tau(t)} + c_{\sigma(s)}
    + \phi^{\text{tooth}}_{it} + \phi^{\text{site}}_{its},
    $$

    with a patient level random intercept $u_i$, a tooth-within-patient random
    intercept $v_{it}$, an effect $a$ of the FDI tooth number and an effect $c$
    of the position of a site around its tooth. All random effects use a
    non-centered parameterization, e.g. $u_i = \sigma_u z_i$ with
    $z_i \sim \mathcal{N}(0, 1)$, which avoids the funnel geometry of the
    centered form.

    The spatial terms $\phi$ are optional and added by the second modeling
    stage. They are Gaussian Markov random fields over the anatomical adjacency
    of teeth and sites, with precision $I + L$, where $L$ is the graph
    Laplacian of the adjacency. This is an intrinsic CAR penalty shrunk towards
    zero, which keeps the prior proper without a hard sum-to-zero constraint. A
    soft sum-to-zero constraint per patient separates the field from the
    patient intercept.

    On the tooth level the structured field replaces part of the variance of
    the tooth intercept, $v_{it} = \sigma_v (\sqrt{1 - \rho}\, z_{it} +
    \sqrt{\rho}\, \phi_{it})$, so that a single variance is split between an
    unstructured and a spatially structured component by the mixing weight
    $\rho$. This avoids the weak identification of two competing variances. On
    the site level the field is purely structured, since an unstructured site
    effect would be an observation level effect that Bernoulli data cannot
    identify.

    Priors are weakly informative on the logit scale. Continuous predictors are
    standardized, so a prior scale of one on $\beta$ is wide but regularizing.

    Inherits:
        - `BaseHierarchicalModel`: Provides configuration and abstract methods.

    Args:
        spatial (Optional[SpatialConfig]): Spatial component of the model.
            Defaults to the configured component.
        tooth_number_effect (Optional[bool]): Models the FDI tooth number as a
            hierarchical effect over the 32 positions. Defaults to the
            configured value of `direct_effects`.
        side_effect (Optional[bool]): Models the site position as a
            hierarchical effect over the six sides. Defaults to the configured
            value of `direct_effects`.
        intercept_sd (Optional[float]): Prior scale of the intercept. Defaults
            to the configured value.
        beta_sd (Optional[float]): Prior scale of the fixed effects. Defaults
            to the configured value.
        sigma_sd (Optional[float]): Prior scale of the random effect standard
            deviations. Defaults to the configured value.
        spatial_sigma_sd (Optional[float]): Prior scale of the spatial standard
            deviations. Defaults to the configured value.
        zero_sum_sd (Optional[float]): Scale of the soft sum-to-zero constraint
            of the spatial fields. Defaults to the configured value.

    Attributes:
        spatial (SpatialConfig): Spatial component of the model.
        tooth_number_effect (bool): Indicates the FDI tooth number effect.
        side_effect (bool): Indicates the site position effect.
        adjacency (SpatialAdjacency): Builder of the anatomical adjacency.

    Methods:
        build: Builds the PyMC model of a split.
        linear_predictor: Computes posterior draws of the linear predictor.

    Example:
        ```
        from periomod.bayes import HierarchicalLogisticModel, SpatialConfig

        model = HierarchicalLogisticModel()
        pymc_model = model.build(data=splits["train"])

        # second stage with a CAR prior on tooth effects
        spatial_model = HierarchicalLogisticModel(
            spatial=SpatialConfig(mode="tooth", name="car_tooth")
        )
        ```
    """

    def __init__(
        self,
        spatial: Optional[SpatialConfig] = None,
        tooth_number_effect: Optional[bool] = None,
        side_effect: Optional[bool] = None,
        intercept_sd: Optional[float] = None,
        beta_sd: Optional[float] = None,
        sigma_sd: Optional[float] = None,
        spatial_sigma_sd: Optional[float] = None,
        zero_sum_sd: Optional[float] = None,
    ) -> None:
        """Initializes the hierarchical logistic model with its priors."""
        super().__init__(spatial=spatial)
        self.tooth_number_effect = (
            self.direct_effects["tooth"]
            if tooth_number_effect is None
            else tooth_number_effect
        )
        self.side_effect = (
            self.direct_effects["side"] if side_effect is None else side_effect
        )
        self.intercept_sd = self.intercept_sd if intercept_sd is None else intercept_sd
        self.beta_sd = self.beta_sd if beta_sd is None else beta_sd
        self.sigma_sd = self.sigma_sd if sigma_sd is None else sigma_sd
        self.spatial_sigma_sd = (
            self.spatial_sigma_sd if spatial_sigma_sd is None else spatial_sigma_sd
        )
        self.zero_sum_sd = self.zero_sum_sd if zero_sum_sd is None else zero_sum_sd
        self.adjacency = SpatialAdjacency()

    def _coords(self, data: HierarchicalData) -> Dict[str, list]:
        """Builds the coordinates of the PyMC model.

        Args:
            data (HierarchicalData): Design matrix and nesting structure.

        Returns:
            Dict[str, list]: Coordinates of the model dimensions.
        """
        return {
            "feature": list(data.feature_names),
            "patient": list(range(data.n_patients)),
            "tooth": list(range(data.n_teeth)),
            "toothnum": list(all_teeth),
            "side": list(range(len(self.side_ring))),
            "obs": list(range(data.n_obs)),
        }

    def _structured_field(
        self,
        name: str,
        node1: np.ndarray,
        node2: np.ndarray,
        group: np.ndarray,
        n_groups: int,
        dims: str,
    ):
        """Adds a Gaussian Markov random field over an adjacency structure.

        The field has precision $I + L$, with $L$ the graph Laplacian of the
        adjacency, and a soft sum-to-zero constraint within every patient.

        Args:
            name (str): Name of the field.
            node1 (np.ndarray): First node of every edge.
            node2 (np.ndarray): Second node of every edge.
            group (np.ndarray): Patient index of every node.
            n_groups (int): Number of patients.
            dims (str): Dimension name of the field.

        Returns:
            TensorVariable: Standardized field of shape (n_nodes,).
        """
        phi = pm.Normal(f"phi_{name}", mu=0.0, sigma=1.0, dims=dims)
        pm.Potential(f"car_{name}", -0.5 * pt.sum(pt.sqr(phi[node1] - phi[node2])))
        sums = pt.inc_subtensor(pt.zeros(n_groups)[group], phi)
        pm.Potential(
            f"zerosum_{name}",
            pm.logp(
                pm.Normal.dist(
                    mu=0.0, sigma=self.zero_sum_sd * group_sizes(group, n_groups)
                ),
                sums,
            ).sum(),
        )
        return phi

    def build(self, data: HierarchicalData) -> pm.Model:
        """Builds the PyMC model of a split.

        Args:
            data (HierarchicalData): Design matrix and nesting structure of the
                training patients.

        Returns:
            pm.Model: Multilevel logistic model of the target sites.
        """
        with pm.Model(coords=self._coords(data=data)) as model:
            alpha = pm.Normal("alpha", mu=0.0, sigma=self.intercept_sd)
            beta = pm.Normal("beta", mu=0.0, sigma=self.beta_sd, dims="feature")
            eta = alpha + pm.math.dot(data.X, beta)

            sigma_patient = pm.HalfNormal("sigma_patient", sigma=self.sigma_sd)
            z_patient = pm.Normal("z_patient", mu=0.0, sigma=1.0, dims="patient")
            eta = eta + (sigma_patient * z_patient)[data.patient_idx]

            sigma_tooth = pm.HalfNormal("sigma_tooth", sigma=self.sigma_sd)
            z_tooth = pm.Normal("z_tooth", mu=0.0, sigma=1.0, dims="tooth")
            tooth_effect = sigma_tooth * z_tooth

            if self.spatial.on_teeth:
                node1, node2 = self.adjacency.tooth_edges(
                    data=data, spatial=self.spatial
                )
                if node1.size:
                    phi = self._structured_field(
                        name="tooth",
                        node1=node1,
                        node2=node2,
                        group=data.tooth_patient,
                        n_groups=data.n_patients,
                        dims="tooth",
                    )
                    rho = pm.Beta("rho_tooth", alpha=1.0, beta=1.0)
                    tooth_effect = sigma_tooth * (
                        pt.sqrt(1.0 - rho) * z_tooth + pt.sqrt(rho) * phi
                    )

            eta = eta + tooth_effect[data.tooth_idx]

            if self.tooth_number_effect:
                sigma_toothnum = pm.HalfNormal("sigma_toothnum", sigma=self.sigma_sd)
                z_toothnum = pm.Normal("z_toothnum", mu=0.0, sigma=1.0, dims="toothnum")
                eta = (
                    eta
                    + (sigma_toothnum * z_toothnum)[data.toothnum_idx[data.tooth_idx]]
                )

            if self.side_effect:
                sigma_side = pm.HalfNormal("sigma_side", sigma=self.sigma_sd)
                z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, dims="side")
                eta = eta + (sigma_side * z_side)[data.side_idx]

            if self.spatial.on_sites:
                node1, node2 = self.adjacency.site_edges(
                    data=data, spatial=self.spatial
                )
                if node1.size:
                    sigma_car_site = pm.HalfNormal(
                        "sigma_car_site", sigma=self.spatial_sigma_sd
                    )
                    eta = eta + sigma_car_site * self._structured_field(
                        name="site",
                        node1=node1,
                        node2=node2,
                        group=data.patient_idx,
                        n_groups=data.n_patients,
                        dims="obs",
                    )

            pm.Bernoulli("y_obs", logit_p=eta, observed=data.y, dims="obs")

        return model

    def linear_predictor(
        self,
        idata,
        data: HierarchicalData,
        in_sample: bool = False,
        random_effects: str = "sample",
        max_draws: Optional[int] = 1000,
        seed: int = 0,
    ) -> np.ndarray:
        """Computes the posterior draws of the linear predictor.

        For the training patients the fitted group effects are used. For unseen
        patients the patient and tooth effects are either drawn from their
        posterior population distribution or set to zero, while the effects of
        the FDI tooth number and of the site position are population level
        effects and therefore carried over. The spatial fields are patient
        specific latent surfaces and cannot be extrapolated, so they enter
        out-of-sample predictions with their prior mean of zero, and the tooth
        effect of an unseen patient carries only its unstructured share of the
        variance.

        Args:
            idata (arviz.InferenceData): Posterior draws of the fitted model.
            data (HierarchicalData): Design matrix and nesting structure.
            in_sample (bool): Uses the fitted group effects if True. Defaults
                to False.
            random_effects (str): Treatment of the group effects for unseen
                patients. Choose 'sample' to draw them from the posterior
                population distribution or 'zero' for the population mean.
                Defaults to "sample".
            max_draws (Optional[int]): Maximum number of posterior draws used.
                Defaults to 1000.
            seed (int): Random state of the drawn group effects. Defaults to 0.

        Returns:
            np.ndarray: Linear predictor of shape (n_draws, n_obs).

        Raises:
            ValueError: If `random_effects` is not 'sample' or 'zero', or if
                in-sample predictions are requested for another split.
        """
        if random_effects not in ("sample", "zero"):
            raise ValueError(
                f"{random_effects} is an invalid treatment of random effects. "
                "Choose 'sample' or 'zero'."
            )
        posterior = idata.posterior.stack(sample=("chain", "draw"))
        if in_sample and posterior.sizes["patient"] != data.n_patients:
            raise ValueError(
                "In-sample predictions require the split the model was fitted "
                "on. Use in_sample=False for validation and test patients."
            )
        keep = thin_draws(n_draws=posterior.sizes["sample"], max_draws=max_draws)
        rng = np.random.default_rng(seed=seed)

        def draws(name: str) -> np.ndarray:
            """Extracts the thinned posterior draws of a variable.

            Args:
                name (str): Name of the model variable.

            Returns:
                np.ndarray: Draws of shape (n_draws,) or (n_draws, n_units).
            """
            values = posterior[name].to_numpy()
            values = values.T if values.ndim > 1 else values
            return values[keep]

        alpha = draws("alpha")
        beta = draws("beta")
        eta = alpha[:, None] + beta @ data.X.T
        n_draws = eta.shape[0]

        structured = "rho_tooth" in posterior
        rho = draws("rho_tooth") if structured else None

        for name, index, n_units in (
            ("patient", data.patient_idx, data.n_patients),
            ("tooth", data.tooth_idx, data.n_teeth),
        ):
            sigma = draws(f"sigma_{name}")
            if in_sample:
                effect = draws(f"z_{name}") * sigma[:, None]
                if name == "tooth" and structured:
                    effect = sigma[:, None] * (
                        np.sqrt(1.0 - rho)[:, None] * draws("z_tooth")
                        + np.sqrt(rho)[:, None] * draws("phi_tooth")
                    )
            elif random_effects == "sample":
                scale = sigma
                if name == "tooth" and structured:
                    scale = sigma * np.sqrt(1.0 - rho)
                effect = rng.standard_normal(size=(n_draws, n_units)) * scale[:, None]
            else:
                continue
            eta += effect[:, index]

        if self.tooth_number_effect:
            effect = draws("z_toothnum") * draws("sigma_toothnum")[:, None]
            eta += effect[:, data.toothnum_idx[data.tooth_idx]]

        if self.side_effect:
            effect = draws("z_side") * draws("sigma_side")[:, None]
            eta += effect[:, data.side_idx]

        if in_sample and "phi_site" in posterior:
            eta += draws("phi_site") * draws("sigma_car_site")[:, None]

        return eta
