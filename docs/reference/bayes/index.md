# periomod.bayes Overview

The `periomod.bayes` module fits Bayesian hierarchical models to site level treatment outcomes. Sites are nested within teeth and teeth within patients, which the multilevel logistic model represents with a patient level and a tooth-within-patient random intercept. A second modeling stage adds an anatomical spatial component, using the same periodontal adjacency as the graph submodule.

## Available Components

| Component              | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| [BaseBayesConfig](basebayesconfig.md)              | Base class providing the Bayesian configuration.       |
| [BaseBayesValidator](basebayesvalidator.md)        | Base class validating classification and criterion.    |
| [BaseHierarchicalTransformer](basetransformer.md)  | Base class for split-aware preprocessing.              |
| [BaseHierarchicalModel](basemodel.md)              | Base class for hierarchical models.                    |
| [BaseBayesTrainer](basetrainer.md)                 | Base class for sampling and evaluation.                |
| [HierarchicalData](hierarchicaldata.md)            | Design matrix and nesting structure of a split.        |
| [PatientSplit](patientsplit.md)                    | Patient identifiers of the fixed split.                |
| [SpatialConfig](spatialconfig.md)                  | Configuration of the anatomical spatial component.     |
| [HierarchicalDataTransformer](transformer.md)      | Split-aware imputation, dummy coding and scaling.      |
| [SpatialAdjacency](spatialadjacency.md)            | Anatomical adjacency of teeth and sites.               |
| [HierarchicalLogisticModel](model.md)              | Multilevel logistic model of site outcomes.            |
| [BayesTrainer](trainer.md)                         | Sampling, prediction, diagnostics and checks.          |
| [BayesExperiment](experiment.md)                   | End-to-end Bayesian experiment.                        |
| [BayesBenchmarker](benchmarker.md)                 | Benchmark across tasks, criteria and model stages.     |
| [group_sizes](group_sizes.md)                      | Number of units per group.                             |
| [thin_draws](thin_draws.md)                        | Slice that thins the posterior draws.                  |
| [load_bayes_config](load_bayes_config.md)          | Composes the Hydra configuration of the submodule.     |
