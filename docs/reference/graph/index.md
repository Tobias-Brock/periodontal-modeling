# periomod.graph Overview

The `periomod.graph` module extends the site level analysis of `periomod` to patient-level heterogeneous graphs. Every patient is represented by a single graph with patient, tooth and site nodes, connected by biologically predefined relations, and a heterogeneous GraphSAGE model predicts the outcome of the target sites.

## Available Components

| Component              | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| [BaseGraphConfig](basegraphconfig.md)         | Base class providing the graph and GNN configuration.       |
| [BaseGraphValidator](basegraphvalidator.md)   | Base class validating classification and criterion.         |
| [BaseGraphTransformer](basegraphtransformer.md) | Base class for split-aware preprocessing.                 |
| [BaseGraphBuilder](basegraphbuilder.md)       | Base class for the construction of patient graphs.          |
| [BaseGraphTrainer](basegraphtrainer.md)       | Base class for training graph neural networks.              |
| [EdgeConfig](edgeconfig.md)                   | Switches for the relations of the patient graph.            |
| [GraphSplit](graphsplit.md)                   | Patient identifiers of the fixed train/val/test split.      |
| [GraphDataTransformer](graphtransformer.md)   | Split-aware imputation, encoding and scaling.               |
| [PatientGraphBuilder](graphbuilder.md)        | Builder of the heterogeneous patient graphs.                |
| [HeteroSitePredictor](sitepredictor.md)       | Heterogeneous GraphSAGE model for site level predictions.   |
| [GraphTrainer](graphtrainer.md)               | Trainer with early stopping on the validation split.        |
| [GraphExperiment](graphexperiment.md)         | End-to-end graph learning experiment.                       |
| [GraphBenchmarker](graphbenchmarker.md)       | Benchmark across tasks, criteria and edge ablations.        |
| [build_model](build_model.md)                 | Instantiates a model matching a patient graph.              |
| [graph_dimensions](graph_dimensions.md)       | Input dimensions per node level of a patient graph.         |
| [graph_loader](graph_loader.md)               | Loader over batches of complete patient graphs.             |
