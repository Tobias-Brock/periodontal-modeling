import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..data import ProcessedDataLoader
from ._basegraph import BaseGraphConfig, BaseGraphValidator, EdgeConfig
from ._builder import PatientGraphBuilder, graph_loader
from ._models import build_model
from ._trainer import GraphTrainer
from ._transform import GraphDataTransformer

try:
    from torch_geometric.loader import DataLoader
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "The graph submodule requires 'torch_geometric'. Install the optional "
        "dependencies with 'pip install periomod[gnn]'."
    ) from error


class GraphExperiment(BaseGraphValidator):
    """Runs a full graph learning experiment on patient graphs.

    The experiment splits patients into fixed train, validation and test sets,
    fits the preprocessing on training patients, builds one heterogeneous graph
    per patient and trains a site level GNN with early stopping on the
    validation split. The test split is only evaluated when explicitly
    requested, so that it stays untouched during model development.

    Inherits:
        - `BaseGraphValidator`: Validates classification and criterion.

    Args:
        data (pd.DataFrame): Processed dataset with one row per site.
        task (str): Task name. Can be 'pocketclosure', 'pocketclosureinf',
            'improvement' or 'pdgrouprevaluation'.
        criterion (str): Evaluation criterion for model selection. Options are
            'f1' and 'macro_f1' for F1 score and 'brier_score' for Brier Score.
            Defaults to "f1".
        edges (Optional[EdgeConfig]): Relations included in the patient graphs.
            Defaults to the configured relations.
        include_infect (Optional[bool]): Includes the engineered infection
            features. Defaults to the configured value.
        target_sites (Optional[str]): Sites entering loss and evaluation.
            Defaults to the configured value.
        hidden_dim (Optional[int]): Hidden dimension of the model. Defaults to
            the configured value.
        num_layers (Optional[int]): Number of message passing stages. Defaults
            to the configured value.
        dropout (Optional[float]): Dropout rate. Defaults to the configured
            value.
        lr (Optional[float]): Learning rate. Defaults to the configured value.
        weight_decay (Optional[float]): Weight decay of the optimizer. Defaults
            to the configured value.
        batch_size (Optional[int]): Number of patient graphs per batch.
            Defaults to the configured value.
        epochs (Optional[int]): Maximum number of training epochs. Defaults to
            the configured value.
        patience (Optional[int]): Number of epochs without improvement before
            early stopping. Defaults to the configured value.
        pos_weight (Optional[float]): Weight of the positive class in binary
            classification. Defaults to None.
        val_size (Optional[float]): Proportion of patients in the validation
            set. Defaults to the configured value.
        test_size (Optional[float]): Proportion of patients in the test set.
            Defaults to the configured value.
        split_seed (Optional[int]): Random state of the patient level split.
            Defaults to the configured value.
        gnn_state (Optional[int]): Random state of the model initialization.
            Defaults to the configured value.
        threshold_tuning (bool): Tunes the decision threshold on validation
            sites. Defaults to True.
        device (Optional[str]): Compute device. Defaults to the configured
            value.
        verbose (bool): Prints preprocessing and training progress. Defaults to
            True.

    Attributes:
        data (pd.DataFrame): Processed dataset used in the experiment.
        task (str): Task name of the experiment.
        classification (str): Classification type derived from the task.
        edges (EdgeConfig): Relations included in the patient graphs.
        transformer (GraphDataTransformer): Split-aware preprocessing.
        builder (PatientGraphBuilder): Builder of the patient graphs.
        trainer (GraphTrainer): Trainer of the site level GNN.
        graphs (Dict[str, list]): Patient graphs per split.
        loaders (Dict[str, DataLoader]): Loaders per split.

    Methods:
        prepare_data: Splits patients, transforms the data and builds graphs.
        perform_evaluation: Trains the model and evaluates the target sites.

    Example:
        ```
        from periomod.data import ProcessedDataLoader
        from periomod.graph import EdgeConfig, GraphExperiment

        df = ProcessedDataLoader.load_data(
            path="data/processed/processed_data.csv"
        )
        experiment = GraphExperiment(
            data=df, task="improvement", criterion="f1", edges=EdgeConfig()
        )
        result = experiment.perform_evaluation()
        print(result["metrics"])
        ```
    """

    def __init__(
        self,
        data: pd.DataFrame,
        task: str,
        criterion: str = "f1",
        edges: Optional[EdgeConfig] = None,
        include_infect: Optional[bool] = None,
        target_sites: Optional[str] = None,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Optional[float] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
        pos_weight: Optional[float] = None,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
        split_seed: Optional[int] = None,
        gnn_state: Optional[int] = None,
        threshold_tuning: bool = True,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Initializes the graph experiment with data, task and settings."""
        self.task = task
        super().__init__(
            classification=self._determine_classification(), criterion=criterion
        )
        self.data = data
        self.edges = self.default_edges() if edges is None else edges
        self.hidden_dim = self.hidden_dim if hidden_dim is None else hidden_dim
        self.num_layers = self.num_layers if num_layers is None else num_layers
        self.dropout = self.dropout if dropout is None else dropout
        self.batch_size = self.batch_size if batch_size is None else batch_size
        self.gnn_state = self.gnn_state if gnn_state is None else gnn_state
        self.verbose = verbose
        self.transformer = GraphDataTransformer(
            task=task,
            include_infect=include_infect,
            target_sites=target_sites,
            val_size=val_size,
            test_size=test_size,
            split_seed=split_seed,
            verbose=verbose,
        )
        self.trainer = GraphTrainer(
            classification=self.classification,
            criterion=criterion,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            pos_weight=pos_weight,
            threshold_tuning=threshold_tuning,
            device=device,
            verbose=verbose,
        )
        self.builder: Optional[PatientGraphBuilder] = None
        self.graphs: Dict[str, list] = {}
        self.loaders: Dict[str, DataLoader] = {}

    def _determine_classification(self) -> str:
        """Determines the classification type based on the task name.

        Returns:
            str: Classification type, either 'binary' or 'multiclass'.

        Raises:
            ValueError: If the task is unknown.
        """
        if self.task in ["pocketclosure", "pocketclosureinf", "improvement"]:
            return "binary"
        if self.task == "pdgrouprevaluation":
            return "multiclass"
        raise ValueError(
            f"Task '{self.task}' is unknown. Unable to determine classification."
        )

    def prepare_data(self) -> Dict[str, DataLoader]:
        """Splits patients, transforms the data and builds the patient graphs.

        Returns:
            Dict[str, DataLoader]: Loaders of the 'train', 'val' and 'test'
                split, yielding batches of complete patient graphs.
        """
        splits = self.transformer.transform_splits(data=self.data)
        self.builder = PatientGraphBuilder(
            feature_names=self.transformer.feature_names, edges=self.edges
        )
        self.graphs = {
            name: self.builder.build(data=split) for name, split in splits.items()
        }
        self.loaders = {
            name: graph_loader(
                graphs=graphs, batch_size=self.batch_size, shuffle=name == "train"
            )
            for name, graphs in self.graphs.items()
        }
        if self.verbose:
            print(
                f"Built patient graphs with relations "
                f"{self.edges.active_relations()} in '{self.edges.mode}' mode."
            )
        return self.loaders

    def perform_evaluation(self, evaluate_test: bool = False) -> Dict[str, Any]:
        """Trains the site level GNN and evaluates the target sites.

        Args:
            evaluate_test (bool): Evaluates the held-out test split in addition
                to the validation split. Only use for the final model.
                Defaults to False.

        Returns:
            Dict[str, Any]: Validation metrics, trained model, decision
                threshold, validation score and site level predictions. Test
                metrics and predictions are added when `evaluate_test` is set.
        """
        if not self.loaders:
            self.prepare_data()

        model = build_model(
            graph=self.graphs["train"][0],
            classification=self.classification,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            conv_aggr=self.conv_aggr,
            relation_aggr=self.relation_aggr,
            layer_norm=self.layer_norm,
            seed=self.gnn_state,
        )
        score, model, threshold = self.trainer.train(
            model=model,
            train_loader=self.loaders["train"],
            val_loader=self.loaders["val"],
        )

        result: Dict[str, Any] = {
            "metrics": self.trainer.evaluate_model(
                model=model, loader=self.loaders["val"], threshold=threshold
            ),
            "model": model,
            "threshold": threshold,
            "val_score": score,
            "predictions": self._site_predictions(model=model, split="val"),
            "edges": self.edges,
        }

        if evaluate_test:
            result["test_metrics"] = self.trainer.evaluate_model(
                model=model, loader=self.loaders["test"], threshold=threshold
            )
            result["test_predictions"] = self._site_predictions(
                model=model, split="test"
            )

        return result

    def _site_predictions(self, model: Any, split: str) -> pd.DataFrame:
        """Collects the site level predictions of a split.

        Args:
            model (Any): Trained heterogeneous GNN.
            split (str): Name of the split, either 'train', 'val' or 'test'.

        Returns:
            pd.DataFrame: Target sites with outcome and predicted probability.
        """
        outcomes, probs, identifiers = self.trainer.predict(
            model=model, loader=self.loaders[split]
        )
        predictions = identifiers.copy()
        predictions["y"] = outcomes
        if self.classification == "binary":
            predictions["prob"] = probs
        else:
            for class_index in range(probs.shape[1]):
                predictions[f"prob_{class_index}"] = probs[:, class_index]
        return predictions


class GraphBenchmarker(BaseGraphConfig):
    """Benchmarks site level GNNs across tasks, criteria and edge ablations.

    The benchmarker runs one `GraphExperiment` per combination of task,
    criterion, edge configuration and model seed, and collects the validation
    metrics in a single DataFrame. It is intended for the edge ablations that
    compare true anatomical relations against removed or randomized ones.

    Inherits:
        - `BaseGraphConfig`: Provides package and graph configuration.

    Args:
        tasks (List[str]): Tasks to benchmark.
        criteria (List[str]): Evaluation criteria for model selection.
        ablations (Optional[List[str]]): Names of the predefined edge
            configurations. Defaults to 'full', 'no_neighbors' and
            'random_neighbors'.
        edge_configs (Optional[Dict[str, EdgeConfig]]): Custom edge
            configurations, which take precedence over `ablations`.
            Defaults to None.
        seeds (Optional[List[int]]): Random states of the model initialization.
            Defaults to the configured value.
        path (Union[str, Path]): Path to the processed dataset. Defaults to
            Path("data/processed/processed_data.csv").
        verbose (bool): Prints the progress of the benchmark. Defaults to True.
        experiment_args (Optional[Dict[str, Any]]): Additional arguments passed
            to every `GraphExperiment`. Defaults to None.

    Attributes:
        tasks (List[str]): Tasks included in the benchmark.
        criteria (List[str]): Evaluation criteria included in the benchmark.
        edge_configs (Dict[str, EdgeConfig]): Edge configurations to compare.
        seeds (List[int]): Random states of the model initialization.
        data (pd.DataFrame): Processed dataset used in the benchmark.
        verbose (bool): Controls verbosity of the benchmark.
        experiment_args (Dict[str, Any]): Additional experiment arguments.

    Methods:
        run_benchmarks: Runs every combination and collects the metrics.

    Example:
        ```
        from periomod.graph import GraphBenchmarker

        benchmarker = GraphBenchmarker(
            tasks=["improvement"],
            criteria=["f1"],
            ablations=["full", "no_neighbors", "random_neighbors"],
            path="data/processed/processed_data.csv",
        )
        results, models = benchmarker.run_benchmarks()
        ```
    """

    def __init__(
        self,
        tasks: List[str],
        criteria: List[str],
        ablations: Optional[List[str]] = None,
        edge_configs: Optional[Dict[str, EdgeConfig]] = None,
        seeds: Optional[List[int]] = None,
        path: Union[str, Path] = Path("data/processed/processed_data.csv"),
        verbose: bool = True,
        experiment_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the benchmarker with tasks, criteria and ablations."""
        super().__init__()
        self.tasks = tasks
        self.criteria = criteria
        self.seeds = [self.gnn_state] if seeds is None else seeds
        self.verbose = verbose
        self.experiment_args = experiment_args or {}
        self.data = ProcessedDataLoader.load_data(path=path)

        if edge_configs is None:
            presets = EdgeConfig.ablations(seed=self.edge_seed)
            names = ablations or ["full", "no_neighbors", "random_neighbors"]
            edge_configs = {name: presets[name] for name in names}
        self.edge_configs = edge_configs

    def run_benchmarks(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Runs every combination of task, criterion, edges and seed.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Validation metrics of every
                combination and the trained models, keyed by configuration.
        """
        results, models = [], {}

        for task, criterion, (name, edges), seed in itertools.product(
            self.tasks, self.criteria, self.edge_configs.items(), self.seeds
        ):
            if (criterion == "macro_f1" and task != "pdgrouprevaluation") or (
                criterion == "f1" and task == "pdgrouprevaluation"
            ):
                print(f"Criterion '{criterion}' and task '{task}' not valid.")
                continue
            if self.verbose:
                print(
                    f"\nRunning graph benchmark for Task: {task}, "
                    f"Criterion: {criterion}, Edges: {name}, Seed: {seed}."
                )

            experiment = GraphExperiment(
                data=self.data,
                task=task,
                criterion=criterion,
                edges=edges,
                gnn_state=seed,
                verbose=self.verbose,
                **self.experiment_args,
            )
            result = experiment.perform_evaluation()
            metrics = {
                metric: value
                for metric, value in result["metrics"].items()
                if metric != "Confusion Matrix"
            }
            key = f"{task}_{criterion}_{name}_{seed}"
            models[key] = result["model"]
            results.append({
                "Task": task,
                "Criterion": criterion,
                "Edges": name,
                "Edge Mode": edges.mode,
                "Seed": seed,
                **metrics,
            })

        return pd.DataFrame(results), models
