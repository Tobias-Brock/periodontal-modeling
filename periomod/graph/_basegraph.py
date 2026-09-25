from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import hydra
import pandas as pd

from ..base import BaseConfig, BaseValidator

ANATOMICAL_RELATIONS = (
    "site_neighbor",
    "tooth_neighbor",
    "interproximal",
    "occlusal",
)
HIERARCHICAL_RELATIONS = ("patient_tooth", "tooth_site")
EDGE_MODES = ("anatomical", "none", "random")
NODE_LEVELS = ("patient", "tooth", "site")


def load_graph_config(
    config_path: str = "../../config", config_name: str = "graphconfig"
):
    """Composes the Hydra configuration of the graph submodule.

    Args:
        config_path (str): Path to the Hydra config directory, relative to this
            file. Defaults to "../../config".
        config_name (str): Name of the configuration file without extension.
            Defaults to "graphconfig".

    Returns:
        DictConfig: Composed configuration containing the 'graph' and 'gnn' groups.
    """
    with hydra.initialize(config_path=config_path, version_base="1.2"):
        return hydra.compose(config_name=config_name)


@dataclass
class EdgeConfig:
    """Switches for the relations included in a patient graph.

    The hierarchical relations ('patient_tooth' and 'tooth_site') define the
    hierarchy of the periodontal chart and are kept by default. The anatomical
    relations ('site_neighbor', 'tooth_neighbor', 'interproximal' and 'occlusal')
    encode biological adjacency and are the subject of the edge ablations, which
    are controlled by `mode`.

    Attributes:
        patient_tooth (bool): Connects the patient node to each tooth node.
        tooth_site (bool): Connects a tooth node to its six site nodes.
        site_neighbor (bool): Connects anatomically neighboring sites around the
            same tooth.
        tooth_neighbor (bool): Connects adjacent teeth within the same arch.
        interproximal (bool): Connects distal sites of a tooth with the mesial
            sites of the adjacent tooth.
        occlusal (bool): Connects occluding teeth of the upper and lower arch.
        mode (str): Treatment of the anatomical relations. Choose 'anatomical'
            for true anatomy, 'none' to drop all anatomical relations or 'random'
            to rewire them randomly within the patient graph.
        seed (int): Random state for the 'random' edge mode.
        name (str): Label of the configuration, used in ablation results.

    Example:
        ```
        from periomod.graph import EdgeConfig

        # true anatomy without occlusal contacts
        edges = EdgeConfig()

        # hierarchy only, all anatomical relations removed
        ablation = EdgeConfig(mode="none", name="no_neighbors")
        ```
    """

    patient_tooth: bool = True
    tooth_site: bool = True
    site_neighbor: bool = True
    tooth_neighbor: bool = True
    interproximal: bool = True
    occlusal: bool = False
    mode: str = "anatomical"
    seed: int = 0
    name: str = "full"

    def __post_init__(self) -> None:
        """Validates the edge mode.

        Raises:
            ValueError: If `mode` is not 'anatomical', 'none' or 'random'.
        """
        if self.mode not in EDGE_MODES:
            raise ValueError(
                f"{self.mode} is an invalid edge mode. "
                "Choose 'anatomical', 'none' or 'random'."
            )

    @classmethod
    def from_config(
        cls, relations: Dict[str, bool], mode: str, seed: int, name: str = "full"
    ) -> "EdgeConfig":
        """Creates an EdgeConfig from the Hydra configuration.

        Args:
            relations (Dict[str, bool]): Relation switches from the config.
            mode (str): Treatment of the anatomical relations.
            seed (int): Random state for the 'random' edge mode.
            name (str): Label of the configuration. Defaults to "full".

        Returns:
            EdgeConfig: Instantiated edge configuration.
        """
        return cls(**dict(relations), mode=mode, seed=seed, name=name)

    @classmethod
    def ablations(
        cls, seed: int = 0, occlusal: bool = False
    ) -> Dict[str, "EdgeConfig"]:
        """Returns the predefined edge configurations for ablation studies.

        Args:
            seed (int): Random state for the randomized configuration.
                Defaults to 0.
            occlusal (bool): Includes occlusal edges in every configuration.
                Defaults to False.

        Returns:
            Dict[str, EdgeConfig]: Mapping of ablation name to edge configuration.
        """
        return {
            "full": cls(occlusal=occlusal, seed=seed, name="full"),
            "no_neighbors": cls(
                occlusal=occlusal, mode="none", seed=seed, name="no_neighbors"
            ),
            "random_neighbors": cls(
                occlusal=occlusal, mode="random", seed=seed, name="random_neighbors"
            ),
            "no_site_neighbor": cls(
                site_neighbor=False,
                occlusal=occlusal,
                seed=seed,
                name="no_site_neighbor",
            ),
            "no_tooth_neighbor": cls(
                tooth_neighbor=False,
                occlusal=occlusal,
                seed=seed,
                name="no_tooth_neighbor",
            ),
            "no_interproximal": cls(
                interproximal=False,
                occlusal=occlusal,
                seed=seed,
                name="no_interproximal",
            ),
        }

    def active_relations(self) -> List[str]:
        """Determines the relations included in the patient graph.

        Anatomical relations are dropped entirely when `mode` is set to 'none'.

        Returns:
            List[str]: Names of the active relations.
        """
        active = [
            relation
            for relation in HIERARCHICAL_RELATIONS + ANATOMICAL_RELATIONS
            if getattr(self, relation)
        ]
        if self.mode == "none":
            return [
                relation for relation in active if relation not in ANATOMICAL_RELATIONS
            ]
        return active


class GraphConfigMixin:
    """Mixin that loads the graph and GNN configuration of the submodule.

    Provides `_init_graph_config`, which attaches all parameters of the
    'graph' and 'gnn' config groups to the instance. It is applied on top of
    `BaseConfig` or `BaseValidator`, which load the main package configuration.

    Attributes:
        exclude_columns (List[str]): Columns dropped from all node features.
        infect_levels (Dict[str, List[str]]): Engineered infection features per
            node level.
        include_infect (bool): Includes the engineered infection features.
        side_ring (List[List[int]]): Anatomical ring of the six sides of a tooth.
        interproximal_aspects (List[List[int]]): Interproximal contacts, given as
            (mesial side, distal side) pairs per aspect.
        relations (Dict[str, bool]): Default relation switches of the graph.
        edge_mode (str): Default treatment of the anatomical relations.
        edge_seed (int): Default random state of the 'random' edge mode.
        val_size (float): Default proportion of patients in the validation set.
        test_size (float): Default proportion of patients in the test set.
        split_seed (int): Default random state of the patient level split.
        target_sites (str): Default site selection for loss and evaluation.
        hidden_dim (int): Default hidden dimension of the GNN.
        num_layers (int): Default number of message passing stages.
        dropout (float): Default dropout rate.
        conv_aggr (str): Default neighborhood aggregation of the convolutions.
        relation_aggr (str): Default aggregation across relations.
        layer_norm (bool): Applies layer normalization between message passing
            stages.
        lr (float): Default learning rate.
        weight_decay (float): Default weight decay of the optimizer.
        batch_size (int): Default number of patient graphs per batch.
        epochs (int): Default maximum number of training epochs.
        patience (int): Default number of epochs without improvement before
            early stopping.
        gnn_state (int): Default random state of model initialization.
        device (str): Default compute device.
    """

    def _init_graph_config(self) -> None:
        """Loads the graph and GNN configuration into instance attributes."""
        cfg = load_graph_config()
        self.exclude_columns = [col.lower() for col in cfg.graph.exclude_columns]
        self.infect_levels = {
            level: [col.lower() for col in cfg.graph.infect_levels[level]]
            for level in NODE_LEVELS
        }
        self.include_infect = cfg.graph.include_infect
        self.side_ring = [tuple(pair) for pair in cfg.graph.side_ring]
        self.interproximal_aspects = [
            tuple(pair) for pair in cfg.graph.interproximal_aspects
        ]
        self.relations = dict(cfg.graph.relations)
        self.edge_mode = cfg.graph.edge_mode
        self.edge_seed = cfg.graph.edge_seed
        self.val_size = cfg.graph.val_size
        self.test_size = cfg.graph.test_size
        self.split_seed = cfg.graph.split_seed
        self.target_sites = cfg.graph.target_sites
        self.hidden_dim = cfg.gnn.hidden_dim
        self.num_layers = cfg.gnn.num_layers
        self.dropout = cfg.gnn.dropout
        self.conv_aggr = cfg.gnn.conv_aggr
        self.relation_aggr = cfg.gnn.relation_aggr
        self.layer_norm = cfg.gnn.layer_norm
        self.lr = cfg.gnn.lr
        self.weight_decay = cfg.gnn.weight_decay
        self.batch_size = cfg.gnn.batch_size
        self.epochs = cfg.gnn.epochs
        self.patience = cfg.gnn.patience
        self.gnn_state = cfg.gnn.gnn_state
        self.device = cfg.gnn.device

    def default_edges(self) -> EdgeConfig:
        """Builds the default edge configuration from the Hydra configuration.

        Returns:
            EdgeConfig: Edge configuration with the configured defaults.
        """
        return EdgeConfig.from_config(
            relations=self.relations, mode=self.edge_mode, seed=self.edge_seed
        )


class BaseGraphConfig(BaseConfig, GraphConfigMixin):
    """Base class providing the package and graph submodule configuration.

    Extends `BaseConfig` with the parameters of the 'graph' and 'gnn' config
    groups, which are composed from `config/graphconfig.yaml`.

    Inherits:
        - `BaseConfig`: Provides configuration settings of the package.
        - `GraphConfigMixin`: Loads the graph and GNN configuration.

    Example:
        ```
        config = BaseGraphConfig()
        print(config.hidden_dim)
        ```
    """

    def __init__(self) -> None:
        """Initializes the package and graph submodule configuration."""
        super().__init__()
        self._init_graph_config()


class BaseGraphValidator(BaseValidator, GraphConfigMixin):
    """Base class validating classification and criterion for graph learning.

    Inherits:
        - `BaseValidator`: Validates instance-level variables and parameters.
        - `GraphConfigMixin`: Loads the graph and GNN configuration.

    Args:
        classification (str): Type of classification, either 'binary' or
            'multiclass'.
        criterion (str): Evaluation criterion (e.g., 'f1', 'macro_f1',
            'brier_score').

    Example:
        ```
        validator = BaseGraphValidator(classification="binary", criterion="f1")
        print(validator.num_layers)
        ```
    """

    def __init__(self, classification: str, criterion: str) -> None:
        """Initializes the validator with the graph submodule configuration."""
        super().__init__(classification=classification, criterion=criterion)
        self._init_graph_config()


class BaseGraphTransformer(BaseGraphConfig, ABC):
    """Abstract base class for split-aware preprocessing of periodontal data.

    Provides the resolution of the dataset columns to the patient, tooth and
    site level of the graph, as well as the patient level split that precedes
    any preprocessing. Fitting of imputation, encoding and scaling is left to
    the subclass and must be restricted to training patients.

    Inherits:
        - `BaseGraphConfig`: Provides package and graph configuration.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        task (str): Task name. Can be 'pocketclosure', 'pocketclosureinf',
            'improvement' or 'pdgrouprevaluation'.
        include_infect (Optional[bool]): Includes the engineered infection
            features. Defaults to the configured value.

    Attributes:
        task (str): Task name used to derive the site level outcome.
        include_infect (bool): Indicates whether engineered infection features
            are included in the node features.
        levels (Dict[str, Dict[str, List[str]]]): Raw columns per node level and
            column kind ('numeric', 'binary' or 'categorical').

    Abstract Methods:
        - `fit`: Fits imputation, encoding and scaling on training patients.
        - `transform_data`: Applies the fitted transformations to a dataset.
    """

    def __init__(self, task: str, include_infect: Optional[bool] = None) -> None:
        """Initializes the transformer with task and feature configuration."""
        super().__init__()
        self.task = task
        self.include_infect = (
            self.include_infect if include_infect is None else include_infect
        )
        self.levels: Dict[str, Dict[str, List[str]]] = {}

    def _level_columns(self, level: str) -> List[str]:
        """Collects the candidate columns of a node level.

        Args:
            level (str): Node level, either 'patient', 'tooth' or 'site'.

        Returns:
            List[str]: Lowercased column names assigned to the node level.

        Raises:
            ValueError: If `level` is not a valid node level.
        """
        if level == "patient":
            columns = list(self.patient_columns)
        elif level == "tooth":
            columns = list(self.tooth_columns)
        elif level == "site":
            columns = list(self.side_columns)
        else:
            raise ValueError(
                f"{level} is an invalid node level. "
                "Choose 'patient', 'tooth' or 'site'."
            )
        columns = [col.lower() for col in columns]
        if self.include_infect:
            columns += self.infect_levels[level]

        excluded = set(self.exclude_columns + list(self.task_cols) + self.no_train_cols)
        return [col for col in columns if col not in excluded]

    def _column_kind(self, column: str) -> str:
        """Determines the kind of a column from the package configuration.

        Args:
            column (str): Lowercased column name.

        Returns:
            str: Column kind, either 'categorical', 'binary' or 'numeric'.
        """
        if column in self.all_cat_vars:
            return "categorical"
        if column in self.bin_vars:
            return "binary"
        return "numeric"

    def resolve_levels(self, data: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        """Assigns the available columns to node levels and column kinds.

        Args:
            data (pd.DataFrame): Processed dataset with lowercased columns.

        Returns:
            Dict[str, Dict[str, List[str]]]: Columns per node level and kind.
        """
        levels: Dict[str, Dict[str, List[str]]] = {}
        for level in NODE_LEVELS:
            columns = [col for col in self._level_columns(level) if col in data.columns]
            levels[level] = {
                kind: [col for col in columns if self._column_kind(col) == kind]
                for kind in ("numeric", "binary", "categorical")
            }
        self.levels = levels
        return levels

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fits imputation, encoding and scaling on training patients.

        Args:
            data (pd.DataFrame): Dataset restricted to training patients.
        """

    @abstractmethod
    def transform_data(self, data: pd.DataFrame):
        """Applies the fitted transformations to a dataset.

        Args:
            data (pd.DataFrame): Dataset to transform.
        """


class BaseGraphBuilder(BaseGraphConfig, ABC):
    """Abstract base class for the construction of patient graphs.

    Inherits:
        - `BaseGraphConfig`: Provides package and graph configuration.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        feature_names (Dict[str, List[str]]): Transformed feature columns per
            node level, as returned by the transformer.
        edges (Optional[EdgeConfig]): Relations included in the graph. Defaults
            to the configured relations.

    Attributes:
        feature_names (Dict[str, List[str]]): Feature columns per node level.
        edges (EdgeConfig): Relations included in the patient graphs.

    Abstract Methods:
        - `build_patient_graph`: Builds the graph of a single patient.
        - `build`: Builds the graphs of all patients in a dataset.
    """

    def __init__(
        self,
        feature_names: Dict[str, List[str]],
        edges: Optional[EdgeConfig] = None,
    ) -> None:
        """Initializes the builder with feature columns and relations."""
        super().__init__()
        self.feature_names = feature_names
        self.edges = self.default_edges() if edges is None else edges

    @abstractmethod
    def build_patient_graph(self, patient_data: pd.DataFrame):
        """Builds the heterogeneous graph of a single patient.

        Args:
            patient_data (pd.DataFrame): Transformed rows of one patient.
        """

    @abstractmethod
    def build(self, data: pd.DataFrame):
        """Builds the heterogeneous graphs of all patients in a dataset.

        Args:
            data (pd.DataFrame): Transformed dataset with one row per site.
        """


class BaseGraphTrainer(BaseGraphValidator, ABC):
    """Abstract base class for training heterogeneous GNNs on patient graphs.

    Inherits:
        - `BaseGraphValidator`: Validates classification and criterion.
        - `ABC`: Specifies abstract methods for subclasses to implement.

    Args:
        classification (str): Type of classification, either 'binary' or
            'multiclass'.
        criterion (str): Evaluation criterion (e.g., 'f1', 'macro_f1',
            'brier_score').

    Abstract Methods:
        - `train`: Trains a model with early stopping on the validation loader.
        - `predict`: Computes site level predictions for a data loader.
    """

    def __init__(self, classification: str, criterion: str) -> None:
        """Initializes the trainer with classification type and criterion."""
        super().__init__(classification=classification, criterion=criterion)

    @abstractmethod
    def train(self, model, train_loader, val_loader):
        """Trains a model with early stopping on the validation loader.

        Args:
            model (Any): Heterogeneous GNN to train.
            train_loader (Any): Loader of the training patient graphs.
            val_loader (Any): Loader of the validation patient graphs.
        """

    @abstractmethod
    def predict(self, model, loader):
        """Computes site level predictions for a data loader.

        Args:
            model (Any): Trained heterogeneous GNN.
            loader (Any): Loader of patient graphs.
        """


@dataclass
class GraphSplit:
    """Patient identifiers of the fixed train, validation and test split.

    Attributes:
        train (List[int]): Patient identifiers of the training set.
        val (List[int]): Patient identifiers of the validation set.
        test (List[int]): Patient identifiers of the test set.

    Example:
        ```
        split = GraphSplit(train=[1, 2], val=[3], test=[4])
        print(split.sizes)
        ```
    """

    train: List[int] = field(default_factory=list)
    val: List[int] = field(default_factory=list)
    test: List[int] = field(default_factory=list)

    @property
    def sizes(self) -> Dict[str, int]:
        """Number of patients per split.

        Returns:
            Dict[str, int]: Mapping of split name to number of patients.
        """
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def items(self) -> List[Tuple[str, List[int]]]:
        """Splits as name and patient identifier pairs.

        Returns:
            List[Tuple[str, List[int]]]: Split name and patient identifiers.
        """
        return [("train", self.train), ("val", self.val), ("test", self.test)]
