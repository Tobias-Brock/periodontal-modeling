from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..base import all_teeth
from ._basebayes import (
    BAYES_TASKS,
    NODE_LEVELS,
    BaseHierarchicalTransformer,
    HierarchicalData,
    PatientSplit,
)


class HierarchicalDataTransformer(BaseHierarchicalTransformer):
    """Split-aware preprocessing of processed data for hierarchical models.

    Patients are split into fixed training, validation and test sets before any
    transformation is fitted, using the same grouped split as the graph
    submodule, so that both model families can be compared on identical
    patients. Imputation, dummy coding and standardization are fitted on
    training patients only.

    Continuous predictors are standardized with training statistics. Binary
    predictors enter as 0/1 indicators. Categorical predictors receive compact
    dummy coding with the first level as reference, except for the predictors
    listed in `direct_effects`, which the model represents as hierarchical
    effects over their levels instead of dummy variables.

    Observations are the target sites of the task, e.g. sites with a baseline
    pocket depth above 3 mm for 'improvement', which reproduces the row
    selection of the tabular pipeline.

    Inherits:
        - `BaseHierarchicalTransformer`: Provides level resolution and config.

    Args:
        task (str): Task name. Can be 'pocketclosure', 'pocketclosureinf' or
            'improvement'.
        include_infect (Optional[bool]): Includes the engineered infection
            features. Defaults to the configured value.
        target_sites (Optional[str]): Sites entering the likelihood. Choose
            'all', 'diseased' for sites with baseline pocket depth above 3 mm
            or 'task' to derive the selection from the task. Defaults to the
            configured value.
        val_size (Optional[float]): Proportion of patients in the validation
            set. Defaults to the configured value.
        test_size (Optional[float]): Proportion of patients in the test set.
            Defaults to the configured value.
        split_seed (Optional[int]): Random state of the patient level split.
            Defaults to the configured value.
        verbose (bool): Prints split and design summaries. Defaults to True.

    Attributes:
        task (str): Task name used to derive the site level outcome.
        target_sites (str): Selection of sites entering the likelihood.
        val_size (float): Proportion of patients in the validation set.
        test_size (float): Proportion of patients in the test set.
        split_seed (int): Random state of the patient level split.
        verbose (bool): Controls verbosity of the preprocessing.
        levels (Dict[str, Dict[str, List[str]]]): Raw columns per level.
        impute_values (Dict[str, Any]): Imputation value per raw column.
        encoders (Dict[str, OneHotEncoder]): Fitted dummy coder per level.
        keep_columns (Dict[str, np.ndarray]): Retained dummy columns per level.
        scalers (Dict[str, StandardScaler]): Fitted standard scaler per level.
        feature_names (List[str]): Columns of the design matrix.
        split (PatientSplit): Patient identifiers of the fixed split.

    Methods:
        split_patients: Splits patients into train, validation and test sets.
        fit: Fits imputation, dummy coding and scaling on training patients.
        transform_data: Builds design matrix and nesting structure of a split.
        transform_splits: Splits patients, fits on training patients and
            transforms all three splits.

    Example:
        ```
        from periomod.bayes import HierarchicalDataTransformer
        from periomod.data import ProcessedDataLoader

        df = ProcessedDataLoader.load_data(
            path="data/processed/processed_data.csv"
        )
        transformer = HierarchicalDataTransformer(task="improvement")
        splits = transformer.transform_splits(data=df)
        print(splits["train"].n_obs, splits["train"].n_teeth)
        ```
    """

    def __init__(
        self,
        task: str,
        include_infect: Optional[bool] = None,
        target_sites: Optional[str] = None,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
        split_seed: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """Initializes the transformer with task, split and design settings."""
        super().__init__(task=task, include_infect=include_infect)
        self.target_sites = self.target_sites if target_sites is None else target_sites
        self.val_size = self.val_size if val_size is None else val_size
        self.test_size = self.test_size if test_size is None else test_size
        self.split_seed = self.split_seed if split_seed is None else split_seed
        self.verbose = verbose
        self._validate_task()
        self._validate_target_sites()
        self.impute_values: Dict[str, Any] = {}
        self.encoders: Dict[str, OneHotEncoder] = {}
        self.keep_columns: Dict[str, np.ndarray] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.split = PatientSplit()
        self.side_positions = sorted({side for pair in self.side_ring for side in pair})
        self.tooth_positions = {tooth: index for index, tooth in enumerate(all_teeth)}

    def _validate_task(self) -> None:
        """Validates the task against the tasks supported by the submodule.

        Raises:
            ValueError: If `self.task` is not a supported binary task.
        """
        if self.task not in BAYES_TASKS:
            raise ValueError(
                f"Task '{self.task}' not supported by the Bayesian submodule. "
                f"Choose one of {list(BAYES_TASKS)}. Multiclass tasks such as "
                "'pdgrouprevaluation' require an ordinal likelihood."
            )

    def _validate_target_sites(self) -> None:
        """Validates the selection of target sites.

        Raises:
            ValueError: If `self.target_sites` is not a valid selection.
        """
        if self.target_sites not in ["all", "diseased", "task"]:
            raise ValueError(
                f"{self.target_sites} is an invalid site selection. "
                "Choose 'all', 'diseased' or 'task'."
            )

    def _level_frame(self, data: pd.DataFrame, level: str) -> pd.DataFrame:
        """Reduces a dataset to unique rows of a level.

        Args:
            data (pd.DataFrame): Dataset with one row per site.
            level (str): Level, either 'patient', 'tooth' or 'site'.

        Returns:
            pd.DataFrame: Dataset with one row per unit of the given level.
        """
        if level == "patient":
            return data.drop_duplicates(subset=[self.group_col])
        if level == "tooth":
            return data.drop_duplicates(subset=[self.group_col, "tooth"])
        return data

    def split_patients(self, data: pd.DataFrame) -> PatientSplit:
        """Splits patients into fixed training, validation and test sets.

        The split is performed on patient identifiers before any preprocessing.
        Both `val_size` and `test_size` refer to the full set of patients.

        Args:
            data (pd.DataFrame): Processed dataset with one row per site.

        Returns:
            PatientSplit: Patient identifiers of the three splits.

        Raises:
            ValueError: If the group column is missing from the dataset.
        """
        if self.group_col not in data.columns:
            raise ValueError(f"Column '{self.group_col}' is missing from the dataset.")

        groups = data[self.group_col]
        test_split = GroupShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.split_seed
        )
        dev_idx, test_idx = next(test_split.split(X=data, groups=groups))
        dev_data = data.iloc[dev_idx]

        val_fraction = self.val_size / (1 - self.test_size)
        val_split = GroupShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=self.split_seed
        )
        train_idx, val_idx = next(
            val_split.split(X=dev_data, groups=dev_data[self.group_col])
        )

        self.split = PatientSplit(
            train=sorted(dev_data.iloc[train_idx][self.group_col].unique().tolist()),
            val=sorted(dev_data.iloc[val_idx][self.group_col].unique().tolist()),
            test=sorted(data.iloc[test_idx][self.group_col].unique().tolist()),
        )
        if self.verbose:
            print(f"Patients per split: {self.split.sizes}")
        return self.split

    def _target_selection(self) -> str:
        """Resolves the selection of target sites for the task.

        Returns:
            str: Site selection, either 'all' or 'diseased'.
        """
        if self.target_sites != "task":
            return self.target_sites
        return "diseased" if self.task in ["improvement", "pocketclosureinf"] else "all"

    def _assign_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds the site level outcome and restricts the data to target sites.

        Args:
            data (pd.DataFrame): Processed dataset with one row per site.

        Returns:
            pd.DataFrame: Target sites with the outcome column 'y'.

        Raises:
            ValueError: If the outcome column of the task is missing.
        """
        column = "pocketclosure" if self.task == "pocketclosureinf" else self.task
        if column not in data.columns:
            raise ValueError(f"Outcome column '{column}' is missing from the dataset.")

        data = data.copy()
        data["y"] = data[column]

        if self._target_selection() == "diseased":
            if "pdgroupbase" in data.columns:
                mask = data["pdgroupbase"].isin([1, 2])
            else:
                mask = data["pdbaseline"] > 3
            data = data[mask]

        return data.sort_values(by=[self.group_col, "tooth", "side"])

    def _impute(self, data: pd.DataFrame, level: str) -> pd.DataFrame:
        """Imputes the columns of a level with the fitted values.

        Args:
            data (pd.DataFrame): Dataset to impute.
            level (str): Level, either 'patient', 'tooth' or 'site'.

        Returns:
            pd.DataFrame: Dataset with imputed columns of the level.
        """
        columns = (
            self.levels[level]["numeric"]
            + self.levels[level]["binary"]
            + self.levels[level]["categorical"]
        )
        imputed = data[columns].copy()
        for column in self.levels[level]["numeric"]:
            imputed[column] = pd.to_numeric(imputed[column], errors="coerce")
        return imputed.fillna(value={col: self.impute_values[col] for col in columns})

    @staticmethod
    def _reference_mask(encoder: OneHotEncoder) -> np.ndarray:
        """Determines the dummy columns retained after dropping references.

        Args:
            encoder (OneHotEncoder): Encoder fitted on the training patients.

        Returns:
            np.ndarray: Boolean mask of the retained dummy columns.
        """
        keep = []
        for categories in encoder.categories_:
            keep += [False] + [True] * (len(categories) - 1)
        return np.array(keep, dtype=bool)

    def fit(self, data: pd.DataFrame) -> "HierarchicalDataTransformer":
        """Fits imputation, dummy coding and scaling on training patients.

        Args:
            data (pd.DataFrame): Dataset restricted to training patients.

        Returns:
            HierarchicalDataTransformer: The fitted transformer.
        """
        data = self._assign_target(data=data)
        self.resolve_levels(data=data)
        self.impute_values = {}
        self.encoders = {}
        self.keep_columns = {}
        self.scalers = {}
        self.feature_names = []

        for level in NODE_LEVELS:
            frame = self._level_frame(data=data, level=level)
            numeric = self.levels[level]["numeric"]
            binary = self.levels[level]["binary"]
            categorical = self.levels[level]["categorical"]

            for column in numeric:
                self.impute_values[column] = float(
                    pd.to_numeric(frame[column], errors="coerce").median()
                )
            for column in binary + categorical:
                modes = frame[column].mode(dropna=True)
                self.impute_values[column] = modes.iloc[0] if not modes.empty else 0

            imputed = self._impute(data=frame, level=level)
            dummy_names: List[str] = []
            if categorical:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(imputed[categorical].astype(str))
                mask = self._reference_mask(encoder=encoder)
                self.encoders[level] = encoder
                self.keep_columns[level] = mask
                names = np.array(encoder.get_feature_names_out(categorical))
                dummy_names = list(names[mask])
            if numeric:
                scaler = StandardScaler()
                scaler.fit(imputed[numeric])
                self.scalers[level] = scaler

            self.feature_names += numeric + binary + dummy_names

        if self.verbose:
            print(f"Fixed effects in the design matrix: {len(self.feature_names)}")
        return self

    def _design_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Builds the design matrix of the fixed effects.

        Args:
            data (pd.DataFrame): Target sites of a split.

        Returns:
            np.ndarray: Design matrix of shape (n_obs, n_features).
        """
        blocks = []
        for level in NODE_LEVELS:
            numeric = self.levels[level]["numeric"]
            binary = self.levels[level]["binary"]
            categorical = self.levels[level]["categorical"]
            imputed = self._impute(data=data, level=level)

            if numeric:
                blocks.append(self.scalers[level].transform(imputed[numeric]))
            if binary:
                blocks.append(imputed[binary].astype(float).to_numpy())
            if categorical:
                encoded = self.encoders[level].transform(
                    imputed[categorical].astype(str)
                )
                blocks.append(encoded[:, self.keep_columns[level]])

        return np.concatenate(blocks, axis=1).astype(float)

    def transform_data(self, data: pd.DataFrame) -> HierarchicalData:
        """Builds design matrix and nesting structure of a dataset.

        Args:
            data (pd.DataFrame): Dataset with one row per site.

        Returns:
            HierarchicalData: Design matrix, outcome and nesting structure of
                the target sites.

        Raises:
            ValueError: If the transformer was not fitted before transforming.
        """
        if not self.feature_names:
            raise ValueError("Transformer must be fitted before transforming data.")

        data = self._assign_target(data=data)
        keys = data[[self.group_col, "tooth", "side"]].reset_index(drop=True)
        patients, patient_idx = np.unique(
            keys[self.group_col].to_numpy(), return_inverse=True
        )
        tooth_keys = patient_idx.astype(np.int64) * 100 + keys[
            "tooth"
        ].to_numpy().astype(np.int64)
        _, first, tooth_idx = np.unique(
            tooth_keys, return_index=True, return_inverse=True
        )
        tooth_number = keys["tooth"].to_numpy()[first]
        side_idx = np.array(
            [self.side_positions.index(side) for side in keys["side"].to_numpy()],
            dtype=np.int64,
        )

        return HierarchicalData(
            X=self._design_matrix(data=data),
            y=data["y"].to_numpy().astype(float),
            feature_names=list(self.feature_names),
            keys=keys,
            patient_idx=patient_idx.astype(np.int64),
            tooth_idx=tooth_idx.astype(np.int64),
            side_idx=side_idx,
            tooth_patient=patient_idx[first].astype(np.int64),
            tooth_number=tooth_number.astype(np.int64),
            toothnum_idx=np.array(
                [self.tooth_positions[tooth] for tooth in tooth_number], dtype=np.int64
            ),
            patients=patients,
        )

    def transform_splits(self, data: pd.DataFrame) -> Dict[str, HierarchicalData]:
        """Splits patients, fits on training patients and transforms all splits.

        Args:
            data (pd.DataFrame): Processed dataset with one row per site.

        Returns:
            Dict[str, HierarchicalData]: Design matrices of the 'train', 'val'
                and 'test' split.
        """
        data = data.rename(columns=str.lower)
        split = self.split_patients(data=data)
        self.fit(data=data[data[self.group_col].isin(split.train)])

        splits: Dict[str, HierarchicalData] = {}
        for name, patients in split.items():
            subset = data[data[self.group_col].isin(patients)]
            splits[name] = self.transform_data(data=subset)
            if self.verbose:
                print(
                    f"Split '{name}': {splits[name].n_patients} patients, "
                    f"{splits[name].n_teeth} teeth, {splits[name].n_obs} target sites."
                )
        return splits
