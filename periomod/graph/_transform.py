from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._basegraph import NODE_LEVELS, BaseGraphTransformer, GraphSplit


class GraphDataTransformer(BaseGraphTransformer):
    """Split-aware preprocessing of processed periodontal data for graph learning.

    Patients are split into fixed training, validation and test sets before any
    transformation is fitted. Imputation, one-hot encoding and standard scaling
    are fitted on training patients only and applied to validation and test
    patients, so that no information crosses the patient level split. Statistics
    of the patient and tooth level are computed on unique patients and unique
    teeth, which avoids weighting them by the number of sites.

    All baseline sites are kept as contextual information. The sites entering
    loss and evaluation are marked by the `target_mask` column, which is derived
    from the task or set explicitly through `target_sites`.

    Inherits:
        - `BaseGraphTransformer`: Provides node level resolution and config.

    Args:
        task (str): Task name. Can be 'pocketclosure', 'pocketclosureinf',
            'improvement' or 'pdgrouprevaluation'.
        include_infect (Optional[bool]): Includes the engineered infection
            features 'side_infected', 'tooth_infected' and 'infected_neighbors'.
            Defaults to the configured value.
        target_sites (Optional[str]): Sites entering loss and evaluation. Choose
            'all', 'diseased' for sites with baseline pocket depth above 3 mm or
            'task' to derive the selection from the task. Defaults to the
            configured value.
        val_size (Optional[float]): Proportion of patients in the validation
            set. Defaults to the configured value.
        test_size (Optional[float]): Proportion of patients in the test set.
            Defaults to the configured value.
        split_seed (Optional[int]): Random state of the patient level split.
            Defaults to the configured value.
        verbose (bool): Prints split and feature summaries. Defaults to True.

    Attributes:
        task (str): Task name used to derive the site level outcome.
        target_sites (str): Selection of sites entering loss and evaluation.
        val_size (float): Proportion of patients in the validation set.
        test_size (float): Proportion of patients in the test set.
        split_seed (int): Random state of the patient level split.
        verbose (bool): Controls verbosity of the preprocessing.
        levels (Dict[str, Dict[str, List[str]]]): Raw columns per node level and
            column kind.
        impute_values (Dict[str, float]): Imputation value per raw column,
            computed on training patients.
        encoders (Dict[str, OneHotEncoder]): Fitted one-hot encoder per level.
        scalers (Dict[str, StandardScaler]): Fitted standard scaler per level.
        feature_names (Dict[str, List[str]]): Transformed feature columns per
            node level.
        split (GraphSplit): Patient identifiers of the fixed split.

    Methods:
        split_patients: Splits patients into train, validation and test sets.
        fit: Fits imputation, encoding and scaling on training patients.
        transform_data: Applies the fitted transformations to a dataset.
        transform_splits: Splits patients, fits on the training patients and
            transforms all three splits.

    Example:
        ```
        from periomod.data import ProcessedDataLoader
        from periomod.graph import GraphDataTransformer

        df = ProcessedDataLoader.load_data(
            path="data/processed/processed_data.csv"
        )
        transformer = GraphDataTransformer(task="improvement")
        splits = transformer.transform_splits(data=df)
        print(transformer.feature_names["site"])
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
        """Initializes the transformer with task, split and feature settings."""
        super().__init__(task=task, include_infect=include_infect)
        self.target_sites = self.target_sites if target_sites is None else target_sites
        self.val_size = self.val_size if val_size is None else val_size
        self.test_size = self.test_size if test_size is None else test_size
        self.split_seed = self.split_seed if split_seed is None else split_seed
        self.verbose = verbose
        self._validate_task()
        self._validate_target_sites()
        self.impute_values: Dict[str, float] = {}
        self.encoders: Dict[str, OneHotEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.split = GraphSplit()

    def _validate_task(self) -> None:
        """Validates the task against the tasks defined in the configuration.

        Raises:
            ValueError: If `self.task` is not a supported task.
        """
        if self.task not in self.task_cols:
            raise ValueError(
                f"Task '{self.task}' not supported. "
                f"Choose one of {list(self.task_cols)}."
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
        """Reduces a dataset to unique rows of a node level.

        Args:
            data (pd.DataFrame): Dataset with one row per site.
            level (str): Node level, either 'patient', 'tooth' or 'site'.

        Returns:
            pd.DataFrame: Dataset with one row per node of the given level.
        """
        if level == "patient":
            return data.drop_duplicates(subset=[self.group_col])
        if level == "tooth":
            return data.drop_duplicates(subset=[self.group_col, "tooth"])
        return data

    def split_patients(self, data: pd.DataFrame) -> GraphSplit:
        """Splits patients into fixed training, validation and test sets.

        The split is performed on patient identifiers before any preprocessing,
        so that all sites of a patient stay within the same split. Both
        `val_size` and `test_size` refer to the full set of patients.

        Args:
            data (pd.DataFrame): Processed dataset with one row per site.

        Returns:
            GraphSplit: Patient identifiers of the three splits.

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

        self.split = GraphSplit(
            train=sorted(dev_data.iloc[train_idx][self.group_col].unique().tolist()),
            val=sorted(dev_data.iloc[val_idx][self.group_col].unique().tolist()),
            test=sorted(data.iloc[test_idx][self.group_col].unique().tolist()),
        )
        if self.verbose:
            print(f"Patients per split: {self.split.sizes}")
        return self.split

    def _assign_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds the site level outcome and the target mask to a dataset.

        Args:
            data (pd.DataFrame): Processed dataset with one row per site.

        Returns:
            pd.DataFrame: Dataset with the 'y' and 'target_mask' columns.

        Raises:
            ValueError: If the outcome column of the task is missing.
        """
        column = "pocketclosure" if self.task == "pocketclosureinf" else self.task
        if column not in data.columns:
            raise ValueError(f"Outcome column '{column}' is missing from the dataset.")

        data = data.copy()
        data["y"] = data[column]

        selection = self.target_sites
        if selection == "task":
            selection = (
                "diseased"
                if self.task in ["improvement", "pocketclosureinf"]
                else "all"
            )
        if selection == "diseased":
            if "pdgroupbase" in data.columns:
                data["target_mask"] = data["pdgroupbase"].isin([1, 2])
            else:
                data["target_mask"] = data["pdbaseline"] > 3
        else:
            data["target_mask"] = True

        return data

    def fit(self, data: pd.DataFrame) -> "GraphDataTransformer":
        """Fits imputation, encoding and scaling on training patients.

        Args:
            data (pd.DataFrame): Dataset restricted to training patients.

        Returns:
            GraphDataTransformer: The fitted transformer.
        """
        self.resolve_levels(data=data)
        self.impute_values = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_names = {}

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
            encoded_names: List[str] = []
            if categorical:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoder.fit(imputed[categorical].astype(str))
                self.encoders[level] = encoder
                encoded_names = list(encoder.get_feature_names_out(categorical))
            if numeric:
                scaler = StandardScaler()
                scaler.fit(imputed[numeric])
                self.scalers[level] = scaler

            self.feature_names[level] = numeric + binary + encoded_names

        if self.verbose:
            counts = {level: len(names) for level, names in self.feature_names.items()}
            print(f"Features per node level: {counts}")
        return self

    def _impute(self, data: pd.DataFrame, level: str) -> pd.DataFrame:
        """Imputes the columns of a node level with the fitted values.

        Args:
            data (pd.DataFrame): Dataset to impute.
            level (str): Node level, either 'patient', 'tooth' or 'site'.

        Returns:
            pd.DataFrame: Dataset with imputed columns of the node level.
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

    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted transformations to a dataset.

        Args:
            data (pd.DataFrame): Dataset with one row per site.

        Returns:
            pd.DataFrame: Dataset with the key columns 'id_patient', 'tooth' and
                'side', the transformed features of all node levels, the outcome
                'y' and the boolean 'target_mask'.

        Raises:
            ValueError: If the transformer was not fitted before transforming.
        """
        if not self.feature_names:
            raise ValueError("Transformer must be fitted before transforming data.")

        data = self._assign_target(data=data)
        transformed = data[[self.group_col, "tooth", "side", "y", "target_mask"]].copy()

        for level in NODE_LEVELS:
            numeric = self.levels[level]["numeric"]
            binary = self.levels[level]["binary"]
            categorical = self.levels[level]["categorical"]
            imputed = self._impute(data=data, level=level)

            if numeric:
                transformed[numeric] = self.scalers[level].transform(imputed[numeric])
            if binary:
                transformed[binary] = imputed[binary].astype(float).to_numpy()
            if categorical:
                encoder = self.encoders[level]
                encoded = encoder.transform(imputed[categorical].astype(str))
                names = list(encoder.get_feature_names_out(categorical))
                transformed[names] = encoded

        return transformed.reset_index(drop=True)

    def transform_splits(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Splits patients, fits on training patients and transforms all splits.

        Args:
            data (pd.DataFrame): Processed dataset with one row per site.

        Returns:
            Dict[str, pd.DataFrame]: Transformed datasets of the 'train', 'val'
                and 'test' split.
        """
        data = data.rename(columns=str.lower)
        split = self.split_patients(data=data)
        train_data = data[data[self.group_col].isin(split.train)]
        self.fit(data=train_data)

        splits: Dict[str, pd.DataFrame] = {}
        for name, patients in split.items():
            subset = data[data[self.group_col].isin(patients)]
            splits[name] = self.transform_data(data=subset)
            if self.verbose:
                mask = splits[name]["target_mask"]
                print(
                    f"Split '{name}': {len(patients)} patients, "
                    f"{len(subset)} sites, {int(np.sum(mask))} target sites."
                )
        return splits
