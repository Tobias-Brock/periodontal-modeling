from typing import Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data import BaseDataLoader


class ProcessedDataLoader(BaseDataLoader):
    """Concrete data loader for loading, transforming, and saving processed data.

    This class implements methods for encoding categorical columns, scaling numeric
    columns, and transforming data based on the specified task. It supports encoding
    types such as 'one_hot' and 'target', with optional scaling of numeric columns.

    Inherits:
        `BaseDataLoader`: Provides core data loading, encoding, and scaling methods.

    Args:
        task (str): The task column name, used to guide specific transformations.
            Can be 'pocketclosure', 'pocketclosureinf', 'improvement', or
            'pdgrouprevaluation'.
        encoding (Optional[str]): Specifies the encoding method for categorical columns.
            Options include 'one_hot', 'target', or None. Defaults to None.
        encode (bool, optional): If True, applies encoding to categorical columns.
            Defaults to True.
        scale (bool, optional): If True, applies scaling to numeric columns.
            Defaults to True.

    Attributes:
        task (str): Task column name used during data transformations. Can be
            'pocketclosure', 'pocketclosureinf', 'improvement', or 'pdgrouprevaluation'.
        encoding (str): Encoding method specified for categorical columns. Options
            include 'one_hot' or 'target'.
        encode (bool): Flag to enable encoding of categorical columns.
        scale (bool): Flag to enable scaling of numeric columns.

    Methods:
        encode_categorical_columns: Encodes categorical columns based on
            the specified encoding method.
        scale_numeric_columns: Scales numeric columns to normalize data.
        transform_data: Executes the complete data processing pipeline,
            including encoding and scaling.

    Inherited Methods:
        - `load_data`: Load processed data from the specified path and file.
        - `save_data`: Save processed data to the specified path and file.

    Example:
        ```
        from periomod.data import ProcessedDataLoader

        # instantiate with one-hot encoding and scale numerical variables
        dataloader = ProcessedDataLoader(
            task="pocketclosure", encoding="one_hot", encode=True, scale=True
        )
        data = dataloader.load_data(path="data/processed/processed_data.csv")
        data = dataloader.transform_data(data=data)
        dataloader.save_data(data=data, path="data/training/training_data.csv")
        ```
    """

    def __init__(
        self,
        task: str,
        encoding: Optional[str] = None,
        encode: bool = True,
        scale: bool = True,
    ) -> None:
        """Initializes the ProcessedDataLoader with the specified task column."""
        super().__init__(task=task, encoding=encoding, encode=encode, scale=scale)
        if self.encoding == "one_hot":
            self.one_hot_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )

    def encode_categorical_columns(
        self, data: pd.DataFrame, fit_encoder: bool = True
    ) -> pd.DataFrame:
        """Encodes categorical columns in the DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame containing categorical columns.
            fit_encoder (bool): Whether to fit the encoder on this dataset
                (only for training data). Defaults to True.

        Returns:
            pd.DataFrame: The DataFrame with encoded categorical columns.

        Raises:
            ValueError: If an invalid encoding type is specified.
        """
        if not self.encode:
            return data

        cat_vars = [col for col in self.all_cat_vars if col in data.columns]
        data[cat_vars] = data[cat_vars].apply(
            lambda col: col.astype(str) if col.dtype in [float, object] else col
        )

        if self.encoding == "one_hot":
            data_reset = data.reset_index(drop=True)

            if fit_encoder or self.one_hot_encoder is None:
                encoded_arr = self.one_hot_encoder.fit_transform(data_reset[cat_vars])
                self.encoded_feature_names = self.one_hot_encoder.get_feature_names_out(
                    cat_vars
                )  # Store feature names
            else:
                encoded_arr = self.one_hot_encoder.transform(data_reset[cat_vars])

            encoded_cols = self.one_hot_encoder.get_feature_names_out(cat_vars)
            encoded_df = pd.DataFrame(
                encoded_arr, columns=encoded_cols, index=data.index
            )

            if not fit_encoder and hasattr(self, "encoded_feature_names"):
                missing_cols = set(self.encoded_feature_names) - set(encoded_df.columns)
                for col in missing_cols:
                    encoded_df[col] = 0  # Add missing features as 0

                encoded_df = encoded_df[list(self.encoded_feature_names)]

            data_encoded = pd.concat(
                [data_reset.drop(cat_vars, axis=1), encoded_df], axis=1
            )

        elif self.encoding == "target":
            data["toothside"] = (
                data["tooth"].astype(str) + "_" + data["side"].astype(str)
            )
            data_encoded = data.drop(columns=["tooth", "side"])

        else:
            raise ValueError(
                f"Invalid encoding '{self.encoding}' specified. "
                "Choose 'one_hot', 'target', or None."
            )

        self._check_encoded_columns(data=data_encoded)
        return data_encoded

    def scale_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scales numeric columns in the DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame containing numeric columns.

        Returns:
            data: The DataFrame with scaled numeric columns.
        """
        scale_vars = [col for col in self.scale_vars if col in data.columns]
        data[scale_vars] = data[scale_vars].apply(pd.to_numeric, errors="coerce")
        scaled_values = StandardScaler().fit_transform(data[scale_vars])
        data[scale_vars] = pd.DataFrame(
            scaled_values, columns=scale_vars, index=data.index
        )
        self._check_scaled_columns(data=data)
        return data

    def transform_data(
        self, data: pd.DataFrame, fit_encoder: bool = True
    ) -> pd.DataFrame:
        """Select task column and optionally, scale and encode.

        Args:
            data (pd.DataFrame): The DataFrame to transform.
            fit_encoder (bool): Whether to fit the encoder on this dataset
                (only for training data). Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with the selected task 'y'.

        Raises:
            ValueError: If self.task is invalid.
        """
        if self.encode:
            data = self.encode_categorical_columns(data=data, fit_encoder=fit_encoder)
        if self.scale:
            data = self.scale_numeric_columns(data=data)

        if self.task not in self.task_cols:
            raise ValueError(f"Task '{self.task}' not supported.")

        if (
            self.task in ["improvement", "pocketclosureinf"]
            and "pdgroupbase" in data.columns
        ):
            data = data.query("pdgroupbase in [1, 2]")
            if self.task == "pocketclosureinf":
                self.task = "pocketclosure"

        cols_to_drop = [
            col for col in self.task_cols if col != self.task and col in data.columns
        ] + self.no_train_cols

        data = data.drop(columns=cols_to_drop, errors="ignore").rename(
            columns={self.task: "y"}
        )

        if "y" not in data.columns:
            raise ValueError(f"task column '{self.task}' was not renamed to 'y'.")

        return data
