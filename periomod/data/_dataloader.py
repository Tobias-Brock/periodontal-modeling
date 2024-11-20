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
        df = dataloader.load_data(path="data/processed/processed_data.csv")
        df = dataloader.transform_data(df=df)
        dataloader.save_data(df=df, path="data/training/training_data.csv")
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

    def encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing categorical columns.

        Returns:
            df: The DataFrame with encoded categorical columns.

        Raises:
            ValueError: If an invalid encoding type is specified.
        """
        if not self.encode:
            return df

        cat_vars = [col for col in self.all_cat_vars if col in df.columns]
        df[cat_vars] = df[cat_vars].apply(
            lambda col: col.astype(int) if col.dtype in [float, object] else col
        )

        if self.encoding == "one_hot":
            df_reset = df.reset_index(drop=True)
            df_reset[cat_vars] = df_reset[cat_vars].astype(str)
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_columns = encoder.fit_transform(df_reset[cat_vars])
            encoded_df = pd.DataFrame(
                encoded_columns, columns=encoder.get_feature_names_out(cat_vars)
            )
            df = pd.concat([df_reset.drop(cat_vars, axis=1), encoded_df], axis=1)
        elif self.encoding == "target":
            df["toothside"] = df["tooth"].astype(str) + "_" + df["side"].astype(str)
            df = df.drop(columns=["tooth", "side"])
        else:
            raise ValueError(
                f"Invalid encoding '{self.encoding}' specified. "
                "Choose 'one_hot', 'target', or None."
            )
        self._check_encoded_columns(df=df)
        return df

    def scale_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scales numeric columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing numeric columns.

        Returns:
            df: The DataFrame with scaled numeric columns.
        """
        scale_vars = [col for col in self.scale_vars if col in df.columns]
        df[scale_vars] = df[scale_vars].apply(pd.to_numeric, errors="coerce")
        scaled_values = StandardScaler().fit_transform(df[scale_vars])
        df[scale_vars] = pd.DataFrame(scaled_values, columns=scale_vars, index=df.index)
        self._check_scaled_columns(df=df)
        return df

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select task column and optionally, scale and encode.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            df: DataFrame with the selected task 'y'.
        """
        if self.encode:
            df = self.encode_categorical_columns(df=df)
        if self.scale:
            df = self.scale_numeric_columns(df=df)

        if self.task not in self.task_cols:
            raise ValueError(f"Task '{self.task}' not supported.")

        if (
            self.task in ["improvement", "pocketclosureinf"]
            and "pdgroupbase" in df.columns
        ):
            df = df.query("pdgroupbase in [1, 2]")
            if self.task == "pocketclosureinf":
                self.task = "pocketclosure"
        cols_to_drop = [
            col for col in self.task_cols if col != self.task and col in df.columns
        ] + self.no_train_cols

        df = df.drop(columns=cols_to_drop, errors="ignore").rename(
            columns={self.task: "y"}
        )

        if "y" not in df.columns:
            raise ValueError(f"task column '{self.task}' was not renamed to 'y'.")

        return df
