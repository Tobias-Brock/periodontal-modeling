import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pamod.base import BaseHydra
from pamod.config import PROCESSED_BASE_DIR


class ProcessedDataLoader(BaseHydra):
    def __init__(
        self, task: str, encoding: str, encode: bool = True, scale: bool = True
    ) -> None:
        """Initializes the ProcessedDataLoader with the specified task column.

        Args:
            task (str): The task column name.
            encoding (str): Specifies the encoding for categorical columns.
                Options: 'one_hot', 'target', or None.
            encode (bool): If True, performs encodign on categorical columns.
                Defaults to True.
            scale (bool): If True, performs scaling on numeric columns.
                Defaults to True.
        """
        super().__init__()
        self.scale = scale
        self.task = task
        self.encode = encode
        self.encoding = encoding

    @staticmethod
    def load_data(
        path: Path = PROCESSED_BASE_DIR, name: str = "processed_data.csv"
    ) -> pd.DataFrame:
        """Loads the processed data from the specified path.

        Args:
            path (str): Directory path for the processed data.
            name (str): File name for the processed data.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        input_file = os.path.join(path, name)
        return pd.read_csv(input_file)

    def encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing categorical columns.

        Returns:
            pd.DataFrame: The DataFrame with encoded categorical columns.

        Raises:
            ValueError: If an invalid encoding type is specified.
        """
        if self.encoding is None:
            return df

        cat_vars = [col for col in self.all_cat_vars if col in df.columns]

        if self.encoding == "one_hot":
            df_reset = df.reset_index(drop=True)
            df_reset[cat_vars] = df_reset[cat_vars].astype(str)
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_columns = encoder.fit_transform(df_reset[cat_vars])
            encoded_df = pd.DataFrame(
                encoded_columns, columns=encoder.get_feature_names_out(cat_vars)
            )
            df_final = pd.concat([df_reset.drop(cat_vars, axis=1), encoded_df], axis=1)
        elif self.encoding == "target":
            df["toothside"] = df["tooth"].astype(str) + "_" + df["side"].astype(str)
            df_final = df.drop(columns=["tooth", "side"])
        else:
            raise ValueError(
                f"Invalid encoding '{self.encoding}' specified. "
                "Choose 'one_hot', 'target', or None."
            )

        return df_final

    def _check_encoded_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verifies that categorical columns were correctly one-hot encoded.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Raises:
            ValueError: If columns are not correctly encoded.
        """
        if self.encoding == "one_hot":
            cat_vars = [col for col in self.all_cat_vars if col in df.columns]

            for col in cat_vars:
                if col in df.columns:
                    raise ValueError(
                        f"Column '{col}' was not correctly one-hot encoded."
                    )
                matching_columns = [c for c in df.columns if c.startswith(f"{col}_")]
                if not matching_columns:
                    raise ValueError(f"No one-hot encoded columns for '{col}'.")
        elif self.encoding == "target":
            if "toothside" not in df.columns:
                raise ValueError("Target encoding for 'toothside' failed.")
        elif self.encoding is None:
            print("No encoding was applied.")
        else:
            raise ValueError(f"Invalid encoding '{self.encoding}'.")

    def scale_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scales numeric columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing numeric columns.

        Returns:
            pd.DataFrame: The DataFrame with scaled numeric columns.
        """
        scale_vars = [col for col in self.scale_vars if col in df.columns]
        df[scale_vars] = df[scale_vars].apply(pd.to_numeric, errors="coerce")
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[scale_vars])
        df[scale_vars] = pd.DataFrame(scaled_values, columns=scale_vars, index=df.index)

        return df

    def _check_scaled_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verifies that scaled columns are within expected ranges.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Raises:
            ValueError: If any columns are not correctly scaled.
        """
        if self.scale:
            for col in self.scale_vars:
                scaled_min = df[col].min()
                scaled_max = df[col].max()
                if scaled_min < -5 or scaled_max > 15:
                    raise ValueError(f"Column {col} is not correctly scaled.")

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select task column, rename to 'y', and delete remaining tasks.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with the selected task 'y'.
        """
        if self.encode:
            df = self.encode_categorical_columns(df)
            self._check_encoded_columns(df)

        if self.scale:
            df = self.scale_numeric_columns(df)
            self._check_scaled_columns(df)

        if self.task not in df.columns:
            raise ValueError(f"task column '{self.task}' not found in the DataFrame.")

        cols_to_drop = [
            col for col in self.task_cols if col != self.task and col in df.columns
        ]
        cols_to_drop.extend(self.no_train_cols)

        if self.task == "improve":
            if "pdgroupbase" in df.columns:
                df = df.query("pdgroupbase in [1, 2]")

        df = df.drop(columns=cols_to_drop, errors="ignore")
        df = df.rename(columns={self.task: "y"})

        if "y" not in df.columns:
            raise ValueError(f"task column '{self.task}' was not renamed to 'y'.")

        if any(col in df.columns for col in cols_to_drop):
            raise ValueError(f"Error removing tasks: Remaining columns: {cols_to_drop}")

        return df
