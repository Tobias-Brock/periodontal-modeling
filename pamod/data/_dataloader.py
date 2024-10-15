import os
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pamod.base import BaseData
from pamod.config import PROCESSED_BASE_DIR


class ProcessedDataLoader(BaseData):
    def __init__(self, target: str, encoding: str, scale: bool = True) -> None:
        """Initializes the ProcessedDataLoader with the specified target column.

        Args:
            target (str): The target column name.
            encoding (str): Specifies the encoding for categorical columns.
                Options: 'one_hot', 'target', or None.
            scale (bool): If True, performs scaling on numeric columns.
                Defaults to True.
        """
        super().__init__()
        self.scale = scale
        self.target = target
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
        df[self.scale_vars] = df[self.scale_vars].apply(pd.to_numeric, errors="coerce")
        scaler = StandardScaler()
        df[self.scale_vars] = scaler.fit_transform(df[self.scale_vars])
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
        """Select target column, rename to 'y', and delete remaining targets.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with the selected target 'y'.
        """
        df = self.encode_categorical_columns(df)
        self._check_encoded_columns(df)

        if self.scale:
            df = self.scale_numeric_columns(df)
            self._check_scaled_columns(df)

        if self.target not in df.columns:
            raise ValueError(
                f"Target column '{self.target}' not found in the DataFrame."
            )

        target_columns = ["pocketclosure", "pdgrouprevaluation", "improve"]

        columns_to_drop = [
            col for col in target_columns if col != self.target and col in df.columns
        ]
        df = df.drop(columns=columns_to_drop)
        df = df.drop(
            columns=["boprevaluation", "pdrevaluation", "pdgroup", "pdgroupbase"],
            errors="ignore",
        )
        if self.target == "improve":
            if "pdgroupbase" in df.columns:
                df = df.query("pdgroupbase == 0")
        df = df.rename(columns={self.target: "y"})

        if "y" not in df.columns:
            raise ValueError(f"Target column '{self.target}' was not renamed to 'y'.")

        if any(col in df.columns for col in columns_to_drop):
            raise ValueError(
                f"Error removing targets: Remaining columns: {columns_to_drop}"
            )

        return df
