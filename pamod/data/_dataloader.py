import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..data import BaseDataLoader


class ProcessedDataLoader(BaseDataLoader):
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
        super().__init__(task=task, encoding=encoding, encode=encode, scale=scale)

    def encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical columns in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing categorical columns.

        Returns:
            pd.DataFrame: The DataFrame with encoded categorical columns.

        Raises:
            ValueError: If an invalid encoding type is specified.
        """
        if not self.encode:
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
        self._check_encoded_columns(df=df_final)
        return df_final

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
        self._check_scaled_columns(df=df)
        return df

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select task column, rename to 'y', and delete remaining tasks.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with the selected task 'y'.
        """
        if self.encode:
            df = self.encode_categorical_columns(df=df)
        if self.scale:
            df = self.scale_numeric_columns(df=df)

        if self.task not in df.columns:
            raise ValueError(f"task column '{self.task}' not found in the DataFrame.")

        cols_to_drop = [
            col for col in self.task_cols if col != self.task and col in df.columns
        ]
        cols_to_drop.extend(self.no_train_cols)

        if self.task == "improve":
            if "pdgroupbase" in df.columns:
                df = df.query("pdgroupbase in [1, 2]")

        df = df.drop(columns=cols_to_drop, errors="ignore").rename(
            columns={self.task: "y"}
        )

        if "y" not in df.columns:
            raise ValueError(f"task column '{self.task}' was not renamed to 'y'.")

        if any(col in df.columns for col in cols_to_drop):
            raise ValueError(f"Error removing tasks: Remaining columns: {cols_to_drop}")

        return df
