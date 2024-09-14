import os

import pandas as pd

from pamod.config import PROCESSED_BASE_DIR


class ProcessedDataLoader:
    def __init__(self, target: str) -> None:
        """
        Initializes the ProcessedDataLoader with the specified target column.

        Args:
            target (str): The target column name.
        """
        self.target = target

    def load_data(
        self, path: str = PROCESSED_BASE_DIR, name: str = "processed_data.csv"
    ) -> pd.DataFrame:
        """
        Loads the processed data from the specified path.

        Args:
            path (str): Directory path for the processed data.
            name (str): File name for the processed data.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        input_file = os.path.join(path, name)
        return pd.read_csv(input_file)

    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by selecting the target column, renaming it to 'y',
        and removing other target columns and specific columns.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame with the selected target column renamed to 'y'.
        """
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in the DataFrame.")

        target_columns = ["pocketclosure", "pdgrouprevaluation", "improve"]

        columns_to_drop = [
            col for col in target_columns if col != self.target and col in df.columns
        ]
        df = df.drop(columns=columns_to_drop)
        df = df.drop(
            columns=["boprevaluation", "pdrevaluation", "pdgroup", "pdgroupbase"], errors="ignore"
        )
        df = df.rename(columns={self.target: "y"})

        if "y" not in df.columns:
            raise ValueError(f"Target column '{self.target}' was not renamed to 'y'.")

        if any(col in df.columns for col in columns_to_drop):
            raise ValueError(
                f"Not all unwanted target columns were successfully removed. Remaining columns: {columns_to_drop}"
            )

        return df
