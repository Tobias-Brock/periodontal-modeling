import os
from pathlib import Path
from typing import Union
import warnings

import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pamod.base import BaseData
from pamod.config import PROCESSED_BASE_DIR, PROCESSED_BEHAVIOR_DIR, RAW_DATA_DIR
from pamod.data import FunctionPreprocessor


class StaticProcessEngine(BaseData):
    """Preprocesses periodontal dataset for machine learning."""

    def __init__(self, scale: bool, behavior: bool) -> None:
        """Initializes the StaticProcessEngine.

        Args:
            scale (bool): If True, performs scaling on numeric columns.
            behavior (bool): If True, includes behavioral columns in processing.
        """
        super().__init__()
        self.behavior = behavior
        self.scale = scale

    def load_data(
        self,
        path: Path = RAW_DATA_DIR,
        name: str = "Periodontitis_ML_Dataset_Renamed.xlsx",
    ) -> pd.DataFrame:
        """Loads the dataset and validates required columns.

        Args:
            path (str, optional): Directory where dataset is located.
                Defaults to RAW_DATA_DIR.
            name (str, optional): Dataset file name. Defaults to
                "Periodontitis_ML_Dataset_Renamed.xlsx".

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Raises:
            ValueError: If any required columns are missing.
        """
        input_file = os.path.join(path, name)
        df = pd.read_excel(input_file, header=[1])

        actual_columns_lower = {col.lower(): col for col in df.columns}
        required_columns_lower = [col.lower() for col in self.required_columns]

        missing_columns = [
            col for col in required_columns_lower if col not in actual_columns_lower
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        actual_required_columns = [
            actual_columns_lower[col] for col in required_columns_lower
        ]

        if self.behavior:
            behavior_columns_lower = [
                col.lower() for col in self.behavior_columns["binary"]
            ] + [col.lower() for col in self.behavior_columns["categorical"]]
            missing_behavior_columns = [
                col for col in behavior_columns_lower if col not in actual_columns_lower
            ]
            if missing_behavior_columns:
                raise ValueError(
                    f"Missing behavior columns: {', '.join(missing_behavior_columns)}"
                )
            actual_required_columns += [
                actual_columns_lower[col] for col in behavior_columns_lower
            ]

        return df[actual_required_columns]

    def _scale_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame with missing values.

        Returns:
            pd.DataFrame: The DataFrame with imputed missing values.
        """
        if df.isnull().values.any():
            missing_values = df.isnull().sum()
            warnings.warn(
                f"Missing values found: \n{missing_values[missing_values > 0]}",
                stacklevel=2,
            )

        df["boprevaluation"] = (
            df["boprevaluation"].replace(["", "NA", "-", " "], np.nan).astype(float)
        )
        df["boprevaluation"] = df["boprevaluation"].fillna(1).astype(float)
        df["recbaseline"] = df["recbaseline"].fillna(1).astype(float)
        df["bop"] = df["bop"].fillna(1).astype(float)
        df["percussion-sensitivity"] = (
            df["percussion-sensitivity"].fillna(1).astype(float)
        )
        df["sensitivity"] = df["sensitivity"].fillna(1).astype(float)
        df["bodymassindex"] = pd.to_numeric(df["bodymassindex"], errors="coerce")
        mean_bmi = df["bodymassindex"].mean()
        df["bodymassindex"] = df["bodymassindex"].fillna(mean_bmi).astype(float)
        df["periofamilyhistory"] = df["periofamilyhistory"].fillna(2).astype(int)
        df["smokingtype"] = df["smokingtype"].fillna(1).astype(int)
        df["cigarettenumber"] = df["cigarettenumber"].fillna(0).astype(float)
        df["diabetes"] = df["diabetes"].fillna(1).astype(int)

        df["stresslvl"] = df["stresslvl"] - 1
        df["stresslvl"] = pd.to_numeric(df["stresslvl"], errors="coerce")
        median_stress = df["stresslvl"].median()
        df["stresslvl"] = df["stresslvl"].fillna(median_stress).astype(float)
        df["stresslvl"] = df["stresslvl"].astype(object)

        conditions_stress = [
            df["stresslvl"] <= 3,
            (df["stresslvl"] >= 4) & (df["stresslvl"] <= 6),
            df["stresslvl"] >= 7,
        ]
        choices_stress = ["low", "medium", "high"]
        df["stresslvl"] = np.select(
            conditions_stress, choices_stress, default="Not Specified"
        )

        return df

    def _create_outcome_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds outcome variables to the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with new outcome variables.
        """
        df["pocketclosure"] = df.apply(
            lambda row: (
                0
                if row["pdrevaluation"] == 4
                and row["boprevaluation"] == 2
                or row["pdrevaluation"] > 4
                else 1
            ),
            axis=1,
        )
        df["pdgroupbase"] = df["pdbaseline"].apply(
            lambda x: 0 if x <= 3 else (1 if x in [4, 5] else 2)
        )
        df["pdgrouprevaluation"] = df["pdrevaluation"].apply(
            lambda x: 0 if x <= 3 else (1 if x in [4, 5] else 2)
        )
        df["improve"] = (df["pdrevaluation"] < df["pdbaseline"]).astype(int)
        return df

    def _check_scaled_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Verifies that scaled columns are within expected ranges.

        Args:
            df (pd.DataFrame): The DataFrame to check.

        Raises:
            ValueError: If any columns are not correctly scaled.
        """
        numeric_columns = [
            "pdbaseline",
            "age",
            "bodymassindex",
            "recbaseline",
            "cigarettenumber",
        ]

        if self.scale:
            for col in numeric_columns:
                scaled_min = df[col].min()
                scaled_max = df[col].max()
                if scaled_min < -5 or scaled_max > 15:
                    raise ValueError(f"Column {col} is not correctly scaled.")
        print("All required columns are correctly processed and present.")

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes dataset with data cleaning, imputations and scaling.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        function_preprocessor = FunctionPreprocessor()
        pd.set_option("future.no_silent_downcasting", True)
        df.columns = [col.lower() for col in df.columns]
        initial_patients = df["id_patient"].nunique()
        initial_rows = len(df)
        print(f"Initial number of patients: {initial_patients}")
        print(f"Initial number of rows: {initial_rows}")

        under_age_or_pregnant = df[(df["age"] < 18) | (df["pregnant"] == 2)]
        removed_patients = under_age_or_pregnant["id_patient"].nunique()
        removed_rows = len(under_age_or_pregnant)
        print(f"Number of unique patients removed: {removed_patients}")
        print(f"Number of rows removed: {removed_rows}")

        df = df[df["age"] >= 18].replace(" ", pd.NA)
        df = df[df["pregnant"] != 2]
        df = df.drop(columns=["pregnant"])

        remaining_patients = df["id_patient"].nunique()
        remaining_rows = len(df)
        print(f"Remaining number of patients: {remaining_patients}")
        print(f"Remaining number of rows: {remaining_rows}")

        df = self._impute_missing_values(df)
        df["side_infected"] = df.apply(
            lambda row: function_preprocessor.check_infection(
                row["pdbaseline"], row["bop"]
            ),
            axis=1,
        )
        df["tooth_infected"] = (
            df.groupby(["id_patient", "tooth"])["side_infected"]
            .transform(lambda x: (x == 1).any())
            .astype(int)
        )

        df = function_preprocessor.get_adjacent_infected_teeth_count(
            df, "id_patient", "tooth", "tooth_infected"
        )

        df = self._create_outcome_variables(df)

        if self.behavior:
            self.bin_var += [col.lower() for col in self.behavior_columns["binary"]]
        df[self.bin_var] = df[self.bin_var].replace({1: 0, 2: 1})

        df.replace("", np.nan, inplace=True)
        df.replace(" ", np.nan, inplace=True)

        df = function_preprocessor.fur_imputation(df)
        df = function_preprocessor.plaque_imputation(df)

        if df.isnull().values.any():
            missing_values = df.isnull().sum()
            warnings.warn(
                f"Missing values: \n{missing_values[missing_values > 0]}", stacklevel=2
            )
            for col in df.columns:
                if df[col].isna().sum() > 0:
                    missing_patients = (
                        df[df[col].isna()]["id_patient"].unique().tolist()
                    )
                    print(f"Patients with missing {col}: {missing_patients}")
        else:
            print("No missing values after imputation.")

        if self.scale:
            df = self._scale_numeric_columns(df)
            self._check_scaled_columns(df)

        return df

    def save_processed_data(
        self, df: pd.DataFrame, file: Union[str, None] = None
    ) -> None:
        """Saves the processed DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The processed DataFrame.
            file (str, optional): The file path to save the CSV. Defaults to
                "processed_data.csv" or "processed_data_b.csv".
        """
        if df.empty:
            raise ValueError("Data must be processed before saving.")

        if self.behavior:
            file = file or "processed_data_b.csv"
            processed_file_path = os.path.join(PROCESSED_BEHAVIOR_DIR, file)
        else:
            file = file or "processed_data.csv"
            processed_file_path = os.path.join(PROCESSED_BASE_DIR, file)

        directory = os.path.dirname(processed_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        df.to_csv(processed_file_path, index=False)
        print(f"Data saved to {processed_file_path}")


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    engine = StaticProcessEngine(behavior=cfg.data.behavior, scale=cfg.data.scale)
    df = engine.load_data()
    df = engine.process_data(df)
    engine.save_processed_data(df)


if __name__ == "__main__":
    main()
