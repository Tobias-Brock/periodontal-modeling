import warnings

import numpy as np
import pandas as pd

from ._basedata import BaseProcessor
from ._helpers import ProcessDataHelper


class StaticProcessEngine(BaseProcessor):
    """Concrete implementation for preprocessing a periodontal dataset for ML.

    This class extends `BaseProcessor` and provides specific implementations
    for imputing missing values, creating tooth-related features, and generating
    outcome variables tailored for periodontal data analysis.

    Inherits:
        - BaseProcessor: Provides core data processing methods and abstract method
            definitions for required preprocessing steps.

    Args:
        behavior (bool): If True, includes behavioral columns in processing.
            Defaults to False.
        verbose (bool): Enables verbose logging of data processing steps if True.
            Defaults to True.

    Attributes:
        behavior (bool): Indicates whether to include behavior columns in processing.
        verbose (bool): Flag to enable or disable verbose logging.

    Methods:
        impute_missing_values: Impute missing values specifically for periodontal
          data.
        create_tooth_features: Generate tooth-related features, leveraging
          domain knowledge of periodontal data.
        create_outcome_variables: Create variables representing clinical outcomes.
        process_data: Execute a full processing pipeline including cleaning,
          imputing, scaling, and feature creation.

    Inherited Methods:
        - `load_data`: Load processed data from the specified path and file.
        - `save_data`: Save processed data to the specified path and file.

    Example:
        ```
        engine = StaticProcessEngine(behavior=True, verbose=True)
        df = engine.load_data()
        df = engine.process_data(df)
        engine.save_data(df)
        ```
    """

    def __init__(self, behavior: bool = False, verbose: bool = True) -> None:
        """Initializes the StaticProcessEngine.

        Args:
            behavior (bool): If True, includes behavioral columns in processing.
                Defaults to False.
            verbose (bool): Activates verbose. Defaults to True.
        """
        super().__init__(behavior=behavior)
        self.verbose = verbose
        self.helper = ProcessDataHelper()

    @staticmethod
    def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
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
            df["boprevaluation"]
            .replace(["", "NA", "-", " "], np.nan)
            .astype(float)
            .fillna(1)
            .astype(float)
        )
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

        df["stresslvl"] = (
            pd.to_numeric(df["stresslvl"] - 1, errors="coerce")
            .fillna(df["stresslvl"].median())
            .astype(float)
        )
        df["stresslvl"] = np.select(
            [
                df["stresslvl"] <= 3,
                (df["stresslvl"] >= 4) & (df["stresslvl"] <= 6),
                df["stresslvl"] >= 7,
            ],
            [0, 1, 2],
            default=-1,
        ).astype(int)

        return df

    def create_tooth_features(
        self, df: pd.DataFrame, neighbors: bool = True, patient_id: bool = True
    ) -> pd.DataFrame:
        """Creates side_infected, tooth_infected, and infected_neighbors columns.

        Args:
            df (pd.DataFrame): The input dataframe containing patient data.
            neighbors (bool): Compute the count of adjacent infected teeth.
                Defaults to True.
            patient_id (bool): Flag to indicate whether 'id_patient' is required
                when creating the 'tooth_infected' column. If True, 'id_patient' is
                included in the grouping; otherwise, it is not. Defaults to True.

        Returns:
            pd.DataFrame: The dataframe with additional tooth-related features.
        """
        df["side_infected"] = df.apply(
            lambda row: self.helper.check_infection(
                depth=row["pdbaseline"], boprevaluation=row["bop"]
            ),
            axis=1,
        )
        if patient_id:
            df["tooth_infected"] = (
                df.groupby([self.group_col, "tooth"])["side_infected"]
                .transform(lambda x: (x == 1).any())
                .astype(int)
            )
        else:
            df["tooth_infected"] = (
                df.groupby("tooth")["side_infected"]
                .transform(lambda x: (x == 1).any())
                .astype(int)
            )
        if neighbors:
            df = self.helper.get_adjacent_infected_teeth_count(
                df=df,
                patient_col=self.group_col,
                tooth_col="tooth",
                infection_col="tooth_infected",
            )

        return df

    @staticmethod
    def create_outcome_variables(df: pd.DataFrame) -> pd.DataFrame:
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

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes dataset with data cleaning, imputations and scaling.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        pd.set_option("future.no_silent_downcasting", True)
        df.columns = [col.lower() for col in df.columns]
        initial_patients = df["id_patient"].nunique()
        initial_rows = len(df)
        under_age_or_pregnant = df[(df["age"] < 18) | (df["pregnant"] == 2)]
        removed_patients = under_age_or_pregnant["id_patient"].nunique()
        removed_rows = len(under_age_or_pregnant)

        df = (
            df[df["age"] >= 18]
            .replace(" ", pd.NA)
            .loc[df["pregnant"] != 2]
            .drop(columns=["pregnant"])
        )

        remaining_patients = df["id_patient"].nunique()
        remaining_rows = len(df)
        if self.verbose:
            print(
                f"Initial number of patients: {initial_patients}, "
                f"Initial number of rows: {initial_rows}"
                f"Number of unique patients removed: {removed_patients}, "
                f"Number of rows removed: {removed_rows}"
                f"Remaining number of patients: {remaining_patients}, "
                f"Remaining number of rows: {remaining_rows}"
            )

        df = self.create_outcome_variables(
            self.create_tooth_features(self.impute_missing_values(df=df))
        )

        if self.behavior:
            self.bin_vars += [col.lower() for col in self.behavior_columns["binary"]]
        bin_vars = [col for col in self.bin_vars if col in df.columns]
        df[bin_vars] = df[bin_vars].replace({1: 0, 2: 1})

        df.replace(["", " "], np.nan, inplace=True)
        df = self.helper.fur_imputation(self.helper.plaque_imputation(df=df))

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
                    if self.verbose:
                        print(f"Patients with missing {col}: {missing_patients}")
        else:
            if self.verbose:
                print("No missing values after imputation.")

        return df
