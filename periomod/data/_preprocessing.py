from typing import Tuple
import warnings

import numpy as np
import pandas as pd

from ._basedata import BaseProcessor
from ._helpers import ProcessDataHelper


def _impute_tooth_features(row: pd.Series) -> Tuple[int, int]:
    """Determines the toothtype and rootnumber based on the tooth number.

    Args:
        row (pd.Series): A row from the DataFrame containing a 'tooth' column.

    Returns:
        Tuple: A tuple with imputed values for 'toothtype' and
            'rootnumber'. Returns (0, 0) for certain incisors and canines,
            (1, 1) for specific premolars, and (2, 1) for molars and others.
    """
    tooth_number = row["tooth"]
    if tooth_number in [11, 12, 21, 22, 31, 32, 41, 42, 13, 23, 33, 43]:
        return 0, 0
    elif tooth_number in [14, 15, 24, 25, 34, 35, 44, 45]:
        return 1, 1
    else:
        return 2, 1


class StaticProcessEngine(BaseProcessor):
    """Concrete implementation for preprocessing a periodontal dataset for ML.

    This class extends `BaseProcessor` and provides specific implementations
    for imputing missing values, creating tooth-related features, and generating
    outcome variables tailored for periodontal data analysis.

    Inherits:
        - `BaseProcessor`: Provides core data processing methods and abstract method
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
        impute_missing_values: Impute missing values specifically for
            periodontal data.
        create_tooth_features: Generate tooth-related features, leveraging
            domain knowledge of periodontal data.
        create_outcome_variables: Create variables representing clinical
            outcomes.
        process_data: Execute a full processing pipeline including cleaning,
            imputing, scaling, and feature creation.

    Inherited Methods:
        - `load_data`: Load processed data from the specified path and file.
        - `save_data`: Save processed data to the specified path and file.

    Example:
        ```
        from periomod.data import StaticProcessEngine

        engine = StaticProcessEngine()
        df = engine.load_data(path="data/raw/raw_data.xlsx")
        df = engine.process_data(df)
        engine.save_data(df=df, path="data/processed/processed_data.csv")
        ```
    """

    def __init__(self, behavior: bool = False, verbose: bool = True) -> None:
        """Initializes the StaticProcessEngine."""
        super().__init__(behavior=behavior)
        self.verbose = verbose
        self.helper = ProcessDataHelper()

    @staticmethod
    def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Imputes missing values in the DataFrame.

        Imputation rules exist for a predefined set of variables.
        The method will only impute the columns present in the dataframe.

        Args:
            df (pd.DataFrame): The DataFrame with missing values.

        Returns:
            df: The DataFrame with imputed missing values.
        """
        pd.set_option("future.no_silent_downcasting", True)
        if df.isnull().values.any():
            missing_values = df.isnull().sum()
            warnings.warn(
                f"Missing values found: \n{missing_values[missing_values > 0]}",
                stacklevel=2,
            )

        imputation_rules = {
            "boprevaluation": lambda x: x.replace(["", "NA", "-", " "], np.nan)
            .fillna(1)
            .astype(float),
            "recbaseline": lambda x: x.fillna(1).astype(float),
            "bop": lambda x: x.fillna(1).astype(float),
            "percussion-sensitivity": lambda x: x.fillna(1).astype(float),
            "sensitivity": lambda x: x.fillna(1).astype(float),
            "bodymassindex": lambda x: pd.to_numeric(x, errors="coerce")
            .fillna(pd.to_numeric(x, errors="coerce").mean())
            .astype(float),
            "periofamilyhistory": lambda x: x.fillna(2).astype(int),
            "restoration": lambda x: x.fillna(0).astype(int),
            "smokingtype": lambda x: x.fillna(1).astype(int),
            "cigarettenumber": lambda x: x.fillna(0).astype(float),
            "diabetes": lambda x: x.fillna(1).astype(int),
            "stresslvl": lambda x: np.select(
                [
                    (x - 1).fillna(x.median()).astype(float) <= 3,
                    ((x - 1).fillna(x.median()).astype(float) >= 4)
                    & ((x - 1).fillna(x.median()).astype(float) <= 6),
                    (x - 1).fillna(x.median()).astype(float) >= 7,
                ],
                [0, 1, 2],
                default=-1,
            ).astype(int),
        }

        for column, impute_func in imputation_rules.items():
            if column in df.columns:
                df[column] = impute_func(df[column])
            else:
                warnings.warn(
                    f"Column '{column}' is missing from DataFrame and was not imputed.",
                    stacklevel=2,
                )
        missing_or_incorrect_tooth_rows = df[
            df["toothtype"].isnull()
            | df["rootnumber"].isnull()
            | df.apply(
                lambda row: (row["toothtype"], row["rootnumber"])
                != _impute_tooth_features(row),
                axis=1,
            )
        ]

        df.loc[missing_or_incorrect_tooth_rows.index, ["toothtype", "rootnumber"]] = (
            missing_or_incorrect_tooth_rows.apply(
                lambda row: _impute_tooth_features(row), axis=1
            ).tolist()
        )

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
            df: The dataframe with additional tooth-related features.
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
            df: The DataFrame with new outcome variables.
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
        df["improvement"] = (df["pdrevaluation"] < df["pdbaseline"]).astype(int)
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes dataset with data cleaning, imputation and transformation.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            df: The imputed Dataframe with added feature and target columns.
        """
        pd.set_option("future.no_silent_downcasting", True)
        df.columns = [col.lower() for col in df.columns]
        initial_patients = df[self.group_col].nunique()
        initial_rows = len(df)
        if "age" in df.columns and "pregnant" in df.columns:
            under_age_or_pregnant = df[(df["age"] < 18) | (df["pregnant"] == 2)]
            removed_patients = under_age_or_pregnant[self.group_col].nunique()
            removed_rows = len(under_age_or_pregnant)

            df = (
                df[df["age"] >= 18]
                .replace(" ", pd.NA)
                .loc[df["pregnant"] != 2]
                .drop(columns=["pregnant"])
            )
        else:
            warnings.warn(
                "Columns 'age' and/or 'pregnant' missing from the dataset.",
                stacklevel=2,
            )
            removed_patients = removed_rows = 0

        if self.verbose:
            print(
                f"Initial number of patients: {initial_patients}\n"
                f"Initial number of rows: {initial_rows}\n"
                f"Number of unique patients removed: {removed_patients}\n"
                f"Number of rows removed: {removed_rows}\n"
                f"Remaining number of patients: {df[self.group_col].nunique()}\n"
                f"Remaining number of rows: {len(df)}\n"
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
                        df[df[col].isna()][self.group_col].unique().tolist()
                    )
                    if self.verbose:
                        print(f"Patients with missing {col}: {missing_patients}")
        else:
            if self.verbose:
                print("No missing values after imputation.")

        return df
