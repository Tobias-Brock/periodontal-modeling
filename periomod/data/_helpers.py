from typing import Union

import numpy as np
import pandas as pd

from ..base import BaseConfig


def _get_side() -> dict:
    """Dict containing tooth-side combinations which should have furcation.

    Returns:
        dict: Dict mapping tooth numbers to corresponding sides with furcation.
    """
    return {
        (14, 24): [1, 3],
        (16, 17, 18, 26, 27, 28): [2, 4, 6],
        (36, 37, 38, 46, 47, 48): [2, 5],
    }


def _get_teeth_neighbors() -> dict:
    """Creates a dictionary assigning each tooth its neighbors.

    Returns:
        dict: Dict mapping tooth number to a list of neighboring tooth numbers.
    """
    return {
        11: [12, 21],
        12: [11, 13],
        13: [12, 14],
        14: [13, 15],
        15: [14, 16],
        16: [15, 17],
        17: [16, 18],
        18: [17],
        21: [11, 22],
        22: [21, 23],
        23: [22, 24],
        24: [23, 25],
        25: [24, 26],
        26: [25, 27],
        27: [26, 28],
        28: [27],
        31: [31, 41],
        32: [31, 33],
        33: [32, 34],
        34: [33, 35],
        35: [34, 36],
        36: [35, 37],
        37: [36, 38],
        38: [37],
        41: [31, 42],
        42: [41, 43],
        43: [42, 44],
        44: [43, 45],
        45: [44, 46],
        46: [45, 47],
        47: [46, 48],
        48: [47],
    }


class ProcessDataHelper(BaseConfig):
    """Helper class for processing periodontal data with utility methods.

    This class provides methods for evaluating tooth infection status,
    calculating adjacent infected teeth, and imputing values for 'plaque' and
    'furcationbaseline' columns based on predefined rules and conditions.

    Inherits:
        - `BaseConfig`: Provides configuration settings for data processing.

    Attributes:
        teeth_neighbors (dict): Dictionary mapping each tooth to its adjacent
            neighbors.
        sides_with_fur (dict): Dictionary specifying teeth with furcations and
            their respective sides.

    Methods:
        check_infection: Evaluates infection status based on pocket depth and
            BOP values.
        get_adjacent_infected_teeth_count: Adds a column to indicate the count
            of adjacent infected teeth for each tooth.
        plaque_imputation: Imputes values in the 'plaque' column.
        fur_imputation: Imputes values in the 'furcationbaseline' column.

    Example:
        ```
        helper = ProcessDataHelper()
        df = helper.plaque_imputation(df)
        df = helper.fur_imputation(df)
        infected_count_df = helper.get_adjacent_infected_teeth_count(
            df, patient_col="id_patient", tooth_col="tooth",
            infection_col="infection"
        )
        ```
    """

    def __init__(self):
        """Initialize Preprocessor with helper data without storing the DataFrame."""
        super().__init__()
        self.teeth_neighbors = _get_teeth_neighbors()
        self.sides_with_fur = _get_side()

    @staticmethod
    def check_infection(depth: int, boprevaluation: int) -> int:
        """Check if a given tooth side is infected.

        Args:
            depth: the depth of the pocket before the therapy
            boprevaluation: the value of BOP evaluation for the tooth side

        Returns:
            1 if the tooth side is infected, otherwise 0.
        """
        if depth > 4:
            return 1
        elif depth == 4 and boprevaluation == 2:
            return 1
        return 0

    def _tooth_neighbor(self, nr: int) -> Union[np.ndarray, str]:
        """Returns adjacent teeth for a given tooth.

        Args:
            nr (int): tooth number (11-48)

        Returns:
            Union[np.ndarray, str]: Array of adjacent teeth, or 'No tooth'
            if input is invalid.
        """
        return np.array(self.teeth_neighbors.get(nr, "No tooth"))

    def get_adjacent_infected_teeth_count(
        self, df: pd.DataFrame, patient_col: str, tooth_col: str, infection_col: str
    ) -> pd.DataFrame:
        """Adds a new column indicating the number of adjacent infected teeth.

        Args:
            df (pd.DataFrame): Dataset to process.
            patient_col (str): Name of column containing ID for patients.
            tooth_col (str): Name of column containing teeth represented in numbers.
            infection_col (str): Name of column indicating whether a tooth is healthy.

        Returns:
            pd.DataFrame: Modified dataset with new column 'infected_neighbors'.
        """
        for patient_id, patient_data in df.groupby(patient_col):
            infected_teeth = set(
                patient_data[patient_data[infection_col] == 1][tooth_col]
            )

            df.loc[df[patient_col] == patient_id, "infected_neighbors"] = patient_data[
                tooth_col
            ].apply(
                lambda tooth, infected_teeth=infected_teeth: sum(
                    1
                    for neighbor in self._tooth_neighbor(nr=tooth)
                    if neighbor in infected_teeth
                )
            )

        return df

    @staticmethod
    def _plaque_values(row: pd.Series, modes_dict: dict) -> int:
        """Calculate new values for the Plaque column.

        Args:
            row (pd.Series): A row from the DataFrame.
            modes_dict (dict): Dict mapping (tooth, side, pdbaseline_grouped)
            to the mode plaque value.

        Returns:
            int: Imputed plaque value for the given row.
        """
        if row["plaque_all_na"] == 1:
            key = (row["tooth"], row["side"], row["pdbaseline_grouped"])
            mode_value = modes_dict.get(key, None)
            if mode_value is not None:
                if isinstance(mode_value, tuple) and 2 in mode_value:
                    return 2
                elif mode_value == 1:
                    return 1
                elif mode_value == 2:
                    return 2
        else:
            if pd.isna(row["plaque"]):
                return 1
            else:
                return row["plaque"]
        return 1

    def plaque_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes values for Plaque without affecting other columns.

        Args:
            df (pd.DataFrame): Input DataFrame with a 'plaque' column.

        Returns:
            pd.DataFrame: The DataFrame with the imputed 'plaque' values.
        """
        df.columns = [col.lower() for col in df.columns]

        if "plaque" not in df.columns:
            raise KeyError("'plaque' column not found in the DataFrame")
        df["plaque"] = pd.to_numeric(df["plaque"], errors="coerce")

        conditions_baseline = [
            df["pdbaseline"] <= 3,
            (df["pdbaseline"] >= 4) & (df["pdbaseline"] <= 5),
            df["pdbaseline"] >= 6,
        ]
        choices_baseline = [0, 1, 2]
        df["pdbaseline_grouped"] = np.select(
            conditions_baseline, choices_baseline, default=-1
        )

        patients_with_all_nas = df.groupby(self.group_col)["plaque"].apply(
            lambda x: all(pd.isna(x))
        )
        df["plaque_all_na"] = df[self.group_col].isin(
            patients_with_all_nas[patients_with_all_nas].index
        )

        grouped_data = df.groupby(["tooth", "side", "pdbaseline_grouped"])

        modes_dict = {}
        for (tooth, side, baseline_grouped), group in grouped_data:
            modes = group["plaque"].mode()
            mode_value = modes.iloc[0] if not modes.empty else None
            modes_dict[(tooth, side, baseline_grouped)] = mode_value

        df["plaque"] = df.apply(
            lambda row: self._plaque_values(row=row, modes_dict=modes_dict), axis=1
        )

        df = df.drop(["pdbaseline_grouped", "plaque_all_na"], axis=1)

        return df

    def _fur_side(self, nr: int) -> Union[np.ndarray, str]:
        """Returns the sides for the input tooth that should have furcations.

        Args:
            nr (int): Tooth number.

        Returns:
            Union[np.ndarray, str]: Sides with furcations, or 'without Furkation'
            if not applicable.
        """
        for key, value in self.sides_with_fur.items():
            if nr in key:
                return np.array(value)
        return "Tooth without Furkation"

    def _fur_values(self, row: pd.Series) -> int:
        """Calculate values for the FurcationBaseline column.

        Args:
            row (pd.Series): A row from the DataFrame.

        Returns:
            int: Imputed value for furcationbaseline.
        """
        tooth_fur = [14, 16, 17, 18, 24, 26, 27, 28, 36, 37, 38, 46, 47, 48]
        if pd.isna(row["pdbaseline"]) or pd.isna(row["recbaseline"]):
            raise ValueError(
                "NaN found in pdbaseline or recbaseline. Check RecBaseline imputation."
            )

        if row["furcationbaseline_all_na"] == 1:
            if row["tooth"] in tooth_fur:
                if row["side"] in self._fur_side(nr=row["tooth"]):
                    if (row["pdbaseline"] + row["recbaseline"]) < 4:
                        return 0
                    elif 3 < (row["pdbaseline"] + row["recbaseline"]) < 6:
                        return 1
                    else:
                        return 2
                else:
                    return 0
            else:
                return 0
        else:
            if pd.isna(row["furcationbaseline"]):
                return 0
            else:
                return row["furcationbaseline"]

    def fur_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute the values in the FurcationBaseline column.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with imputed values for 'furcationbaseline'.
        """
        if "furcationbaseline" not in df.columns:
            raise KeyError("'furcationbaseline' column not found in the DataFrame")

        patients_with_all_nas = df.groupby(self.group_col)["furcationbaseline"].apply(
            lambda x: all(pd.isna(x))
        )
        df["furcationbaseline_all_na"] = df[self.group_col].isin(
            patients_with_all_nas[patients_with_all_nas].index
        )

        df["furcationbaseline"] = df.apply(self._fur_values, axis=1)
        df = df.drop(["furcationbaseline_all_na"], axis=1)

        return df
