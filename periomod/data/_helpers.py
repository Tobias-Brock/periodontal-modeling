from typing import Union
import warnings

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


def _get_gender_map() -> dict:
    """Load the gender mapping for patient data.

    Returns:
        dict: A dictionary of gender mappings
    """
    gender_map = {0: "women", 1: "men"}
    return gender_map


def _get_rootlength_data() -> pd.DataFrame:
    """Load root length data from the specified CSV file.

    Returns:
        pd.DataFrame: DataFrame containing root length percentages.
    """
    rootlength = pd.read_csv("../config/data/root_length_percentages.csv")
    return rootlength


def _get_sideencoding() -> dict:
    """Load the side encoding mapping for tooth surfaces.

    Returns:
        dict: A dictionary mapping surface labels to their corresponding
            site indices.
    """
    sideencoding = {3: "m", 4: "m", 1: "d", 6: "d", 2: "b", 5: "o"}
    return sideencoding


def _get_occluding_teeth() -> set:
    """Return a set of frozensets representing symmetrical occluding tooth pairs."""
    pairs = [
        (17, 47),
        (16, 46),
        (15, 45),
        (14, 44),
        (13, 43),
        (12, 42),
        (11, 41),
        (21, 31),
        (22, 32),
        (23, 33),
        (24, 34),
        (25, 35),
        (26, 36),
        (27, 37),
    ]
    return {frozenset(pair) for pair in pairs}


def _get_adjacent_teeth() -> dict:
    """Get a dictionary mapping each tooth to its adjacent teeth.

    Returns:
        dict: A dictionary where keys are tooth numbers and values
            are lists of adjacent teeth.
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
    }


def _get_surface_label(site_index: int) -> str:
    """Get the surface label based on the site index.

    Args:
        site_index (int): The index of the site (1-6).

    Returns:
        str: The surface label "m", "d", "avg_md", or None

    Raises:
        ValueError: If the site_index is not valid.
    """
    sideencoding = _get_sideencoding()
    if site_index in sideencoding["m"]:
        return "m"
    elif site_index in sideencoding["d"]:
        return "d"
    elif site_index in sideencoding["b"] or site_index in sideencoding["o"]:
        return "avg_md"
    raise ValueError(f"Invalid site_index: {site_index}")


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
        data = helper.plaque_imputation(data)
        data = helper.fur_imputation(data)
        infected_count_data = helper.get_adjacent_infected_teeth_count(
            data, patient_col="id_patient", tooth_col="tooth",
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
        self, data: pd.DataFrame, patient_col: str, tooth_col: str, infection_col: str
    ) -> pd.DataFrame:
        """Adds a new column indicating the number of adjacent infected teeth.

        Args:
            data (pd.DataFrame): Dataset to process.
            patient_col (str): Name of column containing ID for patients.
            tooth_col (str): Name of column containing teeth represented in numbers.
            infection_col (str): Name of column indicating whether a tooth is healthy.

        Returns:
            pd.DataFrame: Modified dataset with new column 'infected_neighbors'.
        """
        for patient_id, patient_data in data.groupby(patient_col):
            infected_teeth = set(
                patient_data[patient_data[infection_col] == 1][tooth_col]
            )

            data.loc[data[patient_col] == patient_id, "infected_neighbors"] = (
                patient_data[tooth_col].apply(
                    lambda tooth, infected_teeth=infected_teeth: sum(
                        1
                        for neighbor in self._tooth_neighbor(nr=tooth)
                        if neighbor in infected_teeth
                    )
                )
            )

        return data

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

    def plaque_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Imputes values for Plaque without affecting other columns.

        Args:
            data (pd.DataFrame): Input DataFrame with a 'plaque' column.

        Returns:
            pd.DataFrame: The DataFrame with the imputed 'plaque' values.

        Raises:
            KeyError: If plaque column is not found in DataFrame.
        """
        data.columns = [col.lower() for col in data.columns]

        if "plaque" not in data.columns:
            raise KeyError("'plaque' column not found in the DataFrame")
        data["plaque"] = pd.to_numeric(data["plaque"], errors="coerce")

        conditions_baseline = [
            data["pdbaseline"] <= 3,
            (data["pdbaseline"] >= 4) & (data["pdbaseline"] <= 5),
            data["pdbaseline"] >= 6,
        ]
        choices_baseline = [0, 1, 2]
        data["pdbaseline_grouped"] = np.select(
            conditions_baseline, choices_baseline, default=-1
        )

        patients_with_all_nas = data.groupby(self.group_col)["plaque"].apply(
            lambda x: all(pd.isna(x))
        )
        data["plaque_all_na"] = data[self.group_col].isin(
            patients_with_all_nas[patients_with_all_nas].index
        )

        grouped_data = data.groupby(["tooth", "side", "pdbaseline_grouped"])

        modes_dict = {}
        for (tooth, side, baseline_grouped), group in grouped_data:
            modes = group["plaque"].mode()
            mode_value = modes.iloc[0] if not modes.empty else None
            modes_dict[(tooth, side, baseline_grouped)] = mode_value

        data["plaque"] = data.apply(
            lambda row: self._plaque_values(row=row, modes_dict=modes_dict), axis=1
        )

        data = data.drop(["pdbaseline_grouped", "plaque_all_na"], axis=1)

        return data

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

        Raises:
            ValueError: If NaN is found in pd- or recbaseline.
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

    def fur_imputation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute the values in the FurcationBaseline column.

        Args:
            data (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with imputed values for 'furcationbaseline'.

        Raises:
            KeyError: If furcationbaseline is not found in DatFrame.
        """
        if "furcationbaseline" not in data.columns:
            raise KeyError("'furcationbaseline' column not found in the DataFrame")

        patients_with_all_nas = data.groupby(self.group_col)["furcationbaseline"].apply(
            lambda x: all(pd.isna(x))
        )
        data["furcationbaseline_all_na"] = data[self.group_col].isin(
            patients_with_all_nas[patients_with_all_nas].index
        )

        data["furcationbaseline"] = data.apply(self._fur_values, axis=1)
        data = data.drop(["furcationbaseline_all_na"], axis=1)

        return data


class StageGradeExtentCalculator(ProcessDataHelper):
    """Class to calculate periodontal stages based on clinical data.

    This class extends the ProcessDataHelper to include methods for calculating
    periodontal stages based on clinical attachment loss (CAL) and other factors.

    Inherits:
        - `ProcessDataHelper`: Provides methods for data processing and
            imputation.

    Attributes:
        group_col (str): Column name used for grouping patient data.

    Methods:
        calculate_occluding_pairs: Calculates the number of occluding pairs of
            teeth per patient.
        calculate_cal: Calculates the Clinical Attachment Level (CAL)
            for each row.
    """

    def __init__(self):
        """Initialize PeriodontalStageGradeExtentCalculator with helper data."""
        super().__init__()
        self.rootlengthmap = _get_rootlength_data()
        self.sideencoding = _get_sideencoding()
        self.occluding_teeth = _get_occluding_teeth()
        self.surface_label = _get_surface_label
        self.gender_map = _get_gender_map()
        self.adjacent_pairs = _get_adjacent_teeth()

    def _calculate_occluding_pairs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the number of unique occluding pairs of teeth per patient.

        This method counts the number of unique occluding pairs of teeth for each
        patient based on a defined set of valid occluding pairs.

        Args:
            data (pd.DataFrame): DataFrame with "id_patient" and "tooth" columns.

        Returns:
            pd.DataFrame: Original DataFrame with a new 'occluding_pairs' column.
        """
        occluding_pairs_set = self.occluding_teeth()

        patient_teeth = (
            data[["id_patient", "tooth"]]
            .drop_duplicates()
            .groupby("id_patient")["tooth"]
            .agg(set)
        )

        def count_occlusions(teeth: set) -> int:
            seen = set()
            count = 0
            for t1 in teeth:
                for t2 in teeth:
                    if t1 == t2:
                        continue
                    pair = frozenset((t1, t2))
                    if pair in occluding_pairs_set and pair not in seen:
                        seen.add(pair)
                        count += 1
            return count

        occlusion_counts = patient_teeth.map(count_occlusions)

        data["occluding_pairs"] = data["id_patient"].map(occlusion_counts)
        return data

    @staticmethod
    def _calculate_missing_teeth_per_patient(data: pd.DataFrame) -> pd.DataFrame:
        """Vectorized calculation of the number of missing teeth per patient.

        This method counts the number of missing teeth for each patient,
        based on the FDI tooth numbering system, excluding wisdom teeth (18, 28,
        38, 48).

        Args:
            data (pd.DataFrame): DataFrame with columns "id_patient" and "tooth".

        Returns:
            pd.DataFrame: Same DataFrame with new column "missing_teeth".
        """
        fdi_teeth = [
            t
            for t in range(11, 49)
            if t % 10 not in [0, 8, 9] and t not in [18, 28, 38, 48]
        ]
        full_set = set(fdi_teeth)

        filtered_data = data[~data["tooth"].isin([18, 28, 38, 48])]

        patient_teeth = (
            filtered_data.drop_duplicates(subset=["id_patient", "tooth"])
            .groupby("id_patient")["tooth"]
            .agg(set)
        )

        missing_counts = patient_teeth.map(lambda present: len(full_set - present))

        all_ids = data["id_patient"].unique()
        missing_counts = missing_counts.reindex(all_ids, fill_value=len(full_set))

        data["missing_teeth"] = data["id_patient"].map(missing_counts)
        return data

    @staticmethod
    def _calculate_cal(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the Clinical Attachment Loss (CAL) for each row in the DataFrame.

        The formula used is:
            CAL = PD + REC if REC > 0 else PD - 3
        where PD is the probing depth and REC is the recession.
        If the calculated CAL is negative, it is set to 0.
        This method adds a new column "CAL" to the DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing patient data with columns
            "pdbaseline" and "recbaseline".

        Returns:
            pd.DataFrame: DataFrame with an additional column "CAL" indicating
            the Clinical Attachment Level.
        """
        cal_raw = np.where(
            data["recbaseline"] > 0,
            data["pdbaseline"] + data["recbaseline"],
            data["pdbaseline"] - 3,
        )
        data["CAL"] = np.maximum(cal_raw, 0)
        return data

    def _calculate_bone_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized calculation of bone loss percentage.

        Adds 'bone_loss_percentage' column to the input DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing patient data with columns:
                - "id_patient", "tooth", "side", "CAL",

        Returns:
            pd.DataFrame: DataFrame with an additional column
                "bone_loss_percentage".

        Raises:
            ValueError: If 'CAL' column is missing in the DataFrame.

        Warnings:
            UserWarning: If 'bone_loss_percentage' already exists in the DataFrame.
        """
        cdf = df.copy()

        if "bone_loss_percentage" in cdf.columns:
            warnings.warn(
                "Column 'bone_loss_percentage' already exists in DataFrame. "
                "Skipping bone loss calculation.",
                UserWarning,
                stacklevel=2,
            )
            return cdf

        if "CAL" not in cdf.columns:
            raise ValueError(
                "Missing required column 'CAL' in DataFrame. "
                "Please calculate Clinical Attachment Loss (CAL) before proceeding."
            )

        temp = cdf[["id_patient", "tooth", "side", "CAL", "gender"]].copy()
        temp["gender_mapped"] = temp["gender"].map(self.gender_map)
        temp["surface_mapped"] = temp["side"].map(self.sideencoding)

        rlm = self.rootlengthmap

        df_b = temp[temp["surface_mapped"].isin(["b", "o"])].copy()

        m_map = rlm[rlm["surface"] == "m"].rename(columns={"R": "R_m"})
        d_map = rlm[rlm["surface"] == "d"].rename(columns={"R": "R_d"})

        df_b = df_b.merge(
            m_map[["tooth", "gender", "R_m"]],
            left_on=["tooth", "gender_mapped"],
            right_on=["tooth", "gender"],
            how="left",
        ).merge(
            d_map[["tooth", "gender", "R_d"]],
            left_on=["tooth", "gender_mapped"],
            right_on=["tooth", "gender"],
            how="left",
        )
        df_b["root_length"] = (df_b["R_m"] + df_b["R_d"]) / 2

        df_md = temp[temp["surface_mapped"].isin(["m", "d"])].copy()
        r_map = rlm.rename(columns={"R": "root_length"})

        df_md = df_md.merge(
            r_map[["tooth", "gender", "surface", "root_length"]],
            left_on=["tooth", "gender_mapped", "surface_mapped"],
            right_on=["tooth", "gender", "surface"],
            how="left",
        )

        df_all = pd.concat([df_b, df_md], axis=0)
        df_all["bone_loss_percentage"] = (df_all["CAL"] / df_all["root_length"]) * 100
        df_all["bone_loss_percentage"] = df_all["bone_loss_percentage"].round(1)

        result = df.merge(
            df_all[["id_patient", "tooth", "side", "bone_loss_percentage"]],
            on=["id_patient", "tooth", "side"],
            how="left",
        )
        return result

    @staticmethod
    def _calculate_boneloss_per_age(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the bone loss per age for each row in the DataFrame.

        The formula used is: Bone Loss per Age = Bone Loss Percentage / Age

        Args:
            df (pd.DataFrame): DataFrame containing patient data with columns
                "bone_loss_percentage" and "age".

        Returns:
            pd.DataFrame: DataFrame with an additional column "bl/age"
                indicating the bone loss per age.

        Raises:
            ValueError: If 'bone_loss_percentage' column is missing in the DataFrame.
        """
        if "bone_loss_percentage" not in df.columns:
            raise ValueError(
                "Missing required column 'bone_loss_percentage' in DataFrame. "
                "Please calculate bone loss percentage before proceeding."
            )

        df["bl_per_age"] = df["bone_loss_percentage"] / df["age"]
        return df

    @staticmethod
    def _calculate_stage_per_site(row: pd.Series) -> int:
        """Calculates the periodontal stage for a given row.

        Args:
            row (pd.Series): A row from the DataFrame containing 'CAL' and
                'missing_teeth' columns.

        Returns:
            int: The periodontal stage:
                - 4 if CAL >= 5 and missing teeth >= 5
                - 3 if CAL >= 5 and missing teeth < 5
                - 2 if CAL in [3, 4]
                - 1 if CAL in [1, 2]
                - 0 otherwise
        """
        cal = row["CAL"]
        occluding_pairs = row.get("occluding_pairs", 0)
        furcation_involvement = row.get("furcationbaseline", 0)

        if cal >= 5 and occluding_pairs < 10:
            return 4
        elif cal >= 5 and occluding_pairs > 9:
            return 3
        elif cal in [3, 4]:
            if furcation_involvement > 1:
                return 3
            else:
                return 2
        elif cal in [1, 2]:
            return 1
        else:
            return 0

    def _calculate_stage_per_patient(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the periodontal stage for each patient based on tooth-level data.

        This method calculates the periodontal stage for each tooth and assigns
        the highest stage per patient. Staging follows 2018 EFP/AAP definitions,
        with an upgrade rule for Stage 2.

        Staging rules:
            - Stage 0: No periodontal disease
            - Stage 1: CAL 1–2 mm, no missing teeth
            - Stage 2: CAL 3–4 mm, no missing teeth
            - Stage 3: CAL ≥ 5 mm, <5 missing teeth
            - Stage 4: CAL ≥ 5 mm, ≥5 missing teeth
            - Upgrade Stage 2 → 3 if PPD ≥ 6 mm at ≥2 non-adjacent teeth

        Args:
            df (pd.DataFrame): DataFrame with columns:
                - id_patient, side, CAL, pdbaseline, tooth, occluding_pairs

        Raises:
            ValueError: If 'CAL' column is missing in the DataFrame.

        Returns:
            pd.DataFrame: Original DataFrame with:
                - 'tooth_stage': row-specific stage
                - 'max_stage': highest stage per patient
        """
        stage_df = df[df["side"].isin([1, 3, 4, 6])].copy()

        if "CAL" not in stage_df.columns:
            raise ValueError(
                "Missing required column 'CAL' in DataFrame. "
                "Please calculate Clinical Attachment Loss (CAL) before proceeding."
            )

        if "tooth_stage" not in stage_df.columns:
            stage_df["tooth_stage"] = stage_df.apply(
                self._calculate_stage_per_site, axis=1
            )

        stage_map = {}

        for patient_id, group in stage_df.groupby("id_patient"):
            max_stage = group["tooth_stage"].max()

            if max_stage == 2:
                ppd_teeth = set(group[group["pdbaseline"] >= 6]["tooth"].unique())
                non_adjacent_teeth = []

                for tooth in ppd_teeth:
                    if all(
                        tooth not in self.adjacent_pairs.get(other, [])
                        for other in ppd_teeth
                        if other != tooth
                    ):
                        non_adjacent_teeth.append(tooth)

                if len(non_adjacent_teeth) >= 2:
                    max_stage = 3

            stage_map[patient_id] = max_stage

        stagedf = df.copy()
        stagedf["tooth_stage"] = stagedf.apply(self._calculate_stage_per_site, axis=1)
        stagedf["max_stage"] = stagedf["id_patient"].map(stage_map)

        return stagedf

    def _calculate_grade_per_site(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized assignment of periodontal grade per row.

        This method calculates the periodontal grade based on baseline age,
        smoking status, and diabetes status. The grade is assigned as follows:

            - Grade A: bl_per_age < 0.25, cigarettenumber < 1, diabetes <= 1
            - Grade B: bl_per_age < 1, cigarettenumber < 10, diabetes <= 1, and
                not Grade A
            - Grade B: bl_per_age < 0.25, (cigarettenumber > 0 or diabetes > 1)
            - Grade C: otherwise

        Args:
            df (pd.DataFrame): Must contain 'bl_per_age', 'cigarettenumber', 'diabetes'.

        Returns:
            pd.DataFrame: With new 'grade' column.
        """
        cdf = df.copy()

        bl_age = cdf["bl_per_age"]
        smoke = cdf["cigarettenumber"]
        diabetes = cdf["diabetes"]

        grade = pd.Series(2, index=df.index)
        grade[(bl_age < 0.25) & (smoke < 1) & (diabetes <= 1)] = 0
        grade[(bl_age < 1) & (grade != 0) & (smoke < 10) & (diabetes <= 1)] = 1
        grade[(bl_age < 0.25) & ((smoke > 0) | (diabetes > 1))] = 1

        cdf["grade"] = grade
        return cdf

    def _calculate_grade_per_patient(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assigns the highest periodontal grade per patient.

        Args:
            df (pd.DataFrame): DataFrame containing patient-level data with
            'id_patient', 'side', and 'bl/age', 'cigarettenumber', 'diabetes'.

        Returns:
            pd.DataFrame: DataFrame with a new 'grade' column assigned per patient.
        """
        grade_df = df[df["side"].isin([1, 3, 4, 6])].copy()
        grade_df["grade_temp"] = grade_df.apply(self._calculate_grade_per_site, axis=1)
        max_grade_per_patient = grade_df.groupby("id_patient")["grade_temp"].max()

        cdf = df.copy()
        cdf["grade"] = cdf["id_patient"].map(max_grade_per_patient)
        return cdf

    @staticmethod
    def _calculate_extent(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the extent of periodontal disease for each patient.

        This method calculates the extent of periodontal disease based on the
        maximum stage of each tooth for each patient. The extent is defined as:
            - 1 if the percentage of teeth at the maximum stage is >= 30%
            - 0 otherwise

        Args:
            data (pd.DataFrame): DataFrame containing patient data with
                columns "id_patient", "side", "tooth_stage", "missing_teeth".

        Returns:
            pd.DataFrame: DataFrame with additional columns "extent" and
                "percent_max_stage".
        """
        extent_map = {}
        percent_map = {}
        df_side = data[data["side"].isin([1, 3, 4, 6])].copy()

        for patient_id in df_side["id_patient"].unique():
            patient_data = df_side[df_side["id_patient"] == patient_id]
            patient_tooth_data = patient_data.groupby("tooth").max(numeric_only=True)

            max_stage = patient_tooth_data["max_stage"].max()
            max_stage_count = (patient_tooth_data["tooth_stage"] == max_stage).sum()
            total_teeth = patient_tooth_data.shape[0]

            percent_max_stage = (
                (max_stage_count / total_teeth * 100) if total_teeth > 0 else 0
            )
            extent = int(percent_max_stage >= 30)

            extent_map[patient_id] = extent
            percent_map[patient_id] = percent_max_stage

        data["extent"] = data["id_patient"].map(extent_map)
        data["percent_max_stage"] = data["id_patient"].map(percent_map)
        return data

    def assign_stage_grade_extent(self, data) -> pd.DataFrame:
        """Assigns periodontal stage, grade, and extent to the DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing patient data

        Returns:
            pd.DataFrame: DataFrame with additional columns for stage,
                grade, and extent.
        """
        cdf = data.copy()
        cdf = self._calculate_missing_teeth_per_patient(cdf)
        cdf = self._calculate_cal(cdf)
        cdf = self._calculate_bone_loss(cdf)
        cdf = self._calculate_boneloss_per_age(cdf)
        cdf = self._calculate_stage_per_patient(cdf)
        cdf = self._calculate_grade_per_site(cdf)
        cdf = self._calculate_grade_per_patient(cdf)
        return cdf
