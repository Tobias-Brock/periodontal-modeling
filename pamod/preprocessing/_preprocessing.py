import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pamod.preprocessing import FunctionPreprocessor
from pamod.config import RAW_DATA_DIR, PROCESSED_BASE_DIR, PROCESSED_BEHAVIOR_DIR
import os
import warnings

import hydra
from omegaconf import DictConfig


class StaticProcessEngine:
    """
    A class used to preprocess periodontal dataset for machine learning.
    """

    def __init__(self, behavior=False, scale=False, encoding=None):
        """
        Initializes the StaticProcessEngine with the given parameters, behavior flag, and options for scaling/encoding.

        Parameters:
        ----------
        behavior : bool, optional
            If True, includes behavioral columns in processing (default is False).
        scale : bool, optional
            If True, performs scaling on the dataset's numeric columns (default is False).
        encoding : str, optional
            Specifies the encoding type to apply to categorical columns ('one_hot', 'target', or None) (default is None).
        """
        self.behavior = behavior
        self.scale = scale
        self.encoding = encoding
        self.required_columns = [
            "ID_patient",
            "Tooth",
            "Toothtype",
            "RootNumber",
            "Mobility",
            "Restoration",
            "Percussion-sensitivity",
            "Sensitivity",
            "FurcationBaseline",
            "Side",
            "PdBaseline",
            "RecBaseline",
            "Plaque",
            "BOP",
            "Age",
            "Gender",
            "BodyMassIndex",
            "PerioFamilyHistory",
            "Diabetes",
            "SmokingType",
            "CigaretteNumber",
            "AntibioticTreatment",
            "Stresslvl",
            "PdRevaluation",
            "BOPRevaluation",
            "Pregnant",
        ]
        self.cat_vars = [
            "side",
            "restoration",
            "periofamilyhistory",
            "diabetes",
            "toothtype",
            "tooth",
            "furcationbaseline",
            "smokingtype",
            "stresslvl",
        ]
        self.bin_var = [
            "antibiotictreatment",
            "boprevaluation",
            "plaque",
            "bop",
            "mobility",
            "percussion-sensitivity",
            "sensitivity",
            "rootnumber",
            "gender",
        ]
        self.scale_vars = ["pdbaseline", "age", "bodymassindex", "recbaseline", "cigarettenumber"]
        self.behavior_columns = {
            "binary": ["Flossing", "IDB", "SweetFood", "SweetDrinks", "ErosiveDrinks"],
            "categorical": ["OrthoddonticHistory", "DentalVisits", "Toothbrushing", "DryMouth"],
        }

    def load_data(self, path=RAW_DATA_DIR, name="Periodontitis_ML_Dataset_Renamed.xlsx"):
        """
        Loads the dataset from the provided directory and validates required columns.

        Parameters:
        ----------
        path : str, optional
            The directory path where the dataset is located (default is RAW_DATA_DIR).
        name : str, optional
            The name of the dataset file (default is "Periodontitis_ML_Dataset_Renamed.xlsx").

        Returns:
        -------
        pd.DataFrame
            The loaded DataFrame with required columns.

        Raises:
        ------
        ValueError
            If the dataset is missing any required columns.
        """
        input_file = os.path.join(path, name)
        df = pd.read_excel(input_file, header=[1])  # Load the dataset using the second row as headers

        actual_columns_lower = {col.lower(): col for col in df.columns}
        required_columns_lower = [col.lower() for col in self.required_columns]

        missing_columns = [col for col in required_columns_lower if col not in actual_columns_lower]
        if missing_columns:
            raise ValueError(f"The following required columns are missing: {', '.join(missing_columns)}")

        actual_required_columns = [actual_columns_lower[col] for col in required_columns_lower]

        if self.behavior:
            behavior_columns_lower = [col.lower() for col in self.behavior_columns["binary"]] + [
                col.lower() for col in self.behavior_columns["categorical"]
            ]
            missing_behavior_columns = [col for col in behavior_columns_lower if col not in actual_columns_lower]
            if missing_behavior_columns:
                raise ValueError(f"The following behavior columns are missing: {', '.join(missing_behavior_columns)}")
            actual_required_columns += [actual_columns_lower[col] for col in behavior_columns_lower]

        return df[actual_required_columns]

    def _scale_numeric_columns(self, df):
        """
        Scales numeric columns in the DataFrame using StandardScaler.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing numeric columns to scale.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with scaled numeric columns.
        """
        df[self.scale_vars] = df[self.scale_vars].apply(pd.to_numeric, errors="coerce")
        scaler = StandardScaler()
        df[self.scale_vars] = scaler.fit_transform(df[self.scale_vars])

        return df

    def _encode_categorical_columns(self, df):
        """
        Encodes categorical columns in the DataFrame based on the specified encoding type.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame containing categorical columns to encode.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with encoded categorical columns.

        Raises:
        ------
        ValueError
            If an invalid encoding type is specified.
        """
        if self.encoding is None:
            return df

        elif self.encoding == "one_hot":
            if self.behavior:
                self.cat_vars += [col.lower() for col in self.behavior_columns["categorical"]]

            df_reset = df.reset_index(drop=True)
            df_reset[self.cat_vars] = df_reset[self.cat_vars].astype(str)
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_columns = encoder.fit_transform(df_reset[self.cat_vars])
            encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(self.cat_vars))
            df_final = pd.concat([df_reset.drop(self.cat_vars, axis=1), encoded_df], axis=1)

        elif self.encoding == "target":
            df["toothside"] = df["tooth"].astype(str) + "_" + df["side"].astype(str)
            df_final = df.drop(columns=["tooth", "side"])

        else:
            raise ValueError(f"Invalid encoding '{self.encoding}' specified. Choose 'one_hot', 'target', or None.")

        return df_final

    def _impute_missing_values(self, df):
        """
        Imputes missing values in the DataFrame.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame with missing values to impute.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        if df.isnull().values.any():
            missing_values = df.isnull().sum()
            warnings.warn(f"Missing values found in the following columns: \n{missing_values[missing_values > 0]}")

        df["boprevaluation"] = df["boprevaluation"].replace(["", "NA", "-", " "], np.nan).astype(float)
        df.loc[:, "boprevaluation"] = pd.to_numeric(df["boprevaluation"], errors="coerce")
        df.loc[:, "boprevaluation"] = df["boprevaluation"].fillna(1).astype(float)
        df.loc[:, "recbaseline"] = df["recbaseline"].fillna(1).astype(float)
        df.loc[:, "bop"] = df["bop"].fillna(1).astype(float)
        df.loc[:, "percussion-sensitivity"] = df["percussion-sensitivity"].fillna(1).astype(float)
        df.loc[:, "sensitivity"] = df["sensitivity"].fillna(1).astype(float)
        df["bodymassindex"] = pd.to_numeric(df["bodymassindex"], errors="coerce")
        mean_bmi = df["bodymassindex"].mean()
        df.loc[:, "bodymassindex"] = df["bodymassindex"].fillna(mean_bmi).astype(float)
        df.loc[:, "periofamilyhistory"] = df["periofamilyhistory"].fillna(2).astype(int)
        df.loc[:, "smokingtype"] = df["smokingtype"].fillna(1).astype(int)
        df.loc[:, "cigarettenumber"] = df["cigarettenumber"].fillna(0).astype(float)
        df.loc[:, "diabetes"] = df["diabetes"].fillna(1).astype(int)

        df.loc[:, "stresslvl"] = df["stresslvl"] - 1
        df.loc[:, "stresslvl"] = pd.to_numeric(df["stresslvl"], errors="coerce")
        median_stress = df["stresslvl"].median()
        df.loc[:, "stresslvl"] = df["stresslvl"].fillna(median_stress).astype(float)
        df["stresslvl"] = df["stresslvl"].astype(object)

        conditions_stress = [
            df["stresslvl"] <= 3,
            (df["stresslvl"] >= 4) & (df["stresslvl"] <= 6),
            df["stresslvl"] >= 7,
        ]
        choices_stress = ["low", "medium", "high"]
        df.loc[:, "stresslvl"] = np.select(conditions_stress, choices_stress, default="Not Specified")

        return df

    def _create_outcome_variables(self, df):
        """
        Adds outcome variables to the DataFrame: pocketclosure, pdgroup, and improve.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to which outcome variables are added.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with new outcome variables.
        """
        df.loc[:, "pocketclosure"] = df.apply(
            lambda row: 0 if row["pdbaseline"] == 4 and row["boprevaluation"] == 2 or row["pdbaseline"] > 4 else 1,
            axis=1,
        )
        df.loc[:, "pdgroupbase"] = df["pdbaseline"].apply(lambda x: 0 if x <= 3 else (1 if x in [4, 5] else 2))
        df.loc[:, "pdgrouprevaluation"] = df["pdrevaluation"].apply(
            lambda x: 0 if x <= 3 else (1 if x in [4, 5] else 2)
        )
        df.loc[:, "improve"] = (df["pdrevaluation"] < df["pdbaseline"]).astype(int)
        return df

    def _check_scaled_columns(self, df):
        """
        Verifies that all required columns are correctly scaled, one-hot encoded, and processed.
        """
        numeric_columns = ["pdbaseline", "age", "bodymassindex", "recbaseline", "cigarettenumber"]

        if self.scale:
            for col in numeric_columns:
                scaled_min = df[col].min()
                scaled_max = df[col].max()
                if scaled_min < -5 or scaled_max > 15:
                    raise ValueError(f"Column {col} is not correctly scaled.")

        print("All required columns are correctly processed and present in the DataFrame.")

    def _check_encoded_columns(self, df):
        """
        Verifies that the categorical columns were correctly one-hot encoded if encoding is 'one_hot'.
        Handles cases when encoding is 'target' or None.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to check for correct encoding of categorical columns.

        Raises:
        ------
        ValueError
            If the columns are not correctly one-hot encoded when 'one_hot' encoding is specified.
        """
        if self.encoding == "one_hot":
            if self.behavior:
                self.cat_vars += [col.lower() for col in self.behavior_columns["categorical"]]

            # Verify that original categorical columns are no longer present
            for col in self.cat_vars:
                if col in df.columns:
                    raise ValueError(
                        f"Column '{col}' was not correctly one-hot encoded and still exists in the DataFrame."
                    )

            # Verify that new one-hot encoded columns are present
            for col in self.cat_vars:
                matching_columns = [c for c in df.columns if c.startswith(f"{col}_")]
                if len(matching_columns) == 0:
                    raise ValueError(f"Column '{col}' does not have one-hot encoded columns in the DataFrame.")
            print("One-hot encoding was successful and all categorical columns were correctly encoded.")

        elif self.encoding == "target":
            if "toothside" not in df.columns:
                raise ValueError("'toothside' column was not correctly created during target encoding.")
            print("Target encoding was successful and 'toothside' column is present.")

        elif self.encoding is None:
            print("No encoding was applied, as None was selected.")

        else:
            raise ValueError(f"Invalid encoding '{self.encoding}' specified.")

    def process_data(self, df):
        """
        Processes the input dataset by performing data cleaning, imputations, scaling, and encoding.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to be processed.

        Returns:
        -------
        pd.DataFrame
            The processed DataFrame.
        """
        function_preprocessor = FunctionPreprocessor()
        pd.set_option("future.no_silent_downcasting", True)
        df.columns = [col.lower() for col in df.columns]
        df = df[df["age"] >= 18].replace(" ", pd.NA)
        df = df[df["pregnant"] != 2]
        df = df.drop(columns=["pregnant"])

        # Impute missing values
        df = self._impute_missing_values(df)
        df.loc[:, "side_infected"] = df.apply(
            lambda row: function_preprocessor.check_infection(row["pdbaseline"], row["boprevaluation"]), axis=1
        )
        df.loc[:, "tooth_infected"] = (
            df.groupby(["id_patient", "tooth"])["side_infected"].transform(lambda x: (x == 1).any()).astype(int)
        )

        df = function_preprocessor.get_adjacent_infected_teeth_count(df, "id_patient", "tooth", "tooth_infected")
        side_infected = df["side_infected"].copy()
        tooth_infected = df["tooth_infected"].copy()
        infected_neighbors = df["infected_neighbors"].copy()

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
            warnings.warn(f"Missing values found in the following columns: \n{missing_values[missing_values > 0]}")
            for col in df.columns:
                if df[col].isna().sum() > 0:
                    missing_patients = df[df[col].isna()]["id_patient"].unique().tolist()
                    print(f"Patients with missing {col}: {missing_patients}")
        else:
            print("No missing values present in the dataset after imputation")

        if self.scale:
            df = self._scale_numeric_columns(df)
            self._check_scaled_columns(df)

        df = self._encode_categorical_columns(df)
        self._check_encoded_columns(df)

        df["side_infected"] = side_infected
        df["tooth_infected"] = tooth_infected
        df["infected_neighbors"] = infected_neighbors

        return df

    def save_processed_data(self, df, file_path=None):
        """
        Saves the processed DataFrame to a CSV file.

        Parameters:
        ----------
        df : pd.DataFrame
            The DataFrame to be saved.
        file_path : str, optional
            The path to save the CSV file. Defaults to "processed_data.csv" or "processed_data_b.csv".
        """
        if df is None or df.empty:
            raise ValueError("Data must be processed and not empty before saving.")

        if self.behavior:
            if file_path is None:
                file_path = "processed_data_b.csv"
            processed_file_path = os.path.join(PROCESSED_BEHAVIOR_DIR, file_path)
        else:
            if file_path is None:
                file_path = "processed_data.csv"
            processed_file_path = os.path.join(PROCESSED_BASE_DIR, file_path)

        directory = os.path.dirname(processed_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        df.to_csv(processed_file_path, index=False)
        print(f"Data saved to {processed_file_path}")


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    engine = StaticProcessEngine(
        behavior=cfg.preprocess.behavior, scale=cfg.preprocess.scale, encoding=cfg.preprocess.encoding
    )
    df = engine.load_data()
    df = engine.process_data(df)
    engine.save_processed_data(df)


if __name__ == "__main__":
    main()
