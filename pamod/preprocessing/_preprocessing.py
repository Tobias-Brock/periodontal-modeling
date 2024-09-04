import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pamod.preprocessing import FunctionPreprocessor
from pamod.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
import os

import hydra
from omegaconf import DictConfig


class StaticProcessEngine:
    """
    A class used to preprocess periodontal dataset for machine learning.

    Attributes:
    ----------
    input_file : str
        Path to the input Excel file containing the dataset.
    behavior : bool
        Determines whether to include behavioral columns in the dataset.
    scale : bool
        Determines whether to perform scaling and encoding in the dataset.
    df : pd.DataFrame
        DataFrame containing the loaded and preprocessed data.
    required_columns : list
        A list of required columns for processing.
    behavior_columns : list
        A list of behavioral columns that can be optionally included in processing.
    function_preprocessor : FunctionPreprocessor
        An instance of the FunctionPreprocessor class to handle specific data transformations.
    """

    def __init__(self, behavior, scale, encoding):
        """
        Initializes the StaticProcessEngine with the given parameters, behavior flag, and loads the dataset.

        Parameters:
        ----------
        behavior : bool, optional
            If True, includes behavioral columns in processing (default is False).
        scale : bool, optional
            If True, performs scaling and encoding on the dataset (default is True).
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
            "PdRevaluation",  # remove in TargetClass
            "BOPRevaluation",
        ]
        self.behavior_columns = {
            "binary": [
                "Flossing",
                "IDB",
                "SweetFood",
                "SweetDrinks",
                "ErosiveDrinks",
            ],
            "categorical": ["OrthoddonticHistory", "DentalVisits", "Toothbrushing", "DryMouth"],
        }
        self.df = self._load_data()
        self.function_preprocessor = FunctionPreprocessor(self.df)

    def _load_data(self):
        """
        Loads data from the provided RAW_DATA_DIR, processes multi-index headers, and validates the required columns.

        Returns:
        -------
        pd.DataFrame
            The loaded DataFrame.

        Raises:
        ------
        ValueError
            If the input file is missing required columns.
        """
        input_file = os.path.join(RAW_DATA_DIR, "Periodontitis_ML_Dataset_Renamed.xlsx")

        # Load the data, but only use the second row as the header
        df = pd.read_excel(input_file, header=[1])  # Skip the first row, use the second row as headers

        actual_columns_lower = {col.lower(): col for col in df.columns}  # Mapping of lowercase to actual column names
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

        df = df[actual_required_columns]

        return df

    def _scale_and_encode(self):
        """
        Performs scaling of numeric variables and either one-hot or target encoding of categorical variables.

        Returns:
        -------
        pd.DataFrame
            The final processed DataFrame after scaling and encoding.

        Raises:
        -------
        ValueError
            If an invalid encoding type is provided.
        """
        # Scale numeric covariates (necessary for models)
        scale_vars = ["pdbaseline", "age", "bodymassindex", "recbaseline", "cigarettenumber"]
        self.df[scale_vars] = self.df[scale_vars].apply(pd.to_numeric, errors="coerce")
        scaler = StandardScaler()
        self.df[scale_vars] = scaler.fit_transform(self.df[scale_vars])

        if self.encoding == "one_hot":
            cat_vars = [
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
            if self.behavior:
                cat_vars += [col.lower() for col in self.behavior_columns["categorical"]]

            df_reset = self.df.reset_index(drop=True)
            df_reset[cat_vars] = df_reset[cat_vars].astype(str)
            encoder = OneHotEncoder(sparse_output=False)
            encoded_columns = encoder.fit_transform(df_reset[cat_vars])
            encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(cat_vars))
            df_final = pd.concat([df_reset.drop(cat_vars, axis=1), encoded_df], axis=1)

        elif self.encoding == "target":
            # Combine Tooth and Side into a new 'toothside' feature
            self.df["toothside"] = self.df["tooth"].astype(str) + "_" + self.df["side"].astype(str)
            df_final = self.df.drop(columns=["tooth", "side"])

        else:
            raise ValueError(f"Invalid encoding '{self.encoding}' specified. Choose either 'one_hot' or 'target'.")

        return df_final

    def _create_outcome_variables(self):
        """
        Adds outcome variables to the DataFrame: pocketclosure, pdgroup, and improve.
        """
        self.df.loc[:, "pocketclosure"] = self.df.apply(
            lambda row: 0 if row["pdbaseline"] == 4 and row["boprevaluation"] == 2 or row["pdbaseline"] > 4 else 1,
            axis=1,
        )
        self.df.loc[:, "pdgroup"] = self.df["pdbaseline"].apply(lambda x: 0 if x <= 3 else (1 if x in [4, 5] else 2))
        self.df.loc[:, "improve"] = (self.df["pdrevaluation"] < self.df["pdbaseline"]).astype(int)
        return self.df

    def process_data(self):
        """
        Processes the input dataset by performing data cleaning, imputations, and optional scaling/encoding.
        Returns:
        -------
        pd.DataFrame
            The processed DataFrame with or without scaling and encoding.
        """
        pd.set_option("future.no_silent_downcasting", True)
        self.df.columns = [col.lower() for col in self.df.columns]
        self.df = self.df[self.df["age"] >= 18].replace(" ", pd.NA)
        self.df.loc[:, "side_infected"] = self.df.apply(
            lambda row: self.function_preprocessor.check_infection(row["pdbaseline"], row["boprevaluation"]), axis=1
        )
        self.df.loc[:, "tooth_infected"] = (
            self.df.groupby(["id_patient", "tooth"])["side_infected"].transform(lambda x: (x == 1).any()).astype(int)
        )
        self.df = self.function_preprocessor.get_adjacent_infected_teeth_count(
            self.df, "id_patient", "tooth", "tooth_infected"
        )

        # Impute missing data
        self.df.loc[:, "recbaseline"] = self.df["recbaseline"].fillna(1).astype(float)
        self.df.loc[:, "bop"] = self.df["bop"].fillna(1).astype(float)
        self.df.loc[:, "percussion-sensitivity"] = self.df["percussion-sensitivity"].fillna(1).astype(float)
        self.df.loc[:, "sensitivity"] = self.df["sensitivity"].fillna(1).astype(float)
        self.df["bodymassindex"] = pd.to_numeric(self.df["bodymassindex"], errors="coerce")
        mean_bmi = self.df["bodymassindex"].mean()
        self.df.loc[:, "bodymassindex"] = self.df["bodymassindex"].fillna(mean_bmi).astype(float)
        self.df.loc[:, "periofamilyhistory"] = self.df["periofamilyhistory"].fillna(2).astype(int)
        self.df.loc[:, "smokingtype"] = self.df["smokingtype"].fillna(1).astype(int)
        self.df.loc[:, "cigarettenumber"] = self.df["cigarettenumber"].fillna(0).astype(float)
        self.df.loc[:, "diabetes"] = self.df["diabetes"].fillna(1).astype(int)

        # Impute stress levels
        self.df.loc[:, "stresslvl"] = self.df["stresslvl"] - 1
        self.df.loc[:, "stresslvl"] = pd.to_numeric(self.df["stresslvl"], errors="coerce")
        median_stress = self.df["stresslvl"].median()
        self.df.loc[:, "stresslvl"] = self.df["stresslvl"].fillna(median_stress).astype(float)

        # Map stress levels
        conditions_stress = [
            self.df["stresslvl"] <= 3,
            (self.df["stresslvl"] >= 4) & (self.df["stresslvl"] <= 6),
            self.df["stresslvl"] >= 7,
        ]
        choices_stress = ["low", "medium", "high"]
        self.df.loc[:, "stresslvl"] = np.select(conditions_stress, choices_stress, default="Not Specified")

        # Plaque and furcation imputations
        self.df = self.function_preprocessor.plaque_imputation()
        self.df = self.function_preprocessor.fur_imputation()
        self.df = self._create_outcome_variables()

        # Define and replace binary variables
        bin_var = [
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
        if self.behavior:
            bin_var += [col.lower() for col in self.behavior_columns["binary"]]
        self.df[bin_var] = self.df[bin_var].replace({1: 0, 2: 1})

        # Perform scaling and encoding if required
        if self.scale:
            self.df = self._scale_and_encode()

        return self.df

    def save_processed_data(self, file_path="processed_data.csv"):
        """
        Saves the processed DataFrame to a CSV file.
        Parameters:
        ----------
        file_path : str
            The path to save the CSV file.
        """
        processed_file_path = os.path.join(PROCESSED_DATA_DIR, file_path)
        self.df.to_csv(processed_file_path, index=False)


@hydra.main(config_path="../../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    engine = StaticProcessEngine(
        behavior=cfg.preprocess.behavior, scale=cfg.preprocess.scale, encoding=cfg.preprocess.encoding
    )
    engine.process_data()
    engine.save_processed_data()


if __name__ == "__main__":
    main()
