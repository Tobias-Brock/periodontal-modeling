import pandas as pd


class InputProcessor:
    """Convert input values to internal code-compatible formats.

    This class provides mappings and processing methods to convert user-friendly
    input values into formats that are compatible with the internal code of the
    periodontal modeling application. This ensures standardized and consistent
    processing of values for various attributes such as task type, learner type,
    tuning methods, HPO methods, and clinical features.

    Attributes:
        task_map (dict): Maps human-readable task names to internal codes.
        learner_map (dict): Maps learner names to internal codes for model selection.
        tuning_map (dict): Maps tuning method names to their respective codes.
        hpo_map (dict): Maps hyperparameter optimization methods to internal codes.
        criteria_map (dict): Maps evaluation criteria names to standardized codes.
        encodings_map (dict): Maps encoding names to their respective codes.
        gender_map (dict): Maps gender input to binary values.
        antibiotics_map (dict): Maps antibiotic treatment values to binary format.
        stresslvl_map (dict): Maps stress level categories to numerical codes.
        periohistory_map (dict): Maps periodontal family history values to codes.
        diabetes_map (dict): Maps diabetes types to numerical codes.
        smokingtype_map (dict): Maps smoking type values to integer codes.
        mobility_map (dict): Maps mobility status to binary values.
        percussion_map (dict): Maps percussion sensitivity to binary values.
        sensitivity_map (dict): Maps sensitivity status to binary values.
        restoration_map (dict): Maps restoration types to integer codes.
        furcation_map (dict): Maps furcation involvement levels to integer codes.
        plaque_map (dict): Maps plaque presence to binary values.
        bop_map (dict): Maps BOP (Bleeding on Probing) values to binary values.

    Methods:
        - `process_task(str) -> str`: Converts a task name to its internal code.
        - `process_learners(list) -> list`: Converts learner names to internal codes.
        - `process_tuning(list) -> list`: Converts tuning method names to codes.
        - `process_hpo(list) -> list`: Converts HPO methods to internal codes.
        - `process_criteria(list) -> list`: Converts criteria names to codes.
        - `process_encoding(str) -> str`: Converts encoding types to internal codes.
        - `process_antibiotics(str) -> int`: Converts antibiotic treatment to code.
        - `process_gender(str) -> int`: Converts gender to binary code.
        - `process_stresslvl(str) -> int`: Converts stress level to numerical code.
        - `process_diabetes(str) -> int`: Converts diabetes type to code.
        - `process_periohistory(str) -> int`: Converts family history to code.
        - `process_smokingtype(str) -> int`: Converts smoking type to code.
        - `process_mobility(str) -> int`: Converts mobility status to binary code.
        - `process_percussion(str) -> int`: Converts percussion sensitivity to code.
        - `process_sensitivity(str) -> int`: Converts sensitivity status to code.
        - `process_restoration(str) -> int`: Converts restoration type to code.
        - `process_furcation(str) -> int`: Converts furcation level to code.
        - `process_plaque(str) -> int`: Converts plaque presence to binary code.
        - `process_bop(str) -> int`: Converts BOP presence to binary code.
        - `transform_predictions(str, pd.DataFrame) -> pd.DataFrame`: Transforms raw
            model predictions to a more user-friendly format based on the task.

    Example Usage:
        ```
        # Example input processing for clinical data
        task = InputProcessor.process_task("Pocket closure")
        learners = InputProcessor.process_learners(["XGBoost", "Random Forest"])
        tuning_methods = InputProcessor.process_tuning(["Holdout", "Cross-Validation"])
        hpo_methods = InputProcessor.process_hpo(["HEBO"])
        criteria = InputProcessor.process_criteria(["F1 Score", "Brier Score"])
        encodings = InputProcessor.process_encoding(["One-hot"])
        gender = InputProcessor.process_gender("Male")
        antibiotics = InputProcessor.process_antibiotics("yes")
        stress_level = InputProcessor.process_stresslvl("high")

        print(task)          # Output: "pocketclosure"
        print(learners)      # Output: ["xgb", "rf"]
        print(tuning_methods) # Output: ["holdout", "cv"]
        print(hpo_methods)    # Output: ["hebo"]
        print(criteria)       # Output: ["f1", "brier_score"]
        print(encodings)      # Output: ["one_hot"]
        print(gender)         # Output: 1
        print(antibiotics)    # Output: 1
        print(stress_level)   # Output: 2
        ```
    """

    task_map = {
        "Pocket closure": "pocketclosure",
        "Pocket closure PdBaseline > 3": "pocketclosureinf",
        "Pocket improvement": "improvement",
        "Pocket groups": "pdgrouprevaluation",
    }

    learner_map = {
        "XGBoost": "xgb",
        "Random Forest": "rf",
        "Logistic Regression": "lr",
        "Multilayer Perceptron": "mlp",
    }

    tuning_map = {
        "Holdout": "holdout",
        "Cross-Validation": "cv",
    }

    hpo_map = {
        "HEBO": "hebo",
        "Random Search": "rs",
    }

    criteria_map = {
        "F1 Score": "f1",
        "Brier Score": "brier_score",
        "Macro F1 Score": "macro_f1",
    }

    encodings_map = {
        "One-hot": "one_hot",
        "Target": "target",
    }

    gender_map = {
        "Male": 1,
        "Female": 0,
    }

    antibiotics_map = {
        "yes": 1,
        "no": 0,
    }

    stresslvl_map = {
        "low": 0,
        "mid": 1,
        "high": 2,
    }

    periohistory_map = {
        "no": 0,
        "unknown": 1,
        "yes": 2,
    }

    diabetes_map = {
        "no": 0,
        "Type I": 1,
        "Type II": 2,
        "Type II med.": 3,
    }

    smokingtype_map = {
        "no": 0,
        "Cigarette": 1,
        "Pipe": 2,
        "Cigarillo": 3,
        "all": 4,
    }

    mobility_map = {
        "yes": 1,
        "no": 0,
    }

    percussion_map = {
        "yes": 1,
        "no": 0,
    }

    sensitivity_map = {
        "yes": 1,
        "no": 0,
    }

    restoration_map = {
        "no": 0,
        "Filling": 1,
        "Crown": 2,
    }

    furcation_map = {
        "no": 0,
        "Palpable": 1,
        "1-3 mm": 2,
        ">3 mm": 3,
    }

    plaque_map = {
        "yes": 1,
        "no": 0,
    }

    bop_map = {
        "yes": 1,
        "no": 0,
    }

    @classmethod
    def process_task(cls, task: str) -> str:
        return cls.task_map.get(task, task)

    @classmethod
    def process_learners(cls, learners: list) -> list:
        return [cls.learner_map[learner] for learner in learners]

    @classmethod
    def process_tuning(cls, tuning_method: str) -> list:
        """Processes a single tuning method string using the tuning_map."""
        return [cls.tuning_map[tuning_method]]

    @classmethod
    def process_hpo(cls, hpo_method: str) -> list:
        """Processes a single HPO method string using the hpo_map."""
        return [cls.hpo_map[hpo_method]]

    @classmethod
    def process_criteria(cls, criterion: str) -> list:
        """Processes a single criterion string using the criteria_map."""
        return [cls.criteria_map[criterion]]

    @classmethod
    def process_encoding(cls, encoding: str) -> str:
        """Processes a single encoding string using the encodings_map."""
        return cls.encodings_map.get(encoding, encoding)

    @classmethod
    def process_antibotics(cls, antibiotics: str) -> int:
        return cls.antibiotics_map.get(antibiotics, -1)

    @classmethod
    def process_gender(cls, gender: str) -> int:
        return cls.gender_map.get(gender, -1)

    @classmethod
    def process_stresslvl(cls, stresslvl: str) -> int:
        return cls.stresslvl_map.get(stresslvl, -1)

    @classmethod
    def process_diabetes(cls, diabetes: str) -> int:
        return cls.diabetes_map.get(diabetes, -1)

    @classmethod
    def process_periohistory(cls, periohistory: str) -> int:
        return cls.periohistory_map.get(periohistory, -1)

    @classmethod
    def process_smokingtype(cls, smokingtype: str) -> int:
        return cls.smokingtype_map.get(smokingtype, -1)

    @classmethod
    def process_mobility(cls, mobility: str) -> int:
        return cls.mobility_map.get(mobility, -1)

    @classmethod
    def process_percussion(cls, percussion: str) -> int:
        return cls.percussion_map.get(percussion, -1)

    @classmethod
    def process_sensitivity(cls, sensitivity: str) -> int:
        return cls.sensitivity_map.get(sensitivity, -1)

    @classmethod
    def process_restoration(cls, restoration: str) -> int:
        return cls.restoration_map.get(restoration, -1)

    @classmethod
    def process_furcation(cls, furcation: str) -> int:
        return cls.furcation_map.get(furcation, -1)

    @classmethod
    def process_plaque(cls, plaque: str) -> int:
        return cls.plaque_map.get(plaque, -1)

    @classmethod
    def process_bop(cls, bop: str) -> int:
        return cls.bop_map.get(bop, -1)

    @classmethod
    def transform_predictions(
        cls, task: str, prediction_output: pd.DataFrame
    ) -> pd.DataFrame:
        """Transforms prediction output based on the task.

        Args:
            task (str): The task name for which the model was trained.
            prediction_output (pd.DataFrame): DataFrame containing a inference output.

        Returns:
            pd.DataFrame: Transformed DataFrame with user-friendly display predictions.
        """
        if task in ["pocketclosure", "pocketclosureinf"]:
            mapping = {1: "Pocket closed", 0: "Pocket not closed"}
        elif task == "improvement":
            mapping = {1: "Pocket improved", 0: "Pocket not improved"}
        elif task == "pdgrouprevaluation":
            mapping = {0: "Pocket < 4", 1: "Pocket 4 or 5", 2: "Pocket > 5"}
        else:
            return prediction_output

        prediction_output["prediction"] = (
            prediction_output["prediction"].astype(int).map(mapping)
        )
        return prediction_output
