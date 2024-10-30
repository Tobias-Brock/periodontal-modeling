class InputProcessor:
    """Convert input values to internal code-compatible formats.

    This class provides mappings and processing methods to convert user-friendly
    input values for tasks, learners, tuning methods, HPO methods, criteria, and
    encodings to formats compatible with internal code. These mappings ensure
    consistency in naming and support the required input standards.

    Attributes:
        task_map (dict): Maps task names to their internal codes.
        learner_map (dict): Maps learner names to their respective codes.
        tuning_map (dict): Maps tuning method names to internal values.
        hpo_map (dict): Maps hyperparameter optimization methods to their codes.
        criteria_map (dict): Maps evaluation criteria names to their codes.
        encodings_map (dict): Maps encoding types to internal format codes.

    Methods:
        - `process_task`: Convert a single task name to its internal code.
        - `process_learners`: Convert a list of learner names to internal codes.
        - `process_tuning`: Convert a list of tuning methods to internal codes.
        - `process_hpo`: Convert a list of HPO methods to internal codes.
        - `process_criteria`: Convert a list of criteria to internal codes.
        - `process_encoding`: Convert a list of encodings to internal codes.

    Example:
        ```
        # Example input processing for an experiment setup
        task = InputProcessor.process_task("Pocket closure")
        learners = InputProcessor.process_learners(["XGBoost", "Random Forest"])
        tuning_methods = InputProcessor.process_tuning(["Holdout", "Cross-Validation"])
        hpo_methods = InputProcessor.process_hpo(["HEBO"])
        criteria = InputProcessor.process_criteria(["F1 Score", "Brier Score"])
        encodings = InputProcessor.process_encoding(["One-hot"])

        print(task)          # Output: "pocketclosure"
        print(learners)      # Output: ["xgb", "rf"]
        print(tuning_methods) # Output: ["holdout", "cv"]
        print(hpo_methods)    # Output: ["hebo"]
        print(criteria)       # Output: ["f1", "brier_score"]
        print(encodings)      # Output: ["one_hot"]
        ```
    """

    task_map = {
        "Pocket closure": "pocketclosure",
        "Pocket improvement": "improve",
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

    @classmethod
    def process_task(cls, task: str) -> str:
        return cls.task_map.get(task, task)

    @classmethod
    def process_learners(cls, learners: list) -> list:
        return [cls.learner_map[learner] for learner in learners]

    @classmethod
    def process_tuning(cls, tuning_methods: list) -> list:
        return [cls.tuning_map[tuning] for tuning in tuning_methods]

    @classmethod
    def process_hpo(cls, hpo_methods: list) -> list:
        return [cls.hpo_map[hpo] for hpo in hpo_methods]

    @classmethod
    def process_criteria(cls, criteria: list) -> list:
        return [cls.criteria_map[criterion] for criterion in criteria]

    @classmethod
    def process_encoding(cls, encodings: list) -> list:
        return [cls.encodings_map[encoding] for encoding in encodings]
