class InputProcessor:
    """Process app inputs to internal code-compatible formats."""

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
    def process_tasks(cls, tasks: list) -> list:
        return [cls.task_map[task] for task in tasks]

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
