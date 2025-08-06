import numpy as np
import pandas as pd

from ..base import Patient, Side, Tooth, all_teeth, patient_to_df
from ._preprocessing import StaticProcessEngine


class DataSimulator:
    """Simulates patient data and defines relationships to pdrevaluation.

    This version includes:
    - More complex, non-linear relationships and stronger effects from variables.
    - Heterogeneity in patient dentition.
    - Ensures all teeth have 6 sides.
    - Ensures at least one side with pdbaseline > 3 per patient, ~90% sides remain ≤ 3.
    - Additional noise and interactions to make the relationship harder to learn.
    """

    def __init__(
        self,
        baseline_coef: float = 2.0,
        bmi_coef: float = 0.1,
        antibiotic_coef: float = -0.5,
        smoke_coef: float = 0.3,
        nonlinear_effects: bool = True,
        random_state: int = 42,
    ) -> None:
        """Initializes the DataSimulator.

        Args:
            baseline_coef (float): Coefficient for the baseline effect on pdrevaluation.
            bmi_coef (float): Coefficient for BMI influence on pdrevaluation.
            antibiotic_coef (float): Coefficient for antibiotic treatment influence.
            smoke_coef (float): Coefficient for smoking influence.
            nonlinear_effects (bool): Whether to apply non-linear transformations.
            random_state (int): Seed for reproducibility.
        """
        self.baseline_coef = baseline_coef
        self.bmi_coef = bmi_coef
        self.antibiotic_coef = antibiotic_coef
        self.smoke_coef = smoke_coef
        self.nonlinear_effects = nonlinear_effects
        self.rng = np.random.default_rng(random_state)
        self.processor = StaticProcessEngine()

    def default_pdreval_func(self, df: pd.DataFrame) -> np.ndarray:
        """Computes pdrevaluation based on multiple patient features.

        Args:
            df (pd.DataFrame): Patient data.

        Returns:
            np.ndarray: Computed pdrevaluation scores.
        """
        pdbase = df["pdbaseline"].fillna(3).astype(float)
        bmi = df["bodymassindex"].astype(float)
        antibiotic = df["antibiotictreatment"].astype(int)
        smoke = df["smokingtype"].astype(int)
        diabetes = df["diabetes"].astype(int)
        cig_num = df["cigarettenumber"].fillna(0).astype(float)
        stress = df["stresslvl"].astype(int)
        furcation = df["furcationbaseline"].astype(int)
        rec = df["recbaseline"].astype(int)
        restoration = df["restoration"].astype(int)
        mobility = df["mobility"].astype(int)
        # tooth = df["tooth"].astype(int)

        base_value = 0.0

        # Main effects
        pdbase_effect = self.baseline_coef * pdbase
        if self.nonlinear_effects:
            pdbase_effect += self.baseline_coef * (pdbase**2)

        bmi_effect = self.bmi_coef * bmi
        stress_effect = 0.1 * stress
        cig_effect = self.smoke_coef * (cig_num**1.5)
        antibiotic_effect = self.antibiotic_coef * antibiotic
        furcation_effect = 0.4 * (furcation**1.3)
        rec_effect = 0.3 * np.sqrt(rec + 1)

        restoration_array = restoration.to_numpy()
        restoration_penalty = np.where(restoration_array == 1, 0.0, 0.4)
        restoration_effect = restoration_penalty

        mobility_effect = 0.7 * mobility

        high_smoke_no_antibiotic = 0.2 * ((smoke >= 3) & (antibiotic == 0)).astype(
            float
        )
        high_risk = 0.2 * ((diabetes > 1) & (bmi > 30)).astype(float)

        linear_term = (
            base_value
            + pdbase_effect
            + bmi_effect
            + stress_effect
            + cig_effect
            + antibiotic_effect
            + furcation_effect
            + rec_effect
            + restoration_effect
            + mobility_effect
            + high_smoke_no_antibiotic
            + high_risk
        )

        noise = self.rng.normal(loc=0.0, scale=2.0, size=len(df))
        pdrevaluation_raw = linear_term + noise

        raw_min = pdrevaluation_raw.min()
        raw_max = pdrevaluation_raw.max()

        if raw_min == raw_max:
            pdrevaluation_scaled = np.full_like(pdrevaluation_raw, 5)
        else:
            pdrevaluation_scaled = 1 + (pdrevaluation_raw - raw_min) * (
                9 / (raw_max - raw_min)
            )

        return np.clip(np.round(pdrevaluation_scaled), 1, 10).astype(int)

    def simulate_cohort(
        self,
        n_patients: int = 100,
        p_wisdom: float = 0.2,
        p_missing: float = 0.1,
        tooth_seed: int = 42,
    ) -> pd.DataFrame:
        """Simulates a cohort of patients with variable dentition.

        Args:
            n_patients (int): Number of patients to simulate.
            p_wisdom (float): Probability of retaining wisdom teeth.
            p_missing (float): Probability of missing additional teeth.
            tooth_seed (int): Seed for tooth-related randomness.

        Returns:
            pd.DataFrame: Simulated dataset.
        """
        rng_teeth = np.random.default_rng(tooth_seed)
        patients = []
        for _ in range(1, n_patients + 1):
            patients.append(
                self.simulate_patient(
                    p_wisdom=p_wisdom,
                    p_missing=p_missing,
                    rng=rng_teeth,
                )
            )

        df_list = []
        for i, patient in enumerate(patients, start=1):
            df_p = patient_to_df(patient)
            df_p["id_patient"] = i
            df_list.append(df_p)

        data = pd.concat(df_list, axis=0, ignore_index=True)

        data["pdrevaluation"] = self.default_pdreval_func(data)
        data["boprevaluation"] = self.rng.integers(0, 2)
        return self.processor.create_outcome_variables(
            self.processor.create_tooth_features(data=data)
        )

    def simulate_patient(
        self,
        p_wisdom: float,
        p_missing: float,
        rng: np.random.Generator,
    ) -> Patient:
        """Simulates a single patient with heterogeneous dentition.

        This method generates a synthetic patient with various health conditions,
        lifestyle factors, and dental structures. It ensures variability in missing
        teeth, periodontal conditions, and overall oral health.

        Args:
            p_wisdom (float): Probability of the patient retaining wisdom teeth.
            p_missing (float): Probability of the patient missing additional teeth.
            rng (np.random.Generator): Random number generator for reproducibility.

        Returns:
            Patient: A `Patient` object containing the simulated demographic,
                health, and dental attributes.
        """
        age = rng.integers(18, 80)
        gender = rng.integers(0, 2) % 2  # binary
        bodymassindex = rng.normal(loc=25, scale=4)
        periofamilyhistory = rng.integers(0, 3)  # 0,1,2
        if rng.random() < 0.8:
            diabetes = 1
            smokingtype = 1
            cigarettenumber = 0
            antibiotictreatment = 1
        else:
            diabetes = rng.integers(0, 4)  # 0 to 3
            smokingtype = rng.integers(0, 5)  # 0 to 4
            cigarettenumber = int(max(0, rng.normal(loc=5, scale=5)))
            antibiotictreatment = rng.integers(0, 2)

        stresslvl = rng.integers(0, 3)  # 0,1,2
        teeth_set = all_teeth.copy()

        # Handle wisdom teeth
        has_wisdom = rng.random() < p_wisdom
        if not has_wisdom:
            teeth_set = [t for t in teeth_set if t not in [18, 28, 38, 48]]

        # Possibly remove additional random teeth
        if rng.random() < p_missing:
            n_missing = rng.integers(1, 5)
            missing_teeth = rng.choice(teeth_set, size=n_missing, replace=False)
            teeth_set = [t for t in teeth_set if t not in missing_teeth]

        teeth_objs = [self.simulate_tooth(tnum, rng) for tnum in teeth_set]
        teeth_objs = self.ensure_one_side_above_3(teeth_objs, rng)

        return Patient(
            age=int(age),
            gender=int(gender),
            bodymassindex=float(bodymassindex),
            periofamilyhistory=int(periofamilyhistory),
            diabetes=int(diabetes),
            smokingtype=int(smokingtype),
            cigarettenumber=int(cigarettenumber),
            antibiotictreatment=int(antibiotictreatment),
            stresslvl=int(stresslvl),
            teeth=teeth_objs,
        )

    def ensure_one_side_above_3(
        self, teeth_objs: list, rng: np.random.Generator
    ) -> list:
        """Ensures at least one side of a patient's teeth has pdbaseline > 3.

        Args:
            teeth_objs (list): List of `Tooth` objects for a patient.
            rng (np.random.Generator): Random generator instance.

        Returns:
            list: Modified list of `Tooth` objects.
        """
        all_sides = [(t, s) for t in teeth_objs for s in t.sides]
        if all(np.array([s.pdbaseline for t, s in all_sides]) <= 3):
            # Pick a random side and bump its pdbaseline
            _, s_choice = rng.choice(all_sides)
            s_choice.pdbaseline = rng.integers(4, 7)  # set to something >3
        return teeth_objs

    def simulate_tooth(self, tooth_number: int, rng: np.random.Generator) -> Tooth:
        """Simulates an individual tooth with predefined features.

        Args:
            tooth_number (int): The numeric identifier of the tooth.
            rng (np.random.Generator): Random generator instance.

        Returns:
            Tooth: Simulated `Tooth` object.
        """
        if tooth_number in [11, 12, 21, 22, 31, 32, 41, 42, 13, 23, 33, 43]:
            toothtype = 0
            rootnumber = 0
        elif tooth_number in [14, 15, 24, 25, 34, 35, 44, 45]:
            toothtype = 1
            rootnumber = 1
        else:
            toothtype = 2
            rootnumber = 1

        if rng.random() < 0.8:
            mobility = rng.integers(0, 1)  # binary
            restoration = rng.integers(0, 1)  # categorical 0,1,2
            percussion = rng.integers(0, 1)  # binary
            sensitivity = rng.integers(0, 1)  # binary
        else:
            mobility = rng.integers(0, 2)  # binary
            restoration = rng.integers(0, 3)  # categorical 0,1,2
            percussion = rng.integers(0, 2)  # binary
            sensitivity = rng.integers(0, 2)  # binary

        # Each tooth has 6 sides
        sides = [self.simulate_side(s, rng) for s in range(1, 7)]

        return Tooth(
            tooth=tooth_number,
            toothtype=toothtype,
            rootnumber=rootnumber,
            mobility=int(mobility),
            restoration=int(restoration),
            percussion=int(percussion),
            sensitivity=int(sensitivity),
            sides=sides,
        )

    def simulate_side(self, side_num: int, rng: np.random.Generator) -> Side:
        """Simulates a side of a tooth.

        Args:
            side_num (int): The numeric identifier for the side.
            rng (np.random.Generator): Random generator instance.

        Returns:
            Side: Simulated `Side` object.
        """
        if rng.random() < 0.9:
            pdbaseline = rng.integers(1, 4)  # mostly ≤3
            recbaseline = rng.integers(1, 4)
            bop = rng.integers(0, 1)
            furcationbaseline = rng.integers(0, 1)
            plaque = rng.integers(0, 1)
        else:
            pdbaseline = rng.integers(4, 10)  # occasionally >3
            recbaseline = rng.integers(4, 10)
            bop = rng.integers(0, 2)
            furcationbaseline = rng.integers(1, 4)
            plaque = rng.integers(0, 2)  # binary

        return Side(
            furcationbaseline=int(furcationbaseline),
            side=int(side_num),
            pdbaseline=int(pdbaseline),
            recbaseline=int(recbaseline),
            plaque=int(plaque),
            bop=int(bop),
        )
