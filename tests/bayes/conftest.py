"""Fixtures for the Bayesian submodule tests."""

import numpy as np
import pandas as pd
import pytest

TEETH = [16, 15, 14, 11, 21]
SIDES = [1, 2, 3, 4, 5, 6]


def synthetic_data(n_patients: int = 20, seed: int = 0) -> pd.DataFrame:
    """Creates a small processed dataset with one row per site.

    Args:
        n_patients (int): Number of simulated patients. Defaults to 20.
        seed (int): Random state of the simulation. Defaults to 0.

    Returns:
        pd.DataFrame: Dataset with patient, tooth and site level columns.
    """
    rng = np.random.default_rng(seed=seed)
    rows = []
    for patient in range(1, n_patients + 1):
        patient_features = {
            "id_patient": patient,
            "age": int(rng.integers(20, 80)),
            "gender": int(rng.integers(0, 2)),
            "bodymassindex": float(rng.normal(25, 4)),
            "periofamilyhistory": int(rng.integers(1, 4)),
            "diabetes": int(rng.integers(1, 5)),
            "smokingtype": int(rng.integers(1, 6)),
            "cigarettenumber": float(rng.integers(0, 20)),
            "antibiotictreatment": int(rng.integers(0, 2)),
            "stresslvl": int(rng.integers(0, 3)),
        }
        for tooth in TEETH:
            tooth_features = {
                "tooth": tooth,
                "toothtype": int(rng.integers(0, 3)),
                "rootnumber": int(rng.integers(0, 2)),
                "mobility": int(rng.integers(0, 2)),
                "restoration": int(rng.integers(0, 3)),
                "percussion-sensitivity": int(rng.integers(0, 2)),
                "sensitivity": int(rng.integers(0, 2)),
            }
            for side in SIDES:
                pdbaseline = int(rng.integers(1, 8))
                pdrevaluation = int(rng.integers(1, 8))
                rows.append({
                    **patient_features,
                    **tooth_features,
                    "side": side,
                    "pdbaseline": pdbaseline,
                    "recbaseline": float(rng.integers(0, 4)),
                    "plaque": int(rng.integers(0, 2)),
                    "bop": int(rng.integers(0, 2)),
                    "furcationbaseline": int(rng.integers(0, 4)),
                    "pdrevaluation": pdrevaluation,
                    "boprevaluation": int(rng.integers(0, 2)),
                    "pdgroupbase": 0
                    if pdbaseline <= 3
                    else (1 if pdbaseline < 6 else 2),
                    "pdgrouprevaluation": (
                        0 if pdrevaluation <= 3 else (1 if pdrevaluation < 6 else 2)
                    ),
                    "pocketclosure": int(pdrevaluation <= 3),
                    "improvement": int(pdrevaluation < pdbaseline),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def data() -> pd.DataFrame:
    """Provides a synthetic processed dataset.

    Returns:
        pd.DataFrame: Dataset with 20 patients, 5 teeth and 6 sites per tooth.
    """
    return synthetic_data()
