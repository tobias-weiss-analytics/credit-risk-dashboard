"""Loading and recoding of the UCI Statlog (German Credit Data) dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "german.data"

# Readable names for the 20 anonymous attributes (order as in german.doc).
FEATURE_NAMES = [
    "Checking_Status", "Duration", "Credit_History", "Purpose", "Credit_Amount",
    "Savings_Account", "Employment_Since", "Installment_Rate", "Sex_Status", "Debtors",
    "Residence_Since", "Property", "Age", "Other_Plans", "Housing",
    "Num_Credits", "Job", "Num_Dependents", "Phone", "Foreign_Worker",
]

# Code -> readable label, following the dataset documentation (german.doc).
CODE_LABELS: dict[str, dict[str, str]] = {
    "Checking_Status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "No checking account"},
    "Credit_History": {
        "A30": "No credits / all paid back", "A31": "All paid back at this bank",
        "A32": "Existing paid back duly", "A33": "Delay in the past", "A34": "Critical / other credits",
    },
    "Purpose": {
        "A40": "Car (new)", "A41": "Car (used)", "A42": "Furniture", "A43": "Radio / TV",
        "A44": "Appliances", "A45": "Repairs", "A46": "Education", "A47": "Vacation",
        "A48": "Retraining", "A49": "Business", "A410": "Others",
    },
    "Savings_Account": {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM",
        "A65": "Unknown / no savings",
    },
    "Employment_Since": {"A71": "Unemployed", "A72": "< 1 year", "A73": "1-4 years", "A74": "4-7 years", "A75": ">= 7 years"},
    "Sex_Status": {
        "A91": "Male: divorced", "A92": "Female: div./married", "A93": "Male: single",
        "A94": "Male: married/widowed", "A95": "Female: single",
    },
    "Debtors": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "Property": {"A121": "Real estate", "A122": "Savings / life insurance", "A123": "Car / other", "A124": "No property"},
    "Other_Plans": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "Housing": {"A151": "Rent", "A152": "Own", "A153": "For free"},
    "Job": {"A171": "Unskilled (non-resident)", "A172": "Unskilled (resident)", "A173": "Skilled employee", "A174": "Management / self-employed"},
    "Phone": {"A191": "No", "A192": "Yes"},
    "Foreign_Worker": {"A201": "Yes", "A202": "No"},
}

NUMERIC_FEATURES = ["Duration", "Credit_Amount", "Installment_Rate", "Residence_Since", "Age", "Num_Credits", "Num_Dependents"]

# Features with an unambiguous order (ascending = more favourable). A category
# outside the order (e.g. "no checking account") is not a point on that scale and
# is passed to the model as a separate indicator instead of being forced into it.
ORDINAL_FEATURES: dict[str, tuple[list[str], str | None]] = {
    "Checking_Status": (["< 0 DM", "0-200 DM", ">= 200 DM"], "No checking account"),
    "Savings_Account": (["< 100 DM", "100-500 DM", "500-1000 DM", ">= 1000 DM"], "Unknown / no savings"),
    "Employment_Since": (["Unemployed", "< 1 year", "1-4 years", "4-7 years", ">= 7 years"], None),
}

NOMINAL_FEATURES = [f for f in FEATURE_NAMES if f not in NUMERIC_FEATURES and f not in ORDINAL_FEATURES]

# Attributes that are legally protected or a proxy for them (sex, marital status,
# nationality). Excluded from the model by default; kept for the fairness check.
PROTECTED_FEATURES = ["Sex_Status", "Foreign_Worker"]

TARGET_LABELS = {1: "Good", 0: "Bad"}


def load_german_credit(path: str | Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return readable features ``X`` and target ``y`` (1 = good credit, 0 = bad credit)."""
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=FEATURE_NAMES + ["Class"])
    X = raw[FEATURE_NAMES].copy()
    for col, mapping in CODE_LABELS.items():
        unknown = set(X[col].unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unexpected codes in {col}: {sorted(unknown)}")
        X[col] = X[col].map(mapping)
    y = raw["Class"].map({1: 1, 2: 0}).rename("Good_Credit")
    return X, y


def sex_from_status(sex_status: pd.Series) -> pd.Series:
    """Derive the applicant's sex from the combined personal-status attribute."""
    return sex_status.str.split(":").str[0]
