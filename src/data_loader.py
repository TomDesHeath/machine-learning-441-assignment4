"""
UNSW-NB15 Data Loading and Feature Management Module.
"""

import pandas as pd
import numpy as np

ATTACK_CATEGORY_MAP = {
    0: 'Normal',
    1: 'Reconnaissance',
    2: 'Backdoor',
    3: 'DoS',
    4: 'Exploits',
    5: 'Analysis',
    6: 'Fuzzers',
    7: 'Worms',
    8: 'Shellcode',
    9: 'Generic'
}

NOMINAL_COLUMNS = ['proto', 'service', 'state']
ID_COLUMN = 'id'
TARGET_COLUMN = 'attack_cat'


def load_raw_dataset(filepath='datasets/networkTraffic.csv'):
    """
    Load raw UNSW-NB15 dataset CSV file and clean basic string representations.

    Parameters:
    -----------
    filepath : str
        Path to the dataset CSV file.

    Returns:
    --------
    pd.DataFrame
        Cleaned raw pandas DataFrame.
    """
    df = pd.read_csv(filepath, low_memory=False)

    # Clean whitespace in nominal string columns
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype) == 'string':
            df[col] = df[col].astype(str).str.strip()

    # Encode missing values in service column explicitly as 'missing'
    if 'service' in df.columns:
        df['service'] = df['service'].replace('?', 'missing')

    # Remove identifier column if present
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    return df


def get_feature_column_names(df):
    """
    Separate nominal and numerical feature column names.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset dataframe.

    Returns:
    --------
    tuple (list, list, str)
        (nominal_cols, numeric_cols, target_col)
    """
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    nominal_cols = [c for c in feature_cols if c in NOMINAL_COLUMNS]
    numeric_cols = [c for c in feature_cols if c not in NOMINAL_COLUMNS]
    return nominal_cols, numeric_cols, TARGET_COLUMN
