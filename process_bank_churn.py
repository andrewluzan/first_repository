"""
process_bank_churn.py

This module contains functions to preprocess bank churn data. It includes:
  - preprocess_data: to split the raw data into training/validation sets, add features,
    encode categorical columns, and optionally scale numeric columns.
  - preprocess_new_data: to process new data using the pre-fitted scaler and encoder.
  
Each function is modularized to perform only one action.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def split_data(
    df: pd.DataFrame, target_col: str, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataframe into training and validation sets.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            - Training input features.
            - Validation input features.
            - Training target series.
            - Validation target series.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_val, y_train, y_val


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataframe.

    The following features are added:
      - 'Age_40_plus': Boolean indicating if Age is 40 or more.
      - 'Zero_Balance': Boolean indicating if Balance equals 0.
      - 'NumOfProducts_str': Conversion of 'NumOfProducts' to string.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: A copy of the dataframe with additional features.
    """
    df = df.copy()
    df['Age_40_plus'] = df['Age'] >= 40
    df['Zero_Balance'] = df['Balance'] == 0
    df['NumOfProducts_str'] = df['NumOfProducts'].astype(str)
    return df


def encode_categorical_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """Encode categorical features using OneHotEncoder.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        val_df (pd.DataFrame): Validation dataframe.
        categorical_cols (List[str]): List of categorical column names to encode.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
            - Transformed training dataframe.
            - Transformed validation dataframe.
            - Fitted OneHotEncoder.
            - List of new encoded column names.
    """
    encoder = OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_df[categorical_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    # Transform the categorical columns and add them to the dataframes.
    train_encoded = encoder.transform(train_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols])

    train_df_encoded = train_df.copy()
    val_df_encoded = val_df.copy()

    train_df_encoded[encoded_cols] = train_encoded
    val_df_encoded[encoded_cols] = val_encoded

    return train_df_encoded, val_df_encoded, encoder, encoded_cols


def scale_numeric_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Scale numeric features using MinMaxScaler.

    Args:
        train_df (pd.DataFrame): Training dataframe.
        val_df (pd.DataFrame): Validation dataframe.
        numeric_cols (List[str]): List of numeric column names to scale.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
            - Training dataframe with scaled numeric features.
            - Validation dataframe with scaled numeric features.
            - Fitted MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    train_df_scaled[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df_scaled[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df_scaled, val_df_scaled, scaler


def preprocess_data(
    raw_df: pd.DataFrame, scaler_numeric: bool = True, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Optional[MinMaxScaler], OneHotEncoder]:
    """Preprocess the raw data for training a decision tree model.

    The function performs the following steps:
      1. Splits the data into training and validation sets.
      2. Applies feature engineering.
      3. Encodes specified categorical features.
      4. Optionally scales numeric features.

    Args:
        raw_df (pd.DataFrame): Raw input dataframe.
        scaler_numeric (bool): If True, scales numeric features using MinMaxScaler.
        test_size (float): Proportion of the data to use for validation.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple containing:
            - X_train (pd.DataFrame): Preprocessed training features.
            - train_targets (pd.DataFrame): Training target variable.
            - X_val (pd.DataFrame): Preprocessed validation features.
            - val_targets (pd.DataFrame): Validation target variable.
            - input_cols (List[str]): List of column names used in X.
            - scaler (Optional[MinMaxScaler]): Fitted scaler (or None if scaling is skipped).
            - encoder (OneHotEncoder): Fitted OneHotEncoder.
    """
    target_col = "Exited"
    # Split the data into training and validation sets.
    X_train, X_val, y_train, y_val = split_data(raw_df, target_col, test_size, random_state)

    # Apply feature engineering.
    train_inputs = add_feature_engineering(X_train)
    val_inputs = add_feature_engineering(X_val)

    # Identify numeric and categorical columns.
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    # All object-type columns (e.g., Geography, Gender, and our engineered string column).
    categorical_cols = train_inputs.select_dtypes(include="object").columns.tolist()

    # Specify categorical columns for one-hot encoding.
    categorical_cols_ohe = [
        "Geography",
        "Gender",
        "NumOfProducts_str",
        "Age_40_plus",
        "Zero_Balance",
        "HasCrCard",
        "IsActiveMember",
    ]
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical_features(
        train_inputs, val_inputs, categorical_cols_ohe
    )

    # Exclude columns that should not be scaled.
    exclude_cols = ["HasCrCard", "Tenure", "Balance", "CustomerId"]
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Optionally scale numeric features.
    if scaler_numeric:
        train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols)
    else:
        scaler = None

    # Combine numeric and encoded categorical features.
    feature_cols = numeric_cols + encoded_cols
    X_train_final = train_inputs[feature_cols]
    X_val_final = val_inputs[feature_cols]

    # Prepare target DataFrames.
    train_targets = y_train.to_frame(name=target_col)
    val_targets = y_val.to_frame(name=target_col)

    return X_train_final, train_targets, X_val_final, val_targets, feature_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame, feature_cols: List[str], scaler: Optional[MinMaxScaler], encoder: OneHotEncoder
) -> pd.DataFrame:
    """Preprocess new data using an already fitted scaler and encoder.

    The function applies the same feature engineering, encoding, and (if provided) scaling
    as during training. This prepares the new data for prediction or evaluation.

    Args:
        new_df (pd.DataFrame): New input dataframe to preprocess.
        feature_cols (List[str]): List of feature column names to retain.
        scaler (Optional[MinMaxScaler]): Pre-fitted MinMaxScaler; if None, scaling is skipped.
        encoder (OneHotEncoder): Pre-fitted OneHotEncoder.

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for prediction or evaluation.
    """
    df_processed = add_feature_engineering(new_df)

    # Identify numeric columns and exclude the ones that were not scaled.
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    exclude_cols = ["HasCrCard", "Tenure", "Balance", "CustomerId"]
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Specify the categorical columns to encode.
    categorical_cols_ohe = [
        "Geography",
        "Gender",
        "NumOfProducts_str",
        "Age_40_plus",
        "Zero_Balance",
        "HasCrCard",
        "IsActiveMember",
    ]
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols_ohe))
    df_encoded = df_processed.copy()
    df_encoded[encoded_cols] = encoder.transform(df_encoded[categorical_cols_ohe])

    # Scale numeric features if a scaler is provided.
    if scaler is not None:
        df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

    # Combine the features to match the training columns.
    final_df = df_encoded[feature_cols].copy()
    return final_df


if __name__ == "__main__":
    # Example usage:
    # For demonstration, we create a dummy dataframe.
    data = {
        "CustomerId": [1, 2, 3, 4, 5],
        "CreditScore": [600, 700, 800, 500, 650],
        "Geography": ["France", "Spain", "Germany", "France", "Spain"],
        "Gender": ["Male", "Female", "Female", "Male", "Male"],
        "Age": [40, 50, 30, 60, 45],
        "Tenure": [3, 5, 2, 8, 4],
        "Balance": [50000, 60000, 0, 80000, 70000],
        "NumOfProducts": [2, 1, 2, 1, 2],
        "HasCrCard": [1, 0, 1, 1, 0],
        "IsActiveMember": [1, 0, 1, 0, 1],
        "EstimatedSalary": [50000, 60000, 55000, 70000, 65000],
        "Exited": [0, 1, 0, 1, 0],
    }
    raw_df = pd.DataFrame(data)

    # Preprocess the raw data.
    X_train, train_targets, X_val, val_targets, input_cols, scaler, encoder = preprocess_data(raw_df, scaler_numeric=True)

    print("Preprocessed Training Features:")
    print(X_train)
    print("\nTraining Targets:")
    print(train_targets)
    print("\nFeature Columns Used:")
    print(input_cols)

    # Example of processing new data (e.g., from test.csv).
    new_data = {
        "CustomerId": [6],
        "CreditScore": [720],
        "Geography": ["Germany"],
        "Gender": ["Female"],
        "Age": [35],
        "Tenure": [4],
        "Balance": [0],
        "NumOfProducts": [1],
        "HasCrCard": [1],
        "IsActiveMember": [0],
        "EstimatedSalary": [60000],
        "Exited": [0],
    }
    new_df = pd.DataFrame(new_data)
    new_processed = preprocess_new_data(new_df, input_cols, scaler, encoder)
    print("\nPreprocessed New Data:")
    print(new_processed)