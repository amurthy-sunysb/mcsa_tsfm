"""
Functions for the classification step of the experiments: splitting the windows into
training and test sets, scaling the features, training the models and evaluating them.
They work on the merged table made by rename_and_merge(), once add_labels_LIAS() has
added the 'class' and 'file_group' columns.

The train-test split only works on LIAS for now, because it stratifies on the LIAS
'file_group' column (condition and speed).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common.feature_merging import FEATURE_TYPES


# ---------------------------------------------------------------------------
# Train-test split (LIAS only, for now)
# ---------------------------------------------------------------------------

def train_test_split_LIAS(df: pd.DataFrame,
                          test_size: float = 0.2,
                          seed: int = 67) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the labelled LIAS windows into training and test sets, by file.

    The files, not the windows, are split, stratified on 'file_group'. All windows of a
    file land on the same side, and every condition-and-speed group has the same share of
    files in both sets. Label the windows with add_labels_LIAS() first.

    Args:
        df (pd.DataFrame): One row per window, with at least 'file_name', 'window_num' and
            'file_group' columns, e.g. the table returned by add_labels_LIAS().
        test_size (float): Share of the files that go into the test set.
        seed (int): Seed for the random split. A different seed puts different files in
            the test set, so the experiment can be repeated with several seeds.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train_df and test_df: the rows of df that belong
        to the training files and to the test files.

    Raises:
        ValueError: If df has no 'file_name', 'window_num' or 'file_group' column, if a
            window appears more than once, or if a file ends up in both sets.
    """
    missing = [col for col in ["file_name", "window_num", "file_group"] if col not in df.columns]
    if missing:
        hint = " add_labels_LIAS() adds 'file_group'." if "file_group" in missing else ""
        raise ValueError(f"df has no {missing} column(s).{hint}")
    if df.duplicated(subset=["file_name", "window_num"]).any():
        raise ValueError("Each window (file_name, window_num) must appear only once in df.")

    # Split the files, not the windows. Windows of the same recording overlap and look alike,
    # so if some were in training and others in test, the test would be too easy.
    file_df = df[["file_name", "file_group"]].drop_duplicates()
    train_files, test_files = train_test_split(file_df["file_name"],
                                               stratify=file_df["file_group"],
                                               test_size=test_size,
                                               random_state=seed)

    train_df = df[df["file_name"].isin(train_files)]
    test_df = df[df["file_name"].isin(test_files)]

    if set(train_df["file_name"]) & set(test_df["file_name"]):
        raise ValueError("A file ended up in both the training and the test set.")

    print(f"Seed {seed}: {len(train_files)} training files ({len(train_df)} windows), "
          f"{len(test_files)} test files ({len(test_df)} windows).")

    return train_df, test_df


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_features(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   feature_type: str) -> tuple[np.ndarray, pd.Series, np.ndarray, pd.Series]:
    """
    Picks the features of one type and standardizes them (mean 0, standard deviation 1).

    The feature columns are the ones named '<feature_type>_...', which is how
    rename_and_merge() names them (e.g. 'moment_a_17'). The scaler learns each feature's
    mean and standard deviation from the training set only, then applies them to both
    sets, so no information from the test set leaks into training.

    There is no seed here because scaling is not random. The randomness of an
    experiment is in the split: pass a different seed to train_test_split_LIAS().

    Args:
        train_df (pd.DataFrame): Training windows, with a 'class' column.
        test_df (pd.DataFrame): Test windows, with a 'class' column.
        feature_type (str): Which features to use: "tsfresh", "moment" or "mantis".

    Returns:
        tuple: X_train_scaled, y_train, X_test_scaled, y_test.

    Raises:
        ValueError: If feature_type is not supported, a table has no 'class' column, or
            there are no feature columns for feature_type.
    """
    if feature_type not in FEATURE_TYPES:
        raise ValueError(f"feature_type must be one of {FEATURE_TYPES}, not '{feature_type}'.")
    for name, df in [("train_df", train_df), ("test_df", test_df)]:
        if "class" not in df.columns:
            raise ValueError(f"{name} has no 'class' column.")

    feature_cols = [col for col in train_df.columns if str(col).startswith(f"{feature_type}_")]
    if not feature_cols:
        raise ValueError(f"train_df has no '{feature_type}_...' columns. "
                         f"Is it really a {feature_type} table?")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])  # learn each feature's mean and std on training data
    X_test_scaled = scaler.transform(test_df[feature_cols])        # apply the same mean and std to the test data

    return X_train_scaled, train_df["class"], X_test_scaled, test_df["class"]


# ---------------------------------------------------------------------------
# Models and evaluation
# ---------------------------------------------------------------------------

def train_svc_model(X_train: pd.DataFrame | np.ndarray,
                    y_train: pd.Series,
                    kernel: str = 'rbf',
                    class_weight: str = 'balanced',
                    random_state: int = 42) -> SVC:
    """
    Trains an SVC classifier and returns the trained model.

    Args:
        X_train (pd.DataFrame | np.ndarray): Training features.
        y_train (pd.Series): Training labels.
        kernel (str): Specifies the kernel type to be used in the algorithm.
        class_weight (str): Weights associated with classes.
        random_state (int): The seed of the pseudo random number generator.

    Returns:
        SVC: The trained SVC model.
    """
    model = SVC(kernel=kernel, class_weight=class_weight, random_state=random_state)
    print(f"Training SVM model with {kernel} kernel and balanced class weights...")
    model.fit(X_train, y_train)
    print(f"SVM model training complete with {kernel} kernel.")
    return model


def train_log_reg_model(X_train: pd.DataFrame | np.ndarray, y_train: pd.Series,
                        solver: str = 'lbfgs', penalty: str = 'l2', C: float = 1.0,
                        class_weight: str = 'balanced', random_state: int = 42,
                        max_iter: int = 1000) -> LogisticRegression:
    """
    Trains a Logistic Regression classifier and returns the trained model.

    Args:
        X_train (pd.DataFrame | np.ndarray): Training features.
        y_train (pd.Series): Training labels.
        solver (str): Algorithm to use in the optimization problem.
        penalty (str): Used to specify the norm of the penalty.
        C (float): Inverse of regularization strength.
        class_weight (str): Weights associated with classes.
        random_state (int): The seed of the pseudo random number generator.
        max_iter (int): Maximum number of iterations for the solver to converge.

    Returns:
        LogisticRegression: The trained Logistic Regression model.
    """
    model = LogisticRegression(solver=solver, penalty=penalty, C=C,
                               class_weight=class_weight, random_state=random_state, max_iter=max_iter)
    print(f"Training Logistic Regression model with solver='{solver}', penalty='{penalty}', C={C} and balanced class weights...")
    model.fit(X_train, y_train)
    print("Logistic Regression model training complete.")
    return model


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, title: str):
    """
    Evaluates a classification model by printing the classification report and confusion matrix.

    Args:
        y_true (pd.Series): True labels.
        y_pred (np.ndarray): Predicted labels.
        title (str): Title for the evaluation output.
    """
    print(f"\nClassification Report ({title}):")
    print(classification_report(y_true, y_pred))
    print(f"\nConfusion Matrix ({title}):")
    print(confusion_matrix(y_true, y_pred))
