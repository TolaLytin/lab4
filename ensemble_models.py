"""Эксперимент с ансамблем: мягкое голосование RF + Gradient Boosting + LogReg."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

# Порядок столбцов german.csv без целевой переменной Creditability.
FEATURE_NAMES = [
    "Account_Balance", "Duration_of_Credit_monthly",
    "Payment_Status_of_Previous_Credit", "Purpose", "Credit_Amount",
    "Value_Savings_Stocks", "Length_of_current_employment",
    "Instalment_per_cent", "Sex_Marital_Status", "Guarantors",
    "Duration_in_Current_address", "Most_valuable_available_asset",
    "Age_years", "Concurrent_Credits", "Type_of_apartment",
    "No_of_Credits_at_this_Bank", "Occupation", "No_of_dependents",
    "Telephone", "Foreign_Worker",
]
# Номинальные признаки (коды категорий без порядка) -> one-hot.
CATEGORICAL = [
    "Account_Balance", "Payment_Status_of_Previous_Credit", "Purpose",
    "Value_Savings_Stocks", "Length_of_current_employment",
    "Sex_Marital_Status", "Guarantors", "Most_valuable_available_asset",
    "Concurrent_Credits", "Type_of_apartment", "Occupation",
]
THRESHOLD = 0.5


def _to_frame(features) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
        if not set(FEATURE_NAMES).issubset(frame.columns):
            frame = pd.DataFrame(frame.to_numpy(), columns=FEATURE_NAMES)
    else:
        frame = pd.DataFrame(np.asarray(features), columns=FEATURE_NAMES)
    return frame[FEATURE_NAMES].reset_index(drop=True).astype(float)


def _add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Производные признаки: логарифм суммы, платёж в месяц и на взнос."""
    frame = frame.copy()
    frame["log_amount"] = np.log1p(frame["Credit_Amount"])
    frame["amount_per_month"] = frame["Credit_Amount"] / frame["Duration_of_Credit_monthly"]
    frame["amount_per_instalment"] = frame["amount_per_month"] / frame["Instalment_per_cent"]
    return frame


def build_model(random_seed: int = 42) -> VotingClassifier:
    numeric = [c for c in _add_features(pd.DataFrame(
        np.ones((2, len(FEATURE_NAMES))), columns=FEATURE_NAMES)).columns
        if c not in CATEGORICAL]

    # Деревьям масштабирование не нужно: one-hot + числа как есть.
    tree_prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", numeric),
    ])
    # Логистической регрессии нужны стандартизованные числа.
    linear_prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), numeric),
    ])

    forest = make_pipeline(tree_prep, RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features=0.2,
        random_state=random_seed, n_jobs=1))
    boosting = make_pipeline(tree_prep, GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.7,
        max_features=0.5, random_state=random_seed))
    logreg = make_pipeline(linear_prep, LogisticRegression(C=0.1, max_iter=2000))

    return VotingClassifier(
        [("random_forest", forest), ("gradient_boosting", boosting),
         ("logistic_regression", logreg)],
        voting="soft",
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает soft-voting ансамбль только на train и считает метрики на test."""
    x_train = _add_features(_to_frame(data.train_features))
    x_test = _add_features(_to_frame(data.test_features))
    y_train = np.asarray(data.train_target).ravel().astype(int)
    y_test = np.asarray(data.test_target).ravel().astype(int)

    model = build_model(data.random_seed)
    model.fit(x_train, y_train)  # test_target здесь не используется

    positive_column = list(model.classes_).index(1)
    probabilities = model.predict_proba(x_test)[:, positive_column]
    predictions = (probabilities >= THRESHOLD).astype(int)

    return EnsembleResult(
        model_name="Soft Voting: RandomForest + GradientBoosting + LogisticRegression",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
