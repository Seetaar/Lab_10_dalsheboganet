import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

DATA_PATH = "data/loan_data.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "feature_config.json")

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score"
]

CATEGORICAL_FEATURES = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file"
]

TARGET_COLUMN = "loan_status"


def load_data(path: str) -> pd.DataFrame:
    """
    Загрузка датасета и проверка колонок
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    df = pd.read_csv(path)
    print(f"Датасет загружен. Размер: {df.shape}")

    # Проверка наличия целевой переменной
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Целевая колонка '{TARGET_COLUMN}' не найдена в датасете.")

    return df


def build_preprocessor():
    """
    Трансформер для предобработки данных
    """

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


def train_and_compare_models(X_train, y_train, X_test, y_test):
    """
    Обучение моделей и сравнение их по ROC-AUC
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    results = {}
    best_model = None
    best_score = -1.0

    print("Сравнение моделей:")

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {"roc_auc": roc_auc, "accuracy": accuracy}
        print(f"Модель: {name}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")

        if roc_auc > best_score:
            best_score = roc_auc
            best_model = pipeline

    print(f"Лучшая модель: {max(results, key=lambda k: results[k]['roc_auc'])} (ROC-AUC: {best_score:.4f})")
    return best_model, results


def save_artifacts(model, feature_names, metrics):
    """
    Сохраняет модель и конфигурацию признаков в папку artifacts.
    """
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)

    joblib.dump(model, MODEL_PATH)
    print(f"Модель сохранена в: {MODEL_PATH}")

    config = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_names_after_preprocessing": feature_names,
        "metrics": metrics
    }

    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Конфигурация сохранена в: {CONFIG_PATH}")


if __name__ == "__main__":
    df = load_data(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    best_pipeline, metrics = train_and_compare_models(X_train, y_train, X_test, y_test)

    preprocessor = best_pipeline.named_steps['preprocessor']

    num_names = NUMERIC_FEATURES

    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)

    all_feature_names = list(num_names) + list(cat_names)

    save_artifacts(best_pipeline, all_feature_names, metrics)
