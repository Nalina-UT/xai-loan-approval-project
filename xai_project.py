import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import shap
from lime.lime_tabular import LimeTabularExplainer


# =========================================================
# SETTINGS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATASET_PATH = os.path.join(BASE_DIR, "synthetic_loan_data.csv")

RANDOM_STATE = 42
N_SAMPLES = 301
TEST_SIZE = 0.30
TARGET_COLUMN = "Loan_Status"


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def create_results_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"Dataset saved to: {path}")


# =========================================================
# SYNTHETIC DATASET GENERATION
# =========================================================
def generate_synthetic_loan_dataset(n_samples: int = 301, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.65, 0.35])
    married = rng.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    education = rng.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.70, 0.30])
    self_employed = rng.choice(["Yes", "No"], size=n_samples, p=[0.15, 0.85])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], size=n_samples, p=[0.35, 0.40, 0.25])
    dependents = rng.choice(["0", "1", "2", "3+"], size=n_samples, p=[0.40, 0.25, 0.20, 0.15])

    applicant_income = rng.integers(1500, 15000, size=n_samples)
    coapplicant_income = rng.integers(0, 8000, size=n_samples)
    loan_amount = rng.integers(50, 400, size=n_samples)
    loan_amount_term = rng.choice([120, 180, 240, 300, 360], size=n_samples, p=[0.10, 0.15, 0.15, 0.20, 0.40])
    credit_history = rng.choice([0, 1], size=n_samples, p=[0.25, 0.75])

    total_income = applicant_income + coapplicant_income
    emi_ratio = loan_amount / np.maximum(total_income / 100, 1)

    score = np.zeros(n_samples, dtype=float)
    score += (credit_history == 1) * 3.5
    score += (education == "Graduate") * 0.8
    score += (married == "Yes") * 0.2
    score += (self_employed == "No") * 0.2
    score += (property_area == "Urban") * 0.4
    score += (property_area == "Semiurban") * 0.5
    score += (total_income > 7000) * 1.0
    score += (total_income > 10000) * 0.8
    score -= (loan_amount > 250) * 1.1
    score -= (loan_amount > 320) * 0.9
    score -= (emi_ratio > 3.5) * 1.5
    score -= (dependents == "3+") * 0.4

    noise = rng.normal(0, 0.7, n_samples)
    score += noise

    loan_status = np.where(score >= 2.5, "Approved", "Rejected")

    df = pd.DataFrame({
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        TARGET_COLUMN: loan_status
    })

    for col in ["Gender", "Married", "LoanAmount", "Credit_History"]:
        missing_idx = rng.choice(df.index, size=max(3, n_samples // 40), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df



def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def prepare_features_and_target(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))

    return X, y_encoded, label_encoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


# =========================================================
# MODEL BUILDING
# =========================================================
def build_random_forest_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def build_logistic_regression_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


# =========================================================
# EVALUATION
# =========================================================
def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test, save_dir: str, model_name: str = "model") -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    }

    print(f"\n=== {model_name.upper()} PERFORMANCE ===")
    for key, value in metrics.items():
        if key != "model":
            print(f"{key}: {value:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="viridis")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return metrics


def run_cross_validation(model: Pipeline, X, y, cv_splits: int = 5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores


def save_metrics_table(metrics_list: list, save_dir: str) -> None:
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(save_dir, "evaluation_metrics.csv"), index=False)
    print("Evaluation metrics saved.")


# =========================================================
# FEATURE NAMES
# =========================================================
def get_feature_names(preprocessor: ColumnTransformer) -> list:
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue

        if hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                feature_names.extend(columns)
        else:
            if hasattr(transformer, "get_feature_names_out"):
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                feature_names.extend(columns)

    return feature_names


# =========================================================
# VISUALISATIONS
# =========================================================
def save_feature_importance(model: Pipeline, feature_names: list, save_dir: str, top_n: int = 15) -> None:
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        print("Selected model does not provide feature importances.")
        return

    importances = classifier.feature_importances_

    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n)

    feat_imp.to_csv(os.path.join(save_dir, "feature_importance_table.csv"), index=False)

    plt.figure(figsize=(11, 7))
    plt.barh(feat_imp["Feature"][::-1], feat_imp["Importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("Feature importance saved.")


def run_shap_analysis(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, save_dir: str) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    feature_names = get_feature_names(preprocessor)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_to_plot,
        X_test_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("SHAP summary saved.")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_to_plot,
        X_test_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance (Bar)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_bar.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("SHAP bar plot saved.")


def run_lime_analysis(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, save_dir: str, instance_index: int = 0) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    feature_names = get_feature_names(preprocessor)
    class_names = [str(cls) for cls in classifier.classes_]

    explainer = LimeTabularExplainer(
        training_data=X_train_transformed,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_test_transformed[instance_index],
        classifier.predict_proba,
        num_features=min(10, len(feature_names))
    )

    exp.save_to_file(os.path.join(save_dir, "lime_explanation.html"))

    fig = exp.as_pyplot_figure()
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lime_explanation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("LIME explanation saved.")


# =========================================================
# MAIN
# =========================================================
def main():
    print("Starting project...")
    create_results_dir(RESULTS_DIR)


    print("Generating synthetic dataset...")
    df = generate_synthetic_loan_dataset(n_samples=N_SAMPLES, random_state=RANDOM_STATE)
    df = clean_dataframe(df)
    save_dataframe(df, DATASET_PATH)

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset size:")
    print(df.shape)

    print("\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts())

    # 2. Prepare data
    print("\nPreparing data...")
    X, y, label_encoder = prepare_features_and_target(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(X)

    # 3. Build models
    rf_model = build_random_forest_pipeline(preprocessor)
    lr_model = build_logistic_regression_pipeline(preprocessor)

    # 4. Train models
    print("\nTraining Random Forest...")
    rf_model.fit(X_train, y_train)

    print("Training Logistic Regression...")
    lr_model.fit(X_train, y_train)

    # 5. Evaluate models
    rf_metrics = evaluate_model(rf_model, X_test, y_test, RESULTS_DIR, model_name="random_forest")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, RESULTS_DIR, model_name="logistic_regression")

    # 6. Cross-validation
    print("\nRunning cross-validation...")
    rf_cv_scores = run_cross_validation(rf_model, X, y, cv_splits=5)
    lr_cv_scores = run_cross_validation(lr_model, X, y, cv_splits=5)

    print(f"Random Forest CV Accuracy: {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")
    print(f"Logistic Regression CV Accuracy: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

    cv_df = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression"],
        "CV_Mean_Accuracy": [rf_cv_scores.mean(), lr_cv_scores.mean()],
        "CV_Std": [rf_cv_scores.std(), lr_cv_scores.std()]
    })
    cv_df.to_csv(os.path.join(RESULTS_DIR, "cross_validation_results.csv"), index=False)

    # 7. Save metrics
    save_metrics_table([rf_metrics, lr_metrics], RESULTS_DIR)

    # 8. Feature importance
    fitted_preprocessor = rf_model.named_steps["preprocessor"]
    feature_names = get_feature_names(fitted_preprocessor)
    save_feature_importance(rf_model, feature_names, RESULTS_DIR)

    # 9. SHAP
    print("\nRunning SHAP analysis...")
    run_shap_analysis(rf_model, X_train, X_test, RESULTS_DIR)

    # 10. LIME
    print("Running LIME analysis...")
    run_lime_analysis(rf_model, X_train, X_test, RESULTS_DIR, instance_index=0)

    # 11. Model comparison table
    comparison_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "Random Forest + XAI"],
        "Accuracy": [
            round(lr_metrics["accuracy"], 4),
            round(rf_metrics["accuracy"], 4),
            round(rf_metrics["accuracy"], 4)
        ],
        "Precision": [
            round(lr_metrics["precision"], 4),
            round(rf_metrics["precision"], 4),
            round(rf_metrics["precision"], 4)
        ],
        "Recall": [
            round(lr_metrics["recall"], 4),
            round(rf_metrics["recall"], 4),
            round(rf_metrics["recall"], 4)
        ],
        "F1_Score": [
            round(lr_metrics["f1_score"], 4),
            round(rf_metrics["f1_score"], 4),
            round(rf_metrics["f1_score"], 4)
        ],
        "Interpretability": ["High", "Low", "High"],
        "Notes": [
            "Baseline interpretable model",
            "More complex ensemble model",
            "Random Forest supported by SHAP and LIME"
        ]
    })
    comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison_table.csv"), index=False)

    print("\nDone.")
    print(f"All results saved in: {RESULTS_DIR}")
    print(f"Synthetic dataset saved in: {DATASET_PATH}")


if __name__ == "__main__":
    main()