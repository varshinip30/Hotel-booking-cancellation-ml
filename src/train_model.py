"""
train_model.py
----------------
Trains a hotel booking cancellation prediction model.

Pipeline overview:
    1. Load raw data
    2. Clean data (missing values, duplicates, invalid rows)
    3. Engineer a few simple, explainable features
    4. Build a scikit-learn preprocessing + model Pipeline
    5. Train / compare Logistic Regression, Random Forest, HistGradientBoosting
    6. Evaluate all models on a held-out test set
    7. Save the best pipeline (preprocessing + model together) with joblib

Run from the project root with:
    python src/train_model.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "hotel_bookings.csv")
MODEL_OUTPUT_PATH = "hotel_cancellation_pipeline.pkl"

# These are the raw columns the final pipeline (and therefore the Streamlit
# app) will expect as input. They are chosen because they are all known
# BEFORE a booking's final outcome (no leakage) and because they are
# meaningful enough for a user to enter through a simple form.
NUMERICAL_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "previous_cancellations",
    "booking_changes",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "total_stay",       # engineered
    "total_guests",     # engineered
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "market_segment",
    "deposit_type",
    "customer_type",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_canceled"


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """Load the raw hotel bookings CSV file."""
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Data leakage removal
# ---------------------------------------------------------------------------

def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that are only known AFTER the booking outcome is decided.

    - reservation_status: literally encodes the outcome (Canceled / Check-Out /
      No-Show). Using it would let the model "cheat" and see the answer.
    - reservation_status_date: the date that status was set, which also only
      exists once the outcome is known.

    We also deliberately choose NOT to use several other raw columns even
    though they are not strictly leakage, because they are either identifiers
    with poor generalization, mostly missing, or not realistically something
    a user filling out a prediction form would know:

    - agent / company: numeric ID codes for the travel agent / company that
      made the booking. 'company' is ~94% missing and both are just IDs used
      internally by the hotel, not something a front-desk user would type
      into a prediction form.
    - country: high-cardinality (177 values) and not something we ask a user
      to guess about a future booking in this simplified interview project.
    - assigned_room_type: this is the room the guest was actually GIVEN,
      which is often only finalized close to or at check-in. Using it risks
      leaking information correlated with the final stay. We keep
      'reserved_room_type' style information out too, to keep the feature
      list compact and leakage-safe, favoring booking-intent features
      instead (lead_time, deposit_type, market_segment, etc.).
    - arrival_date_year / arrival_date_week_number / arrival_date_day_of_month:
      dropped in favor of arrival_date_month, which captures seasonality
      without overfitting to specific years present only in training data.
    - is_repeated_guest / previous_bookings_not_canceled / days_in_waiting_list /
      distribution_channel / meal: reasonable signals, but excluded here to
      keep the user-facing form short (10-15 fields) as requested. They can
      be added back later (see README "Future Improvements").
    """
    leakage_cols = ["reservation_status", "reservation_status_date"]
    existing = [c for c in leakage_cols if c in df.columns]
    print(f"Removing leakage columns: {existing}")
    return df.drop(columns=existing)


# ---------------------------------------------------------------------------
# Step 3: Data cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values, duplicates, and invalid rows."""
    df = df.copy()

    # --- Duplicates -------------------------------------------------------
    # This dataset is known to contain a large number of duplicate rows
    # (bookings that appear to be re-exported / repeated). Since we cannot
    # tell whether these are genuine repeat rows or exact duplicate exports,
    # and keeping them would let identical rows leak between the train and
    # test split, we drop exact duplicates.
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows.")

    # --- children -----------------------------------------------------
    # Only 4 rows have a missing 'children' value out of ~119k, so the
    # simplest, safest choice is to fill with 0 (assume no children were
    # specified rather than dropping data).
    df["children"] = df["children"].fillna(0)

    # --- country / agent / company -------------------------------------
    # These columns are NOT used as model features (see remove leakage
    # notes above / feature selection), so we don't need to impute them for
    # modeling purposes. We simply drop them here to keep the dataframe
    # focused on the columns we actually use, rather than deleting rows.
    for col in ["country", "agent", "company"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # --- Impossible bookings ---------------------------------------------
    # Rows where adults + children + babies == 0 describe a booking with no
    # guests at all, which is not a realistic booking. There are only 180
    # such rows out of ~119k (after de-duplication the number may be lower),
    # so we remove them rather than trying to guess/impute a guest count.
    guest_total = df["adults"] + df["children"] + df["babies"]
    invalid_guest_rows = (guest_total == 0).sum()
    df = df[guest_total > 0]
    print(f"Dropped {invalid_guest_rows} rows with zero total guests.")

    # --- adr (average daily rate) outliers --------------------------------
    # adr has a small number of extreme / invalid values (e.g. a negative
    # value and one extreme outlier around 5400). Negative ADR is not
    # meaningful for a room rate, and a single extreme spike would unfairly
    # dominate scaling. We clip adr to a sensible, data-driven range instead
    # of deleting the rows, since these rows may still be genuine bookings.
    df = df[df["adr"] >= 0]
    upper_adr_limit = df["adr"].quantile(0.999)
    df["adr"] = df["adr"].clip(upper=upper_adr_limit)
    print(f"Clipped adr above the 99.9th percentile ({upper_adr_limit:.2f}).")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small number of clearly explainable engineered features."""
    df = df.copy()

    # total_stay: total nights booked (weekend + week nights). A longer
    # planned stay can influence the likelihood of a booking being changed
    # or cancelled compared to a single-night stay.
    df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

    # total_guests: total number of people on the booking. Family / group
    # bookings can behave differently from solo travelers.
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    return df


# ---------------------------------------------------------------------------
# Step 5: Build preprocessing + model pipeline
# ---------------------------------------------------------------------------

def build_pipeline(model) -> Pipeline:
    """
    Build a full preprocessing + model pipeline.

    Numerical features: median imputation + scaling.
    Categorical features: most-frequent imputation + one-hot encoding.

    Wrapping everything (including the classifier) in a single Pipeline
    guarantees the exact same preprocessing is applied at training time and
    at prediction time inside the Streamlit app.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    return pipeline


# ---------------------------------------------------------------------------
# Step 6 & 8: Train, compare, and evaluate models
# ---------------------------------------------------------------------------

def evaluate_model(name, pipeline, X_test, y_test) -> dict:
    """Compute standard classification metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n=== {name} ===")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key.capitalize():10s}: {metrics[key]:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Canceled", "Canceled"]))

    return metrics


def main():
    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Remove leakage columns
    df = remove_leakage_columns(df)

    # 3. Clean
    df = clean_data(df)

    # 4. Feature engineering
    df = engineer_features(df)

    # Baseline cancellation rate, for comparison against model performance
    cancellation_rate = df[TARGET].mean()
    print(f"\nOverall cancellation rate in cleaned data: {cancellation_rate:.2%}")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # 5. Train/test split (80/20, stratified so cancellation ratio is
    # preserved in both sets — important since the classes are imbalanced
    # ~63/37).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 6. Define candidate models.
    # Modest hyperparameters are used deliberately so the saved pipeline
    # stays small (a few MB) and trains quickly, while still giving solid
    # performance for an interview-level project.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)
        fitted_pipelines[name] = pipeline

    results_df = pd.DataFrame(results).set_index("model")
    print("\n=== Model Comparison ===")
    print(results_df.round(4))

    # 7. Select the final model.
    # Primary ranking metric is ROC AUC (threshold-independent measure of
    # overall ranking quality), backed up by F1, since accuracy alone can be
    # misleading on an imbalanced target (63/37 split after cleaning).
    #
    # Tree ensembles (Random Forest, HistGradientBoosting) both clearly beat
    # the Logistic Regression baseline. Random Forest and HistGradientBoosting
    # typically land within a percentage point or two of each other on this
    # dataset. When the gap is small (<= 1.5 points of ROC AUC), we prefer
    # Random Forest: it exposes simple, built-in feature_importances_, is
    # easier to explain end-to-end in an interview, and is the model the
    # project brief explicitly favors when performance is comparable.
    ranked = results_df.sort_values("roc_auc", ascending=False)
    top_model_name = ranked.index[0]
    top_auc = ranked.iloc[0]["roc_auc"]

    if "Random Forest" in ranked.index:
        rf_auc = ranked.loc["Random Forest", "roc_auc"]
        if top_model_name != "Random Forest" and (top_auc - rf_auc) <= 0.015:
            best_model_name = "Random Forest"
        else:
            best_model_name = top_model_name
    else:
        best_model_name = top_model_name

    print(f"\nSelected final model: {best_model_name} "
          f"(ROC AUC={results_df.loc[best_model_name, 'roc_auc']:.4f}, "
          f"top ROC AUC={top_auc:.4f})")
    best_pipeline = fitted_pipelines[best_model_name]

    # 9. Feature importance (for tree-based models)
    print_feature_importance(best_pipeline, best_model_name)

    # 10. Save the full pipeline (preprocessing + model together)
    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_OUTPUT_PATH}")

    # Also save the model comparison table for reference / README use.
    results_df.round(4).to_csv("model_comparison_results.csv")
    print("Saved model comparison table to: model_comparison_results.csv")


def print_feature_importance(pipeline: Pipeline, model_name: str, top_n: int = 15):
    """Print (and explain) feature importance for tree-based models."""
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        print(f"\n{model_name} does not expose feature_importances_; skipping.")
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    cat_feature_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names

    importances = classifier.feature_importances_
    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(f"\n=== Top {top_n} Feature Importances ({model_name}) ===")
    print(importance_df.head(top_n).to_string(index=False))

    importance_df.to_csv("feature_importance.csv", index=False)
    print("Saved full feature importance table to: feature_importance.csv")

    # Simple bar chart, saved as an image for the README / notebook.
    try:
        import matplotlib.pyplot as plt

        top = importance_df.head(top_n).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top["feature"], top["importance"], color="#2E86AB")
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances ({model_name})")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.close()
        print("Saved feature importance chart to: feature_importance.png")
    except Exception as e:
        print(f"Could not save feature importance chart: {e}")


if __name__ == "__main__":
    main()
