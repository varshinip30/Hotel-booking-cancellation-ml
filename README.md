# Hotel Booking Cancellation Prediction System

A machine learning project that predicts whether a
hotel booking will be cancelled, deployed as an interactive Streamlit web
application.

##  Live Demo

Try the deployed application here:

 **[Launch Hotel Booking Cancellation Predictor](https://hotel-booking-cancellation-ml-4djqhdcgu7mh3b2ihsucyl.streamlit.app/)**

## Problem Statement

Hotel cancellations are costly: they create planning uncertainty for
housekeeping, staffing, and revenue management. This project builds a
binary classification model that estimates, at the time a booking is made,
how likely that booking is to be cancelled — using only information that
would realistically be available before the booking's outcome is known.

## Dataset Description

The dataset contains **119,390 hotel booking records** across two hotel
types (City Hotel and Resort Hotel), with 32 columns describing the
booking, the guest, and the eventual reservation status. It is the
well-known "Hotel Booking Demand" dataset (Antonio, de Almeida & Nunes,
2019).

Target variable: `is_canceled`
- `0` = Booking Not Cancelled
- `1` = Booking Cancelled

Baseline cancellation rate in the raw data is **37.0%**; after cleaning
(removing exact duplicates and invalid rows), it is **27.3%**.

## Project Objective

Train and compare several classification models, select the best one based
on overall performance (not accuracy alone), and ship it inside a
production-style scikit-learn `Pipeline` that a Streamlit app can call
directly — with zero risk of preprocessing drift between training and
deployment.

## ML Workflow

1. Load and explore the raw CSV with pandas.
2. Remove target-leakage columns.
3. Clean missing values, duplicates, and invalid rows.
4. Engineer two simple, explainable features.
5. Build a `ColumnTransformer` + `Pipeline` for preprocessing.
6. Split into train/test (80/20, stratified).
7. Train and compare Logistic Regression, Random Forest, and
   HistGradientBoosting.
8. Evaluate with accuracy, precision, recall, F1, ROC AUC, and a confusion
   matrix.
9. Inspect feature importance for the selected model.
10. Save the full pipeline (preprocessing + model) with `joblib`.
11. Serve predictions through a Streamlit app that loads that same pipeline.

## Exploratory Data Analysis (Key Findings)

- **Shape:** 119,390 rows × 32 columns.
- **Missing values:** `company` (94.3% missing), `agent` (13.7% missing),
  `country` (0.4% missing), `children` (4 rows missing).
- **Duplicates:** 32,252 exact duplicate rows (dropped during cleaning).
- **Target distribution:** ~63% not cancelled / ~37% cancelled in the raw
  data (imbalanced, but not extremely so).
- **Invalid bookings:** 166–180 rows had `adults + children + babies == 0`
  (no guests at all) — removed.
- **ADR (average daily rate):** mostly a reasonable distribution, but
  contained one negative value and one extreme outlier (5,400) — negative
  values removed, extreme high values clipped at the 99.9th percentile.
- **Correlation with cancellation:** `lead_time` (longer lead time → more
  likely to cancel) and `previous_cancellations` correlate positively;
  `total_of_special_requests`, `required_car_parking_spaces`, and
  `booking_changes` correlate negatively (guests who engage more with their
  booking are less likely to cancel).
- City Hotel has a noticeably higher cancellation rate (~42%) than Resort
  Hotel (~28%).

## Data Leakage Prevention

Two columns were removed immediately because they directly encode the
booking's final outcome and would not be known at prediction time:

- **`reservation_status`** — literally states `Canceled`, `Check-Out`, or
  `No-Show`. This is a near-perfect proxy for the target.
- **`reservation_status_date`** — the date that final status was recorded;
  it also only exists once the outcome is known.

Beyond these two obvious leakage columns, every other raw column was
reviewed for realistic availability at prediction time. Several columns
were deliberately **excluded from the model's feature set** (not because
they leak the target, but because they are IDs, high-cardinality, or would
make the user-facing form unwieldy):

| Column | Reason for exclusion |
|---|---|
| `agent`, `company` | Internal numeric ID codes; `company` is 94% missing and neither is something a user filling out a prediction form would know. |
| `country` | 177 distinct values (high cardinality); not requested as a simplified form field. |
| `assigned_room_type` | The room a guest is *actually given*, often finalized close to check-in — using it risks smuggling in outcome-adjacent information. `reserved_room_type` was also left out to keep the feature set compact and focused on booking-intent signals. |
| `arrival_date_year`, `arrival_date_week_number`, `arrival_date_day_of_month` | Replaced by `arrival_date_month`, which captures seasonality without overfitting to the specific years present in this historical dataset. |
| `is_repeated_guest`, `previous_bookings_not_canceled`, `days_in_waiting_list`, `distribution_channel`, `meal` | Reasonable, non-leaky signals, but left out to keep the Streamlit form to a manageable ~15 fields (see "Future Improvements" below). |

## Data Cleaning

- **Duplicates:** 32,252 exact duplicate rows dropped.
- **`children`:** 4 missing values filled with `0`.
- **`country` / `agent` / `company`:** dropped as columns (not used as
  features — see leakage table above), rather than deleting rows to
  preserve them.
- **Impossible bookings:** rows with `adults + children + babies == 0`
  removed (166 rows after de-duplication).
- **`adr` outliers:** 1 negative value removed; values above the 99.9th
  percentile (~335) clipped rather than deleted, to avoid discarding
  otherwise-valid rows.

## Feature Engineering

| Feature | Formula | Rationale |
|---|---|---|
| `total_stay` | `stays_in_weekend_nights + stays_in_week_nights` | Total nights booked; longer stays can behave differently from single-night bookings. |
| `total_guests` | `adults + children + babies` | Total party size; family/group bookings can cancel at different rates than solo travelers. |

## Preprocessing Pipeline

Built with scikit-learn's `ColumnTransformer` and `Pipeline`:

- **Numerical features** (`lead_time`, `stays_in_weekend_nights`,
  `stays_in_week_nights`, `adults`, `children`, `previous_cancellations`,
  `booking_changes`, `adr`, `required_car_parking_spaces`,
  `total_of_special_requests`, `total_stay`, `total_guests`): median
  imputation, then `StandardScaler`.
- **Categorical features** (`hotel`, `arrival_date_month`,
  `market_segment`, `deposit_type`, `customer_type`): most-frequent
  imputation, then `OneHotEncoder(handle_unknown="ignore")`.

The preprocessor and the classifier are combined into **one single
`Pipeline` object**, which is what gets saved with `joblib`. This means the
Streamlit app never needs to manually one-hot encode or scale anything —
it just calls `pipeline.predict()` / `pipeline.predict_proba()` on a raw
DataFrame.

## Train/Test Split

- 80% train / 20% test
- `random_state=42` for reproducibility
- `stratify=y` so the ~27% cancellation rate is preserved identically in
  both the training and test sets — important given the class imbalance.

## Models Compared

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7801 | 0.6806 | 0.3665 | 0.4765 | 0.7850 |
| **Random Forest (selected)** | 0.8024 | 0.7338 | 0.4335 | 0.5450 | 0.8313 |
| HistGradientBoosting | 0.8079 | 0.7144 | 0.4939 | 0.5840 | 0.8425 |

*(Actual results from training on the provided dataset — see
`model_comparison_results.csv` after running `train_model.py`.)*

## Final Selected Model: Random Forest

HistGradientBoosting scored marginally higher on ROC AUC (0.8425 vs.
0.8313), but the gap is small (~1.1 points). Random Forest was selected as
the final model because it:

- Performs close to the top model on every metric.
- Exposes built-in `feature_importances_`, making it far easier to explain
  *why* it made a given prediction — valuable for an interview setting.
- Is simple to reason about and describe end-to-end (a collection of
  decision trees) compared to a boosting algorithm.

This trade-off (giving up ~1 point of ROC AUC for materially better
explainability) is a reasonable, defensible choice to walk through in an
interview.

## Evaluation Metrics — Why Not Just Accuracy?

The dataset is imbalanced (~27% cancellations after cleaning), so a model
that always predicted "not cancelled" would already score ~73% accuracy
while being completely useless. That's why this project reports:

- **Precision** — of the bookings the model flags as "will cancel," how
  many actually do? Matters if the hotel plans to act on the flag (e.g.
  overbooking strategy) — false alarms have a cost.
- **Recall** — of all bookings that actually get cancelled, how many did
  the model catch? Matters if missing a true cancellation is costly.
- **F1** — the balance between precision and recall.
- **ROC AUC** — how well the model ranks cancelled vs. not-cancelled
  bookings across all thresholds, independent of any single cutoff.

The confusion matrix and full classification report for each model are
printed by `train_model.py` and reflect real performance on the held-out
20% test set — no fabricated numbers.

## Feature Importance

Top drivers of predicted cancellation, per the trained Random Forest:

1. `lead_time` — bookings made far in advance are more likely to be
   cancelled.
2. `total_of_special_requests` — guests with more special requests are
   less likely to cancel (higher engagement with the booking).
3. `market_segment_Online TA` — bookings from online travel agencies show
   a distinct cancellation pattern.
4. `required_car_parking_spaces` — requesting parking is associated with
   lower cancellation likelihood.
5. `deposit_type_Non Refund` / `previous_cancellations` / `adr` — deposit
   type and prior cancellation history are also influential.

See `feature_importance.png` / `feature_importance.csv` (generated by
`train_model.py`) for the full ranked chart and table.

**Important:** feature importance shows which inputs the model relies on
most to make predictions — it does **not** prove that any single feature
*causes* cancellations.

## Streamlit Application

The Streamlit app (`app.py`) loads `hotel_cancellation_pipeline.pkl`
directly and exposes a simple form covering the model's real input
features:

- Hotel Type, Arrival Month, Lead Time, Weekend Nights, Week Nights
- Number of Adults, Number of Children
- Market Segment, Deposit Type, Customer Type
- Previous Cancellations, Average Daily Rate (ADR)
- Required Parking Spaces, Number of Special Requests, Booking Changes

Clicking **Predict Cancellation** shows either **Likely to be Cancelled**
or **Likely to be Honored**, along with the model's estimated cancellation
probability (clearly labeled as an estimate, not a guarantee).

Raw form inputs are assembled into a single-row pandas DataFrame with
exactly the raw column names the saved pipeline expects. The pipeline
itself performs all imputation, scaling, and encoding — the app never
manually builds dummy variables or zero-fills unknown columns.

## Project Structure

```
hotel-booking-cancellation/
├── data/
│   └── hotel_bookings.csv
├── notebooks/
│   └── hotel_booking_analysis.ipynb
├── src/
│   └── train_model.py
├── app.py
├── hotel_cancellation_pipeline.pkl
├── model_comparison_results.csv
├── feature_importance.csv
├── feature_importance.png
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation Instructions

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## How to Train the Model

Make sure `data/hotel_bookings.csv` exists, then from the project root run:

```bash
python src/train_model.py
```

This will print EDA summaries, train and compare all three models, print
evaluation metrics, save the final pipeline to
`hotel_cancellation_pipeline.pkl`, and save
`model_comparison_results.csv` / `feature_importance.csv` /
`feature_importance.png`.

## How to Run the Streamlit Application

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`).

## Example Prediction Workflow

1. Launch the app with `streamlit run app.py`.
2. Fill in booking details, e.g.:
   - Hotel Type: City Hotel
   - Lead Time: 300 days
   - Market Segment: Online TA
   - Deposit Type: Non Refund
   - Previous Cancellations: 3
3. Click **Predict Cancellation**.
4. The app displays **Likely to be Cancelled** with a probability such as
   `Cancellation Probability: 85.3%`.
5. Try a low-risk example (short lead time, Direct market segment, No
   Deposit, several special requests) to see **Likely to be Honored** with
   a low probability instead.

## Limitations

- Trained on a specific historical dataset (2015–2017, two Portuguese
  hotels); may not generalize to other markets, currencies, or booking
  eras without retraining.
- Several potentially useful columns (`country`, `agent`, `company`,
  `distribution_channel`, `meal`, guest repeat history) were intentionally
  excluded to keep the form short and avoid high-cardinality / mostly-
  missing features — this trades away some signal for simplicity.
- Class imbalance (~27% cancellations) means recall on the minority
  ("cancelled") class is meaningfully lower than precision; the model
  misses a portion of true cancellations.
- Feature importance reflects correlation-driven model behavior, not
  causal drivers of guest behavior.

## Possible Future Improvements

- Add back `distribution_channel`, `meal`, and guest repeat-history
  features and re-evaluate their impact.
- Tune classification threshold (instead of the default 0.5) to trade off
  precision vs. recall based on business cost.
- Add cross-validation and hyperparameter search (e.g. `GridSearchCV`) for
  more rigorous model selection.
- Group rare `country` values into an "Other" bucket and reintroduce it as
  a low-cardinality feature.
- Add model monitoring / retraining pipeline for production use with fresh
  booking data.
