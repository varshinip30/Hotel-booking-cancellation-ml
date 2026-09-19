"""
app.py
-------
Streamlit web application for the Hotel Booking Cancellation Predictor.

This app loads the SAME pipeline object that was trained and saved by
src/train_model.py (preprocessing + model bundled together). The raw user
inputs collected from the form are placed into a single-row pandas
DataFrame with exactly the raw column names the pipeline expects, and the
pipeline itself handles imputation, scaling, and one-hot encoding.

No manual dummy-variable creation and no manual zero-filling of unknown
columns happens here — the saved pipeline is the single source of truth for
preprocessing, so training and deployment can never drift apart.

Run with:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "hotel_cancellation_pipeline.pkl"

# Dropdown options mirror the categories actually seen in the training data.
MONTH_OPTIONS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
HOTEL_OPTIONS = ["City Hotel", "Resort Hotel"]
MARKET_SEGMENT_OPTIONS = [
    "Online TA", "Offline TA/TO", "Groups", "Direct",
    "Corporate", "Complementary", "Aviation",
]
DEPOSIT_TYPE_OPTIONS = ["No Deposit", "Non Refund", "Refundable"]
CUSTOMER_TYPE_OPTIONS = ["Transient", "Transient-Party", "Contract", "Group"]


@st.cache_resource
def load_pipeline(path: str):
    """Load the trained preprocessing + model pipeline once and cache it."""
    return joblib.load(path)


def build_input_dataframe(
    hotel,
    arrival_month,
    lead_time,
    weekend_nights,
    week_nights,
    adults,
    children,
    market_segment,
    previous_cancellations,
    deposit_type,
    customer_type,
    adr,
    parking_spaces,
    special_requests,
    booking_changes,
) -> pd.DataFrame:
    """
    Assemble a single-row DataFrame with the exact raw feature names the
    saved pipeline was trained on (see NUMERICAL_FEATURES / CATEGORICAL_
    FEATURES in src/train_model.py). The two engineered features
    (total_stay, total_guests) are computed here the same way they were
    computed during training.
    """
    total_stay = weekend_nights + week_nights
    total_guests = adults + children  # babies not collected in the simplified form; treated as 0

    row = {
        "lead_time": lead_time,
        "stays_in_weekend_nights": weekend_nights,
        "stays_in_week_nights": week_nights,
        "adults": adults,
        "children": children,
        "previous_cancellations": previous_cancellations,
        "booking_changes": booking_changes,
        "adr": adr,
        "required_car_parking_spaces": parking_spaces,
        "total_of_special_requests": special_requests,
        "total_stay": total_stay,
        "total_guests": total_guests,
        "hotel": hotel,
        "arrival_date_month": arrival_month,
        "market_segment": market_segment,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
    }
    return pd.DataFrame([row])


def main():
    st.set_page_config(page_title="Hotel Booking Cancellation Predictor", page_icon="🏨")

    st.title("Hotel Booking Cancellation Predictor")
    st.write(
        "Enter the booking details below to estimate the likelihood that "
        "the reservation will be cancelled."
    )

    try:
        pipeline = load_pipeline(MODEL_PATH)
    except FileNotFoundError:
        st.error(
            f"Could not find '{MODEL_PATH}'. Train the model first by running "
            "`python src/train_model.py` from the project root."
        )
        st.stop()

    st.subheader("Booking Details")

    col1, col2 = st.columns(2)

    with col1:
        hotel = st.selectbox("Hotel Type", HOTEL_OPTIONS)
        arrival_month = st.selectbox("Arrival Month", MONTH_OPTIONS)
        lead_time = st.number_input(
            "Lead Time (days between booking and arrival)",
            min_value=0, max_value=800, value=50,
        )
        weekend_nights = st.number_input(
            "Weekend Nights", min_value=0, max_value=20, value=1
        )
        week_nights = st.number_input(
            "Week Nights", min_value=0, max_value=40, value=2
        )
        adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        booking_changes = st.number_input(
            "Number of Booking Changes", min_value=0, max_value=20, value=0
        )

    with col2:
        market_segment = st.selectbox("Market Segment", MARKET_SEGMENT_OPTIONS)
        deposit_type = st.selectbox("Deposit Type", DEPOSIT_TYPE_OPTIONS)
        customer_type = st.selectbox("Customer Type", CUSTOMER_TYPE_OPTIONS)
        previous_cancellations = st.number_input(
            "Previous Cancellations by this Guest", min_value=0, max_value=30, value=0
        )
        adr = st.number_input(
            "Average Daily Rate (ADR, in currency units)",
            min_value=0.0, max_value=1000.0, value=100.0, step=1.0,
        )
        parking_spaces = st.number_input(
            "Required Car Parking Spaces", min_value=0, max_value=5, value=0
        )
        special_requests = st.number_input(
            "Number of Special Requests", min_value=0, max_value=10, value=0
        )

    st.markdown("---")

    if st.button("Predict Cancellation", type="primary"):
        input_df = build_input_dataframe(
            hotel=hotel,
            arrival_month=arrival_month,
            lead_time=lead_time,
            weekend_nights=weekend_nights,
            week_nights=week_nights,
            adults=adults,
            children=children,
            market_segment=market_segment,
            previous_cancellations=previous_cancellations,
            deposit_type=deposit_type,
            customer_type=customer_type,
            adr=adr,
            parking_spaces=parking_spaces,
            special_requests=special_requests,
            booking_changes=booking_changes,
        )

        try:
            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0][1]
        except Exception as e:
            st.error(f"Something went wrong while generating the prediction: {e}")
            st.stop()

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("**Likely to be Cancelled**")
        else:
            st.success("**Likely to be Honored**")

        st.metric("Cancellation Probability (model estimate)", f"{probability * 100:.1f}%")
        st.caption(
            "This probability is an estimate produced by a machine learning "
            "model trained on historical booking data. It is not a guarantee "
            "of the booking's actual outcome."
        )

        with st.expander("See the exact input sent to the model"):
            st.dataframe(input_df)


if __name__ == "__main__":
    main()
