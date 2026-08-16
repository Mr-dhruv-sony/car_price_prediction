"""CarValue AI — Streamlit inference app for used-car price estimates."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "random_forest_regression_model.pkl"
DATA_PATH = APP_DIR / "car.csv"

# The existing model was trained with this reference year. Keeping it fixed in
# both training and inference ensures the feature engineering matches the model.
REFERENCE_YEAR = 2024
FEATURE_COLUMNS = [
    "Present_Price",
    "Kms_Driven",
    "Owner",
    "Years_Since_Manufacture",
    "Fuel_Type_Diesel",
    "Fuel_Type_Petrol",
    "Seller_Type_Individual",
    "Transmission_Manual",
]

FEATURE_LABELS = {
    "Present_Price": "Showroom price",
    "Kms_Driven": "Kilometres driven",
    "Owner": "Previous owners",
    "Years_Since_Manufacture": "Vehicle age",
    "Fuel_Type_Diesel": "Diesel fuel",
    "Fuel_Type_Petrol": "Petrol fuel",
    "Seller_Type_Individual": "Individual seller",
    "Transmission_Manual": "Manual transmission",
}


st.set_page_config(
    page_title="CarValue AI",
    page_icon=":material/directions_car:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model once per Streamlit process."""

    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


@st.cache_data(ttl="1h", max_entries=2, show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the source dataset for validation and the data explorer."""

    return pd.read_csv(DATA_PATH)


def prepare_training_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Recreate the feature engineering used during model training."""

    transformed = data.copy()
    transformed["Years_Since_Manufacture"] = REFERENCE_YEAR - transformed["Year"]
    transformed = transformed.drop(columns=["Year", "Car_Name"], errors="ignore")
    transformed = pd.get_dummies(transformed, drop_first=True)
    transformed = transformed.reindex(
        columns=["Selling_Price", *FEATURE_COLUMNS],
        fill_value=0,
    )
    return transformed[FEATURE_COLUMNS], transformed["Selling_Price"]


def make_prediction_features(
    present_price: float,
    kms_driven: int,
    owner: int,
    year: int,
    fuel_type: str,
    seller_type: str,
    transmission: str,
) -> pd.DataFrame:
    """Build one inference row with the model's exact feature order."""

    features = pd.DataFrame(
        [
            {
                "Present_Price": present_price,
                "Kms_Driven": kms_driven,
                "Owner": owner,
                "Years_Since_Manufacture": REFERENCE_YEAR - year,
                "Fuel_Type_Diesel": int(fuel_type == "Diesel"),
                "Fuel_Type_Petrol": int(fuel_type == "Petrol"),
                "Seller_Type_Individual": int(seller_type == "Individual"),
                "Transmission_Manual": int(transmission == "Manual"),
            }
        ]
    )
    return features[FEATURE_COLUMNS]


@st.cache_data(ttl="1h", max_entries=1, show_spinner=False)
def get_holdout_metrics() -> dict[str, float | int]:
    """Evaluate the saved artifact on the notebook's reproducible test split."""

    data = load_dataset()
    features, target = prepare_training_data(data)
    _, test_features, _, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )
    predictions = load_model().predict(test_features)
    return {
        "rmse": float(np.sqrt(mean_squared_error(test_target, predictions))),
        "mae": float(mean_absolute_error(test_target, predictions)),
        "r2": float(r2_score(test_target, predictions)),
        "test_rows": int(len(test_target)),
    }


def get_tree_consensus(model, features: pd.DataFrame) -> tuple[float, float]:
    """Return a descriptive 10th–90th percentile range across forest trees."""

    if not hasattr(model, "estimators_"):
        prediction = float(model.predict(features)[0])
        return prediction, prediction

    tree_predictions = np.asarray(
        [tree.predict(features)[0] for tree in model.estimators_],
        dtype=float,
    )
    return (
        float(np.percentile(tree_predictions, 10)),
        float(np.percentile(tree_predictions, 90)),
    )


def format_lakh_price(value: float) -> str:
    """Format a model output in the same lakh unit as the training target."""

    return f"₹ {value:,.2f} lakh"


def render_prediction_result() -> None:
    """Render the latest prediction stored in the current user session."""

    result = st.session_state.get("prediction_result")
    if result is None:
        st.info(
            "Enter the car details above and select **Estimate selling price** "
            "to generate a live estimate.",
            icon=":material/lightbulb:",
        )
        return

    with st.container(border=True):
        st.subheader("Estimated market price", divider="blue")
        result_columns = st.columns(3, gap="medium")
        result_columns[0].metric(
            "Predicted selling price",
            format_lakh_price(result["prediction"]),
            icon=":material/payments:",
            border=True,
        )
        result_columns[1].metric(
            "Approximate rupee value",
            f"₹ {result['prediction'] * 100_000:,.0f}",
            icon=":material/currency_rupee:",
            border=True,
        )
        result_columns[2].metric(
            "Forest consensus range",
            f"₹ {result['lower']:.2f}–{result['upper']:.2f} L",
            icon=":material/insights:",
            border=True,
        )
        st.caption(
            "The consensus range shows the 10th–90th percentile of individual "
            "Random Forest tree predictions. It is a model-spread indicator, "
            "not a guaranteed valuation interval."
        )


try:
    model = load_model()
    dataset = load_dataset()
except FileNotFoundError as error:
    st.error(
        f"A required project file is missing: `{error.filename}`. "
        "Place the model and dataset beside `app.py` and restart the app.",
        icon=":material/error:",
    )
    st.stop()

st.session_state.setdefault("prediction_result", None)

with st.sidebar:
    st.markdown("## :material/directions_car: CarValue AI")
    st.caption("A practical Random Forest estimator for used-car prices")
    st.badge("Model online", icon=":material/check_circle:", color="green")
    st.space("small")
    st.metric("Dataset rows", f"{len(dataset):,}", icon=":material/table_rows:")
    st.metric("Model trees", f"{len(getattr(model, 'estimators_', [])):,}", icon=":material/forest:")
    st.caption(
        f"Feature engineering uses {REFERENCE_YEAR} as the reference year, "
        "matching the training artifact."
    )
    st.space("small")
    st.markdown("**Built with**")
    st.caption("Python · pandas · scikit-learn · Streamlit")

st.title("Estimate your car's resale value")
st.write(
    "A clean, explainable estimate powered by the car's showroom price, usage, "
    "age, ownership history, and configuration."
)
st.badge("Live inference", icon=":material/bolt:", color="blue")

prediction_tab, insights_tab, explorer_tab = st.tabs(
    [
        ":material/auto_awesome: Price estimate",
        ":material/analytics: Model insights",
        ":material/database: Data explorer",
    ]
)

with prediction_tab:
    st.space("small")
    with st.container(border=True):
        st.subheader("Describe the car")
        st.caption("All prices are entered and returned in Indian lakh units.")
        with st.form("prediction_form"):
            details_left, details_right = st.columns(2, gap="large")
            with details_left:
                present_price = st.number_input(
                    "Showroom price (₹ lakh)",
                    min_value=0.10,
                    max_value=100.00,
                    value=5.59,
                    step=0.05,
                    format="%.2f",
                    help="The listed showroom price when the car was new.",
                )
                kms_driven = st.number_input(
                    "Kilometres driven",
                    min_value=0,
                    max_value=500_000,
                    value=27_000,
                    step=500,
                )
                year = st.number_input(
                    "Year of manufacture",
                    min_value=1990,
                    max_value=REFERENCE_YEAR,
                    value=2014,
                    step=1,
                    help=f"The model derives vehicle age as {REFERENCE_YEAR} minus this year.",
                )
                owner = st.number_input(
                    "Previous owners",
                    min_value=0,
                    max_value=5,
                    value=0,
                    step=1,
                )
            with details_right:
                fuel_type = st.segmented_control(
                    "Fuel type",
                    ["Petrol", "Diesel", "CNG"],
                    default="Petrol",
                    required=True,
                    key="fuel_type",
                    width="stretch",
                )
                seller_type = st.segmented_control(
                    "Seller type",
                    ["Dealer", "Individual"],
                    default="Dealer",
                    required=True,
                    key="seller_type",
                    width="stretch",
                )
                transmission = st.segmented_control(
                    "Transmission",
                    ["Manual", "Automatic"],
                    default="Manual",
                    required=True,
                    key="transmission",
                    width="stretch",
                )
                st.info(
                    f"The car's modeled age will be **{REFERENCE_YEAR - year} years**.",
                    icon=":material/calendar_month:",
                )

            submitted = st.form_submit_button(
                "Estimate selling price",
                type="primary",
                icon=":material/auto_awesome:",
                width="stretch",
            )

        if submitted and fuel_type and seller_type and transmission:
            feature_row = make_prediction_features(
                present_price=float(present_price),
                kms_driven=int(kms_driven),
                owner=int(owner),
                year=int(year),
                fuel_type=fuel_type,
                seller_type=seller_type,
                transmission=transmission,
            )
            raw_prediction = float(model.predict(feature_row)[0])
            lower, upper = get_tree_consensus(model, feature_row)
            st.session_state["prediction_result"] = {
                "prediction": max(0.0, raw_prediction),
                "lower": max(0.0, lower),
                "upper": max(0.0, upper),
            }
            st.toast("Estimate ready", icon=":material/check_circle:")

    render_prediction_result()
    st.caption(
        "Use the estimate as a data point for negotiation. Actual resale value "
        "can vary with location, condition, service history, and market demand."
    )

with insights_tab:
    st.space("small")
    st.subheader("A transparent baseline")
    st.write(
        "The artifact is a tuned Random Forest regressor. The metrics below use "
        "the same 80/20, random-state-42 holdout split used by the training notebook."
    )
    metrics = get_holdout_metrics()
    metric_columns = st.columns(4, gap="medium")
    metric_columns[0].metric(
        "Holdout R²",
        f"{metrics['r2']:.2f}",
        help="Higher is better; 1.00 is a perfect fit.",
        icon=":material/score:",
        border=True,
    )
    metric_columns[1].metric(
        "Holdout RMSE",
        f"₹ {metrics['rmse']:.2f} L",
        help="Root mean squared error in lakh units; lower is better.",
        icon=":material/speed:",
        border=True,
    )
    metric_columns[2].metric(
        "Holdout MAE",
        f"₹ {metrics['mae']:.2f} L",
        help="Mean absolute error in lakh units; lower is better.",
        icon=":material/straighten:",
        border=True,
    )
    metric_columns[3].metric(
        "Test rows",
        f"{metrics['test_rows']}",
        help="Rows reserved for the reproducible holdout evaluation.",
        icon=":material/science:",
        border=True,
    )

    feature_importance = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[name] for name in model.feature_names_in_],
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    importance_left, importance_right = st.columns([1.25, 1], gap="large")
    with importance_left:
        with st.container(border=True):
            st.subheader("What drives the estimate")
            st.bar_chart(
                feature_importance.set_index("Feature"),
                y="Importance",
                horizontal=True,
                sort=False,
            )
    with importance_right:
        with st.container(border=True):
            st.subheader("Feature engineering")
            st.markdown(
                """
                1. Derive **vehicle age** from the manufacture year.
                2. Drop the high-cardinality `Car_Name` field.
                3. One-hot encode fuel, seller, and transmission categories.
                4. Tune a **Random Forest Regressor** with `RandomizedSearchCV`.
                """
            )
            st.caption(
                "Showroom price is the strongest signal in this dataset, while "
                "configuration and usage provide additional context."
            )

with explorer_tab:
    st.space("small")
    st.subheader("Explore the training data")
    st.write(
        "The public dataset contains 301 historical used-car listings. The model "
        "predicts `Selling_Price` in lakh units."
    )
    display_columns = [
        "Car_Name",
        "Year",
        "Selling_Price",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
    ]
    st.dataframe(
        dataset[display_columns],
        hide_index=True,
        height=360,
        column_config={
            "Selling_Price": st.column_config.NumberColumn(
                "Selling price (₹ L)", format="₹ %.2f"
            ),
            "Present_Price": st.column_config.NumberColumn(
                "Showroom price (₹ L)", format="₹ %.2f"
            ),
            "Kms_Driven": st.column_config.NumberColumn(
                "Kilometres driven", format="%d"
            ),
        },
    )
    st.subheader("Showroom price vs. resale price")
    st.scatter_chart(
        dataset,
        x="Present_Price",
        y="Selling_Price",
        color="Fuel_Type",
        height=420,
    )
