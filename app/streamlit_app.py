"""Streamlit app for mushroom edibility classification."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/mushroom_classifier.joblib")


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError("Trained model not found. Run src/train_model.py first.")
    return joblib.load(path)


def option_list(options: dict[str, str]) -> list[str]:
    # Present a human-readable option but preserve the dataset code.
    return [f"{label} ({code})" for code, label in options.items()]


def extract_code(selection: str) -> str:
    # Streamlit select boxes return the combined string; we keep the code between parentheses.
    if "(" in selection and selection.endswith(")"):
        return selection.split("(")[-1].rstrip(")")
    return selection


def main() -> None:
    st.set_page_config(page_title="Mushroom Classifier", page_icon="🍄")
    st.title("🍄 Mushroom Edibility Classifier")
    st.write(
        "Provide mushroom characteristics below to predict whether the species is edible or poisonous."
    )

    model = None
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    numeric_cols = {
        "cap-diameter": (0.0, 30.0, 0.1, 15.0),
        "stem-height": (0.0, 40.0, 0.1, 15.0),
        "stem-width": (0.0, 30.0, 0.1, 10.0),
    }

    categorical_cols = {
        "cap-shape": {
            "b": "bell",
            "c": "conical",
            "x": "convex",
            "f": "flat",
            "s": "sunken",
            "p": "spherical",
            "o": "other",
        },
        "cap-surface": {
            "i": "fibrous",
            "g": "grooves",
            "y": "scaly",
            "s": "smooth",
            "h": "shiny",
            "l": "leathery",
            "k": "silky",
            "t": "sticky",
            "w": "wrinkled",
            "e": "fleshy",
        },
        "cap-color": {
            "n": "brown",
            "b": "buff",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow",
            "l": "blue",
            "o": "orange",
            "k": "black",
        },
        "does-bruise-or-bleed": {"t": "yes", "f": "no"},
        "gill-attachment": {
            "a": "adnate",
            "x": "adnexed",
            "d": "decurrent",
            "e": "free",
            "s": "sinuate",
            "p": "pores",
            "f": "none",
            "?": "unknown",
        },
        "gill-spacing": {"c": "close", "d": "distant", "f": "none"},
        "gill-color": {
            "n": "brown",
            "b": "buff",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow",
            "l": "blue",
            "o": "orange",
            "k": "black",
            "f": "none",
        },
        "stem-root": {
            "b": "bulbous",
            "s": "swollen",
            "c": "club",
            "u": "cup",
            "e": "equal",
            "z": "rhizomorphs",
            "r": "rooted",
        },
        "stem-surface": {
            "i": "fibrous",
            "g": "grooves",
            "y": "scaly",
            "s": "smooth",
            "h": "shiny",
            "l": "leathery",
            "k": "silky",
            "t": "sticky",
            "w": "wrinkled",
            "e": "fleshy",
            "f": "none",
        },
        "stem-color": {
            "n": "brown",
            "b": "buff",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow",
            "l": "blue",
            "o": "orange",
            "k": "black",
            "f": "none",
        },
        "veil-type": {"p": "partial", "u": "universal"},
        "veil-color": {
            "n": "brown",
            "b": "buff",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow",
            "l": "blue",
            "o": "orange",
            "k": "black",
            "f": "none",
        },
        "has-ring": {"t": "ring", "f": "none"},
        "ring-type": {
            "c": "cobwebby",
            "e": "evanescent",
            "r": "flaring",
            "g": "grooved",
            "l": "large",
            "p": "pendant",
            "s": "sheathing",
            "z": "zone",
            "y": "scaly",
            "m": "movable",
            "f": "none",
            "?": "unknown",
        },
        "spore-print-color": {
            "n": "brown",
            "b": "buff",
            "g": "gray",
            "r": "green",
            "p": "pink",
            "u": "purple",
            "e": "red",
            "w": "white",
            "y": "yellow",
            "l": "blue",
            "o": "orange",
            "k": "black",
        },
        "habitat": {
            "g": "grasses",
            "l": "leaves",
            "m": "meadows",
            "p": "paths",
            "h": "heaths",
            "u": "urban",
            "w": "waste",
            "d": "woods",
        },
        "season": {"s": "spring", "u": "summer", "a": "autumn", "w": "winter"},
    }

    with st.form("prediction_form"):
        st.subheader("Mushroom Features")
        input_data: dict[str, float | str] = {}

        for col, (min_val, max_val, step, default) in numeric_cols.items():
            input_data[col] = st.slider(
                label=col.replace("-", " ").title(),
                min_value=min_val,
                max_value=max_val,
                value=default,
                step=step,
            )

        for col, choices in categorical_cols.items():
            selection = st.selectbox(
                label=col.replace("-", " ").title(),
                options=option_list(choices),
                index=0,
            )
            input_data[col] = extract_code(selection)

        submitted = st.form_submit_button("Classify Mushroom")

    if submitted and model is not None:
        input_df = pd.DataFrame([input_data])
        proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        edible_prob = proba[0]
        poisonous_prob = proba[1]

        st.success(f"Predicted class: **{prediction.capitalize()}**")
        st.metric("Probability - Edible", f"{edible_prob:.2%}")
        st.metric("Probability - Poisonous", f"{poisonous_prob:.2%}")
        st.write("Feature values used:")
        st.json(input_data)

        st.info("Remember: Always cross-check with authoritative sources before consuming wild mushrooms.")


if __name__ == "__main__":
    main()
