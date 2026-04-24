import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="QB PPA Predictor", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('qb_xgb_model.pkl')

model = load_model()

FEATURES = [
    'years_in_college', 
    'prev_pass_yds', 
    'prev_pass_td', 
    'prev_avg_ppa', 
    'dest_off_ppa', 
    'dest_sp_offense', 
    'is_transfer'
]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["The Model", "Predict"])

# --- PAGE 1: EXPLAINER ---
if page == "The Model":
    st.title("QB Performance Explainer")
    st.write("""
    This model uses **XGBoost** to predict a Quarterback's PPA (Predicted Points Added) 
    for the upcoming season based on their previous performance.
    """)
    
    st.subheader("What features are most important to the model?")
    st.image('feature_importance.png')

    st.write("""
    Feature Definitions:
    - *prev_pass_td* = Number of a QB's Passing Touchdowns from previous season
    - *dest_off_ppa* = Destination Team's previous season PPA
    - *prev_pass_yds* = Number of a QB's Passing Yards from previous season
    - *dest_sp_offence* = Destination Team's Offensive SP+ Rating
    - *prev_avg_ppa* = QB's PPA from previous season
    - *is_transfer* = No(0)/Yes(1) selection of whether a QB is a transfer
    - *years_in_college* = Number of years/seasons a QB has played in college
    """)
    st.info("Many offensive stats are highlighted here, which is logical for the position. " \
    "Predicting a QB's PPA is based on the combination of how well the QB and the Destination Team plays offensively.")

    st.warning("Although *years_in_college* has a low importance score, and in the below SHAP value graph " \
    "it has a neglegible effect on the PPA value- this feature was left in as the number of years/seasons a QB has played  " \
    "is an overall important front office statistic when wanting to recruit *any* new player")

    st.subheader("What drives the predictions?")
    st.write("""
    The chart below shows **SHAP values**. 
    - **Red** = High value for that stat.
    - **Blue** = Low value for that stat.
    - **Right side** = Pushes the prediction higher.
    - **Left side** = Pushes the prediction lower.
    """)

    st.image('shap_summary.png')
    st.info("Generally speaking, if the QB and Destination Team have high values across all stats, " \
    "the model is inclined to give a higher PPA score given the QB is *not* a transfer - as seen by the blue along *is_transfer*" \
    "\n\n"
    "However, if all stats have high values but the QB *is* a transfer, the model will apply a **Transfer Penality** resulting in a" \
    "lower PPA as opposed to if the QB were to stay with their original team.")
    
    st.warning("*Note:* If a QB's stats have high values and they are a transfer, the PPA will be lower vs. the QB " \
    "staying at their current school. If you apply the Destination Teams values into the model and change *is_transfer* to " \
    "'NO', the result could be a potentially higher PPA for the QB at their new Destination School.")

# --- PAGE 2: PREDICTION ---
elif page == "Predict":
    st.title("Predict QB Performance")
    st.write("Enter the player's stats from the previous season to predict their next PPA.")

    col1, col2 = st.columns(2)

    with col1:
        years_in_college = st.number_input("Years in College", value=2, step=1)
        prev_yds = st.number_input("Previous Pass Yards", value=2500, step=100)
        prev_tds = st.number_input("Previous Passing TDs", value=20, step=1)
        prev_ppa = st.number_input("Previous Avg PPA", value=0.30, step=0.01)
        
    with col2:
        dest_off_ppa = st.number_input("Destination Team's Previous PPA", value=0.30, step=0.01)
        dest_sp_offense = st.number_input("Destination Offense SP+ Rating", value=50.0, step=1.0)
        is_transfer = st.selectbox("Is the player a Transfer?", options=[0, 1], 
                                    format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

    # Create input dataframe
    input_data = pd.DataFrame([[
    years_in_college, 
    prev_yds, 
    prev_tds, 
    prev_ppa, 
    dest_off_ppa, 
    dest_sp_offense, 
    is_transfer]], columns=FEATURES)

    if st.button("Predict Post-PPA"):
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.metric(label="Predicted Post-PPA", value=f"{prediction:.3f}")
        
        # Give context based on the prediction
        if is_transfer == 1:
            st.warning("Note: The 'Transfer Penalty' we saw in the SHAP analysis is being applied here.")
        else:
            st.success("Note: The model is rewarding this player for staying with their current team.")
