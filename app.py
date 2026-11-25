
import streamlit as st
import pandas as pd
import joblib

# Load trained ML model
model = joblib.load("titanic_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Application heading
st.title("🚢 Titanic Survival Prediction")
st.write("Predict passenger survival using Machine Learning.")
st.divider()

# Passenger information
st.subheader("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sibsp = st.number_input("Siblings / Spouses", 0, 10, 0)
parch = st.number_input("Parents / Children", 0, 10, 0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=30.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Prediction
if st.button("Predict Survival", use_container_width=True):

    family_size = sibsp + parch + 1

    age_group = pd.cut(
        [age],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teenager", "Adult", "MiddleAge", "Senior"]
    )[0]

    passenger = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked],
        "FamilySize": [family_size],
        "IsAlone": [int(family_size == 1)],
        "AgeGroup": [age_group]
    })

    prediction = model.predict(passenger)

    st.divider()

    if prediction[0] == 1:
        st.success("🎉 Passenger may survive!")
    else:
        st.error("Passenger may not survive.")

st.caption("Powered by Random Forest Machine Learning")
