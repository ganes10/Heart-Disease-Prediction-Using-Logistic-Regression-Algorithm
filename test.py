import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Streamlit app title
st.title("Heart Disease Prediction")

# Input fields for each feature
age = st.number_input("Age", min_value=18, max_value=150, step=1, value=18)
resting_BP = st.number_input("Resting Systolic Blood Pressure (mm Hg)", step=1, value=120)
cholesterol = st.number_input("Serum Cholesterol (mm/dl)", step=1, min_value=80, value=200)
MaxHR = st.number_input("Maximum Heart Rate Achieved in Exhaustion Test", step=1, min_value=50, value=140)
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest (mm)", value=-0.1)
sex = st.selectbox("Sex", ["Female", "Male"])
chest_pain = st.selectbox("Does the Patient Experience Chest Pain?", ["No Chest Pain", "Typical Angina Pain", "Atypical Angina Pain", "Non-Anginal Pain"])
fasting_bs = st.selectbox("Blood Sugar After Fast (mg/dl)", ["120 or Under", "Over 120"])
resting_ECG = st.selectbox("Resting Electrocardiogram Results", ["Normal", "ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)", "Showing probable or definite left ventricular hypertrophy by Estes' criteria"])
ExerciseAngina = st.selectbox("Does the Patient Experience Exercise-Induced Angina?", ["No", "Yes"])
ST_Slope = st.selectbox("ST Segment Slope during Exercise", ["Sloping Upwards", "Flat", "Sloping Downwards"])

# Model selection
model_options = ["Logistic Regression"]
selected_model = st.selectbox("Classification Model", model_options)


def convert_categorical_variables(sex_, chest_pain_, fasting_bs_, resting_ECG_, ExerciseAngina_, ST_Slope_):
    """
    Converts categorical variables from their user-friendly names to the names that match the dataset
    """
    sex_conversion = {
        "Male": "M",
        "Female": "F"
    }
    chest_pain_conversion = {
        "Typical Angina Pain": "TA",
        "Atypical Angina Pain": "ATA",
        "Non-Anginal Pain": "NAP",
        "No Chest Pain": "ASY"
    }
    fasting_bs_conversion = {
        "Over 120": 1,
        "120 or Under": 0
    }
    resting_ECG_conversion = {
        "ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)": "ST",
        "Normal": "Normal",
        "Showing probable or definite left ventricular hypertrophy by Estes' criteria": "LVH"
    }
    ExerciseAngina_conversion = {
        "Yes": "Y",
        "No": "N"
    }
    ST_Slope_conversion = {
        "Sloping Upwards": "Up",
        "Flat": "Flat",
        "Sloping Downwards": "Down"
    }
    return sex_conversion[sex_], chest_pain_conversion[chest_pain_], fasting_bs_conversion[fasting_bs_], resting_ECG_conversion[resting_ECG_], ExerciseAngina_conversion[ExerciseAngina_], ST_Slope_conversion[ST_Slope_]


sex, chest_pain, fasting_bs, resting_ECG, ExerciseAngina, ST_Slope = convert_categorical_variables(sex, chest_pain, fasting_bs, resting_ECG, ExerciseAngina, ST_Slope)


def preprocess_new_data():
    # Load the data and preprocess it
    df = pd.read_csv("/Users/praswishbasnet/Desktop/Heart-Disease-Prediction/data/heart_disease_data.csv")

    # Check if 'HeartDisease' column exists and if not, handle the error
    if 'HeartDisease' not in df.columns:
        st.error("'HeartDisease' column not found in the dataset.")
        return None
    
    # Drop 'HeartDisease' column for prediction (features)
    df = df.drop(["HeartDisease"], axis=1)

    # Create new row with the user's data
    new_row = pd.Series({
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_BP,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": oldpeak,
        "ST_Slope": ST_Slope
    })

    df.loc[len(df)] = new_row  # add new row to full dataset

    # Preprocessing pipeline
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    numerical_features = [col for col in df.columns if col not in categorical_features]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

    # Ensure 'HeartDisease' is not included in the feature set
    X = df.drop(columns=['HeartDisease'], errors='ignore')  # Features, with error handling
    X_processed = preprocessor.fit_transform(X)

    return X_processed[-1:]  # Get last row for prediction


to_predict = preprocess_new_data()

# Ensure data was processed correctly
if to_predict is None:
    st.stop()  # Exit if there is an issue with the data

# Load the trained Logistic Regression model
log_reg_model = pickle.load(open('/Users/praswishbasnet/Desktop/Heart-Disease-Prediction/saved models/logistic_regressor1.pkl', 'rb'))  # Ensure your model is saved in this file

# Prediction function
def predict():
    prediction = log_reg_model.predict(to_predict)
    probability_positive = log_reg_model.predict_proba(to_predict)[0][1]

    # Display the prediction result
    if prediction[0] == 1:
        st.success(f"It is predicted that the patient has heart disease.")
    else:
        st.success(f"It is predicted that the patient does not have heart disease.")
    st.success(f"Chance of being heart disease positive: {100*probability_positive:.2f}%")


# Trigger prediction
predict_button = st.button("Predict")
if predict_button:
    # Initialize the progress bar
    progress_bar = st.progress(0)

    for i in range(101):
        # Update the progress bar
        progress_bar.progress(i)
        time.sleep(0.002)

    # Run the prediction function
    predict()

# Add disclaimer
medical_advice_warning = "Developed By: Ganesh Basnet (00020111)"
st.subheader(medical_advice_warning)
