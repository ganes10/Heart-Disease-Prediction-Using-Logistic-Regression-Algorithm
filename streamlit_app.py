import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import os
import time


# streamlit run streamlit_app.py

# Create the Streamlit app title
st.title("Heart Disease Prediction")

# Create input fields for each feature
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
# Select model
model_options = ["Neural Network (Highest Accuracy and Sensitivity)",
                 "Random Forest Classifier (Highest Specificity)"]
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
    # Load in whole dataset
    df = pd.read_csv(os.path.join("data", "heart_disease_data.csv"))  # in data directory
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

    # Augment features the same way I did in training
    numerical_features = df.select_dtypes(include=[np.number])
    numerical_features = numerical_features.drop(["FastingBS"], axis=1)
    continuous_feature_names = numerical_features.columns.tolist()

    categorical_features = df.select_dtypes(include=[object])
    categorical_feature_names = categorical_features.columns.to_list() + ["FastingBS"]

    preprocessed_df = df.copy(deep=True)  # make a copy of the original data which we will modify

    # Initialize the scalers
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()  # not clear this was required for 'Age', 'RestingBP', or, 'MaxHR' because those were already looking pretty close to Gaussian. Further normalization here is unlikely to hurt, however. A further investigation into normality with QQ-plots and the shapiro wilk test could be a future direction and dictate whether those features get StandardScaler applied to them

    # Apply both scalers to each continuous variable
    for feature in continuous_feature_names:
        min_max_scaled_data = min_max_scaler.fit_transform(preprocessed_df[[feature]])  # Perform MinMax scaling
        # Perform Standard scaling on the MinMax scaled data
        min_max_standard_scaled_data = standard_scaler.fit_transform(min_max_scaled_data)
        # Update the original DataFrame with the scaled data
        preprocessed_df[feature] = min_max_standard_scaled_data.flatten()

    # one hot encoding of categorical variables
    preprocessed_df = pd.get_dummies(preprocessed_df, columns=categorical_feature_names, dtype=int)

    # return final row to predict
    return preprocessed_df.tail(1)  # get last row, keep as dataframe structure


to_predict = preprocess_new_data()

# Load the trained models
random_forest_classifier = joblib.load("saved models/random_forest_classifier.pkl")
dl_classifier = tf.keras.models.load_model(os.path.join(os.getcwd(), "saved models/deep_learning_classifier"))


# Define a function to handle the prediction
def predict():
    # Make the prediction
    if selected_model == "Random Forest Classifier (Highest Specificity)":
        prediction = random_forest_classifier.predict(to_predict)
        probability_positive = random_forest_classifier.predict_proba(to_predict)[0][1]  # shouldn't display these numbers becuase model was evaluated assuming binary classification, not based on probabilities
    else:
        tf_predictions = dl_classifier.predict(to_predict)
        prediction = np.round(tf_predictions).astype(int)[0]
        probability_positive = tf_predictions[0][0]

    # Display the prediction result
    if prediction[0] == 1:
        st.success(f"It is predicted that the patient has heart disease.")
    else:
        st.success(f"It is predicted that the patient does not have heart disease.")
    st.success(f"Chance of being heart disease positive: {100*probability_positive:.2f}%")


# Create a button to trigger the prediction
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
medical_advice_warning = "WARNING: Please note that this project is intended as an illustrative example of the potential application of machine learning in assisting medical professionals with heart disease diagnosis. The information and results presented here do not constitute medical advice in any form."
st.subheader(medical_advice_warning)
