# Install Required Packages
# !pip install streamlit pandas numpy matplotlib tensorflow transformers

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import AlbertTokenizer, AlbertForQuestionAnswering
from transformers import pipeline

st.set_page_config(page_title="Insurance Estimation", page_icon=":guardsman:", layout="centered", initial_sidebar_state="expanded")

# Customizing the theme
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #007bff;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the insurance prediction model and scaler
def load_model_and_scaler():
    custom_objects = {"Dropout": tf.keras.layers.Dropout}
    model = tf.keras.models.load_model('/content/insurance_model.h5', custom_objects=custom_objects)
    with open('/content/StandardScaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('/content/feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)
    return model, scaler, feature_names

model, scaler, feature_names = load_model_and_scaler()

# Load Hugging Face's Albert model and tokenizer
def load_qa_model():
    tokenizer = AlbertTokenizer.from_pretrained('twmkn9/albert-base-v2-squad2')
    model = AlbertForQuestionAnswering.from_pretrained('twmkn9/albert-base-v2-squad2')
    return tokenizer, model

tokenizer, qa_model = load_qa_model()

# Question-Answering pipeline
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=tokenizer)

# Streamlit app
st.title("Medical Charges Estimation and Insurance Advice")

st.markdown("""
This app estimates the medical charges for a patient based on various input features.
Fill in the details and click on 'Predict' to see the estimated charges.
Additionally, you can ask questions about the best type of insurance to combine.
""")

# Create columns for input fields
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, help="Enter the age of the individual.")
    children = st.number_input("Children", min_value=0, help="Enter the number of children covered by the insurance plan.")
    smoker = st.selectbox("Smoker", ["yes", "no"], help="Indicate if the individual is a smoker.")

with col2:
    bmi = st.number_input("BMI", min_value=0.0, help="Enter the Body Mass Index of the individual.")
    gender = st.selectbox("Gender", ["male", "female"], help="Select the gender of the individual.")
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], help="Select the region of residence.")

with col3:
    indv_medical_hist = st.selectbox("Individual Medical History", ["Heart disease", "High blood pressure", "No value"], help="Select the individual's medical history.")
    family_medical_history = st.selectbox("Family Medical History", ["Heart disease", "High blood pressure", "No Value"], help="Select the family's medical history.")
    exercise_frequency = st.selectbox("Exercise Frequency", ["Never", "Occasionally", "Rarely"], help="Select the frequency of exercise.")
    occupation = st.selectbox("Occupation", ["Student", "Unemployed", "White collar"], help="Select the occupation of the individual.")
    coverage_level = st.selectbox("Coverage Level", ["Premium", "Standard"], help="Select the level of insurance coverage.")

# Add a predict button
if st.button("Predict"):
    # Prepare input data
    input_data = {
        'age': age,
        'bmi': bmi,
        'enc_children': children,
        'gender_male': 1 if gender == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
        'indv_Heart disease': 1 if indv_medical_hist == 'Heart disease' else 0,
        'indv_High blood pressure': 1 if indv_medical_hist == 'High blood pressure' else 0,
        'indv_No value': 1 if indv_medical_hist == 'No value' else 0,
        'family_Heart disease': 1 if family_medical_history == 'Heart disease' else 0,
        'family_High blood pressure': 1 if family_medical_history == 'High blood pressure' else 0,
        'family_No Value': 1 if family_medical_history == 'No Value' else 0,
        'exercise_Never': 1 if exercise_frequency == 'Never' else 0,
        'exercise_Occasionally': 1 if exercise_frequency == 'Occasionally' else 0,
        'exercise_Rarely': 1 if exercise_frequency == 'Rarely' else 0,
        'occupation_Student': 1 if occupation == 'Student' else 0,
        'occupation_Unemployed': 1 if occupation == 'Unemployed' else 0,
        'occupation_White collar': 1 if occupation == 'White collar' else 0,
        'coverage_Premium': 1 if coverage_level == 'Premium' else 0,
        'coverage_Standard': 1 if coverage_level == 'Standard' else 0
    }

    if age > 0 and bmi > 0 and children >= 0:
        # Prepare input data and make predictions
        input_df = pd.DataFrame([input_data])

        # Reorder the columns to match the training data
        input_df = input_df[feature_names]

        input_df_scaled = scaler.transform(input_df)
        prediction = model.predict(input_df_scaled)

        st.write(f"Estimated Charges: ${prediction[0][0]:.2f}")

        # Recommendation based on the input data
        recommendation = "Based on your input, a Premium coverage level is recommended for more benefits." if prediction[0][0] > 10000 else "Based on your input, a Standard coverage level should be sufficient."
        st.write(f"Recommendation: {recommendation}")
    else:
        st.error("Please fill in all required fields correctly.")

# Add a section for asking questions about insurance
st.markdown("## Ask Questions about Insurance")

context = """
Choosing the best insurance combination depends on multiple factors such as age, BMI, smoking status, medical history, and family medical history. Generally, a premium coverage level provides more benefits but at a higher cost, while standard coverage is more affordable but may offer fewer benefits.
"""

question = st.text_input("Enter your question about insurance:")

if question:
    result = qa_pipeline(question=question, context=context)
    st.write("Answer:", result['answer'])

# Function to plot actual vs. predicted values (optional)
def plot_actual_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Charges')
    st.pyplot(plt)
