import pandas as pd
import streamlit as st
import numpy as np
import joblib
from fpdf import FPDF
import matplotlib.pyplot as plt
import io
from PIL import Image

# Load trained model
model_filename = 'random_forest_model.pkl'  # Retrain model with new features
classifier = joblib.load(model_filename)

# Set OpenAI API Key (Replace with your actual key)
openai.api_key = 'your-openai-api-key'

def generate_guidance(prediction, glucose, hba1c, smoking, activity):
    """Generates AI-based medical guidance based on input features."""
    prompt = f"""
    Patient has glucose level {glucose}, HbA1c {hba1c}, smoking status {smoking}, and physical activity level {activity}.
    Provide a medical recommendation if the patient is diabetic ({prediction == 1}) or non-diabetic ({prediction == 0})."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a medical expert providing personalized diabetes guidance."},
                  {"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Streamlit UI
st.header('Diabetes Prediction System')

# Get input values
gender = st.selectbox("Select Gender", ["Male", "Female"])
glucose = st.text_input('Enter Glucose', '0').strip()
blood_pressure = st.text_input('Enter Blood Pressure', '0').strip()
skin_thickness = st.text_input('Enter Skin Thickness', '0').strip()
insulin = st.text_input('Enter Insulin', '0').strip()
bmi = st.text_input('Enter BMI', '0').strip()
dpf = st.text_input('Enter Diabetes Pedigree Function', '0').strip()
age = st.text_input('Enter Age', '0').strip()
hba1c = st.text_input('Enter HbA1c Level', '5.0').strip()
smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
activity = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])

def validate_input(value):
    try:
        return float(value)
    except ValueError:
        return None

if st.button('Predict'):
    try:
        # Convert inputs to float with validation
        input_values = [glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, hba1c]
        validated_values = [validate_input(val) for val in input_values]
        
        if None in validated_values:
            st.error("Please enter valid numeric values.")
        else:
            inputData = np.array([validated_values + [
                1 if gender == "Male" else 0,  # Encoding gender
                1 if smoking == "Smoker" else 0,
                {"Low": 0, "Medium": 1, "High": 2}[activity]
            ]], dtype=float)
            
            # Prediction
            prediction = classifier.predict(inputData)[0]
            prediction_proba = classifier.predict_proba(inputData)[0]
            
            # AI-Based Guidance
            medical_guidance = generate_guidance(prediction, glucose, hba1c, smoking, activity)
            
            # Determine result and risk level
            result_text = "No Diabetes Detected" if prediction == 0 else "Diabetes Found"
            risk_level = "Low Risk" if prediction == 0 else "High Risk"
            
            # Visualization
            st.subheader("Risk Level")
            st.write(f"Risk Level: **{risk_level}**")
            
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(['Risk Level'], [prediction_proba[1]], color='green' if prediction == 0 else 'red')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability of Diabetes')
            st.pyplot(fig)
            
            # Convert visualization to image
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)
            
            # Save as image for PDF
            image_path = "risk_level_visualization.png"
            Image.open(img_buf).save(image_path)
            
            # Generate PDF Report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, "Diabetes Prediction Report", ln=True, align="C")
            pdf.ln(5)
            pdf.cell(90, 8, "Result", border=1, align="C")
            pdf.cell(90, 8, result_text, border=1, align="C")
            pdf.ln()
            pdf.cell(90, 8, "Risk Level", border=1, align="C")
            pdf.cell(90, 8, risk_level, border=1, align="C")
            pdf.ln(10)
            
            # Patient Data Section
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Patient Input Data", ln=True, align="L")
            pdf.set_font("Arial", size=12)
            input_values = [
                ("Gender", gender),
                ("Glucose", glucose),
                ("Blood Pressure", blood_pressure),
                ("Skin Thickness", skin_thickness),
                ("Insulin", insulin),
                ("BMI", bmi),
                ("Diabetes Pedigree Function", dpf),
                ("Age", age),
                ("HbA1c", hba1c),
                ("Smoking Status", smoking),
                ("Physical Activity", activity)
            ]
            for feature, value in input_values:
                pdf.cell(90, 8, feature, border=1, align="C")
                pdf.cell(90, 8, str(value), border=1, align="C")
                pdf.ln()
            
            # Medical Guidance
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Medical Guidance", ln=True, align="L")
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, medical_guidance)
            
            # Save PDF
            pdf_output = "diabetes_prediction_report.pdf"
            pdf.output(pdf_output)
            
            with open(pdf_output, "rb") as file:
                st.download_button(
                    label="Download Report as PDF",
                    data=file,
                    file_name=pdf_output,
                    mime="application/pdf"
                )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
