import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
#model_path = os.path.join(os.path.dirname(__file__), "best_tour_pkg_prediction_model_v1.joblib")
#model = joblib.load(model_path)

model = joblib.load("./deployment/best_tour_pkg_prediction_model_v1.joblib")

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer choosing a tour package based on his/her profile parameters.
Enter the customer data below to get a prediction.
""")

with st.form(key="customer_info_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, help="Age of the customer.")
        gender = st.selectbox("Gender", ["Female", "Male"], help="Gender of the customer.")
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"], help="Marital status of the customer.")
        typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"], help="The method by which the customer was contacted.")
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15, help="Duration of the sales pitch delivered to the customer.")
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"], help="The type of product pitched to the customer.")
        number_of_trips = st.number_input("Annual Number of Trips", min_value=0, max_value=20, value=2, help="Average number of trips the customer takes annually.")
        preferred_property_star = st.slider("Preferred Property Star Rating", min_value=3, max_value=5, value=3, help="Preferred hotel rating by the customer.")
        number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2, help="Total number of people accompanying the customer on the trip.")

    with col2:
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"], help="Customer's occupation.")
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"], help="Customer's designation in their current organization.")
        monthly_income = st.number_input("Monthly Income", min_value=0.0, step=1000.0, value=50000.0, help="Gross monthly income of the customer.")
        number_of_followups = st.slider("Number of Follow-ups", min_value=1, max_value=10, value=3, help="Total number of follow-ups by the salesperson after the sales pitch.")
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, help="Score indicating the customer's satisfaction with the sales pitch.")
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0, help="The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).")
        number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0, help="Number of children below age 5 accompanying the customer.")
        passport_display = st.radio("Holds Valid Passport?", ["No", "Yes"], index=0, help="Whether the customer holds a valid passport.")
        own_car_display = st.radio("Owns a Car?", ["No", "Yes"], index=0, help="Whether the customer owns a car.")

    # Form Submission Button
    submit_button = st.form_submit_button(label="submit")

# Processing when the user clicks submit
if submit_button:
    # Map 'Yes'/'No' labels to binary integer fields (0 or 1)
    passport = 1 if passport_display == "Yes" else 0
    own_car = 1 if own_car_display == "Yes" else 0

    # Construct the dictionary using exact names from your prompt
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_data)[0]

    result = "Customer will select a tour pkg" if prediction == 1 else "Customer will not select a tour pkg"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
