import streamlit as st
import joblib
import pandas as pd

model = joblib.load("mental_health_ridge.pkl")

st.set_page_config(page_title="Student Mental Health Predictor", layout="centered")

st.markdown("""
<style>

.stApp {
    background-image: url("https://beurownlight.com/wp-content/uploads/2017/02/meditate1.jpg");
    background-size: cover;
}

.hero {
    background: rgba(255,255,255,0.95);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

.good {background:#e8f5e9;padding:18px;border-radius:14px;margin-top:20px;}
.mid {background:#fff8e1;padding:18px;border-radius:14px;margin-top:20px;}
.high {background:#ffebee;padding:18px;border-radius:14px;margin-top:20px;}

div[data-testid="stForm"] {
    background:white;
    padding:40px;
    border-radius:30px;
    box-shadow:0px 6px 25px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
<h1>🎓 Student Mental Health Prediction</h1>
<p><b>Get the help you need before it’s too late.</b></p>
</div>
""", unsafe_allow_html=True)


with st.form("prediction_form"):

    age = st.slider("Age", 18, 30, 21)
    stress = st.slider("Stress Level", 0, 5, 3)
    anxiety = st.slider("Anxiety Score", 0, 5, 3)
    financial = st.slider("Financial Stress", 0, 5, 2)
    semester_load = st.slider("Semester Credit Load", 10, 30, 18)
    cgpa = st.slider("CGPA", 2.0, 4.0, 3.4)

    course = st.radio("Course", ["Computer Science","Engineering","Law","Medical","Business","Others"])
    gender = st.radio("Gender", ["Male","Female"])

    sleep = st.radio("Sleep Quality", ["Good","Average","Poor"])
    activity = st.radio("Physical Activity", ["Low","Moderate","High"])
    diet = st.radio("Diet Quality", ["Good","Average","Poor"])
    social = st.radio("Social Support", ["Low","Moderate","High"])
    relationship = st.radio("Relationship Status", ["Single","Married"])

    substance = st.radio("Substance Use", ["Never","Occasionally","Frequently"])
    counsel = st.radio("Counseling Service Use", ["Never","Occasionally","Frequently"])

    family = st.radio("Family History", ["Yes","No"])
    chronic = st.radio("Chronic Illness", ["Yes","No"])
    residence = st.radio("Residence Type", ["On-Campus","With Family"])

    confirm = st.checkbox("I confirm that the information entered above is accurate")

    submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        if not confirm:
            st.error("Please confirm that your information is accurate before proceeding.")
            st.stop()

        psych = (stress + anxiety + financial) / 3
        academic = (semester_load / cgpa) * 10

        df = pd.DataFrame({
            "Age":[age],
            "Psychological_Score":[psych],
            "Academic_Stress_Index":[academic],
            "Course":[course],
            "Gender":[gender],
            "Sleep_Quality":[sleep],
            "Physical_Activity":[activity],
            "Diet_Quality":[diet],
            "Social_Support":[social],
            "Relationship_Status":[relationship],
            "Substance_Use":[substance],
            "Counseling_Service_Use":[counsel],
            "Family_History":[family],
            "Chronic_Illness":[chronic],
            "Residence_Type":[residence]
        })

        df = pd.get_dummies(df)
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        pred = model.predict(df)[0]

        st.markdown(
            f"<h2 style='margin-top:30px;'>🧠 Depression Score: {pred:.2f}</h2>",
            unsafe_allow_html=True
        )

        if pred < 2:
            st.markdown(
                "<div class='good'>🟢 Low Risk. Continue maintaining healthy habits!</div>",
                unsafe_allow_html=True
            )
        elif pred < 4:
            st.markdown(
                "<div class='mid'>🟡 Moderate Risk. Consider seeking support.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='high'>🔴 High Risk. Please seek professional help.</div>",
                unsafe_allow_html=True
            )

        st.caption("Educational Ridge Regression model. Not medical advice.")
