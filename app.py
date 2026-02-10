import streamlit as st
import joblib
import pandas as pd

model = joblib.load("mental_health_ridge.pkl")

st.set_page_config(page_title="Student Mental Health Predictor", layout="centered")

# ---------------- STYLE ---------------- #

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

.hero h1 {
    color: #2C3E50;
}

div[data-testid="stForm"] {
    background-color: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO ---------------- #

st.markdown("""
<div class="hero">
<h1>🎓 Student Mental Health Prediction</h1>
<p>Understand your mental well-being today.<br>
<b>Get the help you need before it’s too late.</b></p>
</div>
""", unsafe_allow_html=True)

# ---------------- FORM ---------------- #

with st.form("prediction_form"):

    st.subheader("📘 Basic Information")

    age = st.slider("Age", 18, 30, 21)
    stress = st.slider("Stress Level", 0, 5, 3)
    anxiety = st.slider("Anxiety Score", 0, 5, 3)
    financial = st.slider("Financial Stress", 0, 5, 2)

    semester_load = st.slider("Semester Credit Load", 10, 30, 18)
    cgpa = st.slider("CGPA", 2.0, 4.0, 3.4)

    st.subheader("💙 Lifestyle & Background")

    course = st.radio("Course", ["Computer Science","Engineering","Law","Medical","Others"])
    gender = st.radio("Gender", ["Male","Female"])
    sleep = st.radio("Sleep Quality", ["Good","Poor"])
    activity = st.radio("Physical Activity", ["Low","Moderate"])
    diet = st.radio("Diet Quality", ["Good","Poor"])
    substance = st.radio("Substance Use", ["Never","Occasionally"])
    counsel = st.radio("Counselling Service Use", ["Never","Occasionally"])
    family = st.radio("Family History", ["Yes","No"])
    chronic = st.radio("Chronic Illness", ["Yes","No"])
    residence = st.radio("Residence Type", ["On-Campus","With Family"])

    submitted = st.form_submit_button("🔍 Predict Depression Score")

    if submitted:

        if cgpa <= 0:
            st.error("CGPA must be greater than zero.")

        else:

            psych_score = stress + anxiety + financial / 3
            academic_index = (semester_load / cgpa) * 10

            df = pd.DataFrame({
                "Age":[age],
                "Psychological_Score":[psych_score],
                "Academic_Stress_Index":[academic_index],
            })

            ohe = {
                "Course_Computer Science": course=="Computer Science",
                "Course_Engineering": course=="Engineering",
                "Course_Law": course=="Law",
                "Course_Medical": course=="Medical",
                "Course_Others": course=="Others",
                "Gender_Male": gender=="Male",
                "Sleep_Quality_Good": sleep=="Good",
                "Physical_Activity_Low": activity=="Low",
                "Diet_Quality_Good": diet=="Good",
                "Substance_Use_Never": substance=="Never",
                "Counseling_Service_Use_Never": counsel=="Never",
                "Family_History_Yes": family=="Yes",
                "Chronic_Illness_Yes": chronic=="Yes",
                "Residence_Type_On-Campus": residence=="On-Campus",
            }

            for k,v in ohe.items():
                df[k] = int(v)

            df = df.reindex(columns=model.feature_names_in_, fill_value=0)

            pred = model.predict(df)[0]

            st.divider()
            st.subheader("📝 Prediction Result")
            st.success(f"🧠 Predicted Depression Score: {pred:.2f}")

            if pred < 2:
                st.info("🟢 Low Risk. Keep maintaining healthy habits!")
            elif pred < 4:
                st.warning("🟡 Moderate Risk. Consider reaching out for support.")
            else:
                st.error("🔴 High Risk. Please seek professional help.")

            st.caption("This is an educational Ridge Linear Regression model, not medical advice.")
