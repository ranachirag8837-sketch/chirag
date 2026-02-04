import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Student Performance Prediction",
    layout="wide"
)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("student_performance_10000.csv")

df = load_data()

# ------------------ MODEL TRAINING ------------------
features = ["StudyHours", "Attendance", "InternalMarks", "PreviousScore"]
X = df[features]
y_class = df["Result"]
y_marks = df["TotalMarks"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)

clf = LogisticRegression()
clf.fit(X_train, y_train)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_scaled, y_marks)

accuracy = accuracy_score(y_test, clf.predict(X_test))

# ------------------ SIDEBAR ------------------
st.sidebar.title("🎛 Control Panel")
st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")

# ------------------ MAIN UI ------------------
st.markdown("<h1 style='text-align:center'>🎓 AI-Powered Student Performance Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Hybrid ML System with Live Prediction</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    study_hours = st.number_input("📘 Study Hours per Day", 0.0, 12.0, step=0.5)
    attendance = st.number_input("📊 Attendance (%)", 0.0, 100.0)
    
with col2:
    internal = st.number_input("📝 Internal Marks", 0.0, 30.0)
    previous = st.number_input("📈 Previous Score", 0.0, 100.0)

if st.button("🔮 Predict Performance"):
    input_df = pd.DataFrame([[study_hours, attendance, internal, previous]], columns=features)
    input_scaled = scaler.transform(input_df)

    pass_prob = clf.predict_proba(input_scaled)[0][1]
    predicted_marks = reg.predict(input_scaled)[0]

    # Risk category
    if pass_prob < 0.4:
        risk = "🔴 High Risk"
        color = "red"
    elif pass_prob < 0.7:
        risk = "🟡 Medium Risk"
        color = "orange"
    else:
        risk = "🟢 Safe Zone"
        color = "green"

    st.markdown("---")
    st.markdown(f"### 🧠 Prediction Result")
    st.metric("Pass Probability", f"{pass_prob*100:.2f}%")
    st.metric("Estimated Marks", f"{predicted_marks:.2f} / 100")
    st.markdown(f"<h3 style='color:{color}'>{risk}</h3>", unsafe_allow_html=True)

# ------------------ VISUAL ANALYTICS ------------------
st.markdown("---")
st.subheader("📊 Dataset Insights")

fig, ax = plt.subplots()
ax.hist(df["TotalMarks"], bins=30)
ax.set_title("Marks Distribution")
ax.set_xlabel("Total Marks")
ax.set_ylabel("Students")
st.pyplot(fig)

st.markdown("<center><small>AI Student Predictor | Live Deployment Ready</small></center>", unsafe_allow_html=True)
