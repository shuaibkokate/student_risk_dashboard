import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

# Load datasets
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")
degree_df = pd.read_csv("degree_progress.csv")

# Ensure consistent string type for merging
student_df["student_id"] = student_df["student_id"].astype(str)
degree_df["student_id"] = degree_df["student_id"].astype(str)

# Merge degree progress
student_df = pd.merge(student_df, degree_df, on="student_id", how="left")

# Features for clustering
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]
X = student_df[features]

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
student_df["cluster"] = clusters

# Calculate risk score
cluster_centers = kmeans.cluster_centers_
risk_scores = []
for center in cluster_centers:
    score = (1 - center[features.index("attendance_rate")]) + \
            (1 - center[features.index("gpa")] / 4.0) + \
            (1 - center[features.index("assignment_completion")]) + \
            (1 - center[features.index("lms_activity")])
    risk_scores.append(score)

# Map risk labels
sorted_clusters = np.argsort(risk_scores)[::-1]
risk_labels = ["High", "Medium", "Low"]
cluster_to_risk = {cluster: risk_labels[rank] for rank, cluster in enumerate(sorted_clusters)}
student_df["Predicted Risk"] = student_df["cluster"].map(cluster_to_risk)

# Risk reason
def generate_risk_reason(row):
    reasons = []
    if row["attendance_rate"] < 0.75:
        reasons.append(f"Low attendance ({row['attendance_rate']:.0%})")
    if row["gpa"] < 2.0:
        reasons.append(f"Low GPA ({row['gpa']:.2f})")
    if row["assignment_completion"] < 0.7:
        reasons.append(f"Low assignment completion ({row['assignment_completion']:.0%})")
    if row["lms_activity"] < 0.5:
        reasons.append(f"Low LMS activity ({row['lms_activity']:.0%})")
    return " & ".join(reasons) if reasons else "No major concerns detected"

student_df["Risk Reason"] = student_df.apply(generate_risk_reason, axis=1)

# Study tips
def generate_study_tips(row):
    tips = []
    if row["gpa"] < 2.0:
        tips.append("Attend tutoring and review basics.")
    if row["attendance_rate"] < 0.75:
        tips.append("Improve class attendance.")
    if row["assignment_completion"] < 0.7:
        tips.append("Complete assignments on time.")
    if row["lms_activity"] < 0.5:
        tips.append("Engage more in online materials.")
    return " ".join(tips) if tips else "Keep up the good work!"

student_df["Study Tips"] = student_df.apply(generate_study_tips, axis=1)

# Schedule status
def flag_behind_schedule(row):
    if pd.isnull(row['progress_percentage']):
        return "Unknown"
    elif row['progress_percentage'] < 70:
        return "Behind Schedule"
    else:
        return "On Track"

student_df["Schedule Status"] = student_df.apply(flag_behind_schedule, axis=1)

# Streamlit UI
st.set_page_config(page_title="Student Risk Predictor (Clustering)", layout="wide")
st.title("🎓 Student Risk Prediction Dashboard (Unsupervised)")

role = st.selectbox("Select your role:", ["advisor", "chair"])
user_id = st.text_input(f"Enter your {role} ID:")

if user_id:
    if role == "advisor":
        allowed_students = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    else:
        allowed_students = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    filtered_df = student_df[student_df["student_id"].isin(allowed_students)]

    if not filtered_df.empty:
        st.subheader("📊 Predicted Risk for Assigned Students")
        st.dataframe(filtered_df[["student_id"] + features + [
            "Predicted Risk", "Risk Reason", "Study Tips",
            "progress_percentage", "required_credits", "completed_credits", "Schedule Status"
        ]])

        st.markdown("### 📈 Risk Level Distribution")
        fig = px.pie(filtered_df, names="Predicted Risk", title="Risk Level Distribution")
        st.plotly_chart(fig)

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", data=csv, file_name="student_risk_predictions.csv", mime='text/csv')

        st.markdown("### 🔍 Detailed Student View")
        for _, row in filtered_df.iterrows():
            with st.expander(f"Student ID: {row['student_id']}"):
                st.write({
                    col: row[col]
                    for col in features + [
                        'Predicted Risk', 'Risk Reason', 'Study Tips',
                        'progress_percentage', 'completed_credits',
                        'required_credits', 'Schedule Status'
                    ]
                })
    else:
        st.warning("No students found for this user ID.")
