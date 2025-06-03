import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Student Risk Dashboard", layout="wide")

# ---------------------
# Load and Preprocess
# ---------------------
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")
degree_df = pd.read_csv("degree_progress_500.csv")

student_df.columns = student_df.columns.str.strip().str.lower()
mapping_df.columns = mapping_df.columns.str.strip().str.lower()
degree_df.columns = degree_df.columns.str.strip().str.lower()

student_df["student_id"] = student_df["student_id"].astype(str)
mapping_df["student_id"] = mapping_df["student_id"].astype(str)
degree_df["student_id"] = degree_df["student_id"].astype(str)

features = ["attendance_rate", "gpa", "assignment_completion"]

for col in features:
    student_df[col] = student_df[col].fillna(student_df[col].mean())
    degree_df[col] = degree_df[col].fillna(degree_df[col].mean())

# ---------------------
# Custom Thresholds
# ---------------------
st.sidebar.markdown("### 🔢 Custom Risk Thresholds")
threshold_attendance = st.sidebar.slider("Minimum Attendance Rate", 0.0, 1.0, 0.75, 0.01)
threshold_gpa = st.sidebar.slider("Minimum GPA", 0.0, 4.0, 2.0, 0.1)
threshold_assignment = st.sidebar.slider("Minimum Assignment Completion", 0.0, 1.0, 0.7, 0.01)

# ---------------------
# Clustering for Students
# ---------------------
X = student_df[features].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
student_df["cluster"] = kmeans.fit_predict(X)
risk_scores = X.apply(lambda row: (1 - row["attendance_rate"]) + (1 - row["gpa"]/4.0) + (1 - row["assignment_completion"]), axis=1)
quantiles = risk_scores.quantile([0.33, 0.66])
def map_risk(score):
    if score >= quantiles[0.66]: return "High"
    elif score >= quantiles[0.33]: return "Medium"
    else: return "Low"
student_df["predicted_risk"] = risk_scores.apply(map_risk)

# ---------------------
# Clustering for Courses
# ---------------------
X_course = degree_df[features].fillna(0)
kmeans_course = KMeans(n_clusters=3, random_state=42, n_init=10)
degree_df["cluster"] = kmeans_course.fit_predict(X_course)
risk_scores_course = X_course.apply(lambda row: (1 - row["attendance_rate"]) + (1 - row["gpa"]/4.0) + (1 - row["assignment_completion"]), axis=1)
quantiles_course = risk_scores_course.quantile([0.33, 0.66])
def map_course_risk(score):
    if score >= quantiles_course[0.66]: return "High"
    elif score >= quantiles_course[0.33]: return "Medium"
    else: return "Low"
degree_df["predicted_risk"] = risk_scores_course.apply(map_course_risk)

# ---------------------
# Reason and Tips
# ---------------------
def get_reason(row):
    reasons = []
    if row["attendance_rate"] < threshold_attendance: reasons.append("Low attendance")
    if row["gpa"] < threshold_gpa: reasons.append("Low GPA")
    if row["assignment_completion"] < threshold_assignment: reasons.append("Low assignment completion")
    return ", ".join(reasons) or "No major concerns"

def get_tips(row):
    tips = []
    if row["gpa"] < threshold_gpa: tips.append("Attend tutoring.")
    if row["attendance_rate"] < threshold_attendance: tips.append("Attend classes regularly.")
    if row["assignment_completion"] < threshold_assignment: tips.append("Submit assignments.")
    return " ".join(tips) or "Keep up the good work!"

student_df["risk_reason"] = student_df.apply(get_reason, axis=1)
student_df["study_tips"] = student_df.apply(get_tips, axis=1)
degree_df["risk_reason"] = degree_df.apply(get_reason, axis=1)
degree_df["study_tips"] = degree_df.apply(get_tips, axis=1)

# ---------------------
# Schedule Status
# ---------------------
def schedule_flag(row):
    if pd.isnull(row["progress_percentage"]) or pd.isnull(row["expected_progress"]):
        return "Unknown"
    if row["progress_percentage"] < row["expected_progress"] - 10:
        return "Behind Schedule"
    return "On Track"

student_df["schedule_status"] = student_df.apply(schedule_flag, axis=1)

# ---------------------
# Display Setup
# ---------------------
st.title("AI-Enhanced Student Risk Dashboard")
st.markdown("Track at-risk students, analyze academic progress, and intervene early.")

st.sidebar.title("Filter Options")
role = st.sidebar.selectbox("Select Role", ["advisor", "chair"])
user_id = st.sidebar.text_input(f"Enter your {role} ID")

if user_id:
    if role == "advisor":
        assigned_ids = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    else:
        assigned_ids = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    filtered_df = student_df[student_df["student_id"].isin(assigned_ids)]
    filtered_courses_df = degree_df[degree_df["student_id"].isin(assigned_ids)]

    if filtered_df.empty:
        st.warning("⚠️ No students assigned to this ID.")
    else:
        tab1, tab2, tab3 = st.tabs(["Overview", "Student Table", "Download"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            all_risks = ["High", "Medium", "Low"]
            risk_summary = filtered_df["predicted_risk"].value_counts().reindex(all_risks, fill_value=0).reset_index()
            risk_summary.columns = ["risk_level", "count"]

            col1.metric("Total Students", len(filtered_df))
            col2.metric("High Risk", risk_summary[risk_summary["risk_level"] == "High"]["count"].values[0])
            col3.metric("Medium Risk", risk_summary[risk_summary["risk_level"] == "Medium"]["count"].values[0])
            st.metric("Low Risk", risk_summary[risk_summary["risk_level"] == "Low"]["count"].values[0])

            pie_fig = px.pie(risk_summary, names='risk_level', values='count', title="Student Risk Levels")
            st.plotly_chart(pie_fig, use_container_width=True)

            st.markdown("### Average GPA by Risk Category")
            gpa_fig = px.bar(
                filtered_df.groupby("predicted_risk")["gpa"].mean().reindex(all_risks).reset_index(),
                x="predicted_risk", y="gpa", color="predicted_risk",
                labels={"predicted_risk": "Risk", "gpa": "Avg GPA"}
            )
            st.plotly_chart(gpa_fig, use_container_width=True)

        with tab2:
            st.markdown("### Student Summary Table")
            st.dataframe(filtered_df, use_container_width=True, height=400)

            st.markdown("### Course-wise Performance for Selected Student")
            selected_student = st.selectbox("Select a student:", filtered_df["student_id"].unique())
            selected_course_df = filtered_courses_df[filtered_courses_df["student_id"] == selected_student]
            st.dataframe(selected_course_df, use_container_width=True, height=300)

        with tab3:
            st.download_button("Download Student Summary", data=filtered_df.to_csv(index=False), file_name="student_summary.csv")
            st.download_button("Download Course Details", data=filtered_courses_df.to_csv(index=False), file_name="course_details.csv")
else:
    st.info("ℹ️ Please enter your role and ID to continue.")
