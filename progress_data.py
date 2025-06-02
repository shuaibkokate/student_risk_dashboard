import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# ---------------------
# Load and Preprocess
# ---------------------
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")
degree_df = pd.read_csv("degree_progress_500.csv")

# Normalize column names
student_df.columns = student_df.columns.str.strip().str.lower()
mapping_df.columns = mapping_df.columns.str.strip().str.lower()
degree_df.columns = degree_df.columns.str.strip().str.lower()

# Ensure IDs are strings
student_df["student_id"] = student_df["student_id"].astype(str)
mapping_df["student_id"] = mapping_df["student_id"].astype(str)
degree_df["student_id"] = degree_df["student_id"].astype(str)

# ---------------------
# Student-Level Aggregation from Course-Wise Data
# ---------------------
available_columns = degree_df.columns.tolist()
agg_funcs = {}
if "attendance_rate" in available_columns:
    agg_funcs["attendance_rate"] = "mean"
if "gpa" in available_columns:
    agg_funcs["gpa"] = "mean"
if "assignment_completion" in available_columns:
    agg_funcs["assignment_completion"] = "mean"
if "lms_activity" in available_columns:
    agg_funcs["lms_activity"] = "mean"
if "progress_percentage" in available_columns:
    agg_funcs["progress_percentage"] = "mean"
if "expected_progress" in available_columns:
    agg_funcs["expected_progress"] = "mean"
if "required_credits" in available_columns:
    agg_funcs["required_credits"] = "sum"
if "completed_credits" in available_columns:
    agg_funcs["completed_credits"] = "sum"

aggregated = degree_df.groupby("student_id").agg(agg_funcs).reset_index()

# Merge into student_df
student_df = pd.merge(student_df, aggregated, on="student_id", how="left")

# ---------------------
# Clustering for Risk
# ---------------------
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]
X = student_df[features].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42)
student_df["cluster"] = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_
risk_scores = [
    (1 - c[0]) + (1 - c[1]/4.0) + (1 - c[2]) + (1 - c[3])
    for c in centroids
]
cluster_order = np.argsort(risk_scores)[::-1]
cluster_map = {cluster: ["High", "Medium", "Low"][i] for i, cluster in enumerate(cluster_order)}
student_df["predicted_risk"] = student_df["cluster"].map(cluster_map)

# ---------------------
# Reason and Tips
# ---------------------
def get_reason(row):
    reasons = []
    if row["attendance_rate"] < 0.75: reasons.append("Low attendance")
    if row["gpa"] < 2.0: reasons.append("Low GPA")
    if row["assignment_completion"] < 0.7: reasons.append("Low assignment completion")
    if row["lms_activity"] < 0.5: reasons.append("Low LMS activity")
    return ", ".join(reasons) or "No major concerns"

def get_tips(row):
    tips = []
    if row["gpa"] < 2.0: tips.append("Attend tutoring.")
    if row["attendance_rate"] < 0.75: tips.append("Attend classes regularly.")
    if row["assignment_completion"] < 0.7: tips.append("Submit assignments.")
    if row["lms_activity"] < 0.5: tips.append("Increase LMS engagement.")
    return " ".join(tips) or "Keep up the good work!"

student_df["risk_reason"] = student_df.apply(get_reason, axis=1)
student_df["study_tips"] = student_df.apply(get_tips, axis=1)

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
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Student Risk Dashboard", layout="wide")
st.title("🎓 AI-Enhanced Student Risk Dashboard")
st.markdown("Track at-risk students, analyze academic progress, and intervene early.")

# Sidebar Filters
st.sidebar.title("🔍 Filter Options")
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
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "📋 Student Table", 
            "📚 Student Detail - Courses", 
            "📥 Download"
        ])

        # Overview
        with tab1:
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Total Students", len(filtered_df))
            col2.metric("🔥 High Risk", (filtered_df["predicted_risk"] == "High").sum())
            col3.metric("🕒 Behind Schedule", (filtered_df["schedule_status"] == "Behind Schedule").sum())

            st.markdown("### 📈 Risk Distribution")
            fig = px.pie(filtered_df, names="predicted_risk", title="Risk Levels")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🗓️ Schedule Status")
            sched_fig = px.bar(
                filtered_df["schedule_status"].value_counts().reset_index(),
                x="index", y="schedule_status", color="index",
                labels={"index": "Status", "schedule_status": "Count"},
                color_discrete_map={"Behind Schedule": "red", "On Track": "green", "Unknown": "gray"}
            )
            st.plotly_chart(sched_fig, use_container_width=True)

        # Student Summary Table
        with tab2:
            st.markdown("### 📋 Student Summary Table")
            display_cols = [
                "student_id", "attendance_rate", "gpa", "assignment_completion", "lms_activity",
                "predicted_risk", "risk_reason", "study_tips", "progress_percentage",
                "expected_progress", "required_credits", "completed_credits", "schedule_status"
            ]
            st.dataframe(filtered_df[display_cols], use_container_width=True)

        # Student Course Detail
        with tab3:
            st.markdown("### 📚 Course-wise Student Performance")
            st.dataframe(filtered_courses_df, use_container_width=True)

        # Download
        with tab4:
            st.markdown("### 📥 Download Reports")
            st.download_button("Download Student Summary", data=filtered_df.to_csv(index=False),
                               file_name="student_summary.csv", mime="text/csv")
            st.download_button("Download Course-wise Details", data=filtered_courses_df.to_csv(index=False),
                               file_name="course_wise_detail.csv", mime="text/csv")
else:
    st.info("ℹ️ Please enter your role and ID to continue.")
