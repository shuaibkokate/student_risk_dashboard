import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Student Risk Dashboard", layout="wide")

# Load data
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")
degree_df = pd.read_csv("degree_progress_500.csv")

# Normalize column names
student_df.columns = student_df.columns.str.strip().str.lower()
mapping_df.columns = mapping_df.columns.str.strip().str.lower()
degree_df.columns = degree_df.columns.str.strip().str.lower()

# Ensure student_id is string
student_df["student_id"] = student_df["student_id"].astype(str)
mapping_df["student_id"] = mapping_df["student_id"].astype(str)
degree_df["student_id"] = degree_df["student_id"].astype(str)

# Sidebar - Thresholds
st.sidebar.markdown("### 🔢 Custom Risk Thresholds")
threshold_attendance = st.sidebar.slider("Minimum Attendance Rate", 0.0, 1.0, 0.75, 0.01)
threshold_gpa = st.sidebar.slider("Minimum GPA", 0.0, 4.0, 2.0, 0.1)
threshold_assignment = st.sidebar.slider("Minimum Assignment Completion", 0.0, 1.0, 0.7, 0.01)

# Filter based on role and ID
st.sidebar.title("🔍 Filter Options")
role = st.sidebar.selectbox("Select Role", ["advisor", "chair"])
user_id = st.sidebar.text_input(f"Enter your {role} ID")

if user_id:
    if role == "advisor":
        assigned_ids = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    else:
        assigned_ids = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    student_df = student_df[student_df["student_id"].isin(assigned_ids)]
    degree_df = degree_df[degree_df["student_id"].isin(assigned_ids)]

    # Fill missing columns
    for col in ["attendance_rate", "gpa", "assignment_completion"]:
        for df in [student_df, degree_df]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(df[col].mean())

    # Risk Prediction
    def predict_risk(row):
        if row["attendance_rate"] < threshold_attendance or row["gpa"] < threshold_gpa or row["assignment_completion"] < threshold_assignment:
            if row["gpa"] < threshold_gpa - 0.5 or row["attendance_rate"] < threshold_attendance - 0.1:
                return "High"
            else:
                return "Medium"
        return "Low"

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

    def schedule_flag(row):
        if pd.isnull(row["progress_percentage"]) or pd.isnull(row["expected_progress"]):
            return "Unknown"
        if row["progress_percentage"] < row["expected_progress"] - 10:
            return "Behind Schedule"
        return "On Track"

    for df in [student_df, degree_df]:
        df["predicted_risk"] = df.apply(predict_risk, axis=1)
        df["risk_reason"] = df.apply(get_reason, axis=1)
        df["study_tips"] = df.apply(get_tips, axis=1)

    student_df["schedule_status"] = student_df.apply(schedule_flag, axis=1)

    # Dashboard
    st.title("🎓 AI-Enhanced Student Risk Dashboard")

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📋 Student Table", "📥 Download"])

    with tab1:
        summary = student_df["predicted_risk"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0).reset_index()
        summary.columns = ["Risk", "Count"]
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Total Students", len(student_df))
        col2.metric("🔥 High Risk", summary[summary["Risk"] == "High"]["Count"].values[0])
        col3.metric("🔹 Medium Risk", summary[summary["Risk"] == "Medium"]["Count"].values[0])
        st.metric("✅ Low Risk", summary[summary["Risk"] == "Low"]["Count"].values[0])

        st.markdown("### 📊 Risk Category Distribution (Bar Chart)")
        bar_fig = px.bar(
            summary,
            x="Risk",
            y="Count",
            color="Risk",
            title="Risk Category Distribution",
            text_auto=True
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        st.markdown("### 📈 GPA vs Attendance (Risk Colored)")
        fig = px.scatter(
            student_df,
            x="gpa",
            y="attendance_rate",
            color="predicted_risk",
            size="assignment_completion",
            hover_data=["student_id"],
            title="Scatter Plot of GPA vs Attendance Colored by Risk Level"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🕒 Schedule Status Distribution")
        schedule_summary = student_df["schedule_status"].value_counts().reset_index()
        schedule_summary.columns = ["Status", "Count"]
        schedule_fig = px.pie(
            schedule_summary,
            names="Status",
            values="Count",
            title="Schedule Status of Students"
        )
        st.plotly_chart(schedule_fig, use_container_width=True)

    with tab2:
        st.markdown("### 📋 Student Summary")
        st.dataframe(student_df, use_container_width=True, height=400)

        st.markdown("### 📚 Course-wise Performance")
        selected_student = st.selectbox("Select Student", student_df["student_id"].unique())
        selected_courses = degree_df[degree_df["student_id"] == selected_student]
        st.dataframe(selected_courses, use_container_width=True, height=400)

    with tab3:
        st.download_button("📥 Download Student Summary", student_df.to_csv(index=False), file_name="student_summary.csv")
        st.download_button("📥 Download Course-wise Data", degree_df.to_csv(index=False), file_name="course_data.csv")
else:
    st.info("ℹ️ Enter your role and ID to begin.")
