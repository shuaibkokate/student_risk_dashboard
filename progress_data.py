import streamlit as st
import pandas as pd

# Load degree progress data (course-level)
degree_df = pd.read_csv("degree_progress_500.csv")

st.title("🎓 Academic Advisor Dashboard")

# Student ID selection
student_ids = degree_df['student_id'].unique()
selected_student = st.selectbox("Select a student:", student_ids)

# Student-Level Aggregation from Course-Wise Data (Safe)
agg_funcs_all = {
    "attendance_rate": "mean",
    "gpa": "mean",
    "assignment_completion": "mean",
    "lms_activity": "mean",
    "progress_percentage": "mean",
    "expected_progress": "mean",
    "required_credits": "sum",
    "completed_credits": "sum"
}

available_columns = degree_df.columns.tolist()
agg_funcs = {col: func for col, func in agg_funcs_all.items() if col in available_columns}
missing_columns = [col for col in agg_funcs_all.keys() if col not in available_columns]

if missing_columns:
    st.warning(f"⚠️ Warning: The following columns are missing from 'degree_progress_500.csv' and will be skipped in aggregation: {', '.join(missing_columns)}")

aggregated = degree_df.groupby("student_id").agg(agg_funcs).reset_index()

# Merge with selected student data
student_data = aggregated[aggregated['student_id'] == selected_student]

# Compute schedule status
if 'progress_percentage' in student_data.columns and 'expected_progress' in student_data.columns:
    def calculate_schedule_status(row):
        if row['progress_percentage'] >= row['expected_progress']:
            return "On Track"
        elif row['progress_percentage'] >= row['expected_progress'] - 10:
            return "Slightly Behind"
        else:
            return "Behind"
    student_data['schedule_status'] = student_data.apply(calculate_schedule_status, axis=1)
else:
    student_data['schedule_status'] = "N/A"

# Show student summary tab
st.subheader("📋 Student Summary")
st.dataframe(student_data)

# Show student courses tab
st.subheader("📚 Student Course Details")
student_courses = degree_df[degree_df['student_id'] == selected_student]
st.dataframe(student_courses)

# Download for all students (aggregated)
st.subheader("⬇️ Download Full Aggregated Student Data")
st.download_button("Download CSV", aggregated.to_csv(index=False), file_name="student_summary_aggregated.csv")
