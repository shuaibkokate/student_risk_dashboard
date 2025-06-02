import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Student Risk Dashboard", layout="wide")

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
# Aggregation
# ---------------------
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]
agg_funcs = {
    "attendance_rate": "mean",
    "gpa": "mean",
    "assignment_completion": "mean",
    "lms_activity": "mean",
    "progress_percentage": "mean",
    "expected_progress": "mean",
    "required_credits": "sum",
    "completed_credits": "sum"
}
valid_agg_funcs = {k: v for k, v in agg_funcs.items() if k in degree_df.columns}
aggregated = degree_df.groupby("student_id").agg(valid_agg_funcs).reset_index()
student_df = pd.merge(student_df.drop(columns=features, errors='ignore'), aggregated, on="student_id", how="left")

# Fill missing values with column means
for col in features:
    if col not in student_df.columns:
        st.warning(f"⚠️ Column '{col}' not found. Filling with 0.")
        student_df[col] = 0.0
    else:
        student_df[col] = student_df[col].fillna(student_df[col].mean())

# ---------------------
# Custom Thresholds
# ---------------------
st.sidebar.markdown("### 🔢 Custom Risk Thresholds")
threshold_attendance = st.sidebar.slider("Minimum Attendance Rate", 0.0, 1.0, 0.75, 0.01)
threshold_gpa = st.sidebar.slider("Minimum GPA", 0.0, 4.0, 2.0, 0.1)
threshold_assignment = st.sidebar.slider("Minimum Assignment Completion", 0.0, 1.0, 0.7, 0.01)
threshold_lms = st.sidebar.slider("Minimum LMS Activity", 0.0, 1.0, 0.5, 0.01)

# ---------------------
# Clustering
# ---------------------
X = student_df[features].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
student_df["cluster"] = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_
risk_scores = [(1 - c[0]) + (1 - c[1]/4.0) + (1 - c[2]) + (1 - c[3]) for c in centroids]
cluster_order = np.argsort(risk_scores)[::-1]
risk_levels = ["High", "Medium", "Low"]
cluster_map = {cluster: risk_levels[i] for i, cluster in enumerate(cluster_order)}
student_df["predicted_risk"] = student_df["cluster"].map(cluster_map)

# Cluster diagnostics
st.sidebar.markdown("### 🔍 Cluster Distribution")
cluster_counts = student_df["cluster"].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    st.sidebar.write(f"Cluster {cluster_id}: {count} students → Risk: {cluster_map[cluster_id]}")

# ---------------------
# Reason and Tips
# ---------------------
def get_reason(row):
    reasons = []
    if row["attendance_rate"] < threshold_attendance: reasons.append("Low attendance")
    if row["gpa"] < threshold_gpa: reasons.append("Low GPA")
    if row["assignment_completion"] < threshold_assignment: reasons.append("Low assignment completion")
    if row["lms_activity"] < threshold_lms: reasons.append("Low LMS activity")
    return ", ".join(reasons) or "No major concerns"

def get_tips(row):
    tips = []
    if row["gpa"] < threshold_gpa: tips.append("Attend tutoring.")
    if row["attendance_rate"] < threshold_attendance: tips.append("Attend classes regularly.")
    if row["assignment_completion"] < threshold_assignment: tips.append("Submit assignments.")
    if row["lms_activity"] < threshold_lms: tips.append("Increase LMS engagement.")
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
st.title("🎓 AI-Enhanced Student Risk Dashboard")
st.markdown("Track at-risk students, analyze academic progress, and intervene early.")

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
        tab1, tab2, tab4 = st.tabs(["📊 Overview", "📋 Student Table", "📥 Download"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            all_risks = ["High", "Medium", "Low"]
            risk_count = filtered_df["predicted_risk"].value_counts().reindex(all_risks, fill_value=0).reset_index()
            risk_count.columns = ["predicted_risk", "count"]

            col1.metric("🎯 Total Students", len(filtered_df))
            col2.metric("🔥 High Risk", risk_count[risk_count["predicted_risk"] == "High"]["count"].values[0])
            col3.metric("🔹 Medium Risk", risk_count[risk_count["predicted_risk"] == "Medium"]["count"].values[0])
            st.metric("✅ Low Risk", risk_count[risk_count["predicted_risk"] == "Low"]["count"].values[0])

            st.markdown("### 📈 Risk Distribution")
            fig = px.pie(risk_count, names="predicted_risk", values="count", title="Risk Levels")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📊 Average GPA by Risk Category")
            bar_fig = px.bar(
                filtered_df.groupby("predicted_risk")["gpa"].mean().reindex(["High", "Medium", "Low"]).reset_index(),
                x="predicted_risk", y="gpa", color="predicted_risk",
                title="Average GPA for Each Risk Category",
                labels={"predicted_risk": "Risk Category", "gpa": "Average GPA"},
                color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"}
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            st.markdown("### 🗓️ Schedule Status")
            sched_df = filtered_df["schedule_status"].value_counts().reset_index()
            sched_df.columns = ["status", "count"]
            sched_fig = px.bar(
                sched_df,
                x="status", y="count", color="status",
                color_discrete_map={
                    "Behind Schedule": "red", 
                    "On Track": "green", 
                    "Unknown": "gray"
                }
            )
            st.plotly_chart(sched_fig, use_container_width=True)

        with tab2:
            st.markdown("### 📋 Student Summary Table")
            display_cols = [
                "student_id", "attendance_rate", "gpa", "assignment_completion", "lms_activity",
                "predicted_risk", "risk_reason", "study_tips", "progress_percentage",
                "expected_progress", "required_credits", "completed_credits", "schedule_status"
            ]
            existing_cols = [col for col in display_cols if col in filtered_df.columns]
            summary_df = filtered_df[existing_cols]
            st.dataframe(summary_df, use_container_width=True)

            st.markdown("### 📚 Course-wise Performance for Selected Student")
            selected_student = st.selectbox("Select a student to view course details:", filtered_df["student_id"].unique())
            selected_course_df = filtered_courses_df[filtered_courses_df["student_id"] == selected_student]
            if not selected_course_df.empty:
                st.dataframe(selected_course_df, use_container_width=True, height=300)
            else:
                st.info("No course data available for this student.")

            st.markdown("### 📚 Course-wise Performance for Selected Student")
            selected_course_df = filtered_courses_df[filtered_courses_df["student_id"] == selected_student]
            if not selected_course_df.empty:
                st.dataframe(selected_course_df, use_container_width=True, height=300)
            else:
                st.info("No course data available for this student.")

        

        with tab4:
            st.markdown("### 📅 Download Reports")
            st.download_button("Download Student Summary", data=filtered_df.to_csv(index=False),
                               file_name="student_summary.csv", mime="text/csv")
            st.download_button("Download Course-wise Details", data=filtered_courses_df.to_csv(index=False),
                               file_name="course_wise_detail.csv", mime="text/csv")
else:
    st.info("ℹ️ Please enter your role and ID to continue.")
