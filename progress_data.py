import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px

# ---------------------
# Load and Clean Data
# ---------------------
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")
degree_df = pd.read_csv("degree_progress_500.csv")

# Normalize all column names
student_df.columns = student_df.columns.str.strip().str.lower()
degree_df.columns = degree_df.columns.str.strip().str.lower()
mapping_df.columns = mapping_df.columns.str.strip().str.lower()

# Ensure IDs are strings
student_df["student_id"] = student_df["student_id"].astype(str)
degree_df["student_id"] = degree_df["student_id"].astype(str)

# Merge progress data
student_df = pd.merge(student_df, degree_df, on="student_id", how="left")

# ---------------------
# Add Missing Columns if Absent
# ---------------------
for col in ["progress_percentage", "expected_progress", "required_credits", "completed_credits"]:
    if col not in student_df.columns:
        student_df[col] = np.nan

# ---------------------
# Feature Engineering and Clustering
# ---------------------
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]
X = student_df[features]
kmeans = KMeans(n_clusters=3, random_state=42)
student_df["cluster"] = kmeans.fit_predict(X)

# Risk score ranking
centers = kmeans.cluster_centers_
risk_scores = [
    (1 - c[0]) + (1 - c[1]/4.0) + (1 - c[2]) + (1 - c[3])
    for c in centers
]
rank = np.argsort(risk_scores)[::-1]
cluster_to_risk = {clust: ["High", "Medium", "Low"][i] for i, clust in enumerate(rank)}
student_df["predicted_risk"] = student_df["cluster"].map(cluster_to_risk)

# ---------------------
# Risk Reason and Tips
# ---------------------
def get_reason(row):
    reasons = []
    if row["attendance_rate"] < 0.75:
        reasons.append(f"Low attendance ({row['attendance_rate']:.0%})")
    if row["gpa"] < 2.0:
        reasons.append(f"Low GPA ({row['gpa']:.2f})")
    if row["assignment_completion"] < 0.7:
        reasons.append(f"Low assignment completion ({row['assignment_completion']:.0%})")
    if row["lms_activity"] < 0.5:
        reasons.append(f"Low LMS activity ({row['lms_activity']:.0%})")
    return " & ".join(reasons) or "No major concerns"

def get_tips(row):
    tips = []
    if row["gpa"] < 2.0:
        tips.append("Attend tutoring.")
    if row["attendance_rate"] < 0.75:
        tips.append("Attend classes regularly.")
    if row["assignment_completion"] < 0.7:
        tips.append("Submit assignments.")
    if row["lms_activity"] < 0.5:
        tips.append("Use LMS more actively.")
    return " ".join(tips) or "Doing well!"

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
st.title("🎓 Student Risk Dashboard")

role = st.selectbox("Select your role:", ["advisor", "chair"])
user_id = st.text_input(f"Enter your {role} ID:")

if user_id:
    if role == "advisor":
        assigned_ids = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].astype(str).tolist()
    else:
        assigned_ids = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].astype(str).tolist()

    filtered_df = student_df[student_df["student_id"].isin(assigned_ids)]

    if not filtered_df.empty:
        st.subheader("📊 Student Details")
        display_cols = [
            "student_id", "attendance_rate", "gpa", "assignment_completion", "lms_activity",
            "predicted_risk", "risk_reason", "study_tips",
        ]

        # Safely add progress columns if present
        for col in ["progress_percentage", "expected_progress", "required_credits", "completed_credits", "schedule_status"]:
            if col in filtered_df.columns:
                display_cols.append(col)

        st.dataframe(filtered_df[display_cols])

        st.markdown("### 📈 Risk Levels")
        fig = px.pie(filtered_df, names="predicted_risk")
        st.plotly_chart(fig)

        st.markdown("### 🗓️ Schedule Status")
        sched_fig = px.bar(
            filtered_df["schedule_status"].value_counts().reset_index(),
            x="index", y="schedule_status", color="index",
            labels={"index": "Status", "schedule_status": "Count"},
            color_discrete_map={"Behind Schedule": "red", "On Track": "green", "Unknown": "gray"}
        )
        st.plotly_chart(sched_fig)

        st.download_button("📥 Download CSV", filtered_df.to_csv(index=False), "student_risk_report.csv")

    else:
        st.warning("No students assigned to this user ID.")
