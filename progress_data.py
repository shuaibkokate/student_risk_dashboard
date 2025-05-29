import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px

# Load datasets
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")
degree_df = pd.read_csv("degree_progress_500.csv")

student_df["student_id"] = student_df["student_id"].astype(str)
degree_df["student_id"] = degree_df["student_id"].astype(str)

# Merge degree progress into student data
student_df = pd.merge(student_df, degree_df, on="student_id", how="left")

features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]

X = student_df[features]

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
student_df["cluster"] = clusters

cluster_centers = kmeans.cluster_centers_
risk_scores = []
for center in cluster_centers:
    score = (1 - center[features.index("attendance_rate")]) + \
            (1 - center[features.index("gpa")] / 4.0) + \
            (1 - center[features.index("assignment_completion")]) + \
            (1 - center[features.index("lms_activity")])
    risk_scores.append(score)

sorted_clusters = np.argsort(risk_scores)[::-1]

risk_labels = ["High", "Medium", "Low"]
cluster_to_risk = {cluster_idx: risk_labels[rank] for rank, cluster_idx in enumerate(sorted_clusters)}
student_df["Predicted Risk"] = student_df["cluster"].map(cluster_to_risk)

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

def generate_study_tips(row):
    tips = []
    if row["gpa"] < 2.0:
        tips.append("Attend tutoring and review foundational materials.")
    if row["attendance_rate"] < 0.75:
        tips.append("Improve attendance and class engagement.")
    if row["assignment_completion"] < 0.7:
        tips.append("Complete and submit assignments on time.")
    if row["lms_activity"] < 0.5:
        tips.append("Increase participation in online materials.")
    return " ".join(tips) if tips else "Keep up the good work!"

student_df["Study Tips"] = student_df.apply(generate_study_tips, axis=1)

def flag_behind_schedule(row):
    if pd.isnull(row['progress_percentage']) or pd.isnull(row['expected_progress']):
        return "Unknown"
    elif row['progress_percentage'] < row['expected_progress'] - 10:
        return "Behind Schedule"
    else:
        return "On Track"

student_df["Schedule Status"] = student_df.apply(flag_behind_schedule, axis=1)

st.set_page_config(page_title="Student Risk Predictor (Clustering)", layout="wide")
st.title("🎓 Student Risk Prediction Dashboard")

# Sidebar inputs
st.sidebar.header("User Access")
role = st.sidebar.selectbox("Select your role:", ["advisor", "chair"])
user_id = st.sidebar.text_input(f"Enter your {role} ID:")

if user_id:
    if role == "advisor":
        allowed_students = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    else:
        allowed_students = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    filtered_df = student_df[student_df["student_id"].isin(allowed_students)]

    if filtered_df.empty:
        st.warning("No students found for this user ID.")
    else:
        # Summary metrics
        st.markdown("### Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Students", len(filtered_df))
        col2.metric("High Risk", (filtered_df["Predicted Risk"] == "High").sum())
        col3.metric("Medium Risk", (filtered_df["Predicted Risk"] == "Medium").sum())
        col4.metric("Low Risk", (filtered_df["Predicted Risk"] == "Low").sum())

        # Tabs for content
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔍 Detailed View", "📥 Download CSV", "💬 Ask Advisor Bot"])

        with tab1:
            st.subheader("Risk Level Distribution")
            fig = px.pie(filtered_df, names="Predicted Risk", title="Risk Level Distribution",
                         color="Predicted Risk",
                         color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Schedule Status Distribution")
            fig2 = px.bar(filtered_df["Schedule Status"].value_counts().reset_index(),
                          x="index", y="Schedule Status",
                          labels={"index": "Schedule Status", "Schedule Status": "Count"},
                          color="index",
                          color_discrete_map={"Behind Schedule": "red", "On Track": "green", "Unknown": "gray"})
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.subheader("Student Details")
            # Color map for risk and schedule status
            def risk_color(risk):
                return {
                    "High": "background-color: #ffcccc;",
                    "Medium": "background-color: #fff0b3;",
                    "Low": "background-color: #ccffcc;",
                }.get(risk, "")

            def schedule_color(status):
                return {
                    "Behind Schedule": "color: red; font-weight: bold;",
                    "On Track": "color: green; font-weight: bold;",
                    "Unknown": "color: gray;",
                }.get(status, "")

            display_cols = ["student_id"] + features + ["Predicted Risk", "Risk Reason", "Study Tips",
                                                       "progress_percentage", "required_credits", "completed_credits", "Schedule Status"]
            df_display = filtered_df[display_cols].copy()

            # Style dataframe with risk and schedule status colors
            styled_df = df_display.style.applymap(risk_color, subset=["Predicted Risk"]) \
                                       .applymap(schedule_color, subset=["Schedule Status"]) \
                                       .set_properties(**{'text-align': 'center'}) \
                                       .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
            st.dataframe(styled_df, use_container_width=True)

        with tab3:
            st.subheader("Download Filtered Data")
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", data=csv, file_name="student_risk_predictions.csv", mime='text/csv')

        with tab4:
            st.subheader("Ask the Advisor Bot (Coming Soon)")
            user_input = st.text_input("Ask a question about a student or general advising help:")
            if user_input:
                st.info("Advisor Bot feature is not yet implemented.")

else:
    st.info("Please select your role and enter your user ID from the sidebar to see student data.")
