import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import FakeListLLM

# ---------------------
# Load Datasets
# ---------------------
student_df = pd.read_csv("student_risk_predictions.csv")
mapping_df = pd.read_csv("advisor_student_mapping.csv")
degree_df = pd.read_csv("degree_progress_500.csv")

# Convert student_id to string for merging
student_df["student_id"] = student_df["student_id"].astype(str)
degree_df["student_id"] = degree_df["student_id"].astype(str)

# Merge degree progress into main student dataset
student_df = pd.merge(student_df, degree_df, on="student_id", how="left")

# ---------------------
# Feature Engineering & Clustering
# ---------------------
features = ["attendance_rate", "gpa", "assignment_completion", "lms_activity"]
X = student_df[features]
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
student_df["cluster"] = clusters

# Cluster Risk Mapping
cluster_centers = kmeans.cluster_centers_
risk_scores = [
    (1 - center[features.index("attendance_rate")]) +
    (1 - center[features.index("gpa")] / 4.0) +
    (1 - center[features.index("assignment_completion")]) +
    (1 - center[features.index("lms_activity")])
    for center in cluster_centers
]

sorted_clusters = np.argsort(risk_scores)[::-1]
risk_labels = ["High", "Medium", "Low"]
cluster_to_risk = {cluster_idx: risk_labels[rank] for rank, cluster_idx in enumerate(sorted_clusters)}
student_df["Predicted Risk"] = student_df["cluster"].map(cluster_to_risk)

# ---------------------
# Reason Generation
# ---------------------
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
        tips.append("Attend tutoring sessions.")
    if row["attendance_rate"] < 0.75:
        tips.append("Improve attendance.")
    if row["assignment_completion"] < 0.7:
        tips.append("Submit assignments on time.")
    if row["lms_activity"] < 0.5:
        tips.append("Engage with online resources.")
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

# ---------------------
# Dummy LLM (for offline demo)
# ---------------------
class SimpleLLM(FakeListLLM):
    def __init__(self):
        super().__init__(responses=["I'll help you! Please check student's risk, schedule, and credits."])

llm = SimpleLLM()
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Student Risk Dashboard", layout="wide")
st.title("🎓 Student Risk Prediction Dashboard (Unsupervised Clustering)")

role = st.selectbox("Select your role:", ["advisor", "chair"])
user_id = st.text_input(f"Enter your {role} ID:")

if user_id:
    if role == "advisor":
        allowed_students = mapping_df[mapping_df["advisor_id"] == user_id]["student_id"].tolist()
    else:
        allowed_students = mapping_df[mapping_df["program_chair_id"] == user_id]["student_id"].tolist()

    filtered_df = student_df[student_df["student_id"].isin(allowed_students)]

    if not filtered_df.empty:
        st.subheader("📊 Assigned Student Risk Details")
        st.dataframe(filtered_df[[
            "student_id", "attendance_rate", "gpa", "assignment_completion", "lms_activity",
            "Predicted Risk", "Risk Reason", "Study Tips",
            "progress_percentage", "required_credits", "completed_credits", "Schedule Status"
        ]])

        st.markdown("### 📈 Risk Level Distribution")
        fig = px.pie(filtered_df, names="Predicted Risk", title="Risk Level Distribution")
        st.plotly_chart(fig)

        st.markdown("### 🗓️ Schedule Status Summary")
        status_counts = filtered_df["Schedule Status"].value_counts().reset_index()
        status_counts.columns = ['Schedule Status', 'Count']
        fig2 = px.bar(status_counts, x="Schedule Status", y="Count",
                      color="Schedule Status",
                      color_discrete_map={"Behind Schedule": "red", "On Track": "green", "Unknown": "gray"})
        st.plotly_chart(fig2)

        st.download_button("📥 Download CSV", data=filtered_df.to_csv(index=False), file_name="risk_report.csv")

        st.markdown("### 🔍 Student Details")
        for _, row in filtered_df.iterrows():
            with st.expander(f"Student ID: {row['student_id']}"):
                st.write({col: row[col] for col in features + [
                    'Predicted Risk', 'Risk Reason', 'Study Tips',
                    'progress_percentage', 'completed_credits', 'required_credits', 'Schedule Status'
                ]})

        st.markdown("### 💬 Ask the Advisor Bot")
        user_input = st.text_input("Ask something:")
        if user_input:
            if "how many credits" in user_input.lower() and "need" in user_input.lower():
                sid = None
                for student in filtered_df["student_id"]:
                    if student in user_input:
                        sid = student
                        break
                if sid:
                    student_row = filtered_df[filtered_df["student_id"] == sid].iloc[0]
                    needed = student_row["required_credits"] - student_row["completed_credits"]
                    st.success(f"Student {sid} needs {needed} more credits to graduate.")
                else:
                    st.warning("Student ID not found in your query.")
            else:
                response = conversation.predict(input=user_input)
                st.success(response)
    else:
        st.warning("No students assigned to this user ID.")
